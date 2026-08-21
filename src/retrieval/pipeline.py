import logging
import json
from collections import defaultdict
import pickle
from pathlib import Path
from typing import Dict, Any, Optional

from src.indexing.bm25 import bm25_search, bm25_search_subset, tokenize_for_bm25
from src.indexing.embedding import dense_search
from src.indexing.hybrid_search import reciprocal_rank_fusion
from src.query.pipeline import QueryPipeline
from src.retrieval.reranker import Reranker
from src.retrieval.confidence import RetrievalConfidenceChecker, QueryReformulator

logger = logging.getLogger(__name__)


class QueryRetrievalPipeline:
    def __init__(
        self,
        documents_path: str | Path,
        index_dir: str | Path | None = None,
        query_pipeline: Optional[QueryPipeline] = None,
        retrieval_config: Optional[dict] = None,
        embedding_config: Optional[dict] = None,
        reranker_enabled: bool = False,
        reranker_model: str = "BAAI/bge-reranker-v2-m3",
        confidence_threshold: float = 0.5,
        max_reformulate_attempts: int = 2,
        llm_client=None,
    ):
        self.documents_path = Path(documents_path)
        self.index_dir = Path(index_dir) if index_dir else self.documents_path.parent
        self.query_pipeline = query_pipeline
        self.retrieval_config = retrieval_config or {}
        self.embedding_config = embedding_config or {}
        self.max_reformulate_attempts = max_reformulate_attempts

        self.bm25 = None
        self.documents = []
        self.faiss_index = None
        self.dense_documents = []
        self.embed_model = None

        self._confidence_checker = RetrievalConfidenceChecker(bm25_score_threshold=confidence_threshold)
        self._reformulator = QueryReformulator(llm_client=llm_client)
        self._reranker = Reranker(model_name=reranker_model, enabled=reranker_enabled)

        self._by_ticker: dict[str, list[int]] = {}
        self._by_ticker_year: dict[tuple[str, int], list[int]] = {}
        self._by_ticker_year_report: dict[tuple[str, int, str], list[int]] = {}
        self._by_exact_key: dict[tuple[str, int, str, str], list[int]] = {}
        self._notes_by_ticker_year: dict[tuple[str, int], list[int]] = {}

        self._load_indices()
        self._build_metadata_index()

    def _load_indices(self):
        bm25_path = self.index_dir / "bm25.pkl"
        if bm25_path.exists():
            try:
                with open(bm25_path, "rb") as f:
                    payload = pickle.load(f)
                    self.bm25 = payload["bm25"]
                    self.documents = payload["documents"]
                logger.info("Loaded BM25 index from %s", bm25_path)
            except Exception as e:
                logger.error("Error loading BM25 index: %s", e)

        if self.bm25 is None and self.documents_path.exists():
            logger.info("Building BM25 on-the-fly from %s...", self.documents_path)
            try:
                with open(self.documents_path, "r", encoding="utf-8") as f:
                    self.documents = [json.loads(line) for line in f if line.strip()]
                if self.documents:
                    from rank_bm25 import BM25Okapi
                    tokenized = [tokenize_for_bm25(doc.get("text", doc.get("content", ""))) for doc in self.documents]
                    self.bm25 = BM25Okapi(tokenized)
            except Exception as e:
                logger.error("Error building on-the-fly BM25: %s", e)

        faiss_path = self.index_dir / "faiss" / "index.faiss"
        docs_path = self.index_dir / "faiss" / "documents.json"
        if self.retrieval_config.get("dense_enabled", False) and faiss_path.exists() and docs_path.exists():
            import faiss
            try:
                self.faiss_index = faiss.read_index(str(faiss_path))
                with open(docs_path, "r", encoding="utf-8") as f:
                    self.dense_documents = json.load(f)
                from src.indexing.embedding import _get_model
                model_name = self.embedding_config.get("model_name", "bkai-foundation-models/vietnamese-bi-encoder")
                self.embed_model = _get_model(model_name)
                logger.info("Loaded FAISS index from %s", faiss_path)
            except Exception as e:
                logger.error("Error loading FAISS index: %s", e)

    def _build_metadata_index(self) -> None:
        """Index ngược ticker/năm/loại báo cáo -> vị trí document, để lọc trước khi xếp hạng."""
        by_ticker = defaultdict(list)
        by_ticker_year = defaultdict(list)
        by_ticker_year_report = defaultdict(list)
        by_exact_key = defaultdict(list)
        notes_by_ticker_year = defaultdict(list)

        for i, doc in enumerate(self.documents):
            meta = doc.get("metadata", {})
            ticker = str(meta.get("ticker", "")).upper()
            if not ticker:
                continue
            by_ticker[ticker].append(i)

            period = meta.get("period") or meta.get("year")
            try:
                period = int(period)
            except (TypeError, ValueError):
                continue
            by_ticker_year[(ticker, period)].append(i)

            report_type = str(meta.get("report_type", "")).lower()
            if report_type:
                by_ticker_year_report[(ticker, period, report_type)].append(i)

            section = meta.get("section")
            item_code = meta.get("item_code")
            if section and item_code is not None:
                by_exact_key[(ticker, period, str(section), str(item_code))].append(i)

            if meta.get("document_type") == "notes":
                notes_by_ticker_year[(ticker, period)].append(i)

        self._by_ticker = dict(by_ticker)
        self._by_ticker_year = dict(by_ticker_year)
        self._by_ticker_year_report = dict(by_ticker_year_report)
        self._by_exact_key = dict(by_exact_key)
        self._notes_by_ticker_year = dict(notes_by_ticker_year)
        logger.info("Metadata index: %d tickers, %d (ticker, year) cặp",
                    len(self._by_ticker), len(self._by_ticker_year))

    def _exact_lookup(self, query_result) -> list[dict]:
        """Lấy thẳng dòng khớp ticker + period + section + item_code.

        Với câu hỏi suy diễn (ROE, tăng trưởng…) một truy vấn BM25 duy nhất không
        thể kéo về đồng thời mọi thành phần của công thức, nên phải tra riêng
        từng thành phần rồi mới ghép lại.
        """
        if query_result is None or not self._by_exact_key:
            return []

        wanted_report = query_result.entities.report_type or "consolidated"
        hits: list[dict] = []
        seen: set[int] = set()

        for rq in query_result.retrieval_queries:
            ticker = str(rq.ticker).upper()
            key = (ticker, int(rq.year), str(rq.section), str(rq.indicator_code))
            indices = self._by_exact_key.get(key, [])
            if not indices:
                continue

            # Ưu tiên đúng loại báo cáo, nếu không có thì lấy tất cả
            preferred = [
                i for i in indices
                if str(self.documents[i].get("metadata", {}).get("report_type", "")).lower() == wanted_report
            ]
            for i in (preferred or indices):
                if i in seen:
                    continue
                seen.add(i)
                # Điểm cao hơn mọi kết quả BM25 để luôn đứng đầu danh sách
                hits.append({"score": 1000.0, "document": self.documents[i], "match": "exact"})

        if hits:
            logger.info("Exact match: %d dòng khớp (ticker, năm, section, mã chỉ tiêu)", len(hits))
        return hits

    def _candidate_indices(self, filters: dict) -> Optional[list[int]]:
        """Tập document ứng viên theo bộ lọc, nới lỏng dần khi quá hẹp.

        Trả về None nghĩa là không lọc được gì -> tìm trên toàn corpus.
        """
        tickers = [t.upper() for t in filters.get("tickers", []) if t]
        if not tickers or not self._by_ticker:
            return None

        years = [y for y in filters.get("years", []) if y]
        report_type = filters.get("report_type")

        # Chỉ tiêu không có trong từ điển schema -> số liệu nằm trong thuyết minh.
        # Giới hạn vào chunk thuyết minh, nếu không các dòng bảng chính cùng ticker
        # sẽ chiếm hết top-k chỉ vì trùng vài từ khoá.
        if filters.get("sections") == ["NOTES"] and years and self._notes_by_ticker_year:
            hits = [
                i for t in tickers for y in years
                for i in self._notes_by_ticker_year.get((t, int(y)), [])
            ]
            if hits:
                logger.debug("Giới hạn tìm kiếm trong %d chunk thuyết minh", len(hits))
                return sorted(set(hits))

        # Chặt nhất: ticker + năm + loại báo cáo
        if years and report_type:
            hits = [
                i for t in tickers for y in years
                for i in self._by_ticker_year_report.get((t, int(y), report_type), [])
            ]
            if hits:
                return sorted(set(hits))
            logger.debug("Không có doc cho report_type=%s, bỏ ràng buộc này", report_type)

        if years:
            hits = [i for t in tickers for y in years for i in self._by_ticker_year.get((t, int(y)), [])]
            if hits:
                return sorted(set(hits))
            logger.debug("Không có doc cho years=%s, bỏ ràng buộc năm", years)

        hits = [i for t in tickers for i in self._by_ticker.get(t, [])]
        return sorted(set(hits)) if hits else None

    @staticmethod
    def _dedupe_hits(hits: list[dict]) -> list[dict]:
        """Bỏ các hit trùng nhau (cùng chỉ tiêu, cùng giá trị) để không chiếm chỗ top-k."""
        seen = set()
        unique = []
        for h in hits:
            meta = h.get("document", h).get("metadata", {})
            key = (
                meta.get("ticker"), meta.get("period"), meta.get("section"),
                meta.get("item_code"), meta.get("value"), meta.get("report_type"),
                None if meta.get("item_code") else h.get("document", h).get("chunk_id"),
            )
            if key in seen:
                continue
            seen.add(key)
            unique.append(h)
        return unique

    @staticmethod
    def _rank_by_intent(hits: list[dict], query_result) -> list[dict]:
        """Đưa dòng khớp đúng ý định câu hỏi lên đầu.

        BM25 chỉ đo độ giống chữ, nên "Chi phí quản lý doanh nghiệp" có thể xếp
        trên "Chi phí khác" dù câu hỏi đã xác định rõ mã chỉ tiêu IS.32. Khi đã
        biết (section, item_code) và loại báo cáo thì ưu tiên chúng trước điểm BM25.
        """
        if query_result is None or not hits:
            return hits

        wanted_codes = {
            (q.section, str(q.indicator_code)) for q in query_result.retrieval_queries
        }
        wanted_years = set(query_result.entities.years)
        # Không nói rõ thì mặc định lấy báo cáo hợp nhất
        wanted_report = query_result.entities.report_type or "consolidated"

        def sort_key(hit: dict):
            meta = hit.get("document", hit).get("metadata", {})
            section = meta.get("section")
            item_code = str(meta.get("item_code") or "")
            period = meta.get("period")

            code_match = (section, item_code) in wanted_codes
            year_match = (not wanted_years) or (period in wanted_years)
            report_match = str(meta.get("report_type", "")).lower() == wanted_report
            has_value = meta.get("value") is not None

            return (
                not (code_match and year_match),   # khớp mã chỉ tiêu + đúng năm
                not report_match,
                not has_value,
                -float(hit.get("score", 0.0)),
            )

        return sorted(hits, key=sort_key)

    def _filter_results(self, results: list[dict], metadata_filters: dict) -> list[dict]:
        if not metadata_filters:
            return results

        tickers = [t.upper() for t in metadata_filters.get("tickers", [])]
        years = metadata_filters.get("years", [])
        filtered = []

        for r in results:
            doc = r.get("document", r)
            meta = doc.get("metadata", {})

            if tickers:
                doc_ticker = str(meta.get("ticker", "")).upper()
                if doc_ticker and doc_ticker not in tickers:
                    continue

            if years:
                # Ưu tiên `period`: bảng có year=2015 nhưng chứa cột so sánh period=2014,
                # lấy `year` trước sẽ loại nhầm đúng dòng đang cần.
                doc_year = meta.get("period")
                if doc_year is None:
                    doc_year = meta.get("year")
                if doc_year and int(doc_year) not in years:
                    continue

            filtered.append(r)
        return filtered

    def _search(self, search_text: str, top_k: int, filters: Optional[dict] = None) -> tuple[list, list]:
        bm25_res = []
        if self.bm25 and self.documents:
            candidates = self._candidate_indices(filters or {})
            bm25_res = bm25_search_subset(
                search_text, self.bm25, self.documents, candidates, top_k=top_k * 2
            )

        dense_res = []
        if self.faiss_index and self.embed_model and self.dense_documents:
            dense_res = dense_search(search_text, self.embed_model, self.faiss_index, self.dense_documents, top_k=top_k * 2)

        return bm25_res, dense_res

    def _fuse_and_filter(self, bm25_res: list, dense_res: list, filters: dict, top_k: int) -> tuple[list, list]:
        fused = reciprocal_rank_fusion(bm25_res, dense_res, top_k=top_k * 2) if (bm25_res and dense_res) else (bm25_res or dense_res)
        return fused, self._filter_results(fused, filters)

    def process(self, question: str, top_k_per_query: int = 8) -> Dict[str, Any]:
        query_result = self.query_pipeline.process(question) if self.query_pipeline else None

        search_text = query_result.search_text if query_result else question
        filters = query_result.metadata_filters.to_dict() if query_result else {}
        entities = query_result.entities.to_dict() if query_result else {}

        final_hits = []
        confidence_ok = False

        for attempt in range(self.max_reformulate_attempts + 1):
            bm25_res, dense_res = self._search(search_text, top_k_per_query, filters)
            raw_confident, top_score = self._confidence_checker.check_raw_results(bm25_res)
            fused, filtered = self._fuse_and_filter(bm25_res, dense_res, filters, top_k_per_query)
            filter_ok = self._confidence_checker.check_filtered_hits(filtered, len(fused))

            if filter_ok and filtered:
                final_hits = self._reranker.rerank(search_text, filtered, top_k=top_k_per_query)
                confidence_ok = True
                if attempt > 0:
                    logger.info("Re-query attempt %d succeeded", attempt)
                break

            if not filter_ok and fused:
                if filters.get("tickers"):
                    logger.info("Ticker filter '%s' returned 0 hits → no data", filters.get("tickers"))
                    final_hits = []
                    confidence_ok = False
                else:
                    final_hits = self._reranker.rerank(search_text, fused, top_k=top_k_per_query)
                    confidence_ok = raw_confident
                break

            if attempt < self.max_reformulate_attempts:
                new_text = self._reformulator.reformulate(question, entities, attempt)
                if new_text != search_text:
                    logger.info("Low confidence (score=%.3f), reformulate [%d]: '%s' → '%s'", top_score, attempt + 1, search_text[:40], new_text[:40])
                    search_text = new_text
                else:
                    break

        exact_hits = self._exact_lookup(query_result)
        if exact_hits:
            final_hits = exact_hits + final_hits
            confidence_ok = True

        final_hits = self._rank_by_intent(self._dedupe_hits(final_hits), query_result)

        # Công thức suy diễn cần đủ mọi thành phần: cắt cứng ở top_k sẽ làm mất
        # vế còn lại của phép tính, nên nới hạn mức theo số dòng khớp chính xác.
        limit = max(top_k_per_query, len(exact_hits))

        query_info: Dict[str, Any] = query_result.to_dict() if query_result else {"query_type": "single_lookup"}
        query_info["retrieval_confidence_ok"] = confidence_ok

        return {"query": query_info, "hits": final_hits[:limit]}
