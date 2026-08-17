import logging
import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional

from src.indexing.bm25 import bm25_search, tokenize_for_bm25
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

        self._load_indices()

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
                doc_year = meta.get("year") or meta.get("period")
                if doc_year and int(doc_year) not in years:
                    continue

            filtered.append(r)
        return filtered

    def _search(self, search_text: str, top_k: int) -> tuple[list, list]:
        bm25_res = []
        if self.bm25 and self.documents:
            bm25_res = bm25_search(search_text, self.bm25, self.documents, top_k=top_k * 2)

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
            bm25_res, dense_res = self._search(search_text, top_k_per_query)
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

        query_info: Dict[str, Any] = query_result.to_dict() if query_result else {"query_type": "single_lookup"}
        query_info["retrieval_confidence_ok"] = confidence_ok

        return {"query": query_info, "hits": final_hits[:top_k_per_query]}
