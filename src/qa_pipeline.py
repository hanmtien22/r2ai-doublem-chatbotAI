import logging
from pathlib import Path
from typing import Dict, Any, List

from src.retrieval.pipeline import QueryRetrievalPipeline
from src.compute.code_generator import CodeGenerator
from src.compute.sandbox import Sandbox
from src.compute.result_verifier import ResultVerifier
from src.compute.retry_manager import RetryManager

from src.answer.answer_formatter import AnswerFormatter
from src.answer.citation_builder import CitationBuilder
from src.answer.refuse_handler import RefuseHandler
from src.llm.client import LLMClient

logger = logging.getLogger(__name__)


class FullQAPipeline:
    """Pipeline 4 phase: Query Understanding → Retrieval → Compute & Verify → Answer & Citation."""
    def __init__(
        self,
        documents_path: str | Path,
        index_dir: str | Path | None = None,
        llm_model: str = "qwen2.5:3b",
        ollama_host: str = "http://localhost:11434",
        use_llm_router: bool = True,          # Bật Router LLM cho câu hỏi không rõ ràng
        sandbox_timeout: float = 10.0,        # Timeout cho code execution (giây)
        reranker_enabled: bool = False,       # Tắt mặc định (cần download ~570MB)
        reranker_model: str = "BAAI/bge-reranker-v2-m3",
        confidence_threshold: float = 0.5,   # Ngưỡng BM25 score cho confidence check
        max_reformulate_attempts: int = 2,    # Số lần re-query khi low confidence
    ):
        self.shared_llm = LLMClient(model=llm_model, host=ollama_host)

        self.router_llm = self.shared_llm
        self.coder_llm = self.shared_llm
        self.answer_llm = self.shared_llm

        from src.query.pipeline import QueryPipeline
        self.query_pipeline = QueryPipeline(
            llm_client=self.router_llm,
            use_llm_fallback=use_llm_router,   # ← Bật Router LLM
            cache_enabled=True,
        )
        self.qr_pipeline = QueryRetrievalPipeline(
            documents_path=documents_path,
            index_dir=index_dir,
            query_pipeline=self.query_pipeline,
            reranker_enabled=reranker_enabled,
            reranker_model=reranker_model,
            confidence_threshold=confidence_threshold,
            max_reformulate_attempts=max_reformulate_attempts,
            llm_client=self.shared_llm,        # ← LLM cho query reformulation
        )

        self.code_gen = CodeGenerator(llm_client=self.coder_llm)
        self.sandbox = Sandbox(timeout=sandbox_timeout)  # ← Timeout
        self.verifier = ResultVerifier()
        self.compute_manager = RetryManager(self.code_gen, self.sandbox, self.verifier)

        self.answer_formatter = AnswerFormatter(llm_client=self.answer_llm)
        self.citation_builder = CitationBuilder()
        self.refuse_handler = RefuseHandler()

        logger.info(
            "FullQAPipeline initialized: model=%s host=%s llm_router=%s reranker=%s timeout=%.0fs",
            llm_model, ollama_host, use_llm_router, reranker_enabled, sandbox_timeout,
        )


    @staticmethod
    def _extract_unit_from_hits(hits: List[dict], expected_tickers: List[str] = None) -> str:
        """Lấy đơn vị từ hit khớp ticker. Fallback về hit đầu tiên."""
        def _get_unit(h: dict) -> str:
            doc = h.get("document", h)
            meta = doc.get("metadata", h.get("metadata", {}))
            return str(meta.get("unit", "vnd")).lower()

        if expected_tickers:
            norm_tickers = [t.upper().strip() for t in expected_tickers]
            for h in hits:
                doc = h.get("document", h)
                meta = doc.get("metadata", h.get("metadata", {}))
                if str(meta.get("ticker", "")).upper() in norm_tickers:
                    return _get_unit(h)
        return _get_unit(hits[0]) if hits else "vnd"

    @staticmethod
    def _hits_to_tables(hits: List[dict]) -> List[dict]:
        """Gộp tất cả hits thành 1 DataFrame để LLM dễ filter."""
        rows = []
        for h in hits:
            doc = h.get("document", h)
            meta = doc.get("metadata", h.get("metadata", {}))
            text = doc.get("text", h.get("content", ""))
            rows.append([
                meta.get("ticker", ""),
                meta.get("period", meta.get("year", "")),
                meta.get("item_name_raw", meta.get("item_name_normalized", meta.get("item_code", ""))),
                meta.get("item_code", ""),
                meta.get("value", None),
                meta.get("section", ""),
                meta.get("report_type", ""),
                meta.get("unit", "vnd"),
                text,
            ])

        columns = ["ticker", "period", "item_name", "item_code", "value", "section", "report_type", "unit", "text"]
        tickers = list({r[0] for r in rows if r[0]})
        return [{
            "ticker": ", ".join(tickers) if tickers else "UNKNOWN",
            "columns": columns,
            "data": rows,
        }]

    @staticmethod
    def _fast_path_single_lookup(hits: List[dict], expected_tickers: List[str] = None):
        """Lấy trực tiếp value từ hit hợp lệ, ưu tiên hit khớp ticker. Trả về float hoặc None."""
        import math

        def _valid_value(val) -> bool:
            if val is None:
                return False
            try:
                f = float(val)
                return not (math.isnan(f) or math.isinf(f))
            except (ValueError, TypeError):
                return False

        if expected_tickers:
            norm_tickers = [t.upper().strip() for t in expected_tickers]
            for h in hits:
                doc = h.get("document", h)
                meta = doc.get("metadata", h.get("metadata", {}))
                hit_ticker = str(meta.get("ticker", "")).upper()
                if hit_ticker in norm_tickers:
                    val = meta.get("value")
                    if _valid_value(val):
                        return float(val)
            return None
        # Không có expected_tickers: lấy hit đầu tiên có value hợp lệ
        for h in hits:
            doc = h.get("document", h)
            meta = doc.get("metadata", h.get("metadata", {}))
            val = meta.get("value")
            if _valid_value(val):
                return float(val)
        return None

    def run(self, question: str) -> Dict[str, Any]:
        logger.info(f"--- BẮT ĐẦU XỬ LÝ CÂU HỎI: {question} ---")

        try:
            retrieval_res = self.qr_pipeline.process(question)
        except Exception as e:
            logger.error(f"Lỗi trong Phase 1 & 2: {e}")
            return {
                "question": question,
                "answer": self.refuse_handler.handle_refuse("unknown"),
                "citations": "",
                "success": False
            }

        hits = retrieval_res.get("hits", [])
        query_info = retrieval_res.get("query", {})
        query_type = query_info.get("query_type", "single_lookup")

        if not hits:
            logger.warning("Không tìm thấy dữ liệu liên quan.")
            return {
                "question": question,
                "answer": self.refuse_handler.handle_refuse("no_hits"),
                "citations": "",
                "success": False
            }

        expected_tickers = query_info.get("entities", {}).get("tickers", [])

        # Fast-path chỉ dùng cho single_lookup — tránh trả 1 giá trị khi cần so sánh nhiều công ty/năm
        if query_type == "single_lookup":
            fast_result = self._fast_path_single_lookup(hits, expected_tickers)
            if fast_result is not None:
                logger.info(f"Fast-path thành công (query_type={query_type}): {fast_result}")
                unit = self._extract_unit_from_hits(hits, expected_tickers)
                final_answer = self.answer_formatter.format_answer(question, fast_result, unit=unit, is_fast_path=True)
                citations = self.citation_builder.build_citation(hits)
                return {
                    "question": question,
                    "answer": final_answer,
                    "citations": citations,
                    "success": True,
                    "computed_result": fast_result,
                    "code": "# Fast-path: value lấy trực tiếp từ hit có rank cao nhất",
                    "hits": hits
                }

        # Code generation path
        tables_for_compute = self._hits_to_tables(hits)

        is_success, computed_result, final_code, error_msg = self.compute_manager.compute(
            question, tables_for_compute
        )

        if not is_success:
            logger.warning("Quá trình tính toán thất bại.")
            return {
                "question": question,
                "answer": self.refuse_handler.handle_refuse("compute_failed"),
                "citations": self.citation_builder.build_citation(hits),
                "success": False,
                "error": error_msg,
                "code": final_code,
                "hits": hits
            }

        unit = self._extract_unit_from_hits(hits, expected_tickers)
        final_answer = self.answer_formatter.format_answer(question, computed_result, unit=unit)
        citations = self.citation_builder.build_citation(hits)

        logger.info("--- HOÀN THÀNH ---")
        return {
            "question": question,
            "answer": final_answer,
            "citations": citations,
            "success": True,
            "computed_result": computed_result,
            "code": final_code,
            "hits": hits
        }
