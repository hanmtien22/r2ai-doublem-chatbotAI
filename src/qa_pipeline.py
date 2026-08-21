import logging
from pathlib import Path
from typing import Dict, Any, List

from src.retrieval.pipeline import QueryRetrievalPipeline
from src.compute.code_generator import CodeGenerator
from src.compute.notes_table import parse_notes_table, find_value_by_label
from src.compute.sandbox import Sandbox
from src.compute.result_verifier import ResultVerifier
from src.compute.retry_manager import RetryManager

from src.answer.answer_formatter import AnswerFormatter, detect_requested_unit
from src.answer.citation_builder import CitationBuilder
from src.answer.refuse_handler import RefuseHandler
from src.llm.factory import build_llm_client

logger = logging.getLogger(__name__)


class FullQAPipeline:
    """Pipeline 4 phase: Query Understanding → Retrieval → Compute & Verify → Answer & Citation."""
    def __init__(
        self,
        documents_path: str | Path,
        index_dir: str | Path | None = None,
        llm_model: str | None = None,
        ollama_host: str = "http://localhost:11434",
        llm_backend: str = "auto",            # auto | ollama | hf | none
        llm_client=None,                      # truyền sẵn client nếu muốn tự dựng
        use_llm_router: bool = True,          # Bật Router LLM cho câu hỏi không rõ ràng
        sandbox_timeout: float = 10.0,        # Timeout cho code execution (giây)
        reranker_enabled: bool = False,       # Tắt mặc định (cần download ~570MB)
        reranker_model: str = "BAAI/bge-reranker-v2-m3",
        confidence_threshold: float = 0.5,   # Ngưỡng BM25 score cho confidence check
        max_reformulate_attempts: int = 2,    # Số lần re-query khi low confidence
    ):
        # Máy local dùng Ollama; Kaggle/Colab không có server nên nạp model bằng
        # transformers, hoặc chạy hẳn chế độ không LLM (chỉ tra cứu tất định).
        self.shared_llm = llm_client or build_llm_client(
            backend=llm_backend, model=llm_model, host=ollama_host
        )

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
            "FullQAPipeline initialized: llm=%s host=%s llm_router=%s reranker=%s timeout=%.0fs",
            getattr(self.shared_llm, "model", llm_backend), ollama_host,
            use_llm_router, reranker_enabled, sandbox_timeout,
        )


    @staticmethod
    def _notes_rows_from_hits(hits: List[dict]) -> List[dict]:
        """Parse các chunk thuyết minh thành dòng có cấu trúc (label / cột / giá trị)."""
        rows: List[dict] = []
        for h in hits:
            doc = h.get("document", h)
            meta = doc.get("metadata", h.get("metadata", {}))
            if meta.get("value") is not None:
                continue  # đã là dòng số liệu của bảng chính
            text = doc.get("text", h.get("content", ""))
            if not text or "|" not in text:
                continue
            for row in parse_notes_table(text, meta.get("section_title", "")):
                row = dict(row)
                row["ticker"] = meta.get("ticker", "")
                row["year"] = meta.get("year")
                row["report_type"] = meta.get("report_type", "")
                rows.append(row)
        return rows

    @staticmethod
    def _hits_to_tables(hits: List[dict]) -> List[dict]:
        """Dựng DataFrame cho bước sinh code.

        df_0: dòng số liệu của bảng chính (BS/IS/CF).
        df_1: dòng đã parse từ bảng trong thuyết minh — trước đây cả chunk
        thuyết minh bị nhét vào một ô text, code pandas không thể dùng được.
        """
        table_rows = []
        for h in hits:
            doc = h.get("document", h)
            meta = doc.get("metadata", h.get("metadata", {}))
            if meta.get("value") is None:
                continue
            table_rows.append([
                meta.get("ticker", ""),
                meta.get("period", meta.get("year", "")),
                meta.get("item_name_raw", meta.get("item_name_normalized", meta.get("item_code", ""))),
                str(meta.get("item_code", "")),
                meta.get("value", None),
                meta.get("section", ""),
                meta.get("report_type", ""),
                meta.get("unit", "vnd"),
            ])

        tables = [{
            "ticker": ", ".join({r[0] for r in table_rows if r[0]}) or "UNKNOWN",
            "columns": ["ticker", "period", "item_name", "item_code", "value",
                        "section", "report_type", "unit"],
            "data": table_rows,
            "description": "Dòng số liệu từ BCTC chính (cột `value` đơn vị VND)",
        }]

        notes_rows = FullQAPipeline._notes_rows_from_hits(hits)
        if notes_rows:
            columns = ["ticker", "year", "report_type", "note_title", "label",
                       "column", "column_role", "value", "unit", "value_vnd"]
            tables.append({
                "ticker": ", ".join({str(r["ticker"]) for r in notes_rows if r.get("ticker")}) or "UNKNOWN",
                "columns": columns,
                "data": [[r.get(c) for c in columns] for r in notes_rows],
                "description": (
                    "Dòng trong bảng thuyết minh. `label` là tên chỉ tiêu, "
                    "`column_role`='current' là số cuối năm / năm nay, "
                    "'previous' là số đầu năm / năm trước. "
                    "`value_vnd` đã quy đổi về VND — dùng cột này."
                ),
            })
        return tables

    @staticmethod
    def _valid_number(val) -> bool:
        import math

        if val is None:
            return False
        try:
            f = float(val)
        except (ValueError, TypeError):
            return False
        return not (math.isnan(f) or math.isinf(f))

    @staticmethod
    def _plausible_amount(question: str, value) -> bool:
        """Chặn kết quả vô lý từ code do LLM sinh.

        Câu hỏi tính bằng triệu/tỷ đồng mà ra vài trăm nghìn đồng thì gần như
        chắc chắn code đã lọc nhầm dòng hoặc nhầm đơn vị — thà nói không biết
        còn hơn đưa ra một con số sai trông có vẻ hợp lệ.
        """
        requested = detect_requested_unit(question)
        if not requested or requested[1] == "%":
            return True
        try:
            amount = abs(float(value))
        except (ValueError, TypeError):
            return True
        # Kể cả 0: một khoản mục BCTC bằng đúng 0 thì câu trả lời "0 tỷ đồng"
        # cũng vô nghĩa, và hầu như luôn là do filter không khớp dòng nào.
        return amount >= 1_000_000

    @classmethod
    def _fast_path_single_lookup(cls, hits: List[dict], query_info: dict):
        """Lấy value trực tiếp khi hit đầu bảng đúng chỉ tiêu/năm/loại báo cáo đang hỏi.

        Trước đây hàm này lấy hit đầu tiên bất kỳ có `value`, nên chỉ cần BM25 xếp
        nhầm một dòng cùng ticker là trả về số của chỉ tiêu khác.
        """
        entities = query_info.get("entities", {})
        tickers = {t.upper().strip() for t in entities.get("tickers", []) if t}
        years = set(entities.get("years", []))
        wanted_report = entities.get("report_type") or "consolidated"
        wanted_codes = {
            (q.get("section"), str(q.get("indicator_code")))
            for q in query_info.get("retrieval_queries", [])
        }

        # Lượt 1 đòi đúng loại báo cáo; lượt 2 bỏ ràng buộc đó, vì có công ty
        # (vd: công ty chứng khoán) chỉ nộp báo cáo riêng, không có hợp nhất.
        for require_report_type in (True, False):
            for h in hits:
                meta = h.get("document", h).get("metadata", h.get("metadata", {}))
                val = meta.get("value")
                if not cls._valid_number(val):
                    continue
                if tickers and str(meta.get("ticker", "")).upper() not in tickers:
                    continue
                if years and meta.get("period") not in years:
                    continue
                # Chỉ tin fast-path khi biết chắc mã chỉ tiêu; nếu không, để bước
                # sinh code pandas quyết định.
                if wanted_codes and (meta.get("section"), str(meta.get("item_code") or "")) not in wanted_codes:
                    continue
                if require_report_type and str(meta.get("report_type", "")).lower() != wanted_report:
                    continue
                return float(val), str(meta.get("unit", "vnd")).lower()

            # Người dùng đã nói rõ loại báo cáo thì không được lấy loại khác
            if entities.get("report_type"):
                break

        return None, None

    @classmethod
    def _fast_path_notes(cls, hits: List[dict], query_info: dict, question: str = ""):
        """Tra số trong bảng thuyết minh bằng khớp nhãn, khi câu hỏi không map ra mã chỉ tiêu."""
        entities = query_info.get("entities", {})
        core_phrase = entities.get("core_phrase") or ""
        if not core_phrase:
            return None, None

        wanted_report = entities.get("report_type") or "consolidated"
        notes_rows = cls._notes_rows_from_hits(hits)
        if not notes_rows:
            return None, None

        # Câu hỏi về tỷ lệ (%) thì con số trong bảng đã là phần trăm — quy đổi
        # theo đơn vị tiền tệ của bảng sẽ cho ra số vô nghĩa.
        requested = detect_requested_unit(question) if question else None
        is_ratio = bool(requested) and requested[1] == "%"
        value_key = "value" if is_ratio else "value_vnd"

        # Ưu tiên đúng loại báo cáo; không có thì dùng tất cả
        preferred = [r for r in notes_rows if str(r.get("report_type", "")).lower() == wanted_report]
        for rows in (preferred, notes_rows):
            if not rows:
                continue
            match = find_value_by_label(rows, core_phrase)
            if match and cls._valid_number(match.get(value_key)):
                logger.info("Notes fast-path: '%s' -> '%s' = %s",
                            core_phrase, match["label"], match[value_key])
                return float(match[value_key]), ("%" if is_ratio else "vnd")
        return None, None

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

        # Fast-path chỉ dùng cho single_lookup — tránh trả 1 giá trị khi cần so sánh nhiều công ty/năm
        if query_type == "single_lookup":
            fast_result, fast_unit = self._fast_path_single_lookup(hits, query_info)
            source = "bảng chính"
            if fast_result is None:
                fast_result, fast_unit = self._fast_path_notes(hits, query_info, question)
                source = "thuyết minh"

            if fast_result is not None:
                logger.info("Fast-path (%s) thành công: %s", source, fast_result)
                final_answer = self.answer_formatter.format_answer(
                    question, fast_result, unit=fast_unit or "vnd", is_fast_path=True
                )
                citations = self.citation_builder.build_citation(hits)
                return {
                    "question": question,
                    "answer": final_answer,
                    "citations": citations,
                    "success": True,
                    "computed_result": fast_result,
                    "code": f"# Fast-path: value lấy trực tiếp từ {source}",
                    "hits": hits
                }

        # Code generation path
        tables_for_compute = self._hits_to_tables(hits)

        is_success, computed_result, final_code, error_msg = self.compute_manager.compute(
            question, tables_for_compute
        )

        if is_success and not self._plausible_amount(question, computed_result):
            logger.warning("Kết quả %s quá nhỏ so với đơn vị được hỏi -> coi như thất bại",
                           computed_result)
            is_success = False
            error_msg = (
                f"Kết quả {computed_result} không hợp lý với đơn vị tiền tệ được hỏi "
                f"(nhiều khả năng code lọc nhầm dòng hoặc nhầm đơn vị)."
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

        # Prompt sinh code yêu cầu `final_result` tính bằng VND (cột `value` của
        # bảng chính, `value_vnd` của thuyết minh), nên không quy đổi lần nữa.
        final_answer = self.answer_formatter.format_answer(question, computed_result, unit="vnd")
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
