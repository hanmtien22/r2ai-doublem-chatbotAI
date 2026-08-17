from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# Danh sách từ khóa báo hiệu câu hỏi yêu cầu tính toán (Derived Indicator)
_DERIVED_KEYWORDS = [
    "ROE", "ROA", "EPS", "P/E", "P/B", "PER", "PBR",
    "tang truong", "tăng trưởng",
    "ty suat", "tỷ suất",
    "bien loi nhuan", "biên lợi nhuận",
    "margin", "ty le", "tỷ lệ",
    "hieu qua", "hiệu quả",
    "current ratio", "he so thanh toan", "hệ số thanh toán",
    "debt to equity", "no tren von", "nợ trên vốn",
    "vong quay", "vòng quay",
    "asset turnover", "inventory turnover",
]

# Danh sách từ khóa báo hiệu câu hỏi so sánh hoặc tìm Max/Min (Multi Comparison)
_COMPARISON_KEYWORDS = [
    "so sanh", "so sánh",
    "so voi", "so với",
    "hon", "hơn",
    "kem", "kém",
    "cao nhat", "cao nhất",
    "thap nhat", "thấp nhất",
    "lon nhat", "lớn nhất",
    "nho nhat", "nhỏ nhất",
    "chenh lech", "chênh lệch",
    "khac nhau", "khác nhau",
]

# Danh sách từ khóa báo hiệu câu hỏi rác, không liên quan đến hệ thống (Out of Scope)
_OUT_OF_SCOPE_KEYWORDS = [
    "thoi tiet", "thời tiết",
    "bong da", "bóng đá",
    "am nhac", "âm nhạc",
    "phim", "the thao", "thể thao",
]

QUERY_TYPES = ["single_lookup", "multi_comparison", "derived_indicator", "out_of_scope"]


class QueryRouter:
    """
    Bộ định tuyến (Router): Phân loại câu hỏi của người dùng vào 1 trong 4 loại kịch bản.
    Hệ thống sẽ dùng Rule-based (tìm từ khóa) trước vì nó nhanh và rẻ.
    Chỉ khi không chắc chắn, hệ thống mới gọi LLM (Mô hình ngôn ngữ lớn) để phân loại.
    """
    def __init__(self, llm_client=None, use_llm_fallback: bool = True):
        self._llm_client = llm_client
        self._use_llm_fallback = use_llm_fallback

    def classify(self, entities: dict, question: str) -> str:
        """Phân loại bằng Quy tắc (Rule-based)."""
        q_lower = question.lower()

        # 1. Kiểm tra xem có phải câu hỏi ngoài lề không
        if self._is_out_of_scope(q_lower):
            logger.debug("Router: out_of_scope")
            return "out_of_scope"

        # 2. Kiểm tra xem có phải câu hỏi cần tính toán công thức không
        if self._is_derived(q_lower):
            logger.debug("Router: derived_indicator (keyword match)")
            return "derived_indicator"

        # Đếm số lượng thực thể được trích xuất
        num_tickers = len(entities.get("tickers", []))
        num_years = len(entities.get("years", []))
        is_comparison = self._has_comparison_keywords(q_lower)

        # 3. Nếu hỏi nhiều hơn 1 công ty, hoặc nhiều hơn 1 năm, hoặc có từ khóa "so sánh"
        if num_tickers > 1 or num_years > 1 or is_comparison:
            logger.debug("Router: multi_comparison (tickers=%d, years=%d, comparison=%s)",
                         num_tickers, num_years, is_comparison)
            return "multi_comparison"

        # 4. Mặc định là tra cứu đơn giản 1 chỉ tiêu của 1 công ty trong 1 năm
        logger.debug("Router: single_lookup (default)")
        return "single_lookup"

    def route(self, entities: dict, question: str) -> str:
        """Use deterministic routing first and LLM only for ambiguous defaults."""
        rule_result = self.classify(entities, question)
        
        # Nếu rule-based phán đoán là tra cứu đơn giản (single_lookup) nhưng thực ra câu hỏi 
        # rất phức tạp, ta sẽ nhờ LLM kiểm tra lại (nếu cấu hình cho phép).
        if (
            self._use_llm_fallback
            and self._llm_client is not None
            and rule_result == "single_lookup"
        ):
            return self.classify_with_llm(entities, question)
            
        return rule_result

    def _is_derived(self, q_lower: str) -> bool:
        from src.utils.text import remove_diacritics
        q_no_dia = remove_diacritics(q_lower)
        for kw in _DERIVED_KEYWORDS:
            kw_check = kw.lower()
            if kw_check in q_lower or remove_diacritics(kw_check) in q_no_dia:
                return True
        return False

    def _has_comparison_keywords(self, q_lower: str) -> bool:
        from src.utils.text import remove_diacritics
        q_no_dia = remove_diacritics(q_lower)
        for kw in _COMPARISON_KEYWORDS:
            kw_check = kw.lower()
            if kw_check in q_lower or remove_diacritics(kw_check) in q_no_dia:
                return True
        return False

    def _is_out_of_scope(self, q_lower: str) -> bool:
        from src.utils.text import remove_diacritics
        q_no_dia = remove_diacritics(q_lower)
        for kw in _OUT_OF_SCOPE_KEYWORDS:
            kw_check = kw.lower()
            if kw_check in q_lower or remove_diacritics(kw_check) in q_no_dia:
                return True
        return False

    def classify_with_llm(self, entities: dict, question: str) -> str:
        """Dùng LLM (chat API) với tiếng Việt đầy đủ và few-shot examples để phân loại câu hỏi."""
        if not self._llm_client:
            return self.classify(entities, question)

        tickers = entities.get("tickers", [])
        years = entities.get("years", [])
        indicators = entities.get("indicators", [])

        system_prompt = (
            "Bạn là bộ phân loại câu hỏi tài chính. "
            "Phân loại câu hỏi vào MỘT trong 4 loại sau:\n"
            "- single_lookup: Tra cứu 1 chỉ tiêu của 1 công ty trong 1 năm\n"
            "- multi_comparison: So sánh nhiều công ty hoặc nhiều năm\n"
            "- derived_indicator: Chỉ số cần tính toán (ROE, ROA, tăng trưởng, tỷ lệ, biên lợi nhuận)\n"
            "- out_of_scope: Câu hỏi không liên quan đến tài chính\n\n"
            "Trả lời CHỈ bằng một trong 4 giá trị: single_lookup, multi_comparison, derived_indicator, out_of_scope"
        )

        user_prompt = (
            "Ví dụ phân loại:\n"
            "Q: \"Doanh thu của VNM năm 2023 là bao nhiêu?\" → single_lookup\n"
            "Q: \"So sánh lợi nhuận của VNM và HPG năm 2023\" → multi_comparison\n"
            "Q: \"ROE của VNM năm 2023 là bao nhiêu?\" → derived_indicator\n"
            "Q: \"Thời tiết hôm nay thế nào?\" → out_of_scope\n\n"
            f"Câu hỏi cần phân loại: \"{question}\"\n"
            f"Thực thể đã trích xuất:\n"
            f"- Công ty: {tickers}\n"
            f"- Năm: {years}\n"
            f"- Chỉ tiêu: {indicators}\n\n"
            "Phân loại:"
        )

        try:
            result = self._llm_client.generate_chat(
                system_prompt=system_prompt,
                user_message=user_prompt,
                max_tokens=30,
                temperature=0.0,
            ).strip().lower()
            # Dọn dẹp câu trả lời của LLM (chỉ giữ lại chữ cái và dấu gạch dưới)
            result = re.sub(r"[^a-z_]", "", result)
            if result in QUERY_TYPES:
                logger.debug("LLM Router: %s", result)
                return result
            # Thử tìm query type trong output dài hơn
            for qt in QUERY_TYPES:
                if qt in result:
                    logger.debug("LLM Router (extracted): %s", qt)
                    return qt
        except Exception as e:
            logger.warning("LLM Router failed: %s", e)

        # Nếu LLM lỗi, fallback về lại Rule-based
        return self.classify(entities, question)
