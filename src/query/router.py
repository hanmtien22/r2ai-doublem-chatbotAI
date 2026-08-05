from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

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

_OUT_OF_SCOPE_KEYWORDS = [
    "thoi tiet", "thời tiết",
    "bong da", "bóng đá",
    "am nhac", "âm nhạc",
    "phim", "the thao", "thể thao",
]

QUERY_TYPES = ["single_lookup", "multi_comparison", "derived_indicator", "out_of_scope"]


class QueryRouter:
    def __init__(self, llm_client=None, use_llm_fallback: bool = True):
        self._llm_client = llm_client
        self._use_llm_fallback = use_llm_fallback

    def classify(self, entities: dict, question: str) -> str:
        q_lower = question.lower()

        if self._is_out_of_scope(q_lower):
            logger.debug("Router: out_of_scope")
            return "out_of_scope"

        if self._is_derived(q_lower):
            logger.debug("Router: derived_indicator (keyword match)")
            return "derived_indicator"

        num_tickers = len(entities.get("tickers", []))
        num_years = len(entities.get("years", []))
        is_comparison = self._has_comparison_keywords(q_lower)

        if num_tickers > 1 or num_years > 1 or is_comparison:
            logger.debug("Router: multi_comparison (tickers=%d, years=%d, comparison=%s)",
                         num_tickers, num_years, is_comparison)
            return "multi_comparison"

        logger.debug("Router: single_lookup (default)")
        return "single_lookup"

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
        if not self._llm_client:
            return self.classify(entities, question)

        tickers = entities.get("tickers", [])
        years = entities.get("years", [])
        indicators = entities.get("indicators", [])

        prompt = (
            "Ban la mot bo phan loai cau hoi tai chinh. "
            "Hay phan loai cau hoi sau vao MOT trong 4 loai:\n\n"
            "1. single_lookup: Tra cuu 1 chi tieu cua 1 cong ty trong 1 nam\n"
            "2. multi_comparison: So sanh nhieu cong ty hoac nhieu nam\n"
            "3. derived_indicator: Chi so can tinh toan (ROE, ROA, tang truong, ty le, bien loi nhuan)\n"
            "4. out_of_scope: Cau hoi khong lien quan den tai chinh\n\n"
            f"Cau hoi: \"{question}\"\n\n"
            f"Thuc the da trich xuat:\n"
            f"- Cong ty: {tickers}\n"
            f"- Nam: {years}\n"
            f"- Chi tieu: {indicators}\n\n"
            "Tra loi CHI bang mot trong 4 gia tri: single_lookup, multi_comparison, derived_indicator, out_of_scope"
        )

        try:
            result = self._llm_client.generate(prompt, max_tokens=20).strip().lower()
            result = re.sub(r"[^a-z_]", "", result)
            if result in QUERY_TYPES:
                logger.debug("LLM Router: %s", result)
                return result
        except Exception as e:
            logger.warning("LLM Router failed: %s", e)

        return self.classify(entities, question)
