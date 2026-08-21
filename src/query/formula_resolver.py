from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from src.query.models import FormulaInfo, RetrievalQuery
from src.utils.text import remove_diacritics

logger = logging.getLogger(__name__)

_FORMULA_KEYWORD_MAP = {
    "ROE": "ROE",
    "roe": "ROE",
    "ty suat loi nhuan tren von chu so huu": "ROE",
    "return on equity": "ROE",
    "ROA": "ROA",
    "roa": "ROA",
    "ty suat loi nhuan tren tong tai san": "ROA",
    "return on assets": "ROA",
    "EPS": "EPS",
    "eps": "EPS",
    "loi nhuan tren moi co phieu": "EPS",
    "earnings per share": "EPS",
    "bien loi nhuan gop": "gross_margin",
    "gross margin": "gross_margin",
    "gross profit margin": "gross_margin",
    "bien loi nhuan rong": "net_margin",
    "net margin": "net_margin",
    "net profit margin": "net_margin",
    "bien loi nhuan hoat dong": "operating_margin",
    "operating margin": "operating_margin",
    "he so thanh toan ngan han": "current_ratio",
    "current ratio": "current_ratio",
    "he so no tren von chu so huu": "debt_to_equity",
    "debt to equity": "debt_to_equity",
    "no tren von": "debt_to_equity",
    "tang truong doanh thu": "revenue_growth",
    "tang truong loi nhuan": "profit_growth",
    "vong quay tong tai san": "asset_turnover",
    "asset turnover": "asset_turnover",
    "vong quay hang ton kho": "inventory_turnover",
    "inventory turnover": "inventory_turnover",
}

# Pre-compute keywords để tránh tính toán lại mỗi khi detect
_PRECOMPUTED_KEYWORDS = sorted(
    [
        (
            kw.lower(),
            remove_diacritics(kw.lower()),
            formula_key
        )
        for kw, formula_key in _FORMULA_KEYWORD_MAP.items()
    ],
    key=lambda x: len(x[0]),
    reverse=True
)


class FormulaResolver:
    def __init__(self, formula_library_path: Optional[str] = None):
        self._formulas: dict = {}
        self._load_formulas(formula_library_path)
    # Load formula library tu file JSON, neu khong co duong dan thi load tu duong dan mac dinh
    def _load_formulas(self, path: Optional[str]) -> None:
        if path is None:
            from src.paths import dictionary_path

            path = str(dictionary_path("formula_library.json"))
        try:
            with open(path, "r", encoding="utf-8") as f:
                self._formulas = json.load(f)
            logger.info("Loaded %d formulas", len(self._formulas))
        except FileNotFoundError:
            logger.warning("Formula library not found: %s", path)
        except json.JSONDecodeError:
            logger.error("Invalid JSON format in formula library: %s", path)
    # Detect formula key tu cau hoi, tra ve formula key neu tim thay, nguoc lai tra ve None
    def detect_formula(self, question: str) -> Optional[str]:
        q_lower = question.lower()
        q_no_dia = remove_diacritics(q_lower)

        for kw_lower, kw_no_dia, formula_key in _PRECOMPUTED_KEYWORDS:
            # Dùng regex word boundary để tránh match substring (như "tien" trong "tien va cac khoan tuong duong tien")
            import re
            pat1 = r"(?<!\w)" + re.escape(kw_lower) + r"(?!\w)"
            pat2 = r"(?<!\w)" + re.escape(kw_no_dia) + r"(?!\w)"
            if re.search(pat1, q_lower) or re.search(pat2, q_no_dia):
                if formula_key in self._formulas:
                    return formula_key

        return None
    # Giai quyet formula key thanh FormulaInfo va danh sach RetrievalQuery
    def resolve(
        self,
        formula_key: str,
        tickers: list[str],
        years: list[int],
    ) -> tuple[Optional[FormulaInfo], list[RetrievalQuery]]:
        # Chống lỗi KeyError nếu formula_key không tồn tại
        if formula_key not in self._formulas:
            logger.error("Formula key '%s' not found in library", formula_key)
            return None, []

        formula_data = self._formulas[formula_key]

        formula_info = FormulaInfo(
            name = formula_data.get("name", formula_key),
            name_en =formula_data.get("name_en", formula_key),
            formula=formula_data["formula"],
            components=formula_data["components"],
            unit=formula_data["unit"],
            multiply_100=formula_data.get("multiply_100", False),
            requires_previous_year=formula_data.get("requires_previous_year", False),
        )
 
        all_years = set(years)
        if formula_info.requires_previous_year:
            for y in years:
                all_years.add(y - 1)

        # Loại bỏ trùng lặp bằng set
        queries_set = set()
        for ticker in set(tickers):
            for year in sorted(all_years):
                for component in formula_data.get("components", []):
                    if "." in component:
                        section, code = component.split(".", 1)
                        queries_set.add((ticker, year, section, code))

        queries = [
            RetrievalQuery(ticker=t, year=y, section=s, indicator_code=c)
            for t, y, s, c in sorted(queries_set)
        ]

        return formula_info, queries
    def detect_dynamic_formula(self, question: str, extracted_indicators: list[dict]) -> Optional[tuple[FormulaInfo, list[RetrievalQuery]]]:
        if not extracted_indicators:
            return None
            
        q_lower = question.lower()
        q_no_dia = remove_diacritics(q_lower)
        import re
        
        # 1. Ty le A va B (Ratio A/B)
        ratio_keywords = ["ty le", "tỷ lệ", "chiem bao nhieu", "chiếm bao nhiêu", "ty trong", "tỷ trọng"]
        has_ratio = any(
            re.search(r"(?<!\w)" + re.escape(kw) + r"(?!\w)", q_lower) or
            re.search(r"(?<!\w)" + re.escape(remove_diacritics(kw)) + r"(?!\w)", q_no_dia)
            for kw in ratio_keywords
        )
        
        if has_ratio and len(extracted_indicators) >= 2:
            ind1 = extracted_indicators[0]
            ind2 = extracted_indicators[1]
            formula = f"{ind1['indicator_code']} / {ind2['indicator_code']} * 100"
            return FormulaInfo(
                name=f"Tỷ lệ {ind1['name']} / {ind2['name']}",
                name_en="",
                formula=formula,
                components=[ind1['indicator_code'], ind2['indicator_code']],
                unit="%",
                multiply_100=True,
                requires_previous_year=False
            ), []
            
        # 2. Tang truong cua A (Growth of A)
        growth_keywords = ["tang truong", "tăng trưởng", "growth", "thay doi", "thay đổi"]
        has_growth = any(
            re.search(r"(?<!\w)" + re.escape(kw) + r"(?!\w)", q_lower) or
            re.search(r"(?<!\w)" + re.escape(remove_diacritics(kw)) + r"(?!\w)", q_no_dia)
            for kw in growth_keywords
        )
        
        if has_growth and len(extracted_indicators) >= 1:
            ind1 = extracted_indicators[0]
            formula = f"({ind1['indicator_code']}[t] - {ind1['indicator_code']}[t-1]) / {ind1['indicator_code']}[t-1] * 100"
            return FormulaInfo(
                name=f"Tăng trưởng {ind1['name']}",
                name_en="",
                formula=formula,
                components=[ind1['indicator_code']],
                unit="%",
                multiply_100=True,
                requires_previous_year=True
            ), []
            
        return None

    # Detect growth indicator từ câu hỏi
    def detect_growth_indicator(self, question: str) -> Optional[str]:
        q_lower = question.lower()
        q_no_dia = remove_diacritics(q_lower)

        growth_keywords = ["tang truong", "tăng trưởng", "growth"]
        import re
        has_growth = any(
            re.search(r"(?<!\w)" + re.escape(kw) + r"(?!\w)", q_lower) or
            re.search(r"(?<!\w)" + re.escape(remove_diacritics(kw)) + r"(?!\w)", q_no_dia)
            for kw in growth_keywords
        )

        if not has_growth:
            return None

        # Tranh bi trung voi "Tang truong cua A" (chi phi, ...)
        if any(kw in q_lower or remove_diacritics(kw) in q_no_dia for kw in ["doanh thu", "revenue"]) and "chi phi" not in q_no_dia:
            return "revenue_growth"
        if any(kw in q_lower or remove_diacritics(kw) in q_no_dia for kw in ["loi nhuan", "profit", "lợi nhuận"]) and "chi phi" not in q_no_dia:
            return "profit_growth"

        # Chống lỗi nếu không khớp đúng chỉ số
        return None