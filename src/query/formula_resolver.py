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


class FormulaResolver:
    def __init__(self, formula_library_path: Optional[str] = None):
        self._formulas: dict = {}
        self._load_formulas(formula_library_path)

    def _load_formulas(self, path: Optional[str]) -> None:
        if path is None:
            path = str(Path(__file__).resolve().parents[2] / "data" / "formula_library.json")
        try:
            with open(path, "r", encoding="utf-8") as f:
                self._formulas = json.load(f)
            logger.info("Loaded %d formulas", len(self._formulas))
        except FileNotFoundError:
            logger.warning("Formula library not found: %s", path)

    def detect_formula(self, question: str) -> Optional[str]:
        q_lower = question.lower()
        q_no_dia = remove_diacritics(q_lower)

        for keyword, formula_key in sorted(_FORMULA_KEYWORD_MAP.items(), key=lambda x: len(x[0]), reverse=True):
            kw_lower = keyword.lower()
            kw_no_dia = remove_diacritics(kw_lower)
            if kw_lower in q_lower or kw_no_dia in q_no_dia:
                if formula_key in self._formulas:
                    return formula_key

        return None

    def resolve(
        self,
        formula_key: str,
        tickers: list[str],
        years: list[int],
    ) -> tuple[FormulaInfo, list[RetrievalQuery]]:
        formula_data = self._formulas[formula_key]

        formula_info = FormulaInfo(
            name=formula_data.get("name_en", formula_key),
            formula=formula_data["formula"],
            components=formula_data["components"],
            unit=formula_data["unit"],
            multiply_100=formula_data.get("multiply_100", False),
            requires_previous_year=formula_data.get("requires_previous_year", False),
        )

        queries: list[RetrievalQuery] = []
        all_years = set(years)

        if formula_info.requires_previous_year:
            for y in years:
                all_years.add(y - 1)

        for ticker in tickers:
            for year in sorted(all_years):
                for component in formula_data["components"]:
                    section, code = component.split(".")
                    queries.append(RetrievalQuery(
                        ticker=ticker,
                        year=year,
                        section=section,
                        indicator_code=code,
                    ))

        return formula_info, queries

    def detect_growth_indicator(self, question: str) -> Optional[str]:
        q_lower = question.lower()
        q_no_dia = remove_diacritics(q_lower)

        growth_keywords = ["tang truong", "tăng trưởng", "growth"]
        has_growth = any(
            kw in q_lower or remove_diacritics(kw) in q_no_dia
            for kw in growth_keywords
        )

        if not has_growth:
            return None

        if any(kw in q_lower or remove_diacritics(kw) in q_no_dia
               for kw in ["doanh thu", "revenue"]):
            return "revenue_growth"
        if any(kw in q_lower or remove_diacritics(kw) in q_no_dia
               for kw in ["loi nhuan", "profit", "lợi nhuận"]):
            return "profit_growth"

        return "revenue_growth"
