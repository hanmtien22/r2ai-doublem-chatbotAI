from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from src.query.models import FormulaInfo, RetrievalQuery
from src.utils.text import remove_diacritics

logger = logging.getLogger(__name__)

# Bảng ánh xạ từ các từ khóa người dùng hay hỏi sang Mã Công Thức (formula_key)
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
    """
    Xử lý các câu hỏi yêu cầu phải tính toán (Derived Indicators).
    Ví dụ: Biên lợi nhuận, ROE, Tăng trưởng... Các chỉ tiêu này không có sẵn 
    trong báo cáo mà phải lấy các chỉ tiêu thành phần ra để chia cho nhau.
    """
    def __init__(self, formula_library_path: Optional[str] = None):
        self._formulas: dict = {}
        self._load_formulas(formula_library_path)

    def _load_formulas(self, path: Optional[str]) -> None:
        """Load thư viện chứa các công thức toán học (VD: ROE = Lợi nhuận sau thuế / Vốn chủ sở hữu)."""
        if path is None:
            path = str(Path(__file__).resolve().parents[2] / "data" / "dictionaries" / "formula_library.json")
        try:
            with open(path, "r", encoding="utf-8") as f:
                self._formulas = json.load(f)
            logger.info("Loaded %d formulas", len(self._formulas))
        except FileNotFoundError:
            logger.warning("Formula library not found: %s", path)

    def detect_formula(self, question: str) -> Optional[str]:
        """
        Quét câu hỏi xem có chứa từ khóa yêu cầu tính toán không.
        Trả về Mã công thức (formula_key) nếu tìm thấy.
        """
        q_lower = question.lower()
        q_no_dia = remove_diacritics(q_lower)

        # Sắp xếp keyword từ dài đến ngắn để ưu tiên bắt các cụm từ dài chính xác
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
        """
        Dựa vào Mã công thức, sinh ra các truy vấn con (Retrieval Queries) 
        để đi lấy số liệu thành phần từ Database.
        """
        formula_data = self._formulas[formula_key]

        # Lấy thông tin về công thức
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

        # Nếu công thức (như Tăng trưởng) yêu cầu so với năm ngoái, tự động thêm năm ngoái vào list
        if formula_info.requires_previous_year:
            for y in years:
                all_years.add(y - 1)

        # Sinh ra tổ hợp truy vấn: [Mã CP] x [Năm] x [Các chỉ tiêu thành phần]
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
        """Xử lý riêng biệt trường hợp người dùng hỏi chung chung về chữ 'tăng trưởng'."""
        q_lower = question.lower()
        q_no_dia = remove_diacritics(q_lower)

        growth_keywords = ["tang truong", "tăng trưởng", "growth"]
        has_growth = any(
            kw in q_lower or remove_diacritics(kw) in q_no_dia
            for kw in growth_keywords
        )

        if not has_growth:
            return None

        # Đoán xem muốn tăng trưởng gì (Doanh thu hay Lợi nhuận)
        if any(kw in q_lower or remove_diacritics(kw) in q_no_dia
               for kw in ["doanh thu", "revenue"]):
            return "revenue_growth"
        if any(kw in q_lower or remove_diacritics(kw) in q_no_dia
               for kw in ["loi nhuan", "profit", "lợi nhuận"]):
            return "profit_growth"

        # Mặc định là tăng trưởng doanh thu nếu không nói rõ
        return "revenue_growth"
