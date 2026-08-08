from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

from src.utils.text import remove_diacritics

logger = logging.getLogger(__name__)

# Pattern để bắt mã chứng khoán (thường là 3 chữ cái viết hoa)
_TICKER_PATTERN = re.compile(r"\b([A-Z]{3})\b")
# Pattern để bắt năm (từ 2000 - 2029)
_YEAR_PATTERN = re.compile(r"\b(20[0-2]\d)\b")


class EntityExtractor:
    """
    Chịu trách nhiệm duyệt qua câu hỏi và nhặt ra (extract) các thực thể quan trọng: 
    1. Mã cổ phiếu (Tickers)
    2. Năm tài chính (Years)
    3. Tên chỉ tiêu tài chính (Indicators)
    """
    def __init__(
        self,
        entity_dict_path: Optional[str] = None,
        indicator_aliases_path: Optional[str] = None,
        schema_mapping_path: Optional[str] = None,
    ):
        self._entity_dict: dict = {}
        self._indicator_aliases: dict[str, str] = {}
        self._schema_mapping: dict = {}
        self._alias_to_ticker: dict[str, str] = {}
        self._all_indicator_names: list[dict] = []

        base = Path(__file__).resolve().parents[2] / "data"
        self._load_entity_dict(entity_dict_path or str(base / "entity_dictionary.json"))
        self._load_indicator_aliases(indicator_aliases_path or str(base / "indicator_aliases.json"))
        self._load_schema_mapping(schema_mapping_path or str(base / "schema_mapping.json"))

    def _load_entity_dict(self, path: str) -> None:
        """Load từ điển công ty (Tên gốc, viết tắt) map với mã cổ phiếu."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                self._entity_dict = json.load(f)
            for ticker, info in self._entity_dict.items():
                for alias in info.get("aliases", []):
                    self._alias_to_ticker[alias.lower()] = ticker
                self._alias_to_ticker[ticker.lower()] = ticker
                if "full_name" in info:
                    self._alias_to_ticker[info["full_name"].lower()] = ticker
                if "short_name" in info:
                    self._alias_to_ticker[info["short_name"].lower()] = ticker
            logger.info("Loaded %d entities", len(self._entity_dict))
        except FileNotFoundError:
            logger.warning("Entity dictionary not found: %s", path)

    def _load_indicator_aliases(self, path: str) -> None:
        """Load từ điển các cách gọi khác nhau của 1 chỉ tiêu (VD: dt -> doanh thu)."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                self._indicator_aliases = json.load(f)
            logger.info("Loaded %d indicator aliases", len(self._indicator_aliases))
        except FileNotFoundError:
            logger.warning("Indicator aliases not found: %s", path)

    def _load_schema_mapping(self, path: str) -> None:
        """Load danh sách tất cả các chỉ tiêu tài chính chuẩn từ hệ thống."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                self._schema_mapping = json.load(f)
            for section, items in self._schema_mapping.items():
                for code, info in items.items():
                    self._all_indicator_names.append({
                        "section": section,
                        "code": code,
                        "name": info["name"],
                        "name_lower": info["name"].lower(),
                        "name_no_diacritics": remove_diacritics(info["name"].lower()),
                    })
            logger.info("Loaded schema mapping with %d indicators", len(self._all_indicator_names))
        except FileNotFoundError:
            logger.warning("Schema mapping not found: %s", path)

    @property
    def entity_dict(self) -> dict:
        return self._entity_dict

    def extract_tickers(self, question: str) -> list[str]:
        """Tìm mã chứng khoán (VD: 'VNM') hoặc dịch từ tên công ty (VD: 'Vinamilk' -> 'VNM')."""
        tickers: list[str] = []

        # 1. Tìm mã trực tiếp bằng Regex
        direct_matches = _TICKER_PATTERN.findall(question)
        for t in direct_matches:
            if t in self._entity_dict:
                tickers.append(t)

        # 2. Tìm mã thông qua tên gọi/viết tắt
        q_lower = question.lower()
        sorted_aliases = sorted(self._alias_to_ticker.keys(), key=len, reverse=True)
        for alias in sorted_aliases:
            if alias in q_lower:
                ticker = self._alias_to_ticker[alias]
                if ticker not in tickers:
                    tickers.append(ticker)

        return tickers

    def extract_years(self, question: str) -> list[int]:
        """Nhặt tất cả các năm (số) xuất hiện trong câu."""
        matches = _YEAR_PATTERN.findall(question)
        return sorted(set(int(y) for y in matches))

    def extract_indicators(self, question: str) -> list[dict]:
        """
        Quét câu hỏi xem người dùng đang hỏi về chỉ tiêu tài chính nào.
        Trả về danh sách dict chứa thông tin chỉ tiêu (tên, mã số, loại báo cáo).
        """
        indicators: list[dict] = []
        q_lower = question.lower()
        q_no_diacritics = remove_diacritics(q_lower)

        # 1. Ưu tiên quét theo từ khóa alias trước
        sorted_aliases = sorted(self._indicator_aliases.keys(), key=len, reverse=True)
        for alias in sorted_aliases:
            if alias in q_lower or alias in q_no_diacritics:
                code_str = self._indicator_aliases[alias]
                section, code = code_str.split(".")
                already = any(
                    ind["section"] == section and ind["code"] == code
                    for ind in indicators
                )
                if not already:
                    indicators.append({
                        "name": alias,
                        "section": section,
                        "code": code,
                        "indicator_code": code_str,
                    })

        # 2. Nếu không có alias, quét trực tiếp trong danh sách tên chuẩn
        if not indicators:
            for ind_info in self._all_indicator_names:
                if ind_info["name_lower"] in q_lower or ind_info["name_no_diacritics"] in q_no_diacritics:
                    already = any(
                        ind["section"] == ind_info["section"] and ind["code"] == ind_info["code"]
                        for ind in indicators
                    )
                    if not already:
                        indicators.append({
                            "name": ind_info["name"],
                            "section": ind_info["section"],
                            "code": ind_info["code"],
                            "indicator_code": f"{ind_info['section']}.{ind_info['code']}",
                        })

        return indicators

    def extract_all(self, question: str) -> dict:
        """Hàm bọc (Wrapper) gọi tất cả các extract bên trên gộp chung lại."""
        tickers = self.extract_tickers(question)
        years = self.extract_years(question)
        indicators = self.extract_indicators(question)

        return {
            "tickers": tickers,
            "years": years,
            "indicators": [ind["name"] for ind in indicators],
            "indicator_codes": [ind["indicator_code"] for ind in indicators],
            "indicator_details": indicators,
        }