from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

from src.utils.text import normalize_unicode, remove_diacritics

logger = logging.getLogger(__name__)

_TYPO_MAP = {
    "doanh thi": "doanh thu",
    "loi nhuan rong": "loi nhuan sau thue",
    "loi nhuan dong": "loi nhuan rong",
    "tai san ngan hen": "tai san ngan han",
    "tai san dai hen": "tai san dai han",
    "bang can doi": "bang can doi ke toan",
}

_RELATIVE_YEAR_PATTERNS = [
    (re.compile(r"n[aă]m\s+ngo[aá]i", re.IGNORECASE), -1),
    (re.compile(r"n[aă]m\s+tr[uư][oơ]c", re.IGNORECASE), -1),
    (re.compile(r"n[aă]m\s+nay", re.IGNORECASE), 0),
    (re.compile(r"n[aă]m\s+hi[eê]n\s+t[aạ]i", re.IGNORECASE), 0),
]

_N_YEARS_PATTERN = re.compile(
    r"(\d+)\s+n[aă]m\s+(g[aầ]n\s+(?:nh[aấ]t|đ[aâ]y)|qua|tr[oở]\s+l[aạ]i\s+đ[aâ]y)",
    re.IGNORECASE,
)

_YEAR_RANGE_PATTERN = re.compile(
    r"t[uừ]\s+(20[0-2]\d)\s+(?:d[eế]n|đ[eế]n|toi|t[oớ]i)\s+(20[0-2]\d)",
    re.IGNORECASE,
)


class QueryPreprocessor:
    def __init__(
        self,
        abbreviations_path: Optional[str] = None,
        reference_year: int = 2024,
    ):
        self.reference_year = reference_year
        self._abbreviations: dict[str, str] = {}
        self._load_abbreviations(abbreviations_path)

    def _load_abbreviations(self, path: Optional[str]) -> None:
        if path is None:
            path = str(Path(__file__).resolve().parents[2] / "data" / "dictionaries" / "abbreviations.json")
        try:
            with open(path, "r", encoding="utf-8") as f:
                self._abbreviations = json.load(f)
            logger.info("Loaded %d abbreviations", len(self._abbreviations))
        except FileNotFoundError:
            logger.warning("Abbreviations file not found: %s", path)

    def normalize(self, question: str) -> str:
        text = normalize_unicode(question)
        text = self._expand_abbreviations(text)
        text = self._fix_typos(text)
        text = self._normalize_relative_years(text)
        text = re.sub(r"\s+", " ", text).strip()
        logger.debug("Preprocessor: '%s' -> '%s'", question, text)
        return text

    def extract_year_list(self, question: str) -> list[int]:
        years: list[int] = []

        explicit = re.findall(r"\b(20[0-2]\d)\b", question)
        years.extend(int(y) for y in explicit)

        range_match = _YEAR_RANGE_PATTERN.search(question)
        if range_match:
            start, end = int(range_match.group(1)), int(range_match.group(2))
            years.extend(range(start, end + 1))

        n_match = _N_YEARS_PATTERN.search(question)
        if n_match and not explicit and not range_match:
            n = int(n_match.group(1))
            years.extend(range(self.reference_year - n + 1, self.reference_year + 1))

        for pattern, offset in _RELATIVE_YEAR_PATTERNS:
            if pattern.search(question):
                years.append(self.reference_year + offset)

        return sorted(set(years))

    def _expand_abbreviations(self, text: str) -> str:
        sorted_abbrs = sorted(self._abbreviations.keys(), key=len, reverse=True)
        for abbr in sorted_abbrs:
            pattern = re.compile(r"\b" + re.escape(abbr) + r"\b", re.IGNORECASE)
            if pattern.search(text):
                text = pattern.sub(self._abbreviations[abbr], text)
        return text

    def _fix_typos(self, text: str) -> str:
        text_lower = remove_diacritics(text.lower())
        for typo, fix in _TYPO_MAP.items():
            if typo in text_lower:
                pattern = re.compile(re.escape(typo), re.IGNORECASE)
                text = pattern.sub(fix, text)
                text_lower = remove_diacritics(text.lower())
        return text

    def _normalize_relative_years(self, text: str) -> str:
        for pattern, offset in _RELATIVE_YEAR_PATTERNS:
            year = self.reference_year + offset
            text = pattern.sub(f"nam {year}", text)

        def _replace_n_years(m: re.Match) -> str:
            n = int(m.group(1))
            year_list = list(range(self.reference_year - n + 1, self.reference_year + 1))
            return "nam " + ", ".join(str(y) for y in year_list)

        text = _N_YEARS_PATTERN.sub(_replace_n_years, text)
        return text
