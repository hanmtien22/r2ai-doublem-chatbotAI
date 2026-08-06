from __future__ import annotations

import logging
from typing import Optional

from src.utils.text import remove_diacritics

logger = logging.getLogger(__name__)

try:
    from rapidfuzz import fuzz, process as rf_process
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False
    logger.warning("rapidfuzz not installed, fuzzy matching disabled")


class EntityResolver:
    def __init__(
        self,
        entity_dict: dict,
        company_threshold: int = 85,
        indicator_threshold: int = 80,
        llm_client=None,
    ):
        self._entity_dict = entity_dict
        self._company_threshold = company_threshold
        self._indicator_threshold = indicator_threshold
        self._llm_client = llm_client

        self._all_company_names: list[tuple[str, str]] = []
        for ticker, info in entity_dict.items():
            if "full_name" in info:
                self._all_company_names.append((info["full_name"], ticker))
            if "short_name" in info:
                self._all_company_names.append((info["short_name"], ticker))
            for alias in info.get("aliases", []):
                self._all_company_names.append((alias, ticker))

    def resolve_companies_in_text(self, text: str) -> list[str]:
        """Fuzzy-match company names embedded in a complete user question."""
        if not HAS_RAPIDFUZZ:
            return []

        normalized_text = remove_diacritics(text.lower())
        scores: dict[str, float] = {}

        for name, ticker in self._all_company_names:
            normalized_name = remove_diacritics(name.lower().strip())
            if len(normalized_name) < 4:
                continue
            score = fuzz.partial_ratio(normalized_name, normalized_text)
            if score >= self._company_threshold:
                scores[ticker] = max(scores.get(ticker, 0), score)

        return [
            ticker
            for ticker, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)
        ]

    def resolve_company(self, mention: str) -> Optional[str]:
        mention_lower = mention.lower().strip()

        for name, ticker in self._all_company_names:
            if name.lower() == mention_lower:
                logger.debug("Exact match: '%s' -> %s", mention, ticker)
                return ticker

        if HAS_RAPIDFUZZ:
            names = [name for name, _ in self._all_company_names]
            result = rf_process.extractOne(
                mention,
                names,
                scorer=fuzz.partial_ratio,
                score_cutoff=self._company_threshold,
            )
            if result:
                matched_name, score, idx = result
                ticker = self._all_company_names[idx][1]
                logger.debug("Fuzzy match: '%s' -> %s (score=%d)", mention, ticker, score)
                return ticker

        if self._llm_client:
            return self._resolve_company_llm(mention)

        logger.warning("Could not resolve company: '%s'", mention)
        return None

    def _resolve_company_llm(self, mention: str) -> Optional[str]:
        tickers_list = ", ".join(
            f"{t}: {info.get('full_name', '')}"
            for t, info in self._entity_dict.items()
        )
        prompt = (
            f"Trich xuat ticker code cua cong ty tu ten sau.\n"
            f"Ten: \"{mention}\"\n"
            f"Danh sach cong ty: {tickers_list}\n"
            f"Tra loi CHI bang ticker code (3 ky tu in hoa). Neu khong tim thay, tra loi NONE."
        )
        try:
            result = self._llm_client.generate(prompt, max_tokens=10).strip().upper()
            if result in self._entity_dict:
                logger.debug("LLM resolved: '%s' -> %s", mention, result)
                return result
        except Exception as e:
            logger.warning("LLM resolve failed for '%s': %s", mention, e)
        return None

    def resolve_indicator(self, mention: str, all_indicators: list[dict]) -> Optional[dict]:
        mention_lower = mention.lower().strip()
        mention_no_dia = remove_diacritics(mention_lower)

        for ind in all_indicators:
            if ind.get("name_lower", ind["name"].lower()) == mention_lower:
                return ind
            if ind.get("name_no_diacritics", remove_diacritics(ind["name"].lower())) == mention_no_dia:
                return ind

        if HAS_RAPIDFUZZ:
            names = [ind["name"] for ind in all_indicators]
            result = rf_process.extractOne(
                mention,
                names,
                scorer=fuzz.partial_ratio,
                score_cutoff=self._indicator_threshold,
            )
            if result:
                matched_name, score, idx = result
                logger.debug("Fuzzy indicator match: '%s' -> '%s' (score=%d)", mention, matched_name, score)
                return all_indicators[idx]

        return None
