from __future__ import annotations

import logging
from typing import Optional

from src.utils.text import remove_diacritics

logger = logging.getLogger(__name__)

LIST_STOP_WORDS = ["của", "là", "bao", "nhiêu", "tỷ", "đồng", "triệu", "năm", "vào", "cuối", "đầu"]
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
        schema_mapping: dict,
        company_threshold: int = 85,
        indicator_threshold: int = 80,
        llm_client=None,
    ):
        self._entity_dict = entity_dict
        self._company_threshold = company_threshold
        self._indicator_threshold = indicator_threshold
        self._llm_client = llm_client
        self._all_company_names: list[tuple[str, str]] = []
        self._all_indicator_names: list[dict] = []

        for ticker, info in entity_dict.items():
            if "full_name" in info:
                self._all_company_names.append((info["full_name"], ticker))
            if "short_name" in info:
                self._all_company_names.append((info["short_name"], ticker))
            for alias in info.get("aliases", []):
                self._all_company_names.append((alias, ticker))

        self._company_names_list = [name for name, _ in self._all_company_names]
        # Load tat ca cac chi tieu tu indicator_aliases va schema_mapping vao _all_indicator_names
        for section, items in schema_mapping.items():
            if not isinstance(items, dict):
                continue
            for code, info in items.items():
                if not isinstance(info, dict):
                    continue
                raw_name = info.get("name", "")
                if not isinstance(raw_name, str) or not raw_name.strip():
                    continue
                self._all_indicator_names.append({
                    "name": raw_name,
                    "name_lower": raw_name.lower(),
                    "name_no_diacritics": remove_diacritics(raw_name.lower()),
                    "section": section,
                    "code": code,
                })
        
        self._init_tfidf()

    def _init_tfidf(self) -> None:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._tfidf_vectorizer = TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 3),
                stop_words= LIST_STOP_WORDS
            )
            self._indicator_corpus = [ind["name_lower"] for ind in self._all_indicator_names]
            self._tfidf_matrix = self._tfidf_vectorizer.fit_transform(self._indicator_corpus)
            self._has_tfidf = True
            logger.info("Initialized TF-IDF for indicator fallback")
        except ImportError:
            self._has_tfidf = False
            logger.warning("scikit-learn not installed, TF-IDF fallback disabled")

    # Tim ten cong ty tu mention, tra ve ticker neu tim thay, nguoc lai tra ve None
    def resolve_company(self, mention: str) -> Optional[str]:
        mention_clean = mention.strip()
        mention_lower = mention_clean.lower()
        
        for name, ticker in self._all_company_names:
            if name.lower() == mention_lower:
                logger.debug("Exact match: '%s' -> %s", mention, ticker)
                return ticker

        if HAS_RAPIDFUZZ:
            # Tinh diem tuong dong giua mention va danh sach ten cong ty, tra ve ticker neu diem cao hon threshold
            result = rf_process.extractOne(
                mention,
                self._company_names_list,
                scorer=fuzz.WRatio,
                score_cutoff=self._company_threshold,
            )
            if result:
                matched_name, score, idx = result
                ticker = self._all_company_names[idx][1] # Lay ticker tu all_company_names theo index 
                logger.debug("Fuzzy match: '%s' -> %s (score=%d)", mention, ticker, score)
                return ticker
        # Neu khong tim thay trong hai buoc tren va co llm_client, su dung llm de giai quyet
        if self._llm_client:
            return self._resolve_company_llm(mention)
        # Neu khong tim thay ticker, log canh bao va tra ve None
        logger.warning("Could not resolve company: '%s'", mention)
        return None
    # Giai quyet ten cong ty bang LLM
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
    # Giai quyet chi tieu tu mention, tra ve dict chi tieu neu tim thay, nguoc lai tra ve None
    def resolve_indicator(self, mention: str) -> Optional[dict]:
        mention_lower = mention.lower().strip()
        mention_no_dia = remove_diacritics(mention_lower)
        # Tim kiem chi tieu trong all_indicators theo ten thuong va ten khong dau
        for ind in self._all_indicator_names:
            if ind.get("name_lower", ind["name"].lower()) == mention_lower:
                return ind
            if ind.get("name_no_diacritics", remove_diacritics(ind["name"].lower())) == mention_no_dia:
                return ind
        # Tim bang fuzzy matching 
        if HAS_RAPIDFUZZ:
            names = [ind["name"] for ind in self._all_indicator_names]
            result = rf_process.extractOne(
                mention,
                names,
                scorer=fuzz.WRatio,
                score_cutoff=self._indicator_threshold,
            )
            if result:
                matched_name, score, idx = result
                logger.debug("Fuzzy indicator match: '%s' -> '%s' (score=%d)", mention, matched_name, score)
                return self._all_indicator_names[idx]

        logger.warning("Could not resolve indicator: '%s'", mention)
        return None

    def resolve_indicator_fallback(self, question: str, entities_to_remove: list[str] = None) -> Optional[dict]:
        """Sử dụng TF-IDF + cosine similarity để tìm indicator khi exact/alias matching thất bại."""
        if not getattr(self, "_has_tfidf", False):
            return None

        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        import re

        clean_q = question.lower()
        if entities_to_remove:
            for ent in entities_to_remove:
                # Thay the thuc the bang khoang trang (VD: VNM, 2023)
                clean_q = re.sub(rf"\b{re.escape(str(ent).lower())}\b", " ", clean_q)

        query_vec = self._tfidf_vectorizer.transform([clean_q])
        cosine_sim = cosine_similarity(query_vec, self._tfidf_matrix).flatten()
        best_idx = int(np.argmax(cosine_sim))
        best_score = float(cosine_sim[best_idx])

        if best_score > 0.65:
            ind = self._all_indicator_names[best_idx]
            logger.info(
                "TF-IDF Fallback resolved indicator: '%s...' -> %s.%s (score=%.2f)",
                clean_q[:30], ind["section"], ind["code"], best_score
            )
            return ind

        logger.warning("TF-IDF fallback could not resolve indicator: '%s' (best_score=%.2f)", clean_q[:50], best_score)
        return None