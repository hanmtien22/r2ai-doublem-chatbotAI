from __future__ import annotations

import logging
import re
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

_BM25_MIN_SCORE = 0.5
_MIN_FILTERED_HITS = 1


class RetrievalConfidenceChecker:
    def __init__(
        self,
        bm25_score_threshold: float = _BM25_MIN_SCORE,
        min_filtered_hits: int = _MIN_FILTERED_HITS,
    ):
        self.bm25_score_threshold = bm25_score_threshold
        self.min_filtered_hits = min_filtered_hits

    def check_raw_results(self, bm25_results: list[dict]) -> Tuple[bool, float]:
        if not bm25_results:
            return False, 0.0
        top_score = float(bm25_results[0].get("score", 0.0))
        is_ok = top_score >= self.bm25_score_threshold
        if not is_ok:
            logger.info("Low BM25 confidence: top_score=%.3f < threshold=%.3f", top_score, self.bm25_score_threshold)
        return is_ok, top_score

    def check_filtered_hits(self, filtered_hits: list[dict], total_hits: int) -> bool:
        if len(filtered_hits) < self.min_filtered_hits and total_hits > 0:
            logger.info("Filter removed all hits (filtered=%d, total=%d)", len(filtered_hits), total_hits)
            return False
        return True


class QueryReformulator:
    def __init__(self, llm_client=None):
        self._llm = llm_client

    def reformulate(self, original_question: str, entities: dict, attempt: int = 0) -> str:
        if attempt == 0:
            return self._strategy_indicator_only(original_question, entities)
        if attempt == 1:
            return self._strategy_normalized(original_question, entities)
        if attempt >= 2 and self._llm:
            return self._strategy_llm(original_question)
        return original_question

    def _strategy_indicator_only(self, question: str, entities: dict) -> str:
        indicators = entities.get("indicators", [])
        if indicators:
            reformulated = indicators[0]
            logger.info("Reformulate s1: '%s' → '%s'", question[:50], reformulated)
            return reformulated
        return question

    def _strategy_normalized(self, question: str, entities: dict) -> str:
        q = re.sub(r'\b20[0-2]\d\b', '', question)
        for ticker in entities.get("tickers", []):
            q = re.sub(r'\b' + re.escape(ticker) + r'\b', '', q, flags=re.IGNORECASE)
        q = re.sub(r'\s+', ' ', q).strip(' ,?.')
        if q:
            logger.info("Reformulate s2: '%s' → '%s'", question[:50], q)
            return q
        return question

    def _strategy_llm(self, question: str) -> str:
        prompt = (
            "Hãy viết lại câu hỏi tài chính sau bằng từ đồng nghĩa, giữ nguyên ý nghĩa. "
            "Chỉ trả lời bằng câu hỏi đã viết lại.\n\n"
            f"Câu hỏi gốc: {question}"
        )
        try:
            result = self._llm.generate(prompt, max_tokens=100).strip()
            if result and len(result) > 10:
                logger.info("Reformulate s3 (LLM): '%s' → '%s'", question[:50], result[:50])
                return result
        except Exception as e:
            logger.warning("LLM reformulate failed: %s", e)
        return question
