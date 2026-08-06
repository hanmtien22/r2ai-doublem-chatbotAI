from __future__ import annotations

import logging
from typing import Optional

from src.query.models import (
    ExtractedEntities,
    FormulaInfo,
    MetadataFilters,
    QueryResult,
    RetrievalQuery,
)

logger = logging.getLogger(__name__)


class QueryBuilder:
    def build(
        self,
        original_question: str,
        normalized_question: str,
        entities: dict,
        query_type: str,
        formula_info: Optional[FormulaInfo],
        retrieval_queries: Optional[list[RetrievalQuery]],
    ) -> QueryResult:
        extracted = ExtractedEntities(
            tickers=entities.get("tickers", []),
            years=entities.get("years", []),
            indicators=entities.get("indicators", []),
            indicator_codes=entities.get("indicator_codes", []),
        )

        if retrieval_queries is None:
            retrieval_queries = self._build_retrieval_queries(entities)

        sections = list(set(q.section for q in retrieval_queries))

        metadata_filters = MetadataFilters(
            tickers=extracted.tickers,
            years=extracted.years,
            sections=sections,
        )

        result = QueryResult(
            original_question=original_question,
            normalized_question=normalized_question,
            entities=extracted,
            query_type=query_type,
            requires_formula=(query_type == "derived_indicator"),
            formula_info=formula_info,
            retrieval_queries=retrieval_queries,
            search_text=normalized_question,
            metadata_filters=metadata_filters,
        )

        logger.debug("QueryBuilder output: type=%s, queries=%d", query_type, len(retrieval_queries))
        return result

    def _build_retrieval_queries(self, entities: dict) -> list[RetrievalQuery]:
        queries: list[RetrievalQuery] = []
        tickers = entities.get("tickers", [])
        years = entities.get("years", [])
        details = entities.get("indicator_details", [])

        if not tickers or not years or not details:
            return queries

        for ticker in tickers:
            for year in years:
                for ind in details:
                    queries.append(RetrievalQuery(
                        ticker=ticker,
                        year=year,
                        section=ind["section"],
                        indicator_code=ind["code"],
                    ))

        return queries
