from __future__ import annotations

import logging
from typing import Optional

from src.query.preprocessor import QueryPreprocessor
from src.query.entity_extractor import EntityExtractor
from src.query.entity_resolver import EntityResolver
from src.query.router import QueryRouter
from src.query.formula_resolver import FormulaResolver
from src.query.query_builder import QueryBuilder
from src.query.models import QueryResult
from src.llm.client import LLMClient
from src.utils.cache import SimpleCache

logger = logging.getLogger(__name__)


class QueryPipeline:
    def __init__(
        self,
        abbreviations_path: Optional[str] = None,
        entity_dict_path: Optional[str] = None,
        indicator_aliases_path: Optional[str] = None,
        schema_mapping_path: Optional[str] = None,
        formula_library_path: Optional[str] = None,
        reference_year: int = 2024,
        company_threshold: int = 85,
        indicator_threshold: int = 80,
        use_llm_fallback: bool = False,
        llm_client: Optional[LLMClient] = None,
        cache_enabled: bool = True,
        cache_max_size: int = 1000,
    ):
        self.preprocessor = QueryPreprocessor(
            abbreviations_path=abbreviations_path,
            reference_year=reference_year,
        )

        self.entity_extractor = EntityExtractor(
            entity_dict_path=entity_dict_path,
            indicator_aliases_path=indicator_aliases_path,
            schema_mapping_path=schema_mapping_path,
        )

        self.entity_resolver = EntityResolver(
            entity_dict=self.entity_extractor.entity_dict,
            company_threshold=company_threshold,
            indicator_threshold=indicator_threshold,
            llm_client=llm_client,
        )

        self.router = QueryRouter(
            llm_client=llm_client,
            use_llm_fallback=use_llm_fallback,
        )

        self.formula_resolver = FormulaResolver(
            formula_library_path=formula_library_path,
        )

        self.query_builder = QueryBuilder()

        self._cache = SimpleCache(max_size=cache_max_size, enabled=cache_enabled)

        logger.info("QueryPipeline initialized")

    def process(self, question: str) -> QueryResult:
        cached = self._cache.get(question)
        if cached is not None:
            logger.debug("Cache hit for question: '%s'", question[:50])
            return cached

        normalized = self.preprocessor.normalize(question)
        logger.info("Step 1.1 Preprocessor: '%s'", normalized[:80])

        entities = self.entity_extractor.extract_all(normalized)
        fuzzy_tickers = self.entity_resolver.resolve_companies_in_text(normalized)
        for ticker in fuzzy_tickers:
            if ticker not in entities["tickers"]:
                entities["tickers"].append(ticker)
        logger.info("Step 1.2 Entities: tickers=%s, years=%s, indicators=%s",
                     entities["tickers"], entities["years"], entities["indicators"])

        year_list = self.preprocessor.extract_year_list(question)
        if year_list:
            entities["years"] = year_list

        query_type = self.router.route(entities, normalized)
        logger.info("Step 1.3 Router: %s", query_type)

        formula_info = None
        retrieval_queries = None

        if query_type == "derived_indicator":
            formula_key = self.formula_resolver.detect_formula(normalized)
            if formula_key is None:
                formula_key = self.formula_resolver.detect_growth_indicator(normalized)

            if formula_key:
                formula_info, retrieval_queries = self.formula_resolver.resolve(
                    formula_key,
                    entities["tickers"],
                    entities["years"],
                )
                logger.info("Step 1.4 Formula: %s -> %d queries",
                             formula_key, len(retrieval_queries))

        result = self.query_builder.build(
            original_question=question,
            normalized_question=normalized,
            entities=entities,
            query_type=query_type,
            formula_info=formula_info,
            retrieval_queries=retrieval_queries,
        )
        logger.info("Step 1.5 QueryBuilder: %d retrieval queries", len(result.retrieval_queries))

        self._cache.set(question, result)
        return result

    @property
    def cache_stats(self) -> dict:
        return self._cache.stats
