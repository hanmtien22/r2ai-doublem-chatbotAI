from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from src.query.preprocessor import QueryPreprocessor
from src.query.entity_extractor import EntityExtractor
from src.query.entity_resolver import EntityResolver
from src.query.router import QueryRouter
from src.query.formula_resolver import FormulaResolver
from src.query.query_builder import QueryBuilder
from src.query.models import QueryResult, RetrievalQuery
from src.llm.client import LLMClient
from src.utils.cache import SimpleCache
from src.paths import dictionary_path

logger = logging.getLogger(__name__)

def _load_json(path: Path) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning("Data file not found: %s", path)
        return {}


class QueryPipeline:
    def __init__(
        self,
        abbreviations_path: Optional[str] = None,
        entity_dict: Optional[dict] = None,
        indicator_aliases: Optional[dict] = None,
        schema_mapping: Optional[dict] = None,
        formula_library_path: Optional[str] = None,
        reference_year: int = 2024,
        company_threshold: int = 85,
        indicator_threshold: int = 80,
        use_llm_fallback: bool = False,
        llm_client: Optional[LLMClient] = None,
        cache_enabled: bool = True,
        cache_max_size: int = 1000,
    ):
        # dictionary_path dò cả /kaggle/input lẫn repo, nên notebook Kaggle
        # không phải copy từ điển vào đúng chỗ mới chạy được.
        if entity_dict is None:
            entity_dict = _load_json(dictionary_path("entity_dictionary.json"))
        if indicator_aliases is None:
            indicator_aliases = _load_json(dictionary_path("indicator_aliases.json"))
        if schema_mapping is None:
            schema_mapping = _load_json(dictionary_path("schema_mapping.json"))

        self.preprocessor = QueryPreprocessor(
            abbreviations_path=abbreviations_path,
            reference_year=reference_year,
        )

        self.entity_extractor = EntityExtractor(
            entity_dict=entity_dict,
            indicator_aliases=indicator_aliases,
            schema_mapping=schema_mapping,
        )

        self.entity_resolver = EntityResolver(
            entity_dict=self.entity_extractor.entity_dict,
            schema_mapping=schema_mapping,
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
        # Normalize key: trim + lower để tăng cache hit rate
        cache_key = question.strip().lower()
        cached = self._cache.get(cache_key)
        if cached is not None:
            logger.debug("Cache hit for question: '%s'", question[:50])
            return cached

        normalized = self.preprocessor.normalize(question)
        logger.info("Step 1.1 Preprocessor: '%s'", normalized[:80])

        entities = self.entity_extractor.extract_all(normalized)
        logger.info("Step 1.2 Entities: tickers=%s, years=%s, indicators=%s",
                     entities["tickers"], entities["years"], entities["indicators"])
        
        # Nếu indicators trả về là NOTES.UNKNOWN, sử dụng TF-IDF fallback ĐỂ CHECK TRƯỚC. 
        # Nếu điểm cao (>0.6) thì ghi đè, nếu không thì giữ nguyên NOTES.UNKNOWN để Semantic Search
        if entities["indicators"] and entities["indicator_codes"][0] == "NOTES.UNKNOWN" and self.router._use_llm_fallback:
            logger.info("Step 1.2b: Indicator is UNKNOWN, triggering TF-IDF fallback...")
            entities_to_remove = entities["tickers"] + [str(y) for y in entities["years"]]
            fallback_ind = self.entity_resolver.resolve_indicator_fallback(normalized, entities_to_remove)
            
            # TF-IDF return None nếu score <= 0.15. Ta có thể nâng ngưỡng tin cậy (VD: > 0.4)
            if fallback_ind: # Giả sử resolve_indicator_fallback đã xử lý ngưỡng
                entities["indicators"] = [fallback_ind["name"]]
                entities["indicator_codes"] = [f"{fallback_ind['section']}.{fallback_ind['code']}"]
                entities["indicator_details"] = [{
                    "name": fallback_ind["name"],
                    "section": fallback_ind["section"],
                    "code": fallback_ind["code"],
                    "indicator_code": f"{fallback_ind['section']}.{fallback_ind['code']}"
                }]
                logger.info("Step 1.2b Fallback Success: found %s", fallback_ind["name"])
            else:
                logger.info("Step 1.2b Fallback Low Score -> Keep NOTES.UNKNOWN for Phase 2 Semantic Search")

        year_list = self.preprocessor.extract_year_list(question)
        if year_list:
            entities["years"] = year_list

        query_type = self.router.classify(entities, normalized)
        
        # Nếu chỉ có 1 chỉ tiêu chính xác và 1 công ty 1 năm -> ép về single_lookup
        if query_type == "derived_indicator" and len(entities.get("tickers", [])) <= 1 and len(entities.get("years", [])) <= 1:
            # Kiểm tra xem có formula thực sự hay không
            fk = self.formula_resolver.detect_formula(normalized) or self.formula_resolver.detect_growth_indicator(normalized)
            if not fk:
                query_type = "single_lookup"
                
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
            else:
                # Dynamic formula detection (Ty le, tang truong khong co trong library)
                dynamic_res = self.formula_resolver.detect_dynamic_formula(normalized, entities.get("indicator_details", []))
                if dynamic_res:
                    formula_info, _ = dynamic_res
                    # Tao retrieval_queries tu dong
                    retrieval_queries = []
                    all_years = set(entities["years"])
                    if formula_info.requires_previous_year:
                        for y in entities["years"]:
                            all_years.add(y - 1)
                    for ticker in entities["tickers"]:
                        for year in sorted(all_years):
                            for component in formula_info.components:
                                section, code = component.split(".")
                                retrieval_queries.append(RetrievalQuery(
                                    ticker=ticker,
                                    year=year,
                                    section=section,
                                    indicator_code=code,
                                ))
                    logger.info("Step 1.4 Dynamic Formula: %s", formula_info.name)

        result = self.query_builder.build(
            original_question=question,
            normalized_question=normalized,
            entities=entities,
            query_type=query_type,
            formula_info=formula_info,
            retrieval_queries=retrieval_queries,
        )
        logger.info("Step 1.5 QueryBuilder: %d retrieval queries", len(result.retrieval_queries))

        self._cache.set(cache_key, result)
        return result

    @property
    def cache_stats(self) -> dict:
        return self._cache.stats
