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
    """
    Điều phối viên (Orchestrator) của toàn bộ pha xử lý câu hỏi.
    Nó kết nối tuần tự tất cả các class trong thư mục `src/query` lại với nhau 
    để tạo thành một luồng xử lý trôi chảy (Pipeline) từ câu hỏi thô thành dữ liệu truy vấn.
    """
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
        # 1. Khởi tạo bộ tiền xử lý (sửa lỗi chính tả, dịch năm tương đối)
        self.preprocessor = QueryPreprocessor(
            abbreviations_path=abbreviations_path,
            reference_year=reference_year,
        )

        # 2. Khởi tạo bộ trích xuất thực thể
        self.entity_extractor = EntityExtractor(
            entity_dict_path=entity_dict_path,
            indicator_aliases_path=indicator_aliases_path,
            schema_mapping_path=schema_mapping_path,
        )

        # 3. Khởi tạo bộ giải quyết thực thể mập mờ (fuzzy match)
        self.entity_resolver = EntityResolver(
            entity_dict=self.entity_extractor.entity_dict,
            company_threshold=company_threshold,
            indicator_threshold=indicator_threshold,
            llm_client=llm_client,
        )

        # 4. Khởi tạo bộ định tuyến để xác định loại câu hỏi
        self.router = QueryRouter(
            llm_client=llm_client,
            use_llm_fallback=use_llm_fallback,
        )

        # 5. Khởi tạo bộ giải quyết công thức (nếu cần tính toán)
        self.formula_resolver = FormulaResolver(
            formula_library_path=formula_library_path,
        )

        # 6. Khởi tạo bộ đóng gói cuối cùng
        self.query_builder = QueryBuilder()

        # 7. Khởi tạo bộ nhớ tạm (Cache) để trả lời nhanh các câu hỏi trùng lặp
        self._cache = SimpleCache(max_size=cache_max_size, enabled=cache_enabled)

        logger.info("QueryPipeline initialized")

    def process(self, question: str) -> QueryResult:
        """
        Hàm chính chạy toàn bộ luồng xử lý:
        1. Check Cache -> 2. Normalize -> 3. Extract -> 4. Route -> 5. Resolve Formula -> 6. Build
        """
        # Kiểm tra cache xem câu này đã ai hỏi chưa
        cached = self._cache.get(question)
        if cached is not None:
            logger.debug("Cache hit for question: '%s'", question[:50])
            return cached

        # Bước 1: Tiền xử lý
        normalized = self.preprocessor.normalize(question)
        logger.info("Step 1.1 Preprocessor: '%s'", normalized[:80])

        # Bước 2: Trích xuất thực thể
        entities = self.entity_extractor.extract_all(normalized)
        
        # Bước 2.5: Cố gắng tra cứu các công ty bị gõ sai chính tả bằng fuzzy match
        fuzzy_tickers = self.entity_resolver.resolve_companies_in_text(normalized)
        for ticker in fuzzy_tickers:
            if ticker not in entities["tickers"]:
                entities["tickers"].append(ticker)
        logger.info("Step 1.2 Entities: tickers=%s, years=%s, indicators=%s",
                     entities["tickers"], entities["years"], entities["indicators"])

        # Bước 2.6: Cập nhật lại năm một lần nữa đề phòng sót
        year_list = self.preprocessor.extract_year_list(question)
        if year_list:
            entities["years"] = year_list

        # Bước 3: Phân loại câu hỏi
        query_type = self.router.route(entities, normalized)
        logger.info("Step 1.3 Router: %s", query_type)

        formula_info = None
        retrieval_queries = None

        # Bước 4: Xử lý riêng cho câu hỏi dạng công thức
        if query_type == "derived_indicator":
            formula_key = self.formula_resolver.detect_formula(normalized)
            if formula_key is None:
                # Nếu không bắt được tên công thức cụ thể, thử bắt chữ "tăng trưởng"
                formula_key = self.formula_resolver.detect_growth_indicator(normalized)

            if formula_key:
                formula_info, retrieval_queries = self.formula_resolver.resolve(
                    formula_key,
                    entities["tickers"],
                    entities["years"],
                )
                logger.info("Step 1.4 Formula: %s -> %d queries",
                             formula_key, len(retrieval_queries))

        # Bước 5: Tổng hợp kết quả
        result = self.query_builder.build(
            original_question=question,
            normalized_question=normalized,
            entities=entities,
            query_type=query_type,
            formula_info=formula_info,
            retrieval_queries=retrieval_queries,
        )
        logger.info("Step 1.5 QueryBuilder: %d retrieval queries", len(result.retrieval_queries))

        # Lưu lại vào cache để dùng cho lần sau
        self._cache.set(question, result)
        return result

    @property
    def cache_stats(self) -> dict:
        return self._cache.stats
