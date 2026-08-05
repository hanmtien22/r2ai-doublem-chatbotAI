import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from src.query.pipeline import QueryPipeline


@pytest.fixture
def pipeline():
    return QueryPipeline(reference_year=2024, use_llm_fallback=False)


class TestSingleLookup:
    def test_doanh_thu_thuan_vnm(self, pipeline):
        result = pipeline.process("Doanh thu thuan cua VNM nam 2023?")
        assert result.query_type == "single_lookup"
        assert "VNM" in result.entities.tickers
        assert 2023 in result.entities.years
        assert len(result.retrieval_queries) >= 1
        assert result.retrieval_queries[0].section == "IS"
        assert result.retrieval_queries[0].indicator_code == "10"

    def test_loi_nhuan_sau_thue_hpg(self, pipeline):
        result = pipeline.process("Loi nhuan sau thue cua HPG nam 2022?")
        assert result.query_type == "single_lookup"
        assert "HPG" in result.entities.tickers
        assert 2022 in result.entities.years

    def test_abbreviation_lnst(self, pipeline):
        result = pipeline.process("LNST cua VNM nam 2023?")
        assert "VNM" in result.entities.tickers
        assert any(q.indicator_code == "60" and q.section == "IS"
                    for q in result.retrieval_queries)

    def test_alias_vinamilk(self, pipeline):
        result = pipeline.process("Tong tai san cua Vinamilk nam 2023?")
        assert "VNM" in result.entities.tickers


class TestMultiComparison:
    def test_compare_two_companies(self, pipeline):
        result = pipeline.process("So sanh loi nhuan sau thue cua HPG va HSG nam 2023")
        assert result.query_type == "multi_comparison"
        assert "HPG" in result.entities.tickers
        assert "HSG" in result.entities.tickers

    def test_multiple_years(self, pipeline):
        result = pipeline.process("Doanh thu thuan cua VNM tu 2020 den 2023?")
        assert result.query_type == "multi_comparison"
        assert len(result.entities.years) >= 2

    def test_comparison_keyword(self, pipeline):
        result = pipeline.process("Cong ty nao co doanh thu cao nhat nam 2023?")
        assert result.query_type == "multi_comparison"


class TestDerivedIndicator:
    def test_roe(self, pipeline):
        result = pipeline.process("ROE cua VNM nam 2023?")
        assert result.query_type == "derived_indicator"
        assert result.requires_formula is True
        assert result.formula_info is not None
        assert result.formula_info.name == "Return on Equity"
        assert len(result.retrieval_queries) >= 2

    def test_roa(self, pipeline):
        result = pipeline.process("ROA cua HPG nam 2023?")
        assert result.query_type == "derived_indicator"
        assert result.formula_info is not None

    def test_growth(self, pipeline):
        result = pipeline.process("Tang truong doanh thu cua VNM tu 2021 den 2023?")
        assert result.query_type == "derived_indicator"
        assert result.requires_formula is True
        years = {q.year for q in result.retrieval_queries}
        assert 2020 in years

    def test_margin(self, pipeline):
        result = pipeline.process("Bien loi nhuan gop cua FPT nam 2023?")
        assert result.query_type == "derived_indicator"
        assert result.formula_info is not None


class TestOutOfScope:
    def test_weather(self, pipeline):
        result = pipeline.process("Thoi tiet hom nay the nao?")
        assert result.query_type == "out_of_scope"


class TestOutputFormat:
    def test_has_all_fields(self, pipeline):
        result = pipeline.process("Doanh thu thuan cua VNM nam 2023?")
        d = result.to_dict()
        assert "original_question" in d
        assert "normalized_question" in d
        assert "entities" in d
        assert "query_type" in d
        assert "requires_formula" in d
        assert "formula_info" in d
        assert "retrieval_queries" in d
        assert "search_text" in d
        assert "metadata_filters" in d

    def test_to_json(self, pipeline):
        result = pipeline.process("ROE cua VNM nam 2023?")
        json_str = result.to_json()
        assert isinstance(json_str, str)
        assert "ROE" in json_str or "Return on Equity" in json_str


class TestCache:
    def test_cache_hit(self, pipeline):
        result1 = pipeline.process("Doanh thu thuan cua VNM nam 2023?")
        result2 = pipeline.process("Doanh thu thuan cua VNM nam 2023?")
        assert result1.query_type == result2.query_type
        assert pipeline.cache_stats["hits"] >= 1
