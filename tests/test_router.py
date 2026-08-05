import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from src.query.router import QueryRouter


@pytest.fixture
def router():
    return QueryRouter(use_llm_fallback=False)


class TestClassify:
    def test_single_lookup(self, router):
        entities = {"tickers": ["VNM"], "years": [2023], "indicators": ["Doanh thu thuan"]}
        result = router.classify(entities, "Doanh thu thuan cua VNM nam 2023?")
        assert result == "single_lookup"

    def test_multi_comparison_multiple_tickers(self, router):
        entities = {"tickers": ["HPG", "HSG"], "years": [2023], "indicators": ["Loi nhuan sau thue"]}
        result = router.classify(entities, "So sanh loi nhuan sau thue cua HPG va HSG nam 2023")
        assert result == "multi_comparison"

    def test_multi_comparison_multiple_years(self, router):
        entities = {"tickers": ["VNM"], "years": [2022, 2023], "indicators": ["Doanh thu thuan"]}
        result = router.classify(entities, "Doanh thu thuan cua VNM nam 2022 va 2023")
        assert result == "multi_comparison"

    def test_multi_comparison_keyword(self, router):
        entities = {"tickers": [], "years": [2023], "indicators": ["Doanh thu"]}
        result = router.classify(entities, "Cong ty nao co doanh thu cao nhat nam 2023?")
        assert result == "multi_comparison"

    def test_derived_roe(self, router):
        entities = {"tickers": ["MSN"], "years": [2023], "indicators": ["ROE"]}
        result = router.classify(entities, "ROE cua Masan nam 2023?")
        assert result == "derived_indicator"

    def test_derived_roa(self, router):
        entities = {"tickers": ["HPG"], "years": [2023], "indicators": ["ROA"]}
        result = router.classify(entities, "ROA cua HPG nam 2023?")
        assert result == "derived_indicator"

    def test_derived_growth(self, router):
        entities = {"tickers": ["VNM"], "years": [2021, 2022, 2023], "indicators": []}
        result = router.classify(entities, "Tang truong doanh thu cua VNM tu 2021 den 2023")
        assert result == "derived_indicator"

    def test_derived_margin(self, router):
        entities = {"tickers": ["FPT"], "years": [2023], "indicators": []}
        result = router.classify(entities, "Bien loi nhuan gop cua FPT nam 2023?")
        assert result == "derived_indicator"

    def test_derived_ty_suat(self, router):
        entities = {"tickers": ["VNM"], "years": [2022], "indicators": []}
        result = router.classify(entities, "Ty suat loi nhuan tren von chu so huu cua VNM nam 2022?")
        assert result == "derived_indicator"

    def test_out_of_scope(self, router):
        entities = {"tickers": [], "years": [], "indicators": []}
        result = router.classify(entities, "Thoi tiet hom nay the nao?")
        assert result == "out_of_scope"

    def test_default_single_lookup(self, router):
        entities = {"tickers": ["VNM"], "years": [2023], "indicators": ["Doanh thu"]}
        result = router.classify(entities, "Doanh thu cua VNM nam 2023")
        assert result == "single_lookup"
