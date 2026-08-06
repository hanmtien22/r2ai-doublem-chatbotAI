import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from src.query.formula_resolver import FormulaResolver


@pytest.fixture
def resolver():
    return FormulaResolver()


class TestDetectFormula:
    def test_detect_roe(self, resolver):
        assert resolver.detect_formula("ROE cua VNM nam 2023") == "ROE"

    def test_detect_roa(self, resolver):
        assert resolver.detect_formula("ROA cua HPG nam 2023") == "ROA"

    def test_detect_gross_margin(self, resolver):
        assert resolver.detect_formula("Bien loi nhuan gop cua FPT") == "gross_margin"

    def test_detect_net_margin(self, resolver):
        assert resolver.detect_formula("Bien loi nhuan rong cua VNM") == "net_margin"

    def test_detect_current_ratio(self, resolver):
        assert resolver.detect_formula("Current ratio cua MSN") == "current_ratio"

    def test_detect_debt_to_equity(self, resolver):
        assert resolver.detect_formula("He so no tren von chu so huu cua VIC") == "debt_to_equity"

    def test_detect_none(self, resolver):
        assert resolver.detect_formula("Doanh thu thuan cua VNM") is None


class TestDetectGrowthIndicator:
    def test_revenue_growth(self, resolver):
        assert resolver.detect_growth_indicator("Tang truong doanh thu cua VNM") == "revenue_growth"

    def test_profit_growth(self, resolver):
        assert resolver.detect_growth_indicator("Tang truong loi nhuan cua HPG") == "profit_growth"

    def test_no_growth(self, resolver):
        assert resolver.detect_growth_indicator("Doanh thu cua VNM") is None


class TestResolve:
    def test_resolve_roe(self, resolver):
        formula_info, queries = resolver.resolve("ROE", ["VNM"], [2023])
        assert formula_info.name == "Return on Equity"
        assert formula_info.unit == "%"
        assert len(queries) == 2
        sections = {q.section for q in queries}
        assert "IS" in sections
        assert "BS" in sections
        codes = {q.indicator_code for q in queries}
        assert "60" in codes
        assert "400" in codes

    def test_resolve_roa(self, resolver):
        formula_info, queries = resolver.resolve("ROA", ["HPG"], [2023])
        assert formula_info.name == "Return on Assets"
        codes = {q.indicator_code for q in queries}
        assert "60" in codes
        assert "270" in codes

    def test_resolve_growth_adds_previous_year(self, resolver):
        formula_info, queries = resolver.resolve("revenue_growth", ["VNM"], [2023])
        assert formula_info.requires_previous_year is True
        years = {q.year for q in queries}
        assert 2022 in years
        assert 2023 in years

    def test_resolve_growth_multi_year(self, resolver):
        formula_info, queries = resolver.resolve("revenue_growth", ["VNM"], [2021, 2022, 2023])
        years = {q.year for q in queries}
        assert 2020 in years
        assert 2021 in years
        assert 2022 in years
        assert 2023 in years

    def test_resolve_multiple_tickers(self, resolver):
        formula_info, queries = resolver.resolve("ROE", ["VNM", "HPG"], [2023])
        assert len(queries) == 4
        tickers = {q.ticker for q in queries}
        assert tickers == {"VNM", "HPG"}

    def test_resolve_current_ratio(self, resolver):
        formula_info, queries = resolver.resolve("current_ratio", ["MSN"], [2023])
        assert formula_info.unit == "lan"
        codes = {q.indicator_code for q in queries}
        assert "100" in codes
        assert "310" in codes
