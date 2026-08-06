import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from src.query.preprocessor import QueryPreprocessor


@pytest.fixture
def preprocessor():
    return QueryPreprocessor(reference_year=2024)


class TestNormalize:
    def test_unicode_normalization(self, preprocessor):
        text = "Doanh\u00a0thu\u2013thu\u1ea7n"
        result = preprocessor.normalize(text)
        assert "\u00a0" not in result
        assert "\u2013" not in result

    def test_expand_abbreviation_lnst(self, preprocessor):
        result = preprocessor.normalize("LNST cua VNM nam 2023")
        assert "Loi nhuan sau thue" in result

    def test_expand_abbreviation_dtt(self, preprocessor):
        result = preprocessor.normalize("DTT cua HPG")
        assert "Doanh thu thuan" in result

    def test_expand_abbreviation_vcsh(self, preprocessor):
        result = preprocessor.normalize("VCSH cua VNM")
        assert "Von chu so huu" in result

    def test_expand_abbreviation_case_insensitive(self, preprocessor):
        result = preprocessor.normalize("lnst cua VNM")
        assert "Loi nhuan sau thue" in result or "loi nhuan sau thue" in result.lower()

    def test_relative_year_nam_ngoai(self, preprocessor):
        result = preprocessor.normalize("Doanh thu nam ngoai")
        assert "2023" in result

    def test_relative_year_nam_nay(self, preprocessor):
        result = preprocessor.normalize("Doanh thu nam nay")
        assert "2024" in result

    def test_n_years_gan_nhat(self, preprocessor):
        result = preprocessor.normalize("3 nam gan nhat")
        assert "2022" in result
        assert "2023" in result
        assert "2024" in result

    def test_fix_typo(self, preprocessor):
        result = preprocessor.normalize("doanh thi cua VNM")
        assert "doanh thu" in result.lower()

    def test_whitespace_cleanup(self, preprocessor):
        result = preprocessor.normalize("  Doanh   thu   cua  VNM  ")
        assert "  " not in result
        assert result == result.strip()


class TestExtractYearList:
    def test_explicit_year(self, preprocessor):
        years = preprocessor.extract_year_list("Doanh thu VNM nam 2023")
        assert years == [2023]

    def test_multiple_years(self, preprocessor):
        years = preprocessor.extract_year_list("So sanh 2022 va 2023")
        assert years == [2022, 2023]

    def test_year_range(self, preprocessor):
        years = preprocessor.extract_year_list("tu 2020 den 2023")
        assert years == [2020, 2021, 2022, 2023]

    def test_relative_year(self, preprocessor):
        years = preprocessor.extract_year_list("nam ngoai")
        assert 2023 in years

    def test_n_years(self, preprocessor):
        years = preprocessor.extract_year_list("3 nam gan nhat")
        assert len(years) == 3
        assert years == [2022, 2023, 2024]

    def test_no_year(self, preprocessor):
        years = preprocessor.extract_year_list("Doanh thu cua VNM")
        assert years == []
