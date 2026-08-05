import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from src.query.entity_extractor import EntityExtractor


@pytest.fixture
def extractor():
    return EntityExtractor()


class TestExtractTickers:
    def test_direct_ticker(self, extractor):
        tickers = extractor.extract_tickers("Doanh thu cua VNM nam 2023")
        assert "VNM" in tickers

    def test_multiple_tickers(self, extractor):
        tickers = extractor.extract_tickers("So sanh HPG va HSG")
        assert "HPG" in tickers
        assert "HSG" in tickers

    def test_alias_vinamilk(self, extractor):
        tickers = extractor.extract_tickers("Doanh thu cua Vinamilk nam 2023")
        assert "VNM" in tickers

    def test_alias_hoa_phat(self, extractor):
        tickers = extractor.extract_tickers("Loi nhuan cua Hoa Phat")
        assert "HPG" in tickers

    def test_alias_masan(self, extractor):
        tickers = extractor.extract_tickers("ROE cua Masan nam 2023")
        assert "MSN" in tickers

    def test_alias_techcombank(self, extractor):
        tickers = extractor.extract_tickers("No ngan han cua Techcombank")
        assert "TCB" in tickers

    def test_no_ticker(self, extractor):
        tickers = extractor.extract_tickers("Cong ty nao co doanh thu cao nhat?")
        assert tickers == []

    def test_unknown_ticker_ignored(self, extractor):
        tickers = extractor.extract_tickers("XYZ khong ton tai")
        assert "XYZ" not in tickers


class TestExtractYears:
    def test_single_year(self, extractor):
        years = extractor.extract_years("Doanh thu VNM nam 2023")
        assert years == [2023]

    def test_multiple_years(self, extractor):
        years = extractor.extract_years("So sanh 2022 va 2023")
        assert years == [2022, 2023]

    def test_no_year(self, extractor):
        years = extractor.extract_years("Doanh thu cua VNM")
        assert years == []


class TestExtractIndicators:
    def test_doanh_thu_thuan(self, extractor):
        indicators = extractor.extract_indicators("Doanh thu thuan cua VNM")
        codes = [ind["indicator_code"] for ind in indicators]
        assert "IS.10" in codes

    def test_loi_nhuan_sau_thue(self, extractor):
        indicators = extractor.extract_indicators("Loi nhuan sau thue cua HPG")
        codes = [ind["indicator_code"] for ind in indicators]
        assert "IS.60" in codes

    def test_tong_tai_san(self, extractor):
        indicators = extractor.extract_indicators("Tong tai san cua VNM")
        codes = [ind["indicator_code"] for ind in indicators]
        assert "BS.270" in codes

    def test_von_chu_so_huu(self, extractor):
        indicators = extractor.extract_indicators("Von chu so huu cua MSN")
        codes = [ind["indicator_code"] for ind in indicators]
        assert "BS.400" in codes

    def test_hang_ton_kho(self, extractor):
        indicators = extractor.extract_indicators("Hang ton kho cua AAA")
        codes = [ind["indicator_code"] for ind in indicators]
        assert any("BS.140" in c or "BS.141" in c for c in codes)

    def test_no_phai_tra(self, extractor):
        indicators = extractor.extract_indicators("No phai tra cua FPT")
        codes = [ind["indicator_code"] for ind in indicators]
        assert "BS.300" in codes


class TestExtractAll:
    def test_full_extraction(self, extractor):
        result = extractor.extract_all("Doanh thu thuan cua VNM nam 2023")
        assert "VNM" in result["tickers"]
        assert 2023 in result["years"]
        assert "IS.10" in result["indicator_codes"]

    def test_multi_entity(self, extractor):
        result = extractor.extract_all("So sanh loi nhuan sau thue cua HPG va HSG nam 2023")
        assert "HPG" in result["tickers"]
        assert "HSG" in result["tickers"]
        assert 2023 in result["years"]
        assert "IS.60" in result["indicator_codes"]
