import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from src.query.entity_extractor import EntityExtractor
from src.query.entity_resolver import EntityResolver


@pytest.fixture
def resolver():
    extractor = EntityExtractor()
    return EntityResolver(entity_dict=extractor.entity_dict)


class TestResolveCompany:
    def test_exact_match_ticker(self, resolver):
        assert resolver.resolve_company("VNM") == "VNM"

    def test_exact_match_alias(self, resolver):
        assert resolver.resolve_company("Vinamilk") == "VNM"

    def test_exact_match_full_name(self, resolver):
        result = resolver.resolve_company("Cong ty Co phan Sua Viet Nam")
        assert result == "VNM"

    def test_exact_match_hoa_phat(self, resolver):
        assert resolver.resolve_company("Hoa Phat") == "HPG"

    def test_exact_match_masan(self, resolver):
        assert resolver.resolve_company("Masan") == "MSN"

    def test_case_insensitive(self, resolver):
        assert resolver.resolve_company("vinamilk") == "VNM"

    def test_unknown_company(self, resolver):
        result = resolver.resolve_company("Cong ty ABC khong ton tai")
        assert result is None

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("rapidfuzz"),
        reason="rapidfuzz not installed"
    )
    def test_fuzzy_match(self, resolver):
        result = resolver.resolve_company("Hoa Phat Group")
        assert result == "HPG"

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("rapidfuzz"),
        reason="rapidfuzz not installed"
    )
    def test_fuzzy_match_vinhomes(self, resolver):
        result = resolver.resolve_company("Vinhomes")
        assert result == "VHM"
