from pathlib import Path

import pandas as pd

from src.ingestion.chunk_builder import enrich_and_normalize_table
from src.ingestion.pipeline import run_ingestion_pipeline
from src.ingestion.processing import (
    FINAL_COLUMNS,
    collect_dictionary_features,
    create_dictionary_stats,
    detect_tables,
    infer_report_type,
    parse_html_table_line,
    parse_number,
    parse_table_line,
    read_structured_table,
    save_parsed_table,
)
from src.ingestion.schemas import DetectedTable


HTML_NAME_FIRST = """<table>
<tr><th>Chỉ tiêu</th><th>Mã số</th><th>TM</th><th>2015</th><th>2014</th></tr>
<tr><td>Doanh thu bán hàng</td><td>01</td><td>6.1</td><td>1.234</td><td>1.000</td></tr>
<tr><td>Lợi nhuận sau thuế</td><td>60</td><td></td><td>120</td><td>100</td></tr>
</table>"""


def _table() -> DetectedTable:
    return DetectedTable("income_statement", "BÁO CÁO KẾT QUẢ KINH DOANH", 1, 2, [HTML_NAME_FIRST])


def test_semantic_html_columns_preserve_leading_zero_and_note():
    rows = parse_html_table_line(HTML_NAME_FIRST)
    assert rows[0] == {
        "item_code": "01",
        "item_name_raw": "Doanh thu bán hàng",
        "note_reference": "6.1",
        "current_value_raw": "1.234",
        "previous_value_raw": "1.000",
    }
    assert rows[1]["note_reference"] is None


def test_text_fallback_rejects_notes_pages_contracts_and_missing_code():
    rejected = [
        "THUYẾT MINH BÁO CÁO TÀI CHÍNH 2015 2014",
        "===== PAGE 28 =====",
        "Số hợp đồng 01/2015 ngày 31 tháng 12 năm 2015",
        "Doanh thu bán hàng 120 100",
    ]
    assert all(parse_table_line(line) is None for line in rejected)
    assert parse_table_line("01 Doanh thu bán hàng 120 100")["item_code"] == "01"


def test_malformed_rowspan_value_is_rejected_without_aborting():
    malformed = "963.717.122.052237.314.356.418726.402.765.634"
    assert parse_number(malformed) is None
    html = (
        "<table><tr><th>Mã số</th><th>Chỉ tiêu</th><th>Năm nay</th><th>Năm trước</th></tr>"
        f"<tr><td>110</td><td>Tiền</td><td>{malformed}</td><td>291.674.680.985</td></tr></table>"
    )
    rows = parse_html_table_line(html)
    assert rows == [{
        "item_code": "110", "item_name_raw": "Tiền", "note_reference": None,
        "current_value_raw": None, "previous_value_raw": "291.674.680.985",
    }]


def test_detect_tables_stops_at_matching_html_table_and_rejects_notes():
    lines = [
        "BẢNG CÂN ĐỐI KẾ TOÁN",
        "Đơn vị: VND",
        HTML_NAME_FIRST,
        "===== PAGE 28 =====",
        "THÔNG TIN BỔ SUNG CHO CÁC KHOẢN MỤC TRÊN BẢNG CÂN ĐỐI KẾ TOÁN",
        HTML_NAME_FIRST,
    ]
    tables = detect_tables(lines)
    assert len(tables) == 1
    assert tables[0].lines == [HTML_NAME_FIRST]


def test_notes_tables_are_not_redetected_as_primary_statements():
    lines = [
        "BẢNG CÂN ĐỐI KẾ TOÁN", HTML_NAME_FIRST,
        "THUYẾT MINH BÁO CÁO TÀI CHÍNH",
        "Bảng cân đối kế toán tại ngày 31 tháng 12 như sau:", HTML_NAME_FIRST,
    ]
    tables = detect_tables(lines)
    assert len(tables) == 1
    assert tables[0].start_line == 1


def test_glued_ocr_headers_still_get_semantic_mapping():
    html = """<table>
    <tr><th>Másố</th><th>CHỈ TIÊU</th><th>Thuyếtminh</th><th>Nămnay</th><th>Nămtrước</th></tr>
    <tr><td>01</td><td>Doanh thu</td><td>5.1</td><td>120</td><td>100</td></tr>
    </table>"""
    assert parse_html_table_line(html) == [{
        "item_code": "01", "item_name_raw": "Doanh thu",
        "note_reference": "5.1", "current_value_raw": "120",
        "previous_value_raw": "100",
    }]


def test_enrichment_period_report_type_schema_and_dictionary(tmp_path: Path):
    raw = pd.DataFrame(parse_html_table_line(HTML_NAME_FIRST))
    enriched = enrich_and_normalize_table(
        raw,
        "Đơn vị tính: nghìn đồng",
        {
            "sticker": "AAA", "year": 2015, "report_type": "consolidated",
            "source_file": "x.txt", "source_path": "/x.txt",
        },
        _table(),
    )
    assert enriched["item_code"].tolist() == ["01", "01", "60", "60"]
    assert enriched["period"].tolist() == [2015, 2014, 2015, 2014]
    assert enriched["report_type"].unique().tolist() == ["consolidated"]

    output = save_parsed_table(
        enriched, tmp_path, "AAA", 2015, "income_statement",
        report_type="consolidated",
    )
    saved = read_structured_table(output)
    assert saved.columns.tolist() == FINAL_COLUMNS
    assert str(saved["item_code"].dtype) == "string"
    assert saved.loc[0, "item_code"] == "01"
    assert "unit_multiplier" not in saved and "source_file" not in saved

    stats = create_dictionary_stats()
    collect_dictionary_features(enriched, stats)
    assert ("income_statement", "01") in stats["indicators"]
    invalid = enriched.copy()
    invalid["item_code"] = None
    clean_stats = create_dictionary_stats()
    collect_dictionary_features(invalid, clean_stats)
    assert not clean_stats["indicators"]


def test_report_type_separate_is_preserved():
    raw = pd.DataFrame(parse_html_table_line(HTML_NAME_FIRST))
    enriched = enrich_and_normalize_table(
        raw, "VND",
        {
            "sticker": "AAA", "year": 2015, "report_type": "separate",
            "source_file": "x.txt", "source_path": "/x.txt",
        },
        _table(),
    )
    assert set(enriched["report_type"]) == {"separate"}


def test_report_type_is_inferred_from_primary_heading_beyond_front_page():
    lines = ["CÔNG BỐ THÔNG TIN"] * 132 + [
        "BÁO CÁO TÌNH HÌNH TÀI CHÍNH RIÊNG", HTML_NAME_FIRST,
    ]
    tables = detect_tables(lines)
    assert tables
    assert infer_report_type(lines, tables) == "separate"


def test_aaa_2015_consolidated_regression_against_golden(tmp_path: Path):
    project = Path(__file__).resolve().parents[1]
    source = project / "data/financial_statements/AAA/2015/AAA_financial_statements_2015_consolidated"
    if not source.exists():
        return

    output = tmp_path / "parsed"
    result = run_ingestion_pipeline({
        "source_dir": str(source),
        "output_dir": str(output),
        "parser": {"minimum_table_rows": 3, "maximum_null_ratio": 0.7},
        "defaults": {"report_type": "consolidated"},
        "dictionary_builder": {"enabled": False},
        "entity_dictionary": {"rebuild": False},
        "indexing": {"enabled": False},
    })
    balance_files = sorted((output / "tables").glob("AAA_2015_consolidated_balance_sheet*.csv"))
    assert balance_files
    generated = pd.concat(
        [read_structured_table(path) for path in balance_files], ignore_index=True,
    )

    assert result["error_count"] == 0
    assert generated.columns.tolist() == FINAL_COLUMNS
    assert str(generated["item_code"].dtype) == "string"
    assert generated["item_code"].notna().all()
    assert {"270", "300", "400"} <= set(generated["item_code"])
    assert set(generated["period"]) == {2014, 2015}
    assert set(generated["report_type"]) == {"consolidated"}
    suspicious = generated["item_name_raw"].str.contains(
        r"PAGE|THÔNG TIN BỔ SUNG|THUYẾT MINH BÁO CÁO|HỢP ĐỒNG",
        case=False, regex=True, na=False,
    )
    assert not suspicious.any()
