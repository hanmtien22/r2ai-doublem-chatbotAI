import json
from pathlib import Path

import pandas as pd

from src.ingestion.notes import detect_section_header, html_table_to_text
from src.ingestion.pipeline import run_ingestion_pipeline
from src.ingestion.processing import FINAL_COLUMNS, read_structured_table


TABLE = """<table><tr><th>Mã số</th><th>Chỉ tiêu</th><th>Năm nay</th><th>Năm trước</th></tr>
<tr><td>01</td><td>Doanh thu</td><td>120</td><td>100</td></tr>
<tr><td>10</td><td>Doanh thu thuần</td><td>110</td><td>90</td></tr>
<tr><td>60</td><td>Lợi nhuận sau thuế</td><td>12</td><td>10</td></tr></table>"""


def test_section_headers_and_note_table_rendering():
    assert detect_section_header("IV. THÔNG TIN BỔ SUNG").section_level == 1
    assert detect_section_header("1. Tiền và tương đương tiền").section_level == 2
    assert detect_section_header("1.2 Các khoản phải thu").section_level == 3
    assert detect_section_header("a) Tài sản thế chấp").section_level == 3
    assert html_table_to_text("<table><tr><th>Ngân hàng</th><th>Số tiền</th></tr><tr><td>VCB</td><td>100 tỷ</td></tr></table>") == "Ngân hàng | Số tiền\nVCB | 100 tỷ"


def test_pipeline_separates_primary_csv_and_notes_csv(tmp_path: Path):
    source = tmp_path / "source" / "AAA" / "2025" / "report"
    source.mkdir(parents=True)
    (source / "AAA_2025_consolidated.txt").write_text("\n".join([
        "BÁO CÁO KẾT QUẢ HOẠT ĐỘNG KINH DOANH", "Đơn vị: triệu đồng", TABLE,
        "THUYẾT MINH BÁO CÁO TÀI CHÍNH", "IV. THÔNG TIN BỔ SUNG",
        "1. Các khoản vay", "Khoản vay VCB được thế chấp bằng nhà xưởng.",
        "<table><tr><th>Ngân hàng</th><th>Số tiền</th></tr><tr><td>VCB</td><td>100 tỷ</td></tr></table>",
    ]), encoding="utf-8")
    output = tmp_path / "parsed"
    result = run_ingestion_pipeline({
        "source_dir": str(source), "output_dir": str(output),
        "parser": {"minimum_table_rows": 3, "maximum_null_ratio": 0.7},
        "notes": {"enabled": True, "output_dir": "notes", "max_chars": 3000, "overlap_chars": 300, "min_chars": 10},
        "dictionary_builder": {"enabled": False}, "entity_dictionary": {"rebuild": False},
        "indexing": {"enabled": False}, "defaults": {"report_type": "consolidated"},
    })
    assert result["error_count"] == 0
    structured = read_structured_table(output / "tables/AAA_2025_consolidated_income_statement.csv")
    assert structured.columns.tolist() == FINAL_COLUMNS
    assert structured["item_code"].tolist()[:2] == ["01", "01"]
    assert not structured["item_name_raw"].str.contains("vay|thế chấp", case=False).any()

    notes = pd.read_csv(output / "notes/AAA_2025_consolidated_notes.csv")
    assert set(notes["document_type"]) == {"notes"}
    assert notes["text"].str.contains("thế chấp", case=False).any()
    assert notes["text"].str.contains(r"VCB \| 100 tỷ", regex=True).any()

    documents = [json.loads(line) for line in (output / "retrieval_documents.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(document["metadata"].get("document_type") == "notes" for document in documents)
    assert any(document["metadata"].get("table_type") == "income_statement" for document in documents)
