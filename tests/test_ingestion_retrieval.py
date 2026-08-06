import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from src.ingestion.chunk_builder import (
    bm25_search,
    build_bm25_index,
    load_documents,
)
from src.ingestion.embedding import dense_search
from src.ingestion.pipeline import run_ingestion_pipeline
from src.ingestion.processing import build_entity_dictionary, parse_html_table_line
from src.query.pipeline import QueryPipeline
from src.retrieval.pipeline import QueryRetrievalPipeline


SAMPLE_TABLE = """<table>
<tr><td>Mã số</td><td>CHỈ TIÊU</td><td>Thuyết minh</td><td>Năm nay</td><td>Năm trước</td></tr>
<tr><td>01</td><td>Doanh thu bán hàng</td><td></td><td>120</td><td>100</td></tr>
<tr><td>10</td><td>Doanh thu thuần</td><td></td><td>110</td><td>90</td></tr>
<tr><td>60</td><td>Lợi nhuận sau thuế</td><td></td><td>12</td><td>10</td></tr>
</table>""".replace("\n", "")


def _run_sample_ingestion(tmp_path: Path) -> Path:
    source_dir = tmp_path / "source" / "AAA" / "2025" / "report"
    source_dir.mkdir(parents=True)
    (source_dir / "sample_extracted.txt").write_text(
        "\n".join([
            "BÁO CÁO KẾT QUẢ HOẠT ĐỘNG KINH DOANH",
            "Đơn vị tính: VND",
            SAMPLE_TABLE,
        ]),
        encoding="utf-8",
    )
    output_dir = tmp_path / "output"
    result = run_ingestion_pipeline({
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "parser": {"minimum_table_rows": 3, "maximum_null_ratio": 0.7},
    })
    assert result == {
        "file_count": 1,
        "record_count": 6,
        "document_count": 6,
        "error_count": 0,
    }
    return output_dir


def test_parse_embedded_html_table():
    rows = parse_html_table_line(SAMPLE_TABLE)
    assert [row["item_code"] for row in rows] == ["01", "10", "60"]
    assert rows[1]["current_value_raw"] == "110"
    assert rows[1]["previous_value_raw"] == "90"


def test_ingestion_to_structured_retrieval(tmp_path):
    output_dir = _run_sample_ingestion(tmp_path)
    pipeline = QueryRetrievalPipeline(
        output_dir / "retrieval_documents.jsonl",
        query_pipeline=QueryPipeline(reference_year=2025, use_llm_fallback=False),
    )

    result = pipeline.process("Doanh thu thuan cua AAA nam 2025")
    code_10_hits = [
        hit for hit in result["hits"]
        if hit["query"]["indicator_code"] == "10"
    ]

    assert code_10_hits
    assert code_10_hits[0]["document"]["metadata"]["section"] == "IS"
    assert code_10_hits[0]["document"]["metadata"]["period"] == 2025
    assert code_10_hits[0]["document"]["metadata"]["value"] == 110


def test_ingestion_builds_entity_dictionary_and_bm25_index(tmp_path):
    source_dir = tmp_path / "source" / "AAA" / "2025" / "report"
    source_dir.mkdir(parents=True)
    (source_dir / "sample_extracted.txt").write_text(
        "\n".join([
            "BÁO CÁO KẾT QUẢ HOẠT ĐỘNG KINH DOANH",
            "Đơn vị tính: VND",
            SAMPLE_TABLE,
        ]),
        encoding="utf-8",
    )
    companies_path = tmp_path / "companies.csv"
    companies_path.write_text("Mã CK,Tên công ty\nAAA,Công ty Nhựa An Phát\n", encoding="utf-8")
    output_dir = tmp_path / "output"

    result = run_ingestion_pipeline({
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "parser": {"minimum_table_rows": 3, "maximum_null_ratio": 0.7},
        "entity_dictionary": {
            "rebuild": True,
            "source_csv": str(companies_path),
            "output_file": "entity_dictionary.json",
        },
        "indexing": {
            "enabled": True,
            "output_dir": "indexes",
            "bm25_enabled": True,
            "dense_enabled": False,
        },
    })

    assert result["error_count"] == 0
    assert (output_dir / "entity_dictionary.json").exists()
    assert (output_dir / "indexes" / "bm25.pkl").exists()


def test_hybrid_retrieval_falls_back_to_bm25(tmp_path):
    documents = [{
        "chunk_id": "AAA:2025:IS:999",
        "text": "AAA 2025 doanh thu thuan",
        "metadata": {
            "ticker": "AAA",
            "period": 2025,
            "section": "IS",
            "item_code": "999",
            "value": 110,
            "source_file": "sample.txt",
        },
    }]
    documents_path = tmp_path / "retrieval_documents.jsonl"
    documents_path.write_text(json.dumps(documents[0]) + "\n", encoding="utf-8")
    index_dir = tmp_path / "indexes"
    build_bm25_index(documents, index_dir / "bm25.pkl")
    pipeline = QueryRetrievalPipeline(
        documents_path,
        query_pipeline=QueryPipeline(reference_year=2025, use_llm_fallback=False),
        index_dir=index_dir,
    )

    result = pipeline.process("Doanh thu thuan cua AAA nam 2025")

    assert result["hits"]
    assert result["hits"][0]["document"]["metadata"]["item_code"] == "999"


def test_entity_dictionary_accepts_vietnamese_csv_headers(tmp_path):
    csv_path = tmp_path / "companies.csv"
    csv_path.write_text("Mã CK,Tên công ty\nAAA,Công ty Nhựa An Phát\n", encoding="utf-8")

    entities = build_entity_dictionary(csv_path, tmp_path / "entities.json")

    assert entities["AAA"]["full_name"] == "Công ty Nhựa An Phát"
    assert "AAA" in entities["AAA"]["aliases"]
    assert "nhua an phat" in entities["AAA"]["aliases"]


def test_bm25_jsonl_build_and_search(tmp_path):
    documents_path = tmp_path / "documents.jsonl"
    documents = [
        {"chunk_id": "1", "text": "doanh thu thuan", "metadata": {}},
        {"chunk_id": "2", "text": "tong tai san", "metadata": {}},
        {"chunk_id": "3", "text": "von chu so huu", "metadata": {}},
    ]
    documents_path.write_text(
        "".join(json.dumps(document, ensure_ascii=False) + "\n" for document in documents),
        encoding="utf-8",
    )
    loaded = load_documents(documents_path)
    index_path = tmp_path / "bm25.pkl"
    build_bm25_index(loaded, index_path)

    with index_path.open("rb") as file:
        payload = pickle.load(file)
    results = bm25_search("doanh thu", payload["bm25"], payload["documents"], top_k=1)

    assert results[0]["document"]["chunk_id"] == "1"


def test_dense_search_returns_documents():
    class FakeModel:
        def encode(self, *args, **kwargs):
            return np.array([[1.0, 0.0]], dtype=np.float32)

    class FakeIndex:
        def search(self, query, top_k):
            return np.array([[0.9]], dtype=np.float32), np.array([[0]])

    documents = [{"chunk_id": "1", "text": "doanh thu", "metadata": {}}]
    results = dense_search(
        "doanh thu",
        FakeModel(),
        FakeIndex(),
        documents,
        top_k=1,
        normalize_embeddings=True,
    )

    assert results == [{"score": pytest.approx(0.9), "document": documents[0]}]
