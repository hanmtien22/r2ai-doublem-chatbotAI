"""Test cho phần chạy trên Kaggle: dò đường dẫn, chọn backend LLM, kiểm tra index."""
from __future__ import annotations

import importlib
import json

import pytest


@pytest.fixture
def fake_kaggle(tmp_path, monkeypatch):
    """Dựng cây thư mục giống Kaggle rồi nạp lại src.paths để nó đọc env mới."""
    dataset = tmp_path / "input" / "bo-du-lieu-cua-toi" / "parsed_tables"
    dataset.mkdir(parents=True)
    (dataset / "retrieval_documents.jsonl").write_text('{"text": "x"}\n', encoding="utf-8")

    dictionaries = tmp_path / "input" / "bo-du-lieu-cua-toi" / "dictionaries"
    dictionaries.mkdir(parents=True)
    (dictionaries / "entity_dictionary.json").write_text("{}", encoding="utf-8")

    working = tmp_path / "working"
    working.mkdir()

    monkeypatch.setenv("R2AI_KAGGLE_INPUT", str(tmp_path / "input"))
    monkeypatch.setenv("R2AI_KAGGLE_WORKING", str(working))

    import src.paths

    yield importlib.reload(src.paths), tmp_path

    # Nạp lại lần nữa sau khi monkeypatch gỡ env, nếu không module vẫn giữ
    # đường dẫn tmp đã bị xoá và làm hỏng các test chạy sau.
    monkeypatch.undo()
    importlib.reload(src.paths)


def test_tim_duoc_du_lieu_du_ten_thu_muc_bat_ky(fake_kaggle):
    """Tên thư mục trong /kaggle/input do người upload đặt nên phải dò theo tên file."""
    paths, tmp_path = fake_kaggle
    documents = paths.find_documents()
    assert documents is not None
    assert documents.name == "retrieval_documents.jsonl"
    assert "bo-du-lieu-cua-toi" in str(documents)


def test_ghi_ra_thu_muc_working_khi_o_kaggle(fake_kaggle):
    """/kaggle/input là read-only, output phải rơi vào /kaggle/working."""
    paths, tmp_path = fake_kaggle
    assert paths.is_kaggle() is True
    assert paths.writable_dir() == tmp_path / "working"


def test_tim_duoc_tu_dien(fake_kaggle):
    paths, _ = fake_kaggle
    assert paths.dictionary_path("entity_dictionary.json").is_file()


def test_tu_dien_khong_co_thi_tra_ve_duong_dan_repo(fake_kaggle):
    """Không tìm thấy thì trả đường dẫn mặc định để lỗi báo ra rõ ràng."""
    paths, _ = fake_kaggle
    fallback = paths.dictionary_path("khong_ton_tai.json")
    assert fallback.parts[-3:] == ("data", "dictionaries", "khong_ton_tai.json")


# --------------------------------------------------------------------------- #
#  Backend LLM                                                                 #
# --------------------------------------------------------------------------- #
def test_backend_none_tra_ve_ngay_khong_goi_mang():
    """Chế độ không LLM phải trả về tức thì.

    Nếu dùng OllamaClient trỏ vào endpoint không tồn tại thì mỗi câu hỏi phải
    chờ hết 3 lần retry với backoff.
    """
    import time

    from src.llm.factory import NullLLMClient, build_llm_client

    client = build_llm_client("none")
    assert isinstance(client, NullLLMClient)

    started = time.time()
    assert client.generate("bất kỳ") == ""
    assert client.generate_chat("hệ thống", "người dùng") == ""
    assert time.time() - started < 0.1


def test_backend_khong_hop_le_thi_bao_loi():
    from src.llm.factory import build_llm_client

    with pytest.raises(ValueError, match="backend không hợp lệ"):
        build_llm_client("khong-ton-tai")


def test_backend_doc_duoc_tu_bien_moi_truong(monkeypatch):
    from src.llm.factory import NullLLMClient, build_llm_client

    monkeypatch.setenv("R2AI_LLM_BACKEND", "none")
    assert isinstance(build_llm_client("auto"), NullLLMClient)


# --------------------------------------------------------------------------- #
#  Kiểm tra index khớp dữ liệu                                                 #
# --------------------------------------------------------------------------- #
def test_phat_hien_index_khong_khop(tmp_path):
    """Pipeline đọc tài liệu từ trong pickle, nên index lệch phải bị phát hiện."""
    from src.kaggle_setup import _index_matches

    documents = tmp_path / "docs.jsonl"
    documents.write_text('{"a":1}\n{"a":2}\n{"a":3}\n', encoding="utf-8")

    meta = tmp_path / "bm25.meta.json"
    meta.write_text(json.dumps({"documents": 3}), encoding="utf-8")
    assert _index_matches(meta, documents) is True

    meta.write_text(json.dumps({"documents": 999}), encoding="utf-8")
    assert _index_matches(meta, documents) is False


def test_index_khong_co_metadata_thi_van_dung(tmp_path):
    """Index do người dùng tự upload không có metadata -> chấp nhận, chỉ cảnh báo."""
    from src.kaggle_setup import _index_matches

    documents = tmp_path / "docs.jsonl"
    documents.write_text('{"a":1}\n', encoding="utf-8")
    assert _index_matches(tmp_path / "khong-co.json", documents) is True
