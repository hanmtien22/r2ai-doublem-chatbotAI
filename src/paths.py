"""Tìm dữ liệu ở cả máy local lẫn Kaggle.

Trên Kaggle, dataset được mount read-only vào /kaggle/input/<tên-dataset>/ và
tên đó do người upload đặt, nên không thể hard-code đường dẫn. Module này dò
theo tên file đặc trưng thay vì đoán tên thư mục.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Cho phép trỏ sang chỗ khác để test được luồng Kaggle mà không cần máy Kaggle
KAGGLE_INPUT = Path(os.environ.get("R2AI_KAGGLE_INPUT", "/kaggle/input"))
KAGGLE_WORKING = Path(os.environ.get("R2AI_KAGGLE_WORKING", "/kaggle/working"))

# File cần cho pipeline hỏi–đáp
DOCUMENTS_NAME = "retrieval_documents.jsonl"
QUESTIONS_NAME = "questions.jsonl"
BM25_NAME = "bm25.pkl"
DICTIONARY_NAMES = ("entity_dictionary.json", "indicator_aliases.json", "schema_mapping.json")


def is_kaggle() -> bool:
    return KAGGLE_INPUT.exists() or "KAGGLE_KERNEL_RUN_TYPE" in os.environ


def _search_roots() -> list[Path]:
    """Các thư mục gốc để dò, ưu tiên dataset Kaggle rồi tới repo."""
    roots: list[Path] = []
    env_root = os.environ.get("R2AI_DATA_DIR")
    if env_root:
        roots.append(Path(env_root))
    if KAGGLE_INPUT.exists():
        roots.extend(sorted(p for p in KAGGLE_INPUT.iterdir() if p.is_dir()))
    roots.append(PROJECT_ROOT / "data")
    roots.append(PROJECT_ROOT)
    return roots


def _find_file(filename: str, roots: Optional[Iterable[Path]] = None) -> Optional[Path]:
    """Tìm file theo tên, không quá sâu để khỏi quét cả dataset lớn."""
    for root in roots or _search_roots():
        if not root.exists():
            continue
        direct = root / filename
        if direct.is_file():
            return direct
        # rglob có thể rất chậm trên dataset lớn -> giới hạn 4 cấp
        for depth in range(1, 5):
            pattern = "/".join(["*"] * depth) + "/" + filename
            for match in root.glob(pattern):
                if match.is_file():
                    return match
    return None


def find_documents() -> Optional[Path]:
    return _find_file(DOCUMENTS_NAME)


def find_questions() -> Optional[Path]:
    return _find_file(QUESTIONS_NAME)


def find_index_dir() -> Optional[Path]:
    """Thư mục chứa bm25.pkl (nếu đã build sẵn và được upload kèm)."""
    bm25 = _find_file(BM25_NAME)
    return bm25.parent if bm25 else None


def find_dictionaries() -> Optional[Path]:
    entity = _find_file(DICTIONARY_NAMES[0])
    return entity.parent if entity else None


def writable_dir() -> Path:
    """Nơi ghi output/index tạm: /kaggle/working trên Kaggle, repo ở local."""
    if KAGGLE_WORKING.exists() and os.access(KAGGLE_WORKING, os.W_OK):
        return KAGGLE_WORKING
    return PROJECT_ROOT


def resolve_all() -> dict[str, Optional[Path]]:
    """Gom mọi đường dẫn cần thiết, kèm log để dễ chẩn đoán khi thiếu file."""
    resolved = {
        "documents": find_documents(),
        "questions": find_questions(),
        "index_dir": find_index_dir(),
        "dictionaries": find_dictionaries(),
        "output_dir": writable_dir(),
    }
    for name, path in resolved.items():
        logger.info("%-13s: %s", name, path if path else "KHÔNG TÌM THẤY")
    return resolved


def dictionary_path(filename: str) -> Path:
    """Đường dẫn tới một file từ điển, ưu tiên thư mục tìm thấy được.

    Trả về đường dẫn trong repo nếu không tìm thấy, để phần gọi vẫn báo lỗi
    "không tìm thấy file" như cũ thay vì lỗi khó hiểu.
    """
    found = _find_file(filename)
    if found:
        return found
    return PROJECT_ROOT / "data" / "dictionaries" / filename
