"""Chuẩn bị môi trường Kaggle rồi dựng sẵn pipeline hỏi–đáp.

Đặt trong `src/` chứ không phải thư mục `kaggle/` vì Kaggle đã cài sẵn package
`kaggle` (API client) — trùng tên sẽ che mất module này.

Dùng trong notebook:

    import sys; sys.path.insert(0, "/kaggle/working/r2ai-doublem-chatbotAI")
    from src.kaggle_setup import setup, build_pipeline

    setup()                       # cài thiếu gì cài nấy, dò dữ liệu, build BM25
    pipeline = build_pipeline()   # backend LLM tự chọn theo máy đang chạy
    print(pipeline.run("Chi phí phạt của công ty mẹ SCR năm 2017 là bao nhiêu tỷ đồng?")["answer"])
"""
from __future__ import annotations

import json
import logging
import pickle
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Kaggle đã có sẵn numpy/pandas/torch/transformers; đây là phần thường thiếu.
REQUIRED_PACKAGES = {
    "rank_bm25": "rank-bm25",
    "rapidfuzz": "rapidfuzz",
    "yaml": "PyYAML",
    # transformers cần accelerate khi nạp model lên GPU bằng device_map
    "accelerate": "accelerate",
}


def _log(message: str) -> None:
    print(f"[setup] {message}", flush=True)


def ensure_dependencies(extra: Optional[list[str]] = None) -> None:
    """Cài các gói còn thiếu. Kaggle bật sẵn internet mới cài được."""
    missing = []
    for module_name, pip_name in REQUIRED_PACKAGES.items():
        try:
            __import__(module_name)
        except ImportError:
            missing.append(pip_name)
    missing.extend(extra or [])

    if not missing:
        _log("Đã đủ thư viện, không cần cài thêm.")
        return

    _log(f"Đang cài: {', '.join(missing)}")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", *missing],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        _log("Cài xong.")
        return
    # Không dừng cả setup: có gói chỉ là tuỳ chọn (accelerate), và pipeline vẫn
    # chạy được ở chế độ hạn chế. Thiếu gói bắt buộc thì import sau sẽ báo rõ.
    _log(
        "CẢNH BÁO: cài đặt thất bại (Kaggle có bật Internet chưa?).\n"
        f"        {(result.stderr or result.stdout).strip().splitlines()[-1:] or ['']}"
    )


def _count_lines(path: Path) -> int:
    with open(path, "rb") as f:
        return sum(1 for line in f if line.strip())


def _index_matches(meta_path: Path, documents_path: Path) -> bool:
    """Index có khớp bộ tài liệu hiện tại không (dựa trên số tài liệu)."""
    if not meta_path.exists():
        # Index do người dùng tự upload, không có metadata -> tin nhưng cảnh báo
        _log("CẢNH BÁO: index không có bm25.meta.json, không kiểm tra được độ khớp.")
        return True
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return int(meta.get("documents", -1)) == _count_lines(documents_path)
    except Exception:
        return False


def ensure_bm25_index(documents_path: Path, index_dir: Path) -> Path:
    """Trả về thư mục chứa bm25.pkl, tự build nếu chưa có.

    Index đầy đủ nặng ~2.4GB nên không nên upload lên Kaggle. Build lại từ
    `retrieval_documents.jsonl` chỉ mất khoảng 2-3 phút, và bản build ở đây bỏ
    `tokenized_corpus` (chỉ cần lúc build) nên file nhẹ hơn nhiều.
    """
    from rank_bm25 import BM25Okapi

    from src.indexing.bm25 import tokenize_for_bm25

    index_dir.mkdir(parents=True, exist_ok=True)
    bm25_path = index_dir / "bm25.pkl"
    meta_path = index_dir / "bm25.meta.json"

    if bm25_path.exists():
        # Pipeline lấy documents TỪ TRONG pickle, nên một index cũ không khớp bộ
        # tài liệu hiện tại sẽ bị dùng im lặng và trả lời sai. Đối chiếu số dòng.
        if _index_matches(meta_path, documents_path):
            _log(f"Đã có BM25 index khớp dữ liệu: {bm25_path}")
            return index_dir
        _log(f"BM25 index tại {bm25_path} không khớp {documents_path.name} -> build lại")

    _log(f"Chưa có BM25 index, đang build từ {documents_path} ...")
    started = time.time()

    with open(documents_path, "r", encoding="utf-8") as f:
        documents = [json.loads(line) for line in f if line.strip()]
    _log(f"  đọc {len(documents):,} tài liệu ({time.time() - started:.0f}s)")

    tokenized = [tokenize_for_bm25(doc.get("text", doc.get("content", ""))) for doc in documents]
    bm25 = BM25Okapi(tokenized)
    _log(f"  build xong ({time.time() - started:.0f}s), đang ghi ra đĩa ...")

    with open(bm25_path, "wb") as f:
        pickle.dump({"bm25": bm25, "documents": documents}, f, protocol=pickle.HIGHEST_PROTOCOL)
    meta_path.write_text(
        json.dumps({"documents": len(documents), "source": str(documents_path)}),
        encoding="utf-8",
    )

    size_mb = bm25_path.stat().st_size / 1024 / 1024
    _log(f"Xong sau {time.time() - started:.0f}s -> {bm25_path} ({size_mb:.0f} MB)")
    return index_dir


def setup(install: bool = True, build_index: bool = True) -> dict:
    """Dò dữ liệu, cài thư viện, build index. Trả về dict đường dẫn."""
    if install:
        ensure_dependencies()

    from src.paths import is_kaggle, resolve_all

    _log(f"Môi trường Kaggle: {is_kaggle()}")
    paths = resolve_all()

    documents = paths["documents"]
    if documents is None:
        raise FileNotFoundError(
            "Không tìm thấy retrieval_documents.jsonl.\n"
            "Hãy thêm dataset chứa file này vào notebook (Add Input), "
            "hoặc đặt biến môi trường R2AI_DATA_DIR trỏ tới thư mục dữ liệu."
        )
    for name in ("questions", "dictionaries"):
        if paths[name] is None:
            _log(f"CẢNH BÁO: không tìm thấy {name}")

    if build_index:
        # Index trong /kaggle/input là read-only -> luôn build vào thư mục ghi được
        index_dir = paths["index_dir"]
        if index_dir is not None and not _index_matches(index_dir / "bm25.meta.json", documents):
            _log(f"Bỏ qua index không khớp tại {index_dir}")
            index_dir = None
        if index_dir is None:
            # /kaggle/input là read-only -> luôn build vào thư mục ghi được
            index_dir = ensure_bm25_index(documents, Path(paths["output_dir"]) / "indexes")
        paths["index_dir"] = index_dir

    for name, path in paths.items():
        _log(f"{name:13s}: {path}")
    return paths


def build_pipeline(
    llm_backend: str = "auto",
    llm_model: Optional[str] = None,
    use_llm_router: bool = True,
    paths: Optional[dict] = None,
    **kwargs,
):
    """Dựng FullQAPipeline với đường dẫn và backend LLM phù hợp Kaggle."""
    from src.qa_pipeline import FullQAPipeline

    paths = paths or setup()
    return FullQAPipeline(
        documents_path=paths["documents"],
        index_dir=paths["index_dir"],
        llm_backend=llm_backend,
        llm_model=llm_model,
        use_llm_router=use_llm_router,
        **kwargs,
    )
