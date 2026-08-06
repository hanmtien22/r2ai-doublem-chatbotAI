from __future__ import annotations

from pathlib import Path

from src.ingestion.chunk_builder import build_bm25_index
from src.ingestion.embedding import (
    build_embedding,
    build_faiss_index,
    save_faiss_artifacts,
)


def build_search_indexes(
    documents: list[dict],
    output_dir: str | Path,
    indexing_config: dict,
    embedding_config: dict,
) -> dict[str, str]:
    """Build configured sparse and dense indexes from ingestion documents."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {}

    if indexing_config.get("bm25_enabled", True) and documents:
        bm25_path = output_dir / "bm25.pkl"
        build_bm25_index(documents, bm25_path)
        artifacts["bm25"] = str(bm25_path)

    if indexing_config.get("dense_enabled", False) and documents:
        embeddings = build_embedding(
            [document["text"] for document in documents],
            model_name=embedding_config.get("model_name"),
            batch_size=embedding_config.get("batch_size"),
            normalize_embeddings=embedding_config.get("normalize_embeddings"),
        )
        index = build_faiss_index(embeddings)
        dense_dir = output_dir / "faiss"
        save_faiss_artifacts(index, documents, dense_dir)
        artifacts["faiss"] = str(dense_dir / "index.faiss")

    return artifacts
