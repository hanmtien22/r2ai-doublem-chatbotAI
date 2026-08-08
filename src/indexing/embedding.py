import json
import logging
from pathlib import Path
import numpy as np

from src.config_loader import load_config

logger = logging.getLogger(__name__)


def _embedding_config() -> dict:
    logger.debug("Loading embedding configuration...")
    config = load_config().get("embedding")
    if not isinstance(config, dict):
        logger.error("Missing 'embedding' section in config.yaml")
        raise ValueError("Thiếu section 'embedding' trong config.yaml")
    logger.debug("Successfully loaded embedding configuration.")
    return config


def build_embedding(
        texts: list[str],
        model_name: str | None = None,
        batch_size: int | None = None,
        normalize_embeddings: bool | None = None,
): 
    """
    Khởi tạo vector embedding
    
    Parameters:
        texts: văn bản
        model_name: tên mô hình, mặc định lấy từ config.yaml
        batch_size: kích thước lô, mặc định lấy từ config.yaml
        normalize_embeddings: chuẩn hóa vector, mặc định lấy từ config.yaml
        
    Return:
        vector embedding
    """

    config = _embedding_config()
    model_name = model_name or config["model_name"]
    batch_size = batch_size or config["batch_size"]
    if normalize_embeddings is None:
        normalize_embeddings = config["normalize_embeddings"]

    logger.info(
        f"Building embeddings for {len(texts)} texts using model='{model_name}', batch_size={batch_size}, normalize_embeddings={normalize_embeddings}"
    )

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=normalize_embeddings,
        convert_to_numpy=True,
    )

    logger.info(f"Built embeddings with shape: {embeddings.shape}")
    return embeddings.astype("float32")

def build_faiss_index(embeddings: np.ndarray):
    """
    Tạo không gian lưu trữ vector embedding
    
    Parameters:
        embeddings: vector embedding

    Return:
        index
    """
    import faiss

    dimension = embeddings.shape[1]
    vector_count = embeddings.shape[0]
    logger.info(f"Building FAISS index with dimension={dimension} for {vector_count} vectors")

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    logger.info("Successfully built FAISS index")
    return index


def save_faiss_artifacts(
    index,
    documents: list[dict],
    output_dir: str | Path,
):
    """
    Lưu index và metadata
    
    Parameters:
        index: Index faiss
        documents: Tập văn bản
        output_dir: Đường dẫn thư mục
    """

    import faiss

    output_dir = Path(output_dir)
    logger.info(f"Saving FAISS index and {len(documents)} documents to {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    faiss.write_index(
        index,
        str(output_dir / "index.faiss"),
    )

    with (output_dir / "documents.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            documents,
            file,
            ensure_ascii=False,
            indent=2,
        )
    logger.info(f"Successfully saved FAISS artifacts to {output_dir}")

def dense_search(
        query: str,
        model,
        index,
        documents: list[dict],
        top_k: int = 10,
        normalize_embeddings: bool | None = None,
) -> list[dict]:
    """
    Tìm kiếm vector embedding
    
    Parameters:
        query: Yêu cầu
        model: Mô hình
        index: Số chỉ 
        documents: Tập văn bản
        top_k: Lấy 10 kết quả cao nhất
        
    Return:
        Danh sách các từ điển gồm {score, document}

    """

    logger.info(f"Performing dense search for query: '{query}' with top_k={top_k}")
    if normalize_embeddings is None:
        normalize_embeddings = _embedding_config()["normalize_embeddings"]

    query_embedding = model.encode(
        [query],
        normalize_embeddings=normalize_embeddings,
        convert_to_numpy=True,
    ).astype("float32")

    scores, indices = index.search(query_embedding, top_k)

    results = [] 

    for score, index in zip(scores[0], indices[0]):
        if index < 0:
            continue


        results.append({
            "score": float(score),
            "document": documents[index],
        })

    logger.info(f"Dense search completed with {len(results)} results")
    return results
