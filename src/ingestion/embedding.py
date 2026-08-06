import numpy as np
from pathlib import Path
import json

from src.config_loader import load_config


def _embedding_config() -> dict:
    config = load_config().get("embedding")
    if not isinstance(config, dict):
        raise ValueError("Thiếu section 'embedding' trong config.yaml")
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

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=normalize_embeddings,
        convert_to_numpy=True,
    )

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

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

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
        documents: ập văn bản
        output_dir: Đường dẫn thư mục
    """

    import faiss

    output_dir = Path(output_dir)
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

    return results



