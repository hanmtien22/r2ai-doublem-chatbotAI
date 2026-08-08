import json
import logging
import pickle
from pathlib import Path
import numpy as np
from rank_bm25 import BM25Okapi
from src.ingestion.processing import normalize_text

logger = logging.getLogger(__name__)


def tokenize_for_bm25(text: str) -> list[str]:
    return normalize_text(text).split()


def load_documents(json_path: str | Path) -> list[dict]:
    logger.info(f"Loading documents from {json_path}")
    documents = []

    with Path(json_path).open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                documents.append(json.loads(line))

    logger.info(f"Loaded {len(documents)} documents from {json_path}")
    return documents


def build_bm25_index(documents: list[dict], output_path: str | Path) -> None:
    logger.info(f"Building BM25 index for {len(documents)} documents")
    logger.debug("Tokenizing corpus for BM25...")
    tokenized_corpus = [
        tokenize_for_bm25(document["text"])
        for document in documents
    ]
    logger.debug("Creating BM25Okapi object...")
    bm25 = BM25Okapi(tokenized_corpus)
    payload = {
        "bm25": bm25,
        "documents": documents,
        "tokenized_corpus": tokenized_corpus,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving BM25 index payload to {output_path}")
    with output_path.open("wb") as file:
        pickle.dump(payload, file)
    logger.info(f"Successfully saved BM25 index to {output_path}")


def bm25_search(query: str, bm25, documents: list[str], top_k: int) -> list[dict]:
    logger.info(f"Performing BM25 search for query: '{query}' with top_k={top_k}")
    tokens = tokenize_for_bm25(query)
    logger.debug(f"Query tokenized into {len(tokens)} tokens: {tokens}")
    scores = bm25.get_scores(tokens)
    top_indies = np.argsort(scores)[::-1][:top_k]

    results = []
    for index in top_indies:
        results.append({
            "score": scores[index],
            "document": documents[index]
        })

    logger.info(f"BM25 search completed with {len(results)} results")
    return results
