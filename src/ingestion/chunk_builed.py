from pathlib import Path
import json
from rank_bm25 import BM25Okapi
import pickle
import numpy as np

from .processing import normalize_text

def tokenize_for_bm25(text: str) -> str:
    return normalize_text(text)

def load_documents(jsonl_path: str | Path) -> list[dict]:
    documents = []

    with Path(jsonl_path).open("r", encoding="utf-8") as file:
        for line in file:
            documents.append(json.load(line))

    return documents

def build_bm25_index(
        documents: list[dict],
        output_path: str | Path,
):
    tokenized_corpus = [
        tokenize_for_bm25(document["text"] for document in documents)
    ]

    bm25 = BM25Okapi(tokenized_corpus)

    payload = {
        "bm25": bm25,
        "documents": documents,
        "tokenizes_corpus": tokenized_corpus
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("wb") as file:
        pickle.dump(payload, file)


def bm25_search(
        query: str,
        bm25,
        documents: list[dict],
        top_k: int,
) -> list[dict]:
    tokens = tokenize_for_bm25(query)
    scores = bm25.get_scores(tokens)

    top_indices = np.argsort(scores)[::-1][:top_k]

    return [
        {
            "score": float(scores[index]),
            "document": documents[index],
        } 
        for index in top_indices
    ]
