from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    bm25_results: list[dict],
    dense_results: list[dict],
    rrf_k: int = 60,
    top_k: int = 10,
) -> list[dict]:
    logger.info(
        f"Starting RRF fusion with {len(bm25_results)} BM25 results and {len(dense_results)} dense results (rrf_k={rrf_k}, top_k={top_k})"
    )
    scores = defaultdict(float)
    documents = {}

    rankings = [bm25_results, dense_results]

    for ranking in rankings:
        for rank, result in enumerate(ranking, start=1):
            document = result["document"]
            chunk_id = document["chunk_id"]

            scores[chunk_id] += 1.0 / (rrf_k + rank)
            documents[chunk_id] = document

    logger.debug(f"Fused scores across {len(scores)} unique documents")

    sorted_ids = sorted(
        scores,
        key=scores.get,
        reverse=True,
    )[:top_k]

    results = [
        {
            "score": scores[chunk_id],
            "document": documents[chunk_id],
        }
        for chunk_id in sorted_ids
    ]

    logger.info(
        f"Reciprocal Rank Fusion completed with {len(results)} final results from {len(scores)} unique documents"
    )
    return results
