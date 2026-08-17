from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_MODEL_CACHE: dict[str, object] = {}


def _get_reranker(model_name: str):
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]
    try:
        from sentence_transformers import CrossEncoder
        model = CrossEncoder(model_name, max_length=512, device="cpu")
        _MODEL_CACHE[model_name] = model
        logger.info("Reranker loaded: %s", model_name)
        return model
    except Exception as e:
        logger.warning("Cannot load reranker '%s': %s", model_name, e)
        _MODEL_CACHE[model_name] = None
        return None


class Reranker:
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        enabled: bool = True,
        score_threshold: Optional[float] = None,
    ):
        self._model_name = model_name
        self._enabled = enabled
        self._threshold = score_threshold
        self._model = _get_reranker(model_name) if enabled else None

    def rerank(self, query: str, hits: list[dict], top_k: int = 8) -> list[dict]:
        if not self._model or not hits or not query.strip():
            return hits[:top_k]

        try:
            texts = [h.get("document", h).get("text", h.get("document", h).get("content", "")) for h in hits]
            pairs = [(query, t) for t in texts]
            scores = self._model.predict(pairs, show_progress_bar=False)

            scored = []
            for score, hit in zip(scores, hits):
                h_copy = dict(hit)
                h_copy["reranker_score"] = float(score)
                scored.append(h_copy)

            scored.sort(key=lambda x: x["reranker_score"], reverse=True)

            if self._threshold is not None:
                scored = [h for h in scored if h["reranker_score"] >= self._threshold]

            logger.debug("Reranker: %d hits, top_score=%.3f", len(scored), scored[0]["reranker_score"] if scored else 0)
            return scored[:top_k]

        except Exception as e:
            logger.warning("Reranker failed: %s", e)
            return hits[:top_k]
