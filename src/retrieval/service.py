from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
import json
import pickle

from src.ingestion.chunk_builder import load_documents
from src.ingestion.embedding import dense_search
from src.ingestion.hybrid_search import reciprocal_rank_fusion
from src.ingestion.processing import TABLE_TYPE_TO_SECTION
from src.utils.text import remove_diacritics
from src.query.models import QueryResult, RetrievalQuery


@dataclass(frozen=True)
class RetrievalHit:
    query: RetrievalQuery | None
    document: dict
    score: float = 1.0

    def to_dict(self) -> dict:
        return {
            "query": asdict(self.query) if self.query else None,
            "score": self.score,
            "document": self.document,
        }


class DocumentRetriever:
    """Resolve structured query requests against normalized ingestion records."""

    def __init__(self, documents: Iterable[dict]):
        self._documents = list(documents)
        self._exact_index: dict[tuple[str, int, str, str], list[dict]] = {}

        for document in self._documents:
            metadata = document.get("metadata", {})
            key = self._metadata_key(metadata)
            if key is not None:
                self._exact_index.setdefault(key, []).append(document)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "DocumentRetriever":
        return cls(load_documents(path))

    @staticmethod
    def _metadata_key(metadata: dict) -> tuple[str, int, str, str] | None:
        ticker = str(metadata.get("ticker", "")).upper()
        period = metadata.get("period", metadata.get("year"))
        section = metadata.get("section")
        if not section:
            section = TABLE_TYPE_TO_SECTION.get(metadata.get("table_type"))
        item_code = metadata.get("item_code")

        if not ticker or period is None or not section or item_code is None:
            return None

        return ticker, int(period), str(section).upper(), str(item_code)

    @staticmethod
    def _query_key(query: RetrievalQuery) -> tuple[str, int, str, str]:
        return (
            query.ticker.upper(),
            int(query.year),
            query.section.upper(),
            str(query.indicator_code),
        )

    def retrieve_query(self, query: RetrievalQuery, top_k: int = 5) -> list[RetrievalHit]:
        documents = self._exact_index.get(self._query_key(query), [])
        return [
            RetrievalHit(query=query, document=document)
            for document in documents[:top_k]
        ]

    def retrieve(self, result: QueryResult, top_k_per_query: int = 5) -> list[RetrievalHit]:
        hits = []
        seen_chunk_ids = set()

        for query in result.retrieval_queries:
            for hit in self.retrieve_query(query, top_k=top_k_per_query):
                chunk_id = hit.document.get("chunk_id")
                dedupe_key = (self._query_key(query), chunk_id)
                if dedupe_key not in seen_chunk_ids:
                    hits.append(hit)
                    seen_chunk_ids.add(dedupe_key)

        return hits


class HybridDocumentRetriever(DocumentRetriever):
    """Structured retrieval with BM25, optional FAISS, fusion and validation fallback."""

    def __init__(
        self,
        documents: Iterable[dict],
        bm25_payload: dict | None = None,
        dense_model=None,
        dense_index=None,
        dense_documents: list[dict] | None = None,
        reranker=None,
        confidence_threshold: float = 0.5,
        rrf_k: int = 60,
    ):
        super().__init__(documents)
        self._bm25_payload = bm25_payload
        self._dense_model = dense_model
        self._dense_index = dense_index
        self._dense_documents = dense_documents or self._documents
        self._reranker = reranker
        self._confidence_threshold = confidence_threshold
        self._rrf_k = rrf_k

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        index_dir: str | Path | None = None,
        retrieval_config: dict | None = None,
        embedding_config: dict | None = None,
    ) -> "HybridDocumentRetriever":
        path = Path(path)
        documents = load_documents(path)
        index_dir = Path(index_dir) if index_dir else path.parent / "indexes"
        retrieval_config = retrieval_config or {}
        embedding_config = embedding_config or {}

        bm25_payload = None
        bm25_path = index_dir / "bm25.pkl"
        if bm25_path.exists():
            with bm25_path.open("rb") as file:
                bm25_payload = pickle.load(file)

        dense_model = dense_index = reranker = None
        dense_documents = None
        faiss_path = index_dir / "faiss" / "index.faiss"
        dense_documents_path = index_dir / "faiss" / "documents.json"
        if retrieval_config.get("dense_enabled", False) and faiss_path.exists():
            import faiss
            from sentence_transformers import SentenceTransformer

            dense_index = faiss.read_index(str(faiss_path))
            dense_model = SentenceTransformer(embedding_config["model_name"])
            with dense_documents_path.open(encoding="utf-8") as file:
                dense_documents = json.load(file)

        reranker_model_name = retrieval_config.get("reranker_model_name")
        if reranker_model_name:
            from sentence_transformers import CrossEncoder

            reranker = CrossEncoder(reranker_model_name)

        return cls(
            documents=documents,
            bm25_payload=bm25_payload,
            dense_model=dense_model,
            dense_index=dense_index,
            dense_documents=dense_documents,
            reranker=reranker,
            confidence_threshold=retrieval_config.get("confidence_threshold", 0.5),
            rrf_k=retrieval_config.get("rrf_k", 60),
        )

    @staticmethod
    def _valid_document(document: dict) -> bool:
        metadata = document.get("metadata", {})
        required = {"ticker", "period", "section", "item_code", "value", "source_file"}
        return bool(document.get("chunk_id")) and required.issubset(metadata)

    @staticmethod
    def _matches_query(document: dict, query: RetrievalQuery | None) -> bool:
        if query is None:
            return True
        metadata = document.get("metadata", {})
        return (
            str(metadata.get("ticker", "")).upper() == query.ticker.upper()
            and int(metadata.get("period", -1)) == int(query.year)
            and str(metadata.get("section", "")).upper() == query.section.upper()
        )

    def _hybrid_search(self, text: str, top_k: int) -> list[dict]:
        bm25_results = []
        if self._bm25_payload is not None:
            from src.ingestion.chunk_builder import bm25_search

            bm25_results = bm25_search(
                text,
                self._bm25_payload["bm25"],
                self._bm25_payload["documents"],
                top_k,
            )

        dense_results = []
        if self._dense_model is not None and self._dense_index is not None:
            dense_results = dense_search(
                text,
                self._dense_model,
                self._dense_index,
                self._dense_documents,
                top_k=top_k,
            )

        if bm25_results and dense_results:
            results = reciprocal_rank_fusion(
                bm25_results,
                dense_results,
                rrf_k=self._rrf_k,
                top_k=top_k,
            )
        else:
            results = bm25_results or dense_results

        if self._reranker is not None and results:
            scores = self._reranker.predict([
                [text, item["document"]["text"]]
                for item in results
            ])
            for item, score in zip(results, scores):
                item["score"] = float(score)
            results.sort(key=lambda item: item["score"], reverse=True)

        return results

    def _fallback_hits(
        self,
        result: QueryResult,
        query: RetrievalQuery | None,
        top_k: int,
    ) -> list[RetrievalHit]:
        search_text = result.search_text
        if query is not None:
            search_text = (
                f"{search_text} {query.ticker} {query.year} "
                f"{query.section} {query.indicator_code}"
            )

        ranked = self._hybrid_search(search_text, max(top_k * 5, 20))
        filtered = [
            item for item in ranked
            if self._valid_document(item["document"])
            and self._matches_query(item["document"], query)
        ][:top_k]
        if not filtered:
            return []

        query_tokens = set(remove_diacritics(search_text.lower()).split())

        def confidence(item: dict) -> float:
            document_tokens = set(
                remove_diacritics(item["document"].get("text", "").lower()).split()
            )
            lexical_score = len(query_tokens & document_tokens) / max(len(query_tokens), 1)
            metadata_bonus = 0.0
            if query is not None:
                metadata = item["document"]["metadata"]
                metadata_bonus += 0.1 if metadata.get("ticker") == query.ticker else 0.0
                metadata_bonus += 0.1 if metadata.get("period") == query.year else 0.0
                metadata_bonus += 0.1 if metadata.get("section") == query.section else 0.0
            return min(1.0, lexical_score + metadata_bonus)

        return [
            RetrievalHit(
                query=query,
                document=item["document"],
                score=confidence(item),
            )
            for item in filtered
        ]

    def retrieve(self, result: QueryResult, top_k_per_query: int = 5) -> list[RetrievalHit]:
        hits = super().retrieve(result, top_k_per_query=top_k_per_query)
        resolved_keys = {self._query_key(hit.query) for hit in hits}

        for query in result.retrieval_queries:
            if self._query_key(query) not in resolved_keys:
                hits.extend(self._fallback_hits(result, query, top_k_per_query))

        if not result.retrieval_queries and not hits:
            hits.extend(self._fallback_hits(result, None, top_k_per_query))

        if hits and max(hit.score for hit in hits) < self._confidence_threshold:
            reformulated = " ".join([
                result.normalized_question,
                *result.entities.tickers,
                *(str(year) for year in result.entities.years),
                *result.entities.indicators,
            ])
            original_search_text = result.search_text
            result.search_text = reformulated
            retry_hits = []
            retry_queries = result.retrieval_queries or [None]
            for query in retry_queries:
                retry_hits.extend(self._fallback_hits(result, query, top_k_per_query))
            result.search_text = original_search_text
            if retry_hits:
                hits = retry_hits

        return hits
