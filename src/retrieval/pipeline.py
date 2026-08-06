from __future__ import annotations

from pathlib import Path

from src.query.pipeline import QueryPipeline
from src.retrieval.service import HybridDocumentRetriever


class QueryRetrievalPipeline:
    """Run query understanding and exact financial-record retrieval together."""

    def __init__(
        self,
        documents_path: str | Path,
        query_pipeline: QueryPipeline | None = None,
        index_dir: str | Path | None = None,
        retrieval_config: dict | None = None,
        embedding_config: dict | None = None,
    ):
        self.query_pipeline = query_pipeline or QueryPipeline()
        self.retriever = HybridDocumentRetriever.from_jsonl(
            documents_path,
            index_dir=index_dir,
            retrieval_config=retrieval_config,
            embedding_config=embedding_config,
        )

    def process(self, question: str, top_k_per_query: int = 5) -> dict:
        query_result = self.query_pipeline.process(question)
        hits = self.retriever.retrieve(query_result, top_k_per_query=top_k_per_query)
        return {
            "query": query_result.to_dict(),
            "hits": [hit.to_dict() for hit in hits],
        }
