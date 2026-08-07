from __future__ import annotations

from pathlib import Path

from src.query.pipeline import QueryPipeline
from src.retrieval.service import HybridDocumentRetriever


class QueryRetrievalPipeline:
    """
    Điểm chạm cuối cùng (Facade/Orchestrator).
    Nhận đầu vào là câu hỏi thô của người dùng, tự động gọi qua Query Pipeline để "hiểu" câu hỏi,
    rồi đẩy sang Retrieval Pipeline để "tìm" đáp án trong CSDL.
    """

    def __init__(
        self,
        documents_path: str | Path,
        query_pipeline: QueryPipeline | None = None,
        index_dir: str | Path | None = None,
        retrieval_config: dict | None = None,
        embedding_config: dict | None = None,
    ):
        # Khởi tạo QueryPipeline nếu chưa được truyền vào
        self.query_pipeline = query_pipeline or QueryPipeline()
        
        # Khởi tạo bộ máy tìm kiếm (load dữ liệu, model vào RAM)
        self.retriever = HybridDocumentRetriever.from_jsonl(
            documents_path,
            index_dir=index_dir,
            retrieval_config=retrieval_config,
            embedding_config=embedding_config,
        )

    def process(self, question: str, top_k_per_query: int = 5) -> dict:
        """
        Thực thi toàn bộ luồng Chatbot:
        1. Phân tích câu hỏi -> QueryResult.
        2. Dùng QueryResult đi tìm kiếm tài liệu -> Hits.
        3. Trả về format chuẩn bị đưa cho LLM sinh câu trả lời.
        """
        # Bước 1: Hiểu người dùng muốn gì
        query_result = self.query_pipeline.process(question)
        
        # Bước 2: Tìm dữ liệu
        hits = self.retriever.retrieve(query_result, top_k_per_query=top_k_per_query)
        
        # Trả về kết quả
        return {
            "query": query_result.to_dict(),
            "hits": [hit.to_dict() for hit in hits],
        }
