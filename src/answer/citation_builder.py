import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class CitationBuilder:
    """
    Trích xuất metadata từ dữ liệu tìm kiếm để tạo phần trích dẫn nguồn.
    """
    def __init__(self):
        pass

    def build_citation(self, hits: List[Dict[str, Any]]) -> str:
        """
        Tạo văn bản trích dẫn từ danh sách các bảng/documents đã tìm thấy.
        """
        if not hits:
            return ""
            
        citations = []
        for i, hit in enumerate(hits):
            # BM25 trả về {score, document: {...}}, FAISS cũng tương tự
            doc = hit.get("document", hit)
            meta = doc.get("metadata", hit.get("metadata", {}))
            ticker = meta.get("ticker", "UNKNOWN")
            year = meta.get("year", "UNKNOWN")
            doc_type = meta.get("document_type", "BCTC")
            page = meta.get("page", "?")
            table_name = meta.get("table_name", "Bảng dữ liệu")
            
            cite = f"[{i+1}] {ticker} - {year} ({doc_type}), {table_name}, Trang {page}"
            if cite not in citations:
                citations.append(cite)
                
        if not citations:
            return ""
            
        return "\n\nNguồn trích dẫn:\n" + "\n".join(citations)
