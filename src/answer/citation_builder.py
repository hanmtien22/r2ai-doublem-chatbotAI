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
            year = meta.get("period") or meta.get("year") or "UNKNOWN"
            report_type = {"separate": "riêng", "consolidated": "hợp nhất"}.get(
                str(meta.get("report_type", "")).lower(), ""
            )
            # Chunk thuyết minh không có `table_name`, chỉ có tiêu đề mục
            table_name = meta.get("table_name") or meta.get("section_title") or "Bảng dữ liệu"
            line = meta.get("start_line")
            location = f"dòng {line}" if line else "Trang ?"

            label = f"{ticker} - {year}"
            if report_type:
                label += f" (BCTC {report_type})"
            cite = f"[{i+1}] {label}, {table_name}, {location}"
            if cite not in citations:
                citations.append(cite)
                
        if not citations:
            return ""
            
        return "\n\nNguồn trích dẫn:\n" + "\n".join(citations)
