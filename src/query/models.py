from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Optional

"""
Định nghĩa các cấu trúc dữ liệu (Data Models) dùng chung trong toàn bộ Query Pipeline.
Sử dụng dataclasses giúp dữ liệu rõ ràng, dễ validation và dễ chuyển đổi sang JSON.
"""

@dataclass
class RetrievalQuery:
    """Đại diện cho một truy vấn chi tiết dùng để tìm kiếm (Retrieve) trong cơ sở dữ liệu."""
    ticker: str            # Mã cổ phiếu (VD: "VNM")
    year: int              # Năm tài chính (VD: 2023)
    section: str           # Loại báo cáo (VD: "BS" - Cân đối kế toán, "IS" - KQKD)
    indicator_code: str    # Mã chỉ tiêu tài chính (VD: "110", "doanh_thu")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FormulaInfo:
    """Lưu trữ thông tin về công thức tính toán cho các chỉ số phái sinh (Derived indicators)."""
    name: str                           # Tên chỉ số (VD: "ROE", "Tăng trưởng doanh thu")
    formula: str                        # Chuỗi công thức toán học (VD: "loi_nhuan / von_chu_so_huu")
    components: list[str]               # Các mã chỉ tiêu thành phần cần lấy từ DB để tính (VD: ["IS.loi_nhuan", "BS.von_chu_so_huu"])
    unit: str                           # Đơn vị đo lường (VD: "%", "lần")
    multiply_100: bool = False          # Cờ xác định xem có cần nhân 100 để hiển thị phần trăm không
    requires_previous_year: bool = False # Cờ xác định xem công thức có cần dữ liệu của năm trước không (VD: Tính tăng trưởng)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExtractedEntities:
    """Tập hợp các thực thể (entities) được trích xuất từ câu hỏi gốc của người dùng."""
    tickers: list[str] = field(default_factory=list)          # Danh sách mã chứng khoán tìm được
    years: list[int] = field(default_factory=list)            # Danh sách các năm tìm được
    indicators: list[str] = field(default_factory=list)       # Tên các chỉ tiêu (dạng văn bản)
    indicator_codes: list[str] = field(default_factory=list)  # Mã các chỉ tiêu đã chuẩn hóa

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MetadataFilters:
    """Bộ lọc metadata dùng để thu hẹp phạm vi tìm kiếm (Vector/BM25) nhằm tăng độ chính xác."""
    tickers: list[str] = field(default_factory=list)
    years: list[int] = field(default_factory=list)
    sections: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class QueryResult:
    """
    Kết quả cuối cùng của Query Pipeline, đóng gói toàn bộ thông tin
    cần thiết để chuyển sang bước Retrieval Pipeline.
    """
    original_question: str                              # Câu hỏi gốc người dùng gõ
    normalized_question: str                            # Câu hỏi sau khi tiền xử lý (sửa lỗi, bỏ dấu, v.v.)
    entities: ExtractedEntities                         # Các thực thể đã trích xuất
    query_type: str                                     # Loại câu hỏi (single_lookup, multi_comparison, derived_indicator, out_of_scope)
    requires_formula: bool                              # Có yêu cầu tính toán công thức không?
    formula_info: Optional[FormulaInfo]                 # Thông tin công thức (nếu requires_formula = True)
    retrieval_queries: list[RetrievalQuery]             # Danh sách các truy vấn cần gọi xuống DB để lấy số liệu
    search_text: str                                    # Text dùng để đưa vào hệ thống Vector/BM25 (thường là câu đã chuẩn hóa)
    metadata_filters: MetadataFilters                   # Bộ lọc để giới hạn kết quả tìm kiếm

    def to_dict(self) -> dict:
        return {
            "original_question": self.original_question,
            "normalized_question": self.normalized_question,
            "entities": self.entities.to_dict(),
            "query_type": self.query_type,
            "requires_formula": self.requires_formula,
            "formula_info": self.formula_info.to_dict() if self.formula_info else None,
            "retrieval_queries": [q.to_dict() for q in self.retrieval_queries],
            "search_text": self.search_text,
            "metadata_filters": self.metadata_filters.to_dict(),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)
