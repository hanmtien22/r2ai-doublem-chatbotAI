import logging
import re
from typing import Dict, Any, List

import pandas as pd

from src.llm.client import LLMClient

logger = logging.getLogger(__name__)

# Model 3B chỉ có 4096 token context. Prompt quá dài vừa làm chậm vừa bị cắt đuôi,
# khiến phần quy tắc quan trọng nhất biến mất khỏi prompt.
_MAX_SAMPLE_ROWS = 8
_MAX_CELL_CHARS = 60
_MAX_SCHEMA_CHARS = 3500


def _shorten_row(row: dict) -> dict:
    """Cắt ngắn ô chuỗi dài (note_title, label) để prompt không phình ra."""
    shortened = {}
    for key, value in row.items():
        text = str(value)
        shortened[key] = text[:_MAX_CELL_CHARS] + "…" if len(text) > _MAX_CELL_CHARS else value
    return shortened


class CodeGenerator:
    _SYSTEM_PROMPT = (
        "Bạn là chuyên gia Python/Pandas. Chỉ trả về code Python trong khối ```python ... ```, "
        "không thêm giải thích nào khác. "
        "Biến kết quả BẮT BUỘC phải là `final_result` kiểu số (int hoặc float). "
        "KHÔNG tạo DataFrame mới hay đọc file. KHÔNG sử dụng print()."
    )

    _USER_TEMPLATE = """\
Câu hỏi: {question}

Dữ liệu (các DataFrame đã sẵn sàng trong môi trường):
{schemas}

QUY TẮC BẮT BUỘC:
- Gán `final_result` = MỘT số (int hoặc float), đơn vị VND. Không trả về Series/DataFrame.
- Không được để `final_result = None` nếu dữ liệu có dòng khớp.
- `pd` (pandas) đã có sẵn, không cần import. Không đọc file, không tạo dữ liệu mới.
- Lọc đúng `ticker` và đúng năm TRƯỚC khi lấy giá trị.
- Dùng `.dropna(subset=[...])` trước `.iloc[0]`; kiểm tra `len(...) > 0` trước khi index.
- Câu hỏi có "công ty mẹ"/"riêng" -> `report_type == 'separate'`;
  có "hợp nhất" hoặc không nói gì -> `report_type == 'consolidated'`.
- Chọn dòng có `item_name` sát nghĩa nhất với chỉ tiêu được hỏi, không lấy dòng
  chỉ tình cờ chứa cùng vài từ (vd: "Chi phí khác" khác "Chi phí quản lý doanh nghiệp").

Với df_0 (bảng chính) — dùng cột `value`:
```python
mask = (df_0['ticker'] == 'VNM') & (df_0['period'] == 2023) & (df_0['report_type'] == 'consolidated')
mask &= df_0['item_name'].str.contains('Chi phí khác', case=False, na=False)
rows = df_0.loc[mask].dropna(subset=['value'])
final_result = float(rows['value'].iloc[0]) if len(rows) else None
```

Với df_1 (thuyết minh) — dùng cột `value_vnd` (đã quy về VND) và
`column_role == 'current'` cho số cuối năm / năm nay:
```python
mask = (df_1['column_role'] == 'current')
mask &= df_1['label'].str.contains('Tiền gửi tại các TCTD khác', case=False, na=False)
rows = df_1.loc[mask].dropna(subset=['value_vnd'])
final_result = float(rows['value_vnd'].iloc[0]) if len(rows) else None
```

{error_feedback}
"""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def _build_schema_info(self, retrieved_tables: List[Dict[str, Any]]) -> str:
        """Tạo mô tả schema + sample data cho LLM (15 dòng, unique values, dtypes)."""
        schemas_info = ""
        for i, table in enumerate(retrieved_tables):
            df_name = f"df_{i}"
            ticker = table.get("ticker", "UNKNOWN")
            columns = table.get("columns", [])
            data = table.get("data", [])

            try:
                df_full = pd.DataFrame(data, columns=columns) if data and columns else pd.DataFrame()
            except Exception:
                df_full = pd.DataFrame()

            schemas_info += f"DataFrame `{df_name}` (ticker={ticker}, total_rows={len(df_full)}):\n"
            description = table.get("description")
            if description:
                schemas_info += f"  Ý nghĩa: {description}\n"
            schemas_info += f"  Columns: {columns}\n"

            if not df_full.empty:
                for col in ("ticker", "period", "item_code"):
                    if col in df_full.columns:
                        uniq = df_full[col].dropna().unique().tolist()
                        schemas_info += f"  Unique {col}: {uniq[:12]}\n"

                dtype_map = {c: str(df_full[c].dtype) for c in df_full.columns}
                schemas_info += f"  dtypes: {dtype_map}\n"

                df_with_val = df_full[df_full["value"].notna()] if "value" in df_full.columns else df_full
                sample_df = df_with_val.head(_MAX_SAMPLE_ROWS) if len(df_with_val) >= 1 else df_full.head(_MAX_SAMPLE_ROWS)
                schemas_info += f"  Sample data ({len(sample_df)} rows shown):\n"
                for _, row in sample_df.iterrows():
                    schemas_info += f"    {_shorten_row(dict(row))}\n"
            else:
                schemas_info += "  (Không có dữ liệu)\n"

            schemas_info += "\n"

        if len(schemas_info) > _MAX_SCHEMA_CHARS:
            schemas_info = schemas_info[:_MAX_SCHEMA_CHARS] + "\n  … (đã lược bớt)\n"
        return schemas_info

    def generate_code(
        self,
        question: str,
        retrieved_tables: List[Dict[str, Any]],
        error_feedback: str = "",
    ) -> str:
        schemas_info = self._build_schema_info(retrieved_tables)

        feedback_section = ""
        if error_feedback:
            feedback_section = (
                f"LẦN CHẠY TRƯỚC BỊ LỖI — hãy sửa lại:\n"
                f"```\n{error_feedback[:600]}\n```\n"
                f"Viết lại code đúng hơn, chú ý:\n"
                f"- Kiểm tra kiểu dữ liệu cột `value` (phải là float)\n"
                f"- Kiểm tra unique values của `ticker` và `period` trước khi filter\n"
                f"- Dùng `.dropna()` trước `.iloc[0]`"
            )

        user_msg = self._USER_TEMPLATE.format(
            question=question,
            schemas=schemas_info,
            error_feedback=feedback_section,
        )

        logger.info("Generating code for question: %s...", question[:60])

        try:
            raw_code = self.llm_client.generate_chat(
                system_prompt=self._SYSTEM_PROMPT,
                user_message=user_msg,
                max_tokens=600,
            )
        except Exception as e:
            logger.warning("generate_chat failed, falling back: %s", e)
            raw_code = self.llm_client.generate(
                f"{self._SYSTEM_PROMPT}\n\n{user_msg}",
                max_tokens=600,
            )

        return self._extract_code(raw_code)

    @staticmethod
    def _extract_code(raw: str) -> str:
        """Trích xuất Python code từ output của LLM."""
        match = re.search(r"```(?:python)?\s*(.*?)```", raw, re.DOTALL)
        if match:
            return match.group(1).strip()

        lines = raw.split("\n")
        code_lines = []
        in_code = False
        for line in lines:
            stripped = line.strip()
            if not in_code and (
                stripped.startswith("import ")
                or stripped.startswith("final_result")
                or stripped.startswith("df_")
                or stripped.startswith("mask")
                or (stripped.startswith("#") and code_lines)
            ):
                in_code = True
            if in_code and not stripped.startswith("```"):
                code_lines.append(line)

        if code_lines:
            return "\n".join(code_lines).strip()

        return raw.strip()
