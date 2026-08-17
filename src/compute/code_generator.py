import logging
import re
from typing import Dict, Any, List

import pandas as pd

from src.llm.client import LLMClient

logger = logging.getLogger(__name__)


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
- Gán `final_result` = kết quả số (int hoặc float). Không được để `final_result = None` nếu có dữ liệu.
- Cột `value` chứa giá trị số thực (đơn vị VND). Dùng cột này để tính toán.
- Cột `period` chứa năm (int). Cột `item_name` chứa tên chỉ tiêu (string).
- Cột `ticker` chứa mã cổ phiếu (string). Lọc đúng ticker và period trước khi tính.
- Dùng `.dropna(subset=['value'])` trước khi `.iloc[0]` để tránh lỗi.
- Nếu không tìm thấy dữ liệu sau khi filter, gán `final_result = None`.

Ví dụ filter đúng:
```python
mask = (df_0['ticker'] == 'VNM') & (df_0['period'] == 2023) & df_0['value'].notna()
final_result = df_0.loc[mask, 'value'].iloc[0]
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
            schemas_info += f"  Columns: {columns}\n"

            if not df_full.empty:
                for col in ("ticker", "period", "item_code"):
                    if col in df_full.columns:
                        uniq = df_full[col].dropna().unique().tolist()
                        schemas_info += f"  Unique {col}: {uniq[:20]}\n"

                dtype_map = {c: str(df_full[c].dtype) for c in df_full.columns}
                schemas_info += f"  dtypes: {dtype_map}\n"

                df_with_val = df_full[df_full["value"].notna()] if "value" in df_full.columns else df_full
                sample_df = df_with_val.head(15) if len(df_with_val) >= 1 else df_full.head(15)
                schemas_info += f"  Sample data ({len(sample_df)} rows shown):\n"
                for _, row in sample_df.iterrows():
                    schemas_info += f"    {dict(row)}\n"
            else:
                schemas_info += "  (Không có dữ liệu)\n"

            schemas_info += "\n"

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
