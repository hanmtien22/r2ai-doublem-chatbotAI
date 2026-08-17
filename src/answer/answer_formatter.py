import logging
import math
from typing import Any, Optional

from src.llm.client import LLMClient

logger = logging.getLogger(__name__)

_BILLION_VND = 1_000_000_000
_MILLION_VND = 1_000_000


def _auto_format_vnd(value: float, unit: str = "vnd") -> str:
    """Quy về VND rồi format thành chuỗi dễ đọc (tỷ/triệu/nghìn đồng)."""
    unit_lower = (unit or "vnd").lower().strip()

    if unit_lower in ("trieu", "triệu", "million"):
        value_vnd = value * _MILLION_VND
    elif unit_lower in ("ty", "tỷ", "billion"):
        value_vnd = value * _BILLION_VND
    elif unit_lower in ("nghin", "nghìn", "thousand"):
        value_vnd = value * 1_000
    else:
        value_vnd = value

    abs_val = abs(value_vnd)
    sign = "-" if value_vnd < 0 else ""

    if abs_val >= _BILLION_VND:
        return f"{sign}{abs_val / _BILLION_VND:,.2f} tỷ đồng"
    if abs_val >= _MILLION_VND:
        return f"{sign}{abs_val / _MILLION_VND:,.2f} triệu đồng"
    if abs_val >= 1_000:
        return f"{sign}{abs_val / 1_000:,.2f} nghìn đồng"
    return f"{sign}{abs_val:,.0f} đồng"


class AnswerFormatter:
    _SYSTEM_PROMPT = (
        "Bạn là trợ lý tài chính chuyên phân tích báo cáo tài chính thị trường chứng khoán Việt Nam. "
        "Nhiệm vụ: định dạng kết quả số liệu thô thành câu trả lời ngắn gọn, rõ ràng, đúng đơn vị. "
        "Quy tắc bắt buộc:\n"
        "1. Trả lời trực tiếp, không lan man.\n"
        "2. KHÔNG bịa đặt hay suy luận thêm số liệu ngoài dữ liệu đã cung cấp.\n"
        "3. Nếu kết quả là số tiền VND, luôn ghi rõ 'tỷ đồng', 'triệu đồng' v.v.\n"
        "4. Nếu kết quả là tỷ lệ (%), hãy ghi thêm dấu %.\n"
        "5. Trả lời bằng tiếng Việt."
    )

    _USER_TEMPLATE = """\
Câu hỏi: {question}

Kết quả số liệu thu được: {result_str}
Đơn vị gốc của dữ liệu: {unit}

Hãy viết một câu trả lời ngắn gọn, rõ ràng dựa trên kết quả trên.\
"""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def format_answer(self, question: str, computed_result: Any, unit: Optional[str] = None) -> str:
        logger.info("Generating natural language answer...")

        if computed_result is None:
            return "Xin lỗi, không tìm thấy dữ liệu để trả lời câu hỏi này."
        try:
            f_val = float(computed_result)
            if math.isnan(f_val) or math.isinf(f_val):
                return "Xin lỗi, kết quả tính toán không hợp lệ (NaN hoặc vô cực)."
        except (ValueError, TypeError):
            pass

        result_str = str(computed_result)
        try:
            numeric_val = float(computed_result)
            result_str = _auto_format_vnd(numeric_val, unit or "vnd")
            result_str += f" (giá trị gốc: {numeric_val:,.0f})"
        except (ValueError, TypeError):
            pass

        user_msg = self._USER_TEMPLATE.format(
            question=question,
            result_str=result_str,
            unit=unit or "VND",
        )

        try:
            answer = self.llm_client.generate_chat(
                system_prompt=self._SYSTEM_PROMPT,
                user_message=user_msg,
                max_tokens=400,
            )
        except Exception as e:
            logger.warning("generate_chat failed, falling back: %s", e)
            combined = f"{self._SYSTEM_PROMPT}\n\nCâu hỏi: {question}\nKết quả: {result_str}\nĐơn vị: {unit or 'VND'}"
            answer = self.llm_client.generate(combined, max_tokens=400)

        return answer.strip()
