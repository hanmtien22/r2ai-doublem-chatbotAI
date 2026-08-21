import logging
import math
import re
from typing import Any, Optional

from src.llm.client import LLMClient
from src.utils.text import remove_diacritics

logger = logging.getLogger(__name__)

_BILLION_VND = 1_000_000_000
_MILLION_VND = 1_000_000

# Đơn vị người dùng hỏi ("... là bao nhiêu triệu đồng?") -> (hệ số, tên hiển thị)
# "ty" BẮT BUỘC đi kèm "dong"/"vnd": bỏ dấu thì "công ty" cũng thành "cong ty",
# nếu để phần đơn vị là tuỳ chọn thì mọi câu hỏi chứa "công ty" đều bị hiểu nhầm
# là đang hỏi theo đơn vị tỷ đồng.
_REQUESTED_UNITS = [
    (re.compile(r"\bnghin\s*ty\s*(?:dong|vnd)\b", re.IGNORECASE), 1_000 * _BILLION_VND, "nghìn tỷ đồng"),
    (re.compile(r"\bty\s*(?:dong|vnd)\b", re.IGNORECASE), _BILLION_VND, "tỷ đồng"),
    (re.compile(r"\btrieu\s*(?:dong|vnd)?\b", re.IGNORECASE), _MILLION_VND, "triệu đồng"),
    (re.compile(r"\bnghin\s*(?:dong|vnd)\b", re.IGNORECASE), 1_000, "nghìn đồng"),
    (re.compile(r"\bphan tram\b|%", re.IGNORECASE), 1, "%"),
]


def detect_requested_unit(question: str):
    """Đơn vị mà câu hỏi yêu cầu trả lời, hoặc None nếu không nói rõ."""
    normalized = remove_diacritics(question.lower())
    for pattern, factor, label in _REQUESTED_UNITS:
        if pattern.search(normalized):
            return factor, label
    return None


def _to_vnd(value: float, unit: str = "vnd") -> float:
    """Quy giá trị từ đơn vị gốc của dữ liệu về VND."""
    unit_lower = remove_diacritics((unit or "vnd").lower().strip())

    if "trieu" in unit_lower or "million" in unit_lower:
        return value * _MILLION_VND
    if re.search(r"\bty\b", unit_lower) or "billion" in unit_lower:
        return value * _BILLION_VND
    if "nghin" in unit_lower or "ngan" in unit_lower or "thousand" in unit_lower:
        return value * 1_000
    return value


def _auto_format_vnd(value: float, unit: str = "vnd") -> str:
    """Quy về VND rồi format thành chuỗi dễ đọc (tỷ/triệu/nghìn đồng)."""
    value_vnd = _to_vnd(value, unit)

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

    def format_answer(self, question: str, computed_result: Any, unit: Optional[str] = None, is_fast_path: bool = False) -> str:
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
            value_vnd = _to_vnd(numeric_val, unit or "vnd")
            requested = detect_requested_unit(question)
            if str(unit).strip() == "%" or (requested and requested[1] == "%"):
                # Tỷ lệ: giữ nguyên con số, không quy đổi sang đơn vị tiền tệ
                result_str = f"{numeric_val:,.2f}%"
            elif requested:
                # Trả lời đúng đơn vị câu hỏi yêu cầu, kèm số VND gốc để đối chiếu
                factor, label = requested
                result_str = f"{value_vnd / factor:,.2f} {label}"
                result_str += f" (giá trị gốc: {value_vnd:,.0f} VND)"
            else:
                result_str = _auto_format_vnd(numeric_val, unit or "vnd")
                result_str += f" (giá trị gốc: {value_vnd:,.0f})"
        except (ValueError, TypeError):
            pass

        # Fast-path: dùng template thảy vì gọi LLM để tiết kiệm latency và token
        if is_fast_path:
            return result_str

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
