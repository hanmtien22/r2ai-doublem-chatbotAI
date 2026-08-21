"""Parse bảng trong thuyết minh BCTC (notes) từ text sang dòng có cấu trúc.

Chunk notes được lưu dạng text thuần nhưng thực chất là bảng phân tách bằng "|":

    8. TIỀN GỬI TẠI CÁC TCTD KHÁC
    Số cuối năm | Số đầu năm
    Triệu VND | Triệu VND
    Tiền gửi tại các TCTD khác | 39.849.011 | 47.523.973

Đưa về dạng dòng (label, cột, giá trị, đơn vị) thì bước tính toán bằng pandas
mới làm việc được với số liệu trong thuyết minh, thay vì phải đoán từ text.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Nhân với hệ số này để quy mọi giá trị về VND
_UNIT_MULTIPLIERS = {
    "trieu": 1_000_000,
    "million": 1_000_000,
    "nghin": 1_000,
    "ngan": 1_000,
    "thousand": 1_000,
    "ty": 1_000_000_000,
    "billion": 1_000_000_000,
}

# Header cột thường gặp: "Số cuối năm | Số đầu năm", "Năm nay | Năm trước"
_CURRENT_COLUMN_HINTS = ("cuoi nam", "nam nay", "cuoi ky", "31 12", "so cuoi")
_PREVIOUS_COLUMN_HINTS = ("dau nam", "nam truoc", "dau ky", "so dau")

_NUMBER_RE = re.compile(r"^\(?-?[\d.,]+\)?$")


def _strip_diacritics(text: str) -> str:
    from src.utils.text import remove_diacritics

    return remove_diacritics(text.lower())


def parse_number(token: str) -> Optional[float]:
    """Đổi '39.849.011' -> 39849011.0, '(60.295)' -> -60295.0, '-' -> None."""
    token = token.strip()
    if not token or token in {"-", "–", "—", "N/A"}:
        return None
    if not _NUMBER_RE.match(token):
        return None

    negative = token.startswith("(") and token.endswith(")")
    token = token.strip("()").strip()
    if token.startswith("-"):
        negative = True
        token = token[1:]

    # Số Việt Nam: '.' phân nhóm nghìn, ',' phân thập phân
    if "," in token and "." in token:
        token = token.replace(".", "").replace(",", ".")
    elif "," in token:
        token = token.replace(",", ".") if len(token.split(",")[-1]) != 3 else token.replace(",", "")
    else:
        parts = token.split(".")
        if len(parts) > 1 and all(len(p) == 3 for p in parts[1:]):
            token = "".join(parts)

    try:
        value = float(token)
    except ValueError:
        return None
    return -value if negative else value


def _normalize_header(header: str) -> str:
    """Chuẩn hoá header để so khớp.

    Text trích từ PDF hay dính liền số với chữ ("31/12/2018Triệu VND"), nên phải
    tách chữ khỏi số trước, nếu không `\btrieu\b` sẽ không bao giờ khớp.
    """
    normalized = _strip_diacritics(header)
    normalized = re.sub(r"(\d)([a-z])", r"\1 \2", normalized)
    normalized = re.sub(r"([a-z])(\d)", r"\1 \2", normalized)
    normalized = re.sub(r"[^\w\s]", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


_MAX_HEADER_CELL_CHARS = 60

# Dòng tổng của một thuyết minh không có nhãn (chỉ toàn số). Về ngữ nghĩa, tên
# của nó chính là tiêu đề thuyết minh: tổng của "30. CHI PHÍ TÀI CHÍNH" chính
# là chi phí tài chính.
_TOTAL_LABEL = "TỔNG CỘNG"

# Đơn vị phải nằm ở CUỐI ô ("Triệu VND", "VND", "Số dư 31/12/2018Triệu VND").
# Nếu chỉ dò từ khoá ở bất kỳ đâu thì "Chi phí lãi từ hợp đồng hợp tác đầu tư
# tại Công ty" cũng khớp ("dong", "ty") và bị nhân sai hệ số.
_UNIT_SUFFIX_RE = re.compile(
    r"\b(?:(trieu|million|nghin|ngan|thousand|ty|billion)\s+)?(vnd|dong|usd)\s*$"
)


def _unit_from_header(header: str) -> Optional[tuple[float, str]]:
    """('Triệu VND') -> (1e6, 'trieu vnd'); None nếu ô không phải đơn vị tiền tệ."""
    match = _UNIT_SUFFIX_RE.search(_normalize_header(header))
    if match is None:
        return None
    scale = match.group(1)
    return float(_UNIT_MULTIPLIERS.get(scale, 1)), match.group(0).strip()


def _is_currency_header(header: str) -> bool:
    """Ô có ghi đơn vị tiền tệ ("Triệu VND", "VND", "Triệu đồng")."""
    return _unit_from_header(header) is not None


def _unit_multiplier(header: str) -> tuple[float, str]:
    """('Triệu VND') -> (1e6, 'trieu vnd')."""
    unit = _unit_from_header(header)
    if unit is not None:
        return unit
    return 1.0, _normalize_header(header) or "vnd"


def _column_role(header: str) -> str:
    """'current' = số cuối năm/năm nay, 'previous' = số đầu năm/năm trước."""
    normalized = _normalize_header(header)
    if any(hint in normalized for hint in _CURRENT_COLUMN_HINTS):
        return "current"
    if any(hint in normalized for hint in _PREVIOUS_COLUMN_HINTS):
        return "previous"
    return "unknown"


def _column_roles(headers: list[str]) -> list[str]:
    """Gán vai trò cho từng cột; cột có năm lớn nhất là số cuối kỳ."""
    roles = [_column_role(h) for h in headers]

    # Header dạng "31/12/2018 | 31/12/2017": năm lớn hơn là kỳ hiện tại
    years = []
    for header in headers:
        found = re.findall(r"\b(20[0-2]\d)\b", _normalize_header(header))
        years.append(int(found[-1]) if found else None)

    known = [y for y in years if y is not None]
    if len(known) >= 2 and len(set(known)) > 1:
        newest = max(known)
        roles = [
            ("current" if y == newest else "previous") if y is not None else role
            for role, y in zip(roles, years)
        ]
    return roles


def _phrase_in(phrase: str, text: str) -> bool:
    """Cụm `phrase` xuất hiện nguyên vẹn trong `text` (theo ranh giới từ)."""
    return bool(re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text))


def _split_cells(line: str) -> list[str]:
    return [c.strip() for c in line.split("|")]


def parse_notes_table(text: str, section_title: str = "") -> list[dict[str, Any]]:
    """Trả về danh sách dòng: label + giá trị theo từng cột của bảng thuyết minh."""
    rows: list[dict[str, Any]] = []
    if not text:
        return rows

    column_headers: list[str] = []
    column_units: list[str] = []
    unit_multipliers: list[float] = []
    column_roles: list[str] = []
    current_title = section_title

    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue

        if "|" not in line:
            # Dòng không có cột: hoặc là tiêu đề mục con, hoặc là văn bản diễn giải
            if len(line) < 200 and re.match(r"^\d+(\.\d+)*[.)]?\s+\S", line):
                current_title = line
                column_headers, column_units, unit_multipliers, column_roles = [], [], [], []
            continue

        cells = _split_cells(line)
        numeric = [parse_number(c) for c in cells]
        has_number = any(n is not None for n in numeric)

        if not has_number:
            # Dòng không đọc được số chưa chắc là header: text trích từ PDF hay
            # dính hai số vào nhau ("9.277.775.342(1.265.429.612)") khiến cả dòng
            # dữ liệu trông như header. Header thật thì mọi ô đều ngắn.
            if any(len(c) > _MAX_HEADER_CELL_CHARS for c in cells):
                continue

            # Dòng header. Header đơn vị ("Triệu VND") đi ngay sau header kỳ.
            if _is_currency_header(cells[0]) and column_headers:
                column_units = [_unit_multiplier(c)[1] for c in cells]
                unit_multipliers = [_unit_multiplier(c)[0] for c in cells]
            else:
                column_headers = cells
                column_roles = _column_roles(cells)
                # Đơn vị có thể nằm ngay trong header ("Số dư 31/12/2018Triệu VND")
                # thay vì ở một dòng riêng bên dưới.
                column_units = [_unit_multiplier(c)[1] for c in cells]
                unit_multipliers = [
                    _unit_multiplier(c)[0] if _is_currency_header(c) else 1.0 for c in cells
                ]
            continue

        # Dòng dữ liệu: ô đầu là nhãn nếu không phải số (dòng tổng cộng thì toàn số)
        if numeric[0] is None:
            label = cells[0]
            value_cells = numeric[1:]
        else:
            label = _TOTAL_LABEL
            value_cells = numeric

        # Có bảng ghi cả header cho cột nhãn ("Công ty con | Số cuối năm | Số đầu năm"),
        # có bảng thì không ("Số cuối năm | Số đầu năm"). Lệch 1 thì phải dịch header.
        offset = 1 if len(column_headers) == len(value_cells) + 1 else 0

        for i, value in enumerate(value_cells):
            if value is None:
                continue
            j = i + offset
            header = column_headers[j] if j < len(column_headers) else f"col_{i}"
            multiplier = unit_multipliers[j] if j < len(unit_multipliers) else 1.0
            unit = column_units[j] if j < len(column_units) else "vnd"
            rows.append({
                "note_title": current_title,
                "label": label,
                "column": header,
                "column_role": column_roles[j] if j < len(column_roles) else _column_role(header),
                "value": value,
                "unit": unit,
                "value_vnd": value * multiplier,
            })

    return rows


def find_value_by_label(
    rows: list[dict[str, Any]],
    query: str,
    prefer_role: str = "current",
    threshold: int = 80,
) -> Optional[dict[str, Any]]:
    """Tìm dòng có nhãn giống `query` nhất, ưu tiên cột số cuối năm."""
    if not rows or not query:
        return None
    try:
        from rapidfuzz import fuzz
    except ImportError:
        return None

    query_norm = _strip_diacritics(query)
    query_tokens = set(query_norm.split())
    scored: list[tuple[float, dict[str, Any]]] = []

    for row in rows:
        if str(row["label"]) == _TOTAL_LABEL:
            # So khớp dòng tổng bằng tiêu đề thuyết minh (bỏ phần đánh số đầu dòng)
            title = re.sub(r"^\d+(\.\d+)*[.)]?\s*", "", str(row.get("note_title", "")))
            label_norm = _normalize_header(title)
        else:
            label_norm = _strip_diacritics(str(row["label"]))
        if not label_norm:
            continue

        label_tokens = set(label_norm.split())
        similarity = max(
            fuzz.token_set_ratio(query_norm, label_norm),
            fuzz.partial_ratio(query_norm, label_norm),
        )
        # Ngưỡng xét trên chính độ giống của nhãn. Nếu cộng điểm thưởng trước
        # rồi mới so ngưỡng thì một nhãn chỉ giống 50% vẫn lọt, và hệ thống trả
        # về một con số bất kỳ trong thuyết minh thay vì nhận là không biết.
        if similarity < threshold:
            continue

        # Nhãn phải giải thích được phần lớn câu hỏi, không chỉ trùng vài từ chung
        coverage = len(query_tokens & label_tokens) / max(len(query_tokens), 1)
        if coverage < 0.5:
            # Ngoại lệ: bảng phân loại ("9.6 Theo ngành nghề kinh doanh") có nhãn
            # chỉ là tên hạng mục ("Thương mại"). Nhãn ngắn nhưng xuất hiện nguyên
            # cụm trong câu hỏi vẫn là khớp đúng.
            if len(label_tokens) < 2 or not _phrase_in(label_norm, query_norm):
                continue
            # Nhưng tiêu đề bảng cũng phải liên quan tới câu hỏi. Nếu không, một
            # cái tên trùng trong bảng khác hẳn chủ đề (vd: giao dịch bên liên
            # quan thay vì thù lao) sẽ bị nhận nhầm là câu trả lời.
            title_tokens = set(_strip_diacritics(str(row.get("note_title", ""))).split())
            if not (title_tokens & query_tokens):
                continue

        # token_set_ratio/partial_ratio đều cho 100 khi câu hỏi là tập con của
        # nhãn, nên "chi phí tài chính" khớp ngang nhau với chính nó và với
        # "chi phí hoạt động tài chính với các bên liên quan". `ratio` so cả
        # chuỗi nên ưu tiên nhãn sát nghĩa nhất thay vì nhãn dài hơn.
        tightness = fuzz.ratio(query_norm, label_norm)

        rank = similarity + coverage * 10 + tightness * 0.25
        if row["column_role"] == prefer_role:
            rank += 15
        elif row["column_role"] == "unknown":
            rank += 5

        # Cùng một nhãn có thể nằm ở nhiều thuyết minh khác nhau ("Tiền gửi tại
        # các TCTD khác" vừa ở mục Tiền gửi, vừa ở mục Tiền tương đương tiền).
        # Tiêu đề thuyết minh nào sát câu hỏi hơn thì dòng đó đáng tin hơn.
        title_tokens = set(_strip_diacritics(str(row.get("note_title", ""))).split())
        title_overlap = len(title_tokens & query_tokens) / max(len(query_tokens), 1)
        rank += title_overlap * 8

        scored.append((rank, row))

    if not scored:
        return None

    best_rank = max(rank for rank, _ in scored)
    top = [row for rank, row in scored if rank >= best_rank - 1e-9]

    # Cùng một nhãn có thể xuất hiện ở nhiều bảng khác nhau trong cùng chunk
    # (vd: "Chu Thị Bình" vừa ở bảng giao dịch bên liên quan vừa ở bảng thù lao).
    # Khi các ứng viên tốt nhất cho giá trị khác nhau thì không có cơ sở để chọn:
    # trả về None để hệ thống nói không biết, thay vì đoán bừa một con số.
    distinct = {round(float(r["value_vnd"]), 6) for r in top}
    if len(distinct) > 1:
        logger.info("Notes label '%s' khớp %d dòng có giá trị khác nhau -> bỏ qua",
                    query, len(distinct))
        return None

    best = top[0]
    logger.debug("Notes label match: '%s' -> '%s' (rank=%.1f)", query, best["label"], best_rank)
    return best
