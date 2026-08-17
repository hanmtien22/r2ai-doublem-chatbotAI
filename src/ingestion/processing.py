"""Strict parsing and persistence for primary financial statements.

Only validated BS/IS/CF/EQ rows belong in this module. Notes and disclosures are
handled by :mod:`src.ingestion.notes` and are never interpreted as statement rows.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from html.parser import HTMLParser
import json
import logging
from pathlib import Path
import re
import unicodedata
from typing import Any

import pandas as pd

from .schemas import DetectedTable

logger = logging.getLogger(__name__)

ITEM_CODE_PATTERN = re.compile(r"^\d{2,4}[A-Za-z]?$")
FINANCIAL_VALUE_PATTERN = re.compile(
    r"^(?:"
    r"\d+|"                              # integer
    r"\d{1,3}(?:[.,]\d{3})+|"           # grouped thousands
    r"\d{1,3}(?:\s\d{3})+|"             # space-grouped thousands
    r"\d+[.,]\d{1,2}"                    # decimal
    r")$"
)
FINAL_COLUMNS = [
    "item_code", "item_name_raw", "item_name_normalized", "value_raw",
    "value", "unit", "ticker", "year", "period", "report_type",
    "table_type", "table_name", "note_reference",
]
STRUCTURED_DTYPES = {
    "item_code": "string", "ticker": "string", "report_type": "string",
    "table_type": "string", "note_reference": "string",
}

TABLE_PATTERNS = {
    "balance_sheet": ("bang can doi ke toan", "bao cao tinh hinh tai chinh"),
    "income_statement": (
        "bao cao ket qua hoat dong kinh doanh", "bao cao ket qua kinh doanh",
        "bao cao ket qua hoat dong",
    ),
    "cash_flow": ("bao cao luu chuyen tien te", "luu chuyen tien te"),
    "equity_statement": (
        "bao cao thay doi von chu so huu", "bien dong von chu so huu",
        "bao cao tinh hinh bien dong von chu so huu",
        "bao cao bien dong von chu so huu",
    ),
}
TABLE_TYPE_TO_SECTION = {
    "balance_sheet": "BS", "income_statement": "IS",
    "cash_flow": "CF", "equity_statement": "EQ",
}
NOTE_MARKERS = (
    "thuyet minh bao cao tai chinh", "thong tin bo sung",
    "thong tin bo sung cho cac khoan muc", "chinh sach ke toan",
    "cac khoan vay", "tai san the chap", "giao dich voi cac ben lien quan",
    "cam ket", "nghia vu", "phu luc",
)
NOTES_START_MARKERS = (
    "thuyet minh bao cao tai chinh", "thong tin bo sung",
    "chinh sach ke toan", "phu luc",
)
TRASH_MARKERS = (
    "nguoi lap bieu", "ke toan truong", "tong giam doc", "giam doc", "chu ky",
)
NULL_VALUES = {"", "-", "--", "—", "–", "n/a", "na", "null", "none"}
COMPANY_STOPWORDS = (
    "cong ty co phan", "cong ty trach nhiem huu han", "tong cong ty",
    "tap doan", "cong ty", "ctcp", "tnhh",
)


def remove_vietnamese_accents(text: str) -> str:
    value = unicodedata.normalize("NFD", str(text))
    value = "".join(char for char in value if unicodedata.category(char) != "Mn")
    return value.replace("đ", "d").replace("Đ", "D")


def normalize_text(text: str) -> str:
    value = remove_vietnamese_accents(str(text).lower())
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", value)).strip()


def scan_financial_files(root_path: str | Path) -> list[Path]:
    root = Path(root_path)
    if not root.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục: {root}")
    return sorted(root.rglob("*.txt"))


def extract_metadata(path: Path) -> dict[str, Any]:
    """Extract ticker/year/report type from directories, filename and path."""
    normalized = normalize_text(str(path))
    years = re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", str(path))
    if not years:
        raise ValueError(f"Không xác định được year từ path: {path}")
    year = int(years[-1])

    ticker = None
    parts = list(path.parts)
    for index, part in enumerate(parts):
        if part == str(year) and index > 0:
            candidate = re.sub(r"[^A-Za-z0-9]", "", parts[index - 1]).upper()
            if 2 <= len(candidate) <= 10:
                ticker = candidate
    if ticker is None:
        stem_match = re.match(r"([A-Za-z][A-Za-z0-9]{1,9})[_\-]", path.name)
        ticker = stem_match.group(1).upper() if stem_match else None
    if ticker is None:
        raise ValueError(f"Không xác định được ticker từ path: {path}")

    if "consolidated" in normalized or "hop nhat" in normalized:
        report_type = "consolidated"
    elif any(value in normalized for value in ("separate", "bao cao rieng", "cong ty me")):
        report_type = "separate"
    else:
        report_type = None
    return {
        "sticker": ticker, "ticker": ticker, "year": year,
        "report_type": report_type, "source_file": path.name,
        "source_path": str(path),
    }


def infer_report_type(lines: list[str], tables: list[DetectedTable]) -> str | None:
    """Infer report scope from detected primary headings, then document front matter."""
    heading_text = normalize_text("\n".join(table.table_name for table in tables))
    if "hop nhat" in heading_text:
        return "consolidated"
    if "rieng" in heading_text or "cong ty me" in heading_text:
        return "separate"

    # Some OCR files have incomplete primary headings. Search front matter up to
    # the first real statement (not an arbitrary fixed number of lines).
    first_table_line = min((table.start_line for table in tables), default=min(len(lines), 600))
    front_matter = normalize_text("\n".join(lines[:first_table_line]))
    if "bao cao tai chinh hop nhat" in front_matter:
        return "consolidated"
    if any(marker in front_matter for marker in (
        "bao cao tai chinh rieng", "bao cao tinh hinh tai chinh rieng",
        "bao cao ket qua hoat dong rieng", "bao cao luu chuyen tien te rieng",
    )):
        return "separate"
    return None


def identify_table_type(line: str) -> str | None:
    if "<table" in line.lower():
        return None
    normalized = normalize_text(line)
    if not normalized or len(normalized) > 180 or is_note_section(line):
        return None
    for table_type, patterns in TABLE_PATTERNS.items():
        if any(normalized.startswith(pattern) for pattern in patterns):
            return table_type
    return None


def is_note_section(line: str) -> bool:
    normalized = normalize_text(line)
    return any(marker in normalized for marker in NOTE_MARKERS)


def detect_notes_start(lines: list[str], after_line: int = 0) -> int | None:
    """Find the first standalone notes heading, ignoring TOC and audit prose.

    A substring match is deliberately insufficient: tables of contents and audit
    opinions mention "thuyết minh báo cáo tài chính" before primary statements.
    """
    for index in range(max(0, after_line), len(lines)):
        raw = lines[index].strip()
        if not raw or "<table" in raw.lower() or len(raw) > 220:
            continue
        normalized = normalize_text(re.sub(r"<[^>]+>", " ", raw))
        # Allow numbered/Roman prefixes, but require the disclosure marker to be
        # at heading position rather than buried in a sentence.
        normalized = re.sub(r"^(?:[ivxlcdm]+|\d+(?:\s+\d+)*)\s+", "", normalized)
        if any(normalized.startswith(marker) for marker in NOTES_START_MARKERS):
            return index
    return None


def is_page_marker(line: str) -> bool:
    normalized = normalize_text(line)
    return bool(re.fullmatch(r"(?:page|trang)\s*\d+", normalized))


def _extract_html_table(lines: list[str], heading_index: int, max_distance: int = 20):
    """Return only the nearest complete ``<table>...</table>`` after a heading."""
    limit = min(len(lines), heading_index + max_distance + 1)
    start = next((i for i in range(heading_index + 1, limit) if "<table" in lines[i].lower()), None)
    if start is None:
        return None
    parts: list[str] = []
    depth = 0
    for index in range(start, min(len(lines), start + 201)):
        line = lines[index]
        lower = line.lower()
        if index == start:
            offset = lower.find("<table")
            line, lower = line[offset:], lower[offset:]
        depth += len(re.findall(r"<table\b", lower))
        depth -= len(re.findall(r"</table\s*>", lower))
        if depth <= 0 and "</table" in lower:
            close = lower.rfind("</table")
            close_end = lower.find(">", close)
            parts.append(line[:close_end + 1])
            return start, index + 1, parts
        parts.append(line)
    return None


def _extract_text_block(lines: list[str], heading_index: int):
    """Bound a fallback block without ever crossing into another statement/notes."""
    end = len(lines)
    for index in range(heading_index + 1, len(lines)):
        if identify_table_type(lines[index]) or is_note_section(lines[index]):
            end = index
            break
    block = lines[heading_index + 1:end]
    # A fallback is admitted only if it already contains several strict rows.
    if sum(parse_table_line(line) is not None for line in block) < 3:
        return None
    return heading_index + 1, end, block


def detect_tables(lines: list[str]) -> list[DetectedTable]:
    tables: list[DetectedTable] = []
    consumed: set[tuple[int, int]] = set()
    notes_start = detect_notes_start(lines)
    for heading_index, heading in enumerate(lines):
        if notes_start is not None and heading_index >= notes_start:
            break
        table_type = identify_table_type(heading)
        if table_type is None:
            continue
        match = _extract_html_table(lines, heading_index) or _extract_text_block(lines, heading_index)
        if match is None:
            logger.debug("Ignored primary-heading candidate without a valid table at line %d", heading_index + 1)
            continue
        start, end, table_lines = match
        if (start, end) in consumed:
            continue
        consumed.add((start, end))
        context_start = max(0, heading_index - 5)
        detected = DetectedTable(
            type_table=table_type, table_name=heading.strip(),
            start_line=start, end_line=end, lines=table_lines,
            context_lines=lines[context_start:start],
        )
        if (
            tables and tables[-1].type_table == table_type
            and "tiep theo" in normalize_text(heading)
        ):
            # Continuation pages are one logical statement. Merging avoids
            # rejecting a last page merely because it contains fewer than the
            # configured minimum number of rows.
            tables[-1].end_line = end
            tables[-1].lines.extend(table_lines)
            tables[-1].context_lines.extend(detected.context_lines)
        else:
            tables.append(detected)
    return tables


class _HTMLTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag.lower() == "tr":
            self._row = []
        elif tag.lower() in {"td", "th"} and self._row is not None:
            self._cell = []

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"td", "th"} and self._row is not None and self._cell is not None:
            self._row.append(re.sub(r"\s+", " ", "".join(self._cell)).strip())
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if self._row:
                self.rows.append(self._row)
            self._row = None


def _header_role(cell: str) -> str | None:
    value = normalize_text(cell)
    compact = value.replace(" ", "")
    if compact in {"ma", "maso", "meso", "ms"}:
        return "item_code"
    if compact in {"thuyetminh", "tm", "note", "ghichu"}:
        return "note_reference"
    if any(key in value for key in ("chi tieu", "tai san", "nguon von", "khoan muc")):
        return "item_name"
    if any(key in compact for key in ("namnay", "cuoinam", "kynay", "3112")):
        return "current_value"
    if any(key in compact for key in ("namtruoc", "daunam", "kytruoc", "0101")):
        return "previous_value"
    return None


def _semantic_mapping(rows: list[list[str]]) -> tuple[dict[str, int], int] | None:
    best: tuple[dict[str, int], int] | None = None
    for row_index, cells in enumerate(rows[:8]):
        mapping = {role: index for index, cell in enumerate(cells) if (role := _header_role(cell))}
        if "item_name" not in mapping and "item_code" in mapping:
            unused = [index for index, cell in enumerate(cells) if index not in mapping.values() and not cell.strip()]
            if unused:
                mapping["item_name"] = unused[0]
        if {"item_code", "item_name"} <= mapping.keys():
            # Reports often label value columns with bare years. Their order is
            # current then previous; infer only within an otherwise semantic header.
            year_columns = [
                index for index, cell in enumerate(cells)
                if re.fullmatch(r"(?:nam\s*)?(?:19|20)\d{2}", normalize_text(cell))
            ]
            if "current_value" not in mapping and year_columns:
                mapping["current_value"] = year_columns[0]
            if "previous_value" not in mapping and len(year_columns) > 1:
                mapping["previous_value"] = year_columns[1]
        if {"item_code", "item_name"} <= mapping.keys() and "current_value" in mapping:
            best = (mapping, row_index)
    return best


def _strong_fallback_mapping(rows: list[list[str]]) -> dict[str, int] | None:
    widths = Counter(len(row) for row in rows if len(row) >= 4)
    if not widths:
        return None
    width, count = widths.most_common(1)[0]
    candidates = [row for row in rows if len(row) == width]
    for code_col in range(width):
        code_ratio = sum(bool(ITEM_CODE_PATTERN.fullmatch(row[code_col].strip())) for row in candidates) / len(candidates)
        if code_ratio < 0.8:
            continue
        for name_col in range(width):
            if name_col == code_col:
                continue
            if sum(_is_valid_item_name(row[name_col]) for row in candidates) / len(candidates) < 0.8:
                continue
            value_cols = [index for index in range(width) if index not in {code_col, name_col} and sum(_is_financial_value(row[index]) for row in candidates) / len(candidates) >= 0.8]
            if value_cols:
                mapping = {"item_code": code_col, "item_name": name_col, "current_value": value_cols[-2] if len(value_cols) > 1 else value_cols[-1]}
                if len(value_cols) > 1:
                    mapping["previous_value"] = value_cols[-1]
                return mapping
    return None


def _is_financial_value(value: str) -> bool:
    text = str(value).strip().lower()
    if text in NULL_VALUES:
        return True
    text = re.sub(r"(?:vnd|vnđ|đồng|dong)$", "", text).strip()
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1].strip()
    if text.startswith("-"):
        text = text[1:].strip()
    # Do not salvage malformed OCR/rowspan concatenations. For example,
    # 963.717.122.052237.314... contains a six-digit separator group and
    # represents several values glued into one HTML cell.
    return bool(FINANCIAL_VALUE_PATTERN.fullmatch(text))


def _is_valid_item_name(value: str) -> bool:
    raw = re.sub(r"<[^>]+>", " ", str(value))
    normalized = normalize_text(raw)
    if not normalized or len(normalized) < 2 or len(normalized) > 220:
        return False
    if is_page_marker(raw) or is_note_section(raw) or any(marker in normalized for marker in TRASH_MARKERS):
        return False
    if not re.search(r"[A-Za-zÀ-ỹ]", raw):
        return False
    return len(raw.split()) <= 35


def parse_html_table_line(html: str) -> list[dict]:
    parser = _HTMLTableParser()
    parser.feed(html)
    semantic = _semantic_mapping(parser.rows)
    if semantic:
        mapping, header_index = semantic
        rows = parser.rows[header_index + 1:]
    else:
        rows = parser.rows
        mapping = _strong_fallback_mapping(rows)
    if not mapping:
        return []

    parsed: list[dict] = []
    required_max = max(mapping.values())
    for cells in rows:
        if len(cells) <= required_max:
            continue
        code = cells[mapping["item_code"]].strip()
        name = cells[mapping["item_name"]].strip()
        if not ITEM_CODE_PATTERN.fullmatch(code) or not _is_valid_item_name(name):
            continue
        current = cells[mapping["current_value"]].strip() if "current_value" in mapping else None
        previous = cells[mapping["previous_value"]].strip() if "previous_value" in mapping else None
        if not any(_is_financial_value(value) for value in (current, previous) if value is not None):
            continue
        note = cells[mapping["note_reference"]].strip() if "note_reference" in mapping else ""
        parsed.append({
            "item_code": code, "item_name_raw": name,
            "note_reference": note or None,
            "current_value_raw": current if current and _is_financial_value(current) else None,
            "previous_value_raw": previous if previous and _is_financial_value(previous) else None,
        })
    return parsed


def parse_table_line(line: str) -> dict | None:
    """Strict text fallback: code + meaningful name + one/two terminal values."""
    clean = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", line)).strip()
    if not clean or is_page_marker(clean) or is_note_section(clean):
        return None
    normalized = normalize_text(clean)
    if any(marker in normalized for marker in TRASH_MARKERS):
        return None
    code_match = re.match(r"^(\d{2,4}[A-Za-z]?)\s+", clean)
    if not code_match:
        # Common statement layout has code between item name and values.
        candidates = list(re.finditer(r"\s(\d{2,4}[A-Za-z]?)\s+(?=(?:\(?-?\d|[-—–]))", clean))
        code_match = candidates[-1] if candidates else None
        if code_match:
            tail = clean[code_match.end():].split()
            # This prevents "Doanh thu 120 100" from treating 120 as a code.
            if sum(_is_financial_value(token) for token in tail[-2:]) < 2:
                code_match = None
    if not code_match:
        return None
    code = code_match.group(1)
    if code_match.start() == 0:
        remainder = clean[code_match.end():]
    else:
        remainder = (clean[:code_match.start()] + " " + clean[code_match.end():]).strip()
    tokens = remainder.split()
    values: list[str] = []
    while tokens and len(values) < 2 and _is_financial_value(tokens[-1]):
        values.insert(0, tokens.pop())
    name = " ".join(tokens).strip()
    if not values or not _is_valid_item_name(name):
        return None
    return {
        "item_code": code, "item_name_raw": name, "note_reference": None,
        "current_value_raw": values[0],
        "previous_value_raw": values[1] if len(values) == 2 else None,
    }


def parse_table_lines(lines: list[str]) -> pd.DataFrame:
    joined = "\n".join(lines)
    rows = parse_html_table_line(joined) if "<table" in joined.lower() else [
        parsed for line in lines if (parsed := parse_table_line(line)) is not None
    ]
    return pd.DataFrame(rows)


def parse_number(value: str | int | float | None) -> Decimal | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (int, float, Decimal)):
        result = Decimal(str(value))
        return result if result.is_finite() else None
    raw_text = str(value).strip().lower()
    if not _is_financial_value(raw_text):
        logger.debug("Ignored malformed financial value: %r", value)
        return None
    text = re.sub(r"\s+", "", raw_text)
    if text in NULL_VALUES:
        return None
    negative = text.startswith("-") or (text.startswith("(") and text.endswith(")"))
    text = re.sub(r"[()\-]", "", text)
    text = re.sub(r"(?:vnd|vnđ|đồng|dong)$", "", text)
    if re.fullmatch(r"\d{1,3}(?:[.,]\d{3})+", text):
        text = re.sub(r"[.,]", "", text)
    elif re.fullmatch(r"\d+,[0-9]{1,2}", text):
        text = text.replace(",", ".")
    else:
        text = re.sub(r"[^0-9.]", "", text)
    if not text:
        return None
    try:
        result = Decimal(text)
    except InvalidOperation:
        # Parsing one damaged OCR cell must not abort the entire source file.
        logger.debug("Ignored financial value that Decimal cannot parse: %r", value)
        return None
    return -result if negative else result


def detect_unit(text: str) -> tuple[str, int]:
    normalized = normalize_text(text)
    for name, multiplier in (
        ("ty dong", 1_000_000_000), ("trieu dong", 1_000_000),
        ("nghin dong", 1_000), ("ngan dong", 1_000),
        ("vnd", 1), ("dong", 1),
    ):
        if name in normalized:
            return name, multiplier
    return "unknown", 1


def normalize_to_vnd(value: Decimal | None, multiplier: int) -> int | None:
    return None if value is None else int(value * multiplier)


def normalize_item_name(item_name: str) -> str:
    return normalize_text(item_name).replace(" ", "_")


def validate_table(df: pd.DataFrame, minimum_table_rows: int = 3, maximum_null_ratio: float = 0.7) -> list[str]:
    errors: list[str] = []
    missing = set(FINAL_COLUMNS) - set(df.columns)
    if missing:
        return [f"Thiếu cột: {sorted(missing)}"]
    if df.empty:
        errors.append("Bảng rỗng")
        return errors
    source_row_count = len(df.drop_duplicates(subset=["item_code", "item_name_raw"]))
    if source_row_count < minimum_table_rows:
        errors.append("Bảng có quá ít dòng")
    valid_code_ratio = df["item_code"].astype("string").str.fullmatch(ITEM_CODE_PATTERN).fillna(False).mean()
    if valid_code_ratio < 0.95:
        errors.append(f"Tỷ lệ item_code hợp lệ quá thấp: {valid_code_ratio:.2%}")
    null_ratio = df["value"].isna().mean()
    if null_ratio > maximum_null_ratio:
        errors.append(f"Tỷ lệ value null quá cao: {null_ratio:.2%}")
    suspicious = df["item_name_raw"].astype("string").map(lambda value: not _is_valid_item_name(value)).mean()
    if suspicious > 0.05:
        errors.append(f"Tỷ lệ text nghi ngờ quá cao: {suspicious:.2%}")
    if not set(df["table_type"].dropna()).issubset(TABLE_PATTERNS):
        errors.append("table_type không hợp lệ")
    return errors


def save_parsed_table(df: pd.DataFrame, output_dir: str | Path, ticker: str, year: int,
                      table_type: str, report_type: str = "consolidated",
                      table_id: str | int | None = None) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{int(table_id):03d}" if table_id is not None else ""
    output_path = output_dir / f"{ticker}_{year}_{report_type}_{table_type}{suffix}.csv"
    missing = set(FINAL_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Không thể save CSV; thiếu final columns: {sorted(missing)}")
    saved = df[FINAL_COLUMNS].copy()
    for column in STRUCTURED_DTYPES:
        saved[column] = saved[column].astype("string")
    saved["value"] = pd.to_numeric(saved["value"], errors="coerce")
    saved["year"] = pd.to_numeric(saved["year"], errors="raise").astype("int64")
    saved["period"] = pd.to_numeric(saved["period"], errors="raise").astype("int64")
    saved.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def read_structured_table(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=STRUCTURED_DTYPES, encoding="utf-8-sig")


def scan_tickers_from_csv(table_dir: str | Path) -> set[str]:
    tickers: set[str] = set()
    for path in sorted(Path(table_dir).rglob("*.csv")) if Path(table_dir).exists() else []:
        try:
            frame = pd.read_csv(path, usecols=["ticker"], dtype={"ticker": "string"})
        except (ValueError, OSError, pd.errors.ParserError) as error:
            logger.warning("Cannot read ticker from %s: %s", path, error)
            continue
        tickers.update(str(value).strip().upper() for value in frame["ticker"].dropna() if str(value).strip())
    return tickers


def _build_short_company_name(name: str) -> str:
    result = normalize_text(name)
    for stopword in COMPANY_STOPWORDS:
        result = re.sub(rf"\b{re.escape(stopword)}\b", " ", result)
    return re.sub(r"\s+", " ", result).strip()


def _question_pairs(path: str | Path) -> list[tuple[str, str]]:
    source = Path(path)
    if not source.exists():
        return []
    pairs: list[tuple[str, str]] = []
    pattern = re.compile(r"([^\n()]{2,180}?)\s*\(([A-Z][A-Z0-9]{1,9})\)")
    company_prefix = re.compile(
        r"(?i)(công\s+ty\s+cổ\s+phần|công\s+ty\s+tnhh|công\s+ty|ctcp|"
        r"tổng\s+ctcp|tổng\s+công\s+ty|tập\s+đoàn|ngân\s+hàng(?:\s+tmcp)?)"
    )
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                question = str(json.loads(line).get("question", ""))
            except json.JSONDecodeError:
                continue
            for name, ticker in pattern.findall(question):
                name = re.split(r"[;,:]", name)[-1].strip(" -–—,.;:")
                prefixes = list(company_prefix.finditer(name))
                if prefixes:
                    name = name[prefixes[-1].start():].strip()
                # Without a company marker, only accept a compact proper name.
                if prefixes or len(name.split()) <= 8:
                    pairs.append((name, ticker.upper()))
    return pairs


def build_entity_dictionary(csv_path: str | Path, output_path: str | Path,
                            questions_path: str | Path | None = None,
                            structured_dir: str | Path | None = None, **legacy) -> dict:
    """Build entities from canonical CSV; questions add aliases, never tickers."""
    if structured_dir is None:
        structured_dir = legacy.get("table_dir")
    source = pd.read_csv(csv_path)
    columns = {normalize_text(column): column for column in source.columns}
    ticker_col = next((columns[key] for key in ("ticker", "sticker", "ma ck") if key in columns), None)
    name_col = next((columns[key] for key in ("company_name", "ten cong ty") if key in columns), None)
    if ticker_col is None or name_col is None:
        raise ValueError("Thiếu cột ticker/Mã CK hoặc company_name/Tên công ty")
    canonical = {
        str(row[ticker_col]).strip().upper(): (str(row[name_col]).strip(), row)
        for _, row in source.iterrows() if pd.notna(row[ticker_col]) and pd.notna(row[name_col])
    }
    allowed = set(canonical)
    if structured_dir is not None:
        allowed &= scan_tickers_from_csv(structured_dir)
    question_aliases: dict[str, set[str]] = defaultdict(set)
    if questions_path:
        for alias, ticker in _question_pairs(questions_path):
            if ticker in allowed:
                question_aliases[ticker].add(alias)
    alias_col = columns.get("alias")
    entities = {}
    for ticker in sorted(allowed):
        company_name, row = canonical[ticker]
        aliases = {ticker, company_name, normalize_text(company_name), _build_short_company_name(company_name)}
        if alias_col and pd.notna(row.get(alias_col)):
            aliases.update(part.strip() for part in str(row[alias_col]).split("|") if part.strip())
        for alias in question_aliases[ticker]:
            aliases.update((alias, normalize_text(alias), _build_short_company_name(alias)))
        entities[ticker] = {"full_name": company_name, "short_name": _build_short_company_name(company_name), "aliases": sorted(filter(None, aliases))}
    save_json(entities, output_path)
    return entities


def create_dictionary_stats() -> dict:
    return {"indicators": defaultdict(Counter), "table_names": defaultdict(Counter), "units": Counter(), "tickers": Counter()}


def collect_dictionary_features(df: pd.DataFrame, stats: dict) -> None:
    if df.empty:
        return
    for row in df.drop_duplicates(subset=["table_type", "item_code", "item_name_raw"]).to_dict("records"):
        code = str(row.get("item_code", "")).strip()
        if ITEM_CODE_PATTERN.fullmatch(code):
            stats["indicators"][(row["table_type"], code)][row["item_name_raw"]] += 1
    for row in df.drop_duplicates(subset=["table_type", "table_name"]).to_dict("records"):
        stats["table_names"][row["table_type"]][row["table_name"]] += 1
    stats["units"].update(df["unit"].dropna().astype(str).unique())
    stats["tickers"].update(df["ticker"].dropna().astype(str).unique())


def build_indicator_aliases_from_stats(stats: dict, min_count: int = 5) -> dict:
    result = {}
    for (table_type, code), names in stats["indicators"].items():
        section = TABLE_TYPE_TO_SECTION[table_type]
        for name, count in names.items():
            if count >= min_count:
                result[normalize_text(name)] = f"{section}.{code}"
    return result


def build_schema_mapping_from_stats(stats: dict, min_count: int = 5) -> dict:
    result: dict[str, dict[str, dict[str, str]]] = {}
    for (table_type, code), names in stats["indicators"].items():
        eligible = [(count, name) for name, count in names.items() if count >= min_count]
        if eligible:
            section = TABLE_TYPE_TO_SECTION[table_type]
            result.setdefault(section, {})[code] = {"name": max(eligible)[1]}
    return result


def build_dictionary_report(stats: dict) -> dict:
    return {
        "indicator_count": len(stats["indicators"]),
        "table_types": {key: sum(value.values()) for key, value in stats["table_names"].items()},
        "units": dict(stats["units"]), "tickers": dict(stats["tickers"]),
    }


def save_json(value: Any, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def relative_difference(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1)
