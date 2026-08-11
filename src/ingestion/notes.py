"""Section-aware extraction and chunking for notes/disclosures."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from html.parser import HTMLParser
import re
from pathlib import Path

import pandas as pd

from .processing import NOTE_MARKERS, detect_notes_start, is_page_marker, normalize_text
from .schemas import DetectedTable

NOTES_COLUMNS = [
    "chunk_id", "ticker", "year", "report_type", "document_type",
    "section_id", "section_title", "section_level", "chunk_index", "text",
    "source_file", "start_line", "end_line",
]


@dataclass(frozen=True)
class SectionHeader:
    section_id: str
    section_title: str
    section_level: int


@dataclass
class _Line:
    number: int
    text: str
    end_number: int | None = None


class _ReadableTableParser(HTMLParser):
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
        if tag.lower() in {"td", "th"} and self._row is not None and self._cell is not None:
            self._row.append(re.sub(r"\s+", " ", "".join(self._cell)).strip())
            self._cell = None
        elif tag.lower() == "tr" and self._row is not None:
            if any(self._row):
                self.rows.append(self._row)
            self._row = None


def html_table_to_text(html: str) -> str:
    parser = _ReadableTableParser()
    parser.feed(html)
    return "\n".join(" | ".join(cell for cell in row if cell) for row in parser.rows)


def detect_section_header(line: str) -> SectionHeader | None:
    """Recognize Roman, decimal and lettered Vietnamese note headings."""
    clean = re.sub(r"<[^>]+>", " ", line)
    clean = re.sub(r"\s+", " ", clean).strip()
    if not clean or len(clean) > 220:
        return None
    patterns = (
        (r"^([IVXLCDM]+)[.)]\s+(.+)$", 1),
        (r"^(\d+(?:\.\d+)+)\s*[.)]?\s+(.+)$", None),
        (r"^(\d+)[.)]\s+(.+)$", 2),
        (r"^([a-zA-Z])[.)]\s+(.+)$", 3),
    )
    for pattern, fixed_level in patterns:
        match = re.match(pattern, clean)
        if not match:
            continue
        identifier, title = match.groups()
        if len(title) < 2 or not re.search(r"[A-Za-zÀ-ỹ]", title):
            return None
        level = fixed_level if fixed_level is not None else identifier.count(".") + 2
        return SectionHeader(identifier, title.strip(), level)
    normalized = normalize_text(clean)
    if any(normalized.startswith(marker) for marker in NOTE_MARKERS):
        return SectionHeader(normalized.replace(" ", "_"), clean, 1)
    return None


def _clean_lines(lines: list[str], excluded: set[int]) -> list[_Line]:
    output: list[_Line] = []
    html_buffer: list[str] = []
    html_start = 0
    in_table = False
    repeated = {}
    for line in lines:
        value = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", line)).strip()
        if value and len(value) < 160:
            repeated[normalize_text(value)] = repeated.get(normalize_text(value), 0) + 1

    for index, raw in enumerate(lines):
        if index in excluded:
            continue
        lower = raw.lower()
        if not in_table and "<table" in lower:
            in_table, html_start, html_buffer = True, index, [raw]
            if "</table" not in lower:
                continue
        elif in_table:
            html_buffer.append(raw)
            if "</table" not in lower:
                continue
        if in_table:
            readable = html_table_to_text("\n".join(html_buffer)).strip()
            if readable:
                output.append(_Line(html_start + 1, readable, index + 1))
            in_table, html_buffer = False, []
            continue
        clean = re.sub(r"<[^>]+>", " ", raw)
        clean = re.sub(r"\s+", " ", clean).strip()
        normalized = normalize_text(clean)
        if not clean or is_page_marker(clean):
            continue
        # Remove short page furniture repeated at least three times. Meaningful
        # note content (numbers, contracts, dates) remains untouched.
        if repeated.get(normalized, 0) >= 3 and len(clean) < 100 and detect_section_header(clean) is None:
            continue
        output.append(_Line(index + 1, clean))
    return output


def _split_long(lines: list[_Line], max_chars: int, overlap_chars: int) -> list[list[_Line]]:
    if len("\n".join(line.text for line in lines)) <= max_chars:
        return [lines]
    chunks: list[list[_Line]] = []
    current: list[_Line] = []
    current_length = 0
    for line in lines:
        addition = len(line.text) + (1 if current else 0)
        if current and current_length + addition > max_chars:
            chunks.append(current)
            overlap: list[_Line] = []
            overlap_length = 0
            for prior in reversed(current):
                if overlap and overlap_length + len(prior.text) + 1 > overlap_chars:
                    break
                overlap.insert(0, prior)
                overlap_length += len(prior.text) + 1
            current, current_length = overlap, overlap_length
        # A single pathological line is split without losing source attribution.
        if len(line.text) > max_chars:
            for start in range(0, len(line.text), max(1, max_chars - overlap_chars)):
                piece = line.text[start:start + max_chars]
                if current:
                    chunks.append(current)
                    current, current_length = [], 0
                chunks.append([_Line(line.number, piece, line.end_number)])
            continue
        current.append(line)
        current_length += addition
    if current:
        chunks.append(current)
    return chunks


def _coalesce_short_chunks(chunks: list[list[_Line]], min_chars: int) -> list[list[_Line]]:
    """Avoid tiny fragments while preserving section boundaries."""
    if min_chars <= 0 or len(chunks) <= 1:
        return chunks
    result: list[list[_Line]] = []
    index = 0
    while index < len(chunks):
        chunk = chunks[index]
        length = len("\n".join(line.text for line in chunk))
        if length < min_chars and index + 1 < len(chunks):
            result.append(chunk + chunks[index + 1])
            index += 2
        elif length < min_chars and result:
            result[-1].extend(chunk)
            index += 1
        else:
            result.append(chunk)
            index += 1
    return result


def build_notes_chunks(lines: list[str], tables: list[DetectedTable], metadata: dict,
                       max_chars: int = 3000, overlap_chars: int = 300,
                       min_chars: int = 100) -> pd.DataFrame:
    """Remove primary table spans, then chunk disclosures by semantic section."""
    excluded = {index for table in tables for index in range(table.start_line, table.end_line)}
    cleaned = _clean_lines(lines, excluded)
    # Notes begin only at an explicit disclosure marker; front matter and primary
    # statement headings therefore cannot leak into the notes corpus.
    source_start = detect_notes_start(lines)
    if source_start is None:
        return pd.DataFrame(columns=NOTES_COLUMNS)
    cleaned = [line for line in cleaned if line.number >= source_start + 1]

    sections: list[tuple[SectionHeader, list[_Line]]] = []
    header = SectionHeader("notes", cleaned[0].text, 1)
    content: list[_Line] = []
    for line in cleaned:
        detected = detect_section_header(line.text)
        if detected:
            if content and any(detect_section_header(item.text) is None for item in content):
                sections.append((header, content))
            header, content = detected, [line]
        else:
            content.append(line)
    if content and any(detect_section_header(item.text) is None for item in content):
        sections.append((header, content))

    records = []
    ticker = metadata["sticker"]
    for section_number, (section, section_lines) in enumerate(sections):
        split_chunks = _coalesce_short_chunks(
            _split_long(section_lines, max_chars, overlap_chars), min_chars,
        )
        for chunk_index, chunk_lines in enumerate(split_chunks):
            text = "\n".join(line.text for line in chunk_lines).strip()
            if not text or (len(text) < min_chars and is_page_marker(text)):
                continue
            records.append({
                "chunk_id": f"{ticker}:{metadata['year']}:{metadata['report_type']}:notes:{section_number}:{chunk_index}",
                "ticker": ticker, "year": int(metadata["year"]),
                "report_type": metadata["report_type"], "document_type": "notes",
                "section_id": section.section_id, "section_title": section.section_title,
                "section_level": section.section_level, "chunk_index": chunk_index,
                "text": text, "source_file": metadata["source_file"],
                "start_line": chunk_lines[0].number,
                "end_line": chunk_lines[-1].end_number or chunk_lines[-1].number,
            })
    return pd.DataFrame(records, columns=NOTES_COLUMNS)


def save_notes_chunks(df: pd.DataFrame, output_dir: str | Path, metadata: dict) -> Path | None:
    if df.empty:
        return None
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{metadata['sticker']}_{metadata['year']}_{metadata['report_type']}_notes.csv"
    df[NOTES_COLUMNS].to_csv(path, index=False, encoding="utf-8-sig")
    return path


def build_notes_retrieval_documents(df: pd.DataFrame) -> list[dict]:
    documents = []
    for row in df.to_dict("records"):
        metadata = {key: value for key, value in row.items() if key not in {"text", "chunk_id"}}
        documents.append({"chunk_id": row["chunk_id"], "text": row["text"], "metadata": metadata})
    return documents
