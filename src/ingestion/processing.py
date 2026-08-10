# import logging
# from pathlib import Path
# from typing import Any
# from html.parser import HTMLParser
# import re
# import unicodedata
# import json
# import pandas as pd
# from .schemas import DetectedTable
# from decimal import Decimal, InvalidOperation
# from collections import defaultdict, Counter

# logger = logging.getLogger(__name__)

# MAX_INT64 = 9_223_372_036_854_775_807
# MIN_INT64 = -9_223_372_036_854_775_808
# NUMBER_TOKEN = r"(?:\(?-?[\d.,]+\)?|[-—–])"

# COMPANY_STOPWORDS = [
#     "cong ty",
#     "cong ty co phan",
#     "cong ty trach nhiem huu han",
#     "tong cong ty",
#     "tap doan",
#     "ctcp",
#     "tnhh",
# ]

# TABLE_PATTERNS = {
#     "balance_sheet": [
#         "bang can doi ke toan",
#         "bao cao tinh hinh tai chinh",
#     ],
#     "income_statement": [
#         "bao cao ket qua hoat dong kinh doanh",
#         "bao cao ket qua kinh doanh",
#     ],
#     "cash_flow": [
#         "bao cao luu chuyen tien te",
#         "luu chuyen tien te",
#     ],
#     "equity_statement": [
#         "bao cao thay doi von chu so huu",
#         "bien dong von chu so huu",
#     ],
# }

# NULL_VALUES = {
#     "",
#     "-",
#     "--",
#     "—",
#     "–",
#     "n/a",
#     "na",
#     "null",
#     "none",
# }

# UNIT_MULTIPLIERS = {
#     "vnd": 1,
#     "dong": 1,
#     "nghin dong": 1_000,
#     "ngan dong": 1_000,
#     "trieu dong": 1_000_000,
#     "ty dong": 1_000_000_000,
# }

# ITEM_ALIASES = {
#     "doanh_thu_thuan": [
#         "doanh thu thuan",
#         "doanh thu thuan ve ban hang va cung cap dich vu",
#     ],
#     "loi_nhuan_sau_thue": [
#         "loi nhuan sau thue",
#         "loi nhuan sau thue thu nhap doanh nghiep",
#         "lnst",
#     ],
#     "tong_tai_san": [
#         "tong cong tai san",
#         "tong tai san",
#     ],
#     "von_chu_so_huu": [
#         "von chu so huu",
#         "tong cong von chu so huu",
#     ],
# }

# DATE_PATTERNS = [
#     r"\b\d{1,2}/\d{1,2}/\d{4}\b",
#     r"\b\d{1,2}-\d{1,2}-\d{4}\b",
#     r"\bnăm\s+\d{4}\b",
#     r"\b\d{4}\b",
# ]

# REQUIRED_COLUMNS = {
#     "item_name_raw",
#     "item_name_normalized",
#     "value",
#     "period",
#     "ticker",
#     "year",
#     "table_type",
#     "section",
# }

# TABLE_TYPE_TO_SECTION = {
#     "balance_sheet": "BS",
#     "income_statement": "IS",
#     "cash_flow": "CF",
#     "equity_statement": "EQ",
# }


# class _HTMLTableParser(HTMLParser):
#     """Collect rows and cells from an HTML table embedded in extracted text."""

#     def __init__(self) -> None:
#         super().__init__(convert_charrefs=True)
#         self.rows: list[list[str]] = []
#         self._row: list[str] | None = None
#         self._cell_parts: list[str] | None = None

#     def handle_starttag(self, tag: str, attrs) -> None:
#         if tag == "tr":
#             self._row = []
#         elif tag in {"td", "th"} and self._row is not None:
#             self._cell_parts = []

#     def handle_data(self, data: str) -> None:
#         if self._cell_parts is not None:
#             self._cell_parts.append(data)

#     def handle_endtag(self, tag: str) -> None:
#         if tag in {"td", "th"} and self._row is not None and self._cell_parts is not None:
#             cell = re.sub(r"\s+", " ", "".join(self._cell_parts)).strip()
#             self._row.append(cell)
#             self._cell_parts = None
#         elif tag == "tr" and self._row is not None:
#             self.rows.append(self._row)
#             self._row = None


# def scan_financial_files(root_path: str | Path) -> list[Path]:
#     """
#     Tìm tất cả các file txt
    
#     Parameters: 
#         root_path: Đường dẫn thư mục data/financial_statements
    
#     Return:
#         list[Path]: Danh sách file
    
#     """

#     if isinstance(root_path, str):
#         root_path = Path(root_path)

#     if not root_path.exists():
#         logger.error("Thư mục không tồn tại: %s", root_path)
#         raise FileNotFoundError(f"Không tìm thấy thư mục: {root_path}")

#     files = sorted(root_path.rglob("*.txt"))
#     logger.info("Scanned financial files in root_path=%s: found %d files", root_path, len(files))
#     return files


# def extract_meatdata(path: Path) -> dict[str, Any]:
#     """
#     Tách sticker, year, company code từ đường dẫn/ tên file
    
#     Parameters:
#         path: Đường dẫn file
        
#     Return:
#         dict: Từ điển bao gồm {
#             sticker,
#             year,
#             source_file,
#             source_path
#             }
        
#     """

#     parts = path.parts

#     sticker = None
#     year = None

#     for part in parts:
#         cleaned = part.strip().upper()
#         if cleaned.isdigit() and len(cleaned) == 4:
#             year_value = int(cleaned)

#             if 1990 <= year_value <= 2100:
#                 year = year_value


#     if year is not None:
#         year_index = parts.index(str(year))

#         if year_index > 0:
#             sticker = parts[year_index - 1].upper()

#     if sticker is None:
#         logger.error("Không xác định được sticker từ path: %s", path)
#         raise ValueError(f"Không xác định được sticker từ path: {path}")

#     if year is None:
#         logger.error("Không xác định được year từ path: %s", path)
#         raise ValueError(f"Không xác định được year từ path: {path}")

#     logger.debug("Extracted metadata from path=%s: sticker=%s, year=%s", path, sticker, year)
#     return {
#         "sticker": sticker,
#         "year" : year,
#         "source_file" : path.name,
#         "source_path" : str(path)
#     }

# def remove_vietnamese_accents(text: str) -> str:
#     # Tách các ký tự có 
#     text = unicodedata.normalize("NFD", text)

#     # Xóa dấu và ghép các ký tự không dấu
#     text = "".join(
#         char for char in text if unicodedata.category(char) != "Mn"
#     )

#     # Gộp về định dạng chuẩn 
#     # text = unicodedata.normalize("NFC", text)

#     return text.replace("đ", "d").replace("Đ", "D")



# def normalize_text(text: str) -> str:
#     # Chuẩn hóa văn bản 

#     text = remove_vietnamese_accents(text.lower())
#     text = re.sub(r"[^a-z0-9\s]", " ", text)
#     text = re.sub(r"\s+", " ", text).strip()
#     return text

# def build_entity_dictionary(
#         csv_path: str | Path,
#         output_path: str | Path
# ) -> dict:
#     logger.info("Building entity dictionary from csv_path=%s", csv_path)
#     df = pd.read_csv(csv_path)
    
#     normalized_columns = {
#         normalize_text(str(column)): column
#         for column in df.columns
#     }
#     ticker_column = next(
#         (normalized_columns[name] for name in ("ticker", "sticker", "ma ck") if name in normalized_columns),
#         None,
#     )
#     company_column = next(
#         (normalized_columns[name] for name in ("company_name", "ten cong ty") if name in normalized_columns),
#         None,
#     )

#     missing = []
#     if ticker_column is None:
#         missing.append("ticker/Mã CK")
#     if company_column is None:
#         missing.append("company_name/Tên công ty")
#     if missing:
#         logger.error("Thiếu cột trong entity dictionary csv (%s): %s", csv_path, missing)
#         raise ValueError(f"Thiếu cột trong code_stock.csv: {missing}")

#     entities = {}

#     for _, row in df.iterrows():
#         ticker = str(row[ticker_column]).strip().upper()
#         company_name = str(row[company_column]).strip()
#         normalized_company_name = normalize_text(company_name)
#         short_name = normalized_company_name
#         for stopword in sorted(COMPANY_STOPWORDS, key=len, reverse=True):
#             short_name = re.sub(
#                 rf"\b{re.escape(stopword)}\b",
#                 " ",
#                 short_name,
#             )
#         short_name = re.sub(r"\s+", " ", short_name).strip()

#         aliases = [ticker, company_name, normalized_company_name, short_name]

#         alias_column = normalized_columns.get("alias")
#         if alias_column and pd.notna(row.get(alias_column)):
#             aliases.extend(
#                 item.strip()
#                 for item in str(row[alias_column]).split("|")
#                 if item.strip()
#             )

#         aliases = sorted({alias for alias in aliases if alias})

#         entities[ticker] = {
#             "full_name": company_name,
#             "short_name": short_name or company_name,
#             "aliases": aliases
#         }


#     output_path = Path(output_path)
#     output_path.parent.mkdir(parents=True, exist_ok=True)

#     output_path.write_text(
#         json.dumps(entities, ensure_ascii=False, indent=2),
#         encoding="utf-8",
#     )

#     logger.info("Built entity dictionary with %d entities at output_path=%s", len(entities), output_path)

#     return entities


# def identify_table_type(line: str) -> str | None:
#     if "<table" in line.lower():
#         return None

#     normalized = normalize_text(line)

#     if len(normalized) > 160:
#         return None

#     for table_type, patterns in TABLE_PATTERNS.items():
#         if any(normalized.startswith(pattern) for pattern in patterns):
#             return table_type

#     return None

# # Nhận diện bảng 
# def detect_tables(lines: list[str]) -> list[DetectedTable]:
#     starts : list[tuple[int, str, str]] = []

#     for index, line in enumerate(lines):
#         type_table = identify_table_type(line=line)
#         has_nearby_table = any(
#             "<table" in candidate.lower()
#             for candidate in lines[index + 1:index + 16]
#         )
#         if type_table and has_nearby_table:
#             starts.append((index, type_table, line.strip()))

#     tables = []
#     for position, (start, type_table, table_name) in enumerate(starts):
#         if position + 1 < len(starts):
#             end = starts[position + 1][0]

#         else:
#             end = len(lines)

#         try: 
#             if start < 0 or end < 0 or start > end:
#                 raise ValueError(f"Dòng bắt đầu {start} hoặc dòng kết thức {end} không hợp lệ.")
            
#             tables.append(DetectedTable(
#                 type_table=type_table,
#                 table_name=table_name,
#                 start_line=start,
#                 end_line=end,
#                 lines=lines[start:end]
#             ))
#         except ValueError as e:
#             logger.warning("Dữ liệu không hợp lệ khi detect tables: %s", e)
#             print(f"Dữ liệu không hợp lệ: {e}.")

#     table_types = [t.type_table for t in tables]
#     logger.info("Detected %d tables from %d lines: table_types=%s", len(tables), len(lines), table_types)

#     return tables


# def parse_number(value: str | int | float | None) -> Decimal:

#     if value is None: 
#         return None

#     if isinstance(value, (int, float)):
#         return Decimal(str(value))

#     text = str(value).strip().lower()

#     if text in NULL_VALUES:
#         return None

#     pattern = r"\(?-?\d{1,3}(?:[.,]\d{3})+\)?"
#     matches = re.findall(pattern, text)

#     if matches:
#         text = matches[0]

#     else:
#         match = re.search(r"\(?-?\d[\d.,]*\)?", text)

#         if match:
#             text = match.group(0)

#     negative = False

#     if "(" in text or ")" in text or text.startswith("-"):
#         negative = True

#     text = re.sub(r"[()\-]", "", text).strip()
#     # Loại bỏ khoảng trắng và ký hiệu tiền tệ
#     text = re.sub(r"\s+", "", text)
#     text = re.sub(r"(vnd|vnđ|đồng|dong)$", "", text)


#     # 1.234.567 hoặc 1,234,567
#     if re.fullmatch(r"\d{1,3}([.,]\d{3})+", text):
#         text = re.sub(r"[.,]", "", text)

#     # 1234,56: dấu phẩy là dấu thập phân
#     elif re.fullmatch(r"\d+,\d{1,2}", text):
#         text = text.replace(",", ".")

#     # Loại bỏ ký tự còn lại không hợp lệ
#     text = re.sub(r"[^0-9.]", "", text)

#     if not text or not re.search(r"\d", text):
#         return None

#     try:
#         number = Decimal(text)
#     except InvalidOperation as error:
#         raise ValueError(f"Không parse được số: {value}") from error

#     return -number if negative else number


# def detect_unit(text: str) -> tuple[str, int]:
#     """
#     Nhận diện đơn vị
#     """

#     normalized = normalize_text(text)

#     ordered_units = [
#         ("ty dong", 1_000_000_000),
#         ("trieu dong", 1_000_000),
#         ("nghin dong", 1_000),
#         ("ngan dong", 1_000),
#         ("vnd", 1),
#         ("dong", 1),
#     ]

#     for unit_name, multiplier in ordered_units:
#         if unit_name in normalized:
#             return unit_name, multiplier

#     return "unknown", 1

# def normalize_to_vnd(value: Decimal | None, multiplier: int,) -> int | None:

#     if value is None:
#         return None

#     normalized = value * multiplier
#     return int(normalized)

# def normalize_item_name(item_name: str) -> str:
#     """
#     Chuẩn hóa tên chỉ tiêu
    
#     """

#     normalized = normalize_text(item_name)

#     for canonical_name, aliases in ITEM_ALIASES.items():
#         if normalized in aliases:
#             return canonical_name

#     return normalized.replace(" ", "_")

# def parse_table_line(line: str) -> dict | None:
#     line = re.sub(r"\s+", " ", line).strip()

#     numbers = list(re.finditer(NUMBER_TOKEN, line))

#     if len(numbers) < 2:
#         return None

#     # Giả định hai số cuối là hai kỳ báo cáo
#     current_value_raw = numbers[-2].group()
#     previous_value_raw = numbers[-1].group()

#     item_part = line[:numbers[-2].start()].strip()

#     # Mã số thường nằm cuối phần tên chỉ tiêu
#     code_match = re.search(r"\b(\d{2,4})\b\s*$", item_part)

#     item_code = None

#     if code_match:
#         item_code = code_match.group(1)
#         item_name = item_part[:code_match.start()].strip()
#     else:
#         item_name = item_part

#     if not item_name:
#         return None

#     return {
#         "item_code": item_code,
#         "item_name_raw": item_name,
#         "current_value_raw": current_value_raw,
#         "previous_value_raw": previous_value_raw,
#     }


# def parse_html_table_line(line: str) -> list[dict]:
#     """Parse financial rows from a single embedded HTML table."""
#     parser = _HTMLTableParser()
#     parser.feed(line)
#     parsed_rows = []

#     for cells in parser.rows:
#         if len(cells) < 4:
#             continue

#         item_code = cells[0].strip()
#         if not re.fullmatch(r"\d{2,4}[a-z]?", item_code, re.IGNORECASE):
#             continue

#         item_name = cells[1].strip()
#         if not item_name:
#             continue

#         parsed_rows.append({
#             "item_code": item_code,
#             "item_name_raw": item_name,
#             "current_value_raw": cells[-2],
#             "previous_value_raw": cells[-1],
#         })

#     return parsed_rows


# def parse_table_lines(lines: list[str]) -> pd.DataFrame:
#     rows = []
#     html_lines = [line for line in lines if "<table" in line.lower()]

#     if html_lines:
#         logger.debug("Parsing table lines using HTML parser (lines_count=%d)", len(lines))
#         for line in html_lines:
#             rows.extend(parse_html_table_line(line))
#         df = pd.DataFrame(rows)
#         logger.info("Parsed %d rows from HTML table lines", len(df))
#         return df

#     logger.debug("Parsing table lines using text parser (lines_count=%d)", len(lines))
#     pending_item_name = ""

#     for line in lines:
#         parsed = parse_table_line(line)

#         if parsed is None:
#             # Có thể đây là dòng tên chỉ tiêu bị xuống dòng
#             normalized_line = re.sub(r"\s+", " ", line).strip()

#             if normalized_line:
#                 pending_item_name = (
#                     f"{pending_item_name} {normalized_line}"
#                 ).strip()

#             continue

#         if pending_item_name:
#             parsed["item_name_raw"] = (
#                 f"{pending_item_name} {parsed['item_name_raw']}"
#             ).strip()
#             pending_item_name = ""

#         rows.append(parsed)

#     df = pd.DataFrame(rows)
#     logger.info("Parsed %d rows from text table lines", len(df))
#     return df

# def is_header_line(line: str) -> bool:
#     normalized = normalize_text(line)

#     keywords = {
#         "ma so",
#         "thuyet minh",
#         "so cuoi nam",
#         "so dau nam",
#         "nam nay",
#         "nam truoc",
#         "don vi tinh",
#     }

#     return any(keyword in normalized for keyword in keywords)


# def is_footer_line(line: str) -> bool:
#     normalized = normalize_text(line)

#     keywords = {
#         "nguoi lap bieu",
#         "ke toan truong",
#         "tong giam doc",
#         "giam doc",
#     }

#     return any(keyword in normalized for keyword in keywords)


# def validate_table(
#     df: pd.DataFrame,
#     minimum_table_rows: int = 3,
#     maximum_null_ratio: float = 0.7,
# ) -> list[str]:
#     errors = []

#     missing = REQUIRED_COLUMNS - set(df.columns)

#     if missing:
#         errors.append(f"Thiếu cột: {sorted(missing)}")

#     if len(df) < minimum_table_rows:
#         errors.append("Bảng có quá ít dòng")

#     if "value" in df.columns:
#         null_ratio = df["value"].isna().mean()

#         if null_ratio > maximum_null_ratio:
#             errors.append(
#                 f"Tỷ lệ value null quá cao: {null_ratio:.2%}"
#             )

#     if "item_name_raw" in df.columns:
#         duplicate_columns = ["item_name_raw"]
#         if "period" in df.columns:
#             duplicate_columns.append("period")
#         duplicate_ratio = df.duplicated(subset=duplicate_columns).mean()

#         if duplicate_ratio > 0.5:
#             errors.append(
#                 f"Tỷ lệ tên chỉ tiêu trùng quá cao: {duplicate_ratio:.2%}"
#             )

#     if errors:
#         logger.warning("Table validation found errors (shape=%s): %s", df.shape, errors)
#     else:
#         logger.debug("Table validation passed for dataframe shape=%s", df.shape)

#     return errors


# def save_parsed_table(
#     df: pd.DataFrame,
#     output_dir: str | Path,
#     ticker: str,
#     year: int,
#     table_type: str,
#     table_id: str | int | None = None,
# ):
#     output_dir = Path(output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)

#     suffix = f"_{table_id}" if table_id is not None else ""
#     output_path = output_dir / f"{ticker}_{year}_{table_type}{suffix}.parquet"

#     df_to_save = df.copy()

#     if "value" in df_to_save.columns:
#         df_to_save["value"] = pd.to_numeric(
#             df_to_save["value"],
#             errors="coerce"
#         )

#     for col in df_to_save.columns:
#         if pd.api.types.is_integer_dtype(df_to_save[col]):
#             if (df_to_save[col] > MAX_INT64).any() or (df_to_save[col] < MIN_INT64).any():
#                 df_to_save[col] = df_to_save[col].astype("float64")

#         elif df_to_save[col].dtype == "object":
#             df_to_save[col] = df_to_save[col].astype(str)

#     logger.info("Saved parsed table to %s (ticker=%s, year=%d, type=%s)", output_path, ticker, year, table_type)

#     return output_path



# def relative_difference(a: float, b: float) -> float:
#     denomitor = max(abs(a), abs(b), 1)
#     return abs(a - b) / denomitor

# def create_dictionary_stats() -> dict:
#     return {
#         "indicators": defaultdict(Counter),
#         "table_name": defaultdict(Counter),
#         "units": Counter(),
#         "tickers": Counter(),
#     }

# def collect_dictionary_features(
#     df: pd.DataFrame,
#     stats: dict,
# ) -> None:

#     for row in df.to_dict(orient="records"):
#         table_type = row.get("table_type")
#         item_code = row.get("item_code")
#         item_name = row.get("item_name_raw")
#         unit = row.get("unit")
#         ticker = row.get("ticker")
#         table_name = row.get("table_name")

#         if table_type and item_code and item_name:
#             key = (table_type, str(item_code))

#             stats["indicators"][key][item_name] += 1

#         if table_type and table_name:
#             stats["table_names"][table_type][table_name] += 1

#         if unit:
#             stats["units"][unit] += 1

#         if ticker:
#             stats["tickers"][ticker] += 1


import logging
from pathlib import Path
from typing import Any
from html.parser import HTMLParser
import re
import unicodedata
import json
import pandas as pd
from .schemas import DetectedTable
from decimal import Decimal, InvalidOperation
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)

MAX_INT64 = 9_223_372_036_854_775_807
MIN_INT64 = -9_223_372_036_854_775_808
NUMBER_TOKEN = r"(?:\(?-?[\d.,]+\)?|[-—–])"

COMPANY_STOPWORDS = [
    "cong ty",
    "cong ty co phan",
    "cong ty trach nhiem huu han",
    "tong cong ty",
    "tap doan",
    "ctcp",
    "tnhh",
]

TABLE_PATTERNS = {
    "balance_sheet": [
        "bang can doi ke toan",
        "bao cao tinh hinh tai chinh",
    ],
    "income_statement": [
        "bao cao ket qua hoat dong kinh doanh",
        "bao cao ket qua kinh doanh",
    ],
    "cash_flow": [
        "bao cao luu chuyen tien te",
        "luu chuyen tien te",
    ],
    "equity_statement": [
        "bao cao thay doi von chu so huu",
        "bien dong von chu so huu",
    ],
}

NULL_VALUES = {
    "",
    "-",
    "--",
    "—",
    "–",
    "n/a",
    "na",
    "null",
    "none",
}

UNIT_MULTIPLIERS = {
    "vnd": 1,
    "dong": 1,
    "nghin dong": 1_000,
    "ngan dong": 1_000,
    "trieu dong": 1_000_000,
    "ty dong": 1_000_000_000,
}

ITEM_ALIASES = {
    "doanh_thu_thuan": [
        "doanh thu thuan",
        "doanh thu thuan ve ban hang va cung cap dich vu",
    ],
    "loi_nhuan_sau_thue": [
        "loi nhuan sau thue",
        "loi nhuan sau thue thu nhap doanh nghiep",
        "lnst",
    ],
    "tong_tai_san": [
        "tong cong tai san",
        "tong tai san",
    ],
    "von_chu_so_huu": [
        "von chu so huu",
        "tong cong von chu so huu",
    ],
}

DATE_PATTERNS = [
    r"\b\d{1,2}/\d{1,2}/\d{4}\b",
    r"\b\d{1,2}-\d{1,2}-\d{4}\b",
    r"\bnăm\s+\d{4}\b",
    r"\b\d{4}\b",
]

REQUIRED_COLUMNS = {
    "item_name_raw",
    "item_name_normalized",
    "value",
    "period",
    "ticker",
    "year",
    "table_type",
    "section",
}

TABLE_TYPE_TO_SECTION = {
    "balance_sheet": "BS",
    "income_statement": "IS",
    "cash_flow": "CF",
    "equity_statement": "EQ",
}


class _HTMLTableParser(HTMLParser):
    """Collect rows and cells from an HTML table embedded in extracted text."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell_parts: list[str] | None = None

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == "tr":
            self._row = []
        elif tag in {"td", "th"} and self._row is not None:
            self._cell_parts = []

    def handle_data(self, data: str) -> None:
        if self._cell_parts is not None:
            self._cell_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self._row is not None and self._cell_parts is not None:
            cell = re.sub(r"\s+", " ", "".join(self._cell_parts)).strip()
            self._row.append(cell)
            self._cell_parts = None
        elif tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None


def scan_financial_files(root_path: str | Path) -> list[Path]:
    """
    Tìm tất cả các file txt
    
    Parameters: 
        root_path: Đường dẫn thư mục data/financial_statements
    
    Return:
        list[Path]: Danh sách file
    
    """

    if isinstance(root_path, str):
        root_path = Path(root_path)

    if not root_path.exists():
        logger.error("Thư mục không tồn tại: %s", root_path)
        raise FileNotFoundError(f"Không tìm thấy thư mục: {root_path}")

    files = sorted(root_path.rglob("*.txt"))
    logger.info("Scanned financial files in root_path=%s: found %d files", root_path, len(files))
    return files


def extract_meatdata(path: Path) -> dict[str, Any]:
    """
    Tách sticker, year, company code từ đường dẫn/ tên file
    
    Parameters:
        path: Đường dẫn file
        
    Return:
        dict: Từ điển bao gồm {
            sticker,
            year,
            source_file,
            source_path
            }
        
    """

    parts = path.parts

    sticker = None
    year = None

    for part in parts:
        cleaned = part.strip().upper()
        if cleaned.isdigit() and len(cleaned) == 4:
            year_value = int(cleaned)

            if 1990 <= year_value <= 2100:
                year = year_value


    if year is not None:
        year_index = parts.index(str(year))

        if year_index > 0:
            sticker = parts[year_index - 1].upper()

    if sticker is None:
        logger.error("Không xác định được sticker từ path: %s", path)
        raise ValueError(f"Không xác định được sticker từ path: {path}")

    if year is None:
        logger.error("Không xác định được year từ path: %s", path)
        raise ValueError(f"Không xác định được year từ path: {path}")

    logger.debug("Extracted metadata from path=%s: sticker=%s, year=%s", path, sticker, year)
    return {
        "sticker": sticker,
        "year" : year,
        "source_file" : path.name,
        "source_path" : str(path)
    }

def remove_vietnamese_accents(text: str) -> str:
    # Tách các ký tự có 
    text = unicodedata.normalize("NFD", text)

    # Xóa dấu và ghép các ký tự không dấu
    text = "".join(
        char for char in text if unicodedata.category(char) != "Mn"
    )

    # Gộp về định dạng chuẩn 
    # text = unicodedata.normalize("NFC", text)

    return text.replace("đ", "d").replace("Đ", "D")



def normalize_text(text: str) -> str:
    # Chuẩn hóa văn bản 

    text = remove_vietnamese_accents(text.lower())
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def _build_short_company_name(company_name: str) -> str:
    normalized_company_name = normalize_text(company_name)
    short_name = normalized_company_name

    for stopword in sorted(COMPANY_STOPWORDS, key=len, reverse=True):
        short_name = re.sub(
            rf"\b{re.escape(stopword)}\b",
            " ",
            short_name,
        )

    return re.sub(r"\s+", " ", short_name).strip()


def scan_tickers_from_parquet(parquet_dir: str | Path) -> set[str]:
    """Return tickers that are actually present in parsed parquet files."""
    parquet_dir = Path(parquet_dir)
    if not parquet_dir.exists():
        logger.warning("Parquet directory does not exist: %s", parquet_dir)
        return set()

    tickers: set[str] = set()
    parquet_files = sorted(parquet_dir.rglob("*.parquet"))

    for parquet_path in parquet_files:
        try:
            ticker_df = pd.read_parquet(parquet_path, columns=["ticker"])
        except Exception as error:
            logger.warning(
                "Cannot read ticker column from parquet %s: %s",
                parquet_path,
                error,
            )
            continue

        for value in ticker_df["ticker"].dropna().unique():
            ticker = str(value).strip().upper()
            if ticker:
                tickers.add(ticker)

    logger.info(
        "Found %d canonical tickers in %d parquet files",
        len(tickers),
        len(parquet_files),
    )
    return tickers


def _extract_company_ticker_pairs_from_questions(
    questions_path: str | Path,
) -> list[tuple[str, str]]:
    """
    Extract only explicit company-name/ticker pairs of the form:
        Company Name (TICKER)

    This function does not infer or create new tickers.
    """
    questions_path = Path(questions_path)
    if not questions_path.exists():
        logger.warning("Questions file does not exist: %s", questions_path)
        return []

    pairs: list[tuple[str, str]] = []

    # Capture a reasonably-sized text span immediately before an uppercase ticker.
    # The leading span is later trimmed to the nearest company-like prefix/separator.
    pair_pattern = re.compile(
        r"([^\n()]{2,180}?)\s*\(([A-Z][A-Z0-9]{1,9})\)"
    )

    company_prefix_pattern = re.compile(
        r"(?i)("
        r"công\s+ty\s+cổ\s+phần|"
        r"công\s+ty\s+tnhh|"
        r"công\s+ty|"
        r"ctcp|"
        r"tổng\s+ctcp|"
        r"tổng\s+công\s+ty|"
        r"tập\s+đoàn|"
        r"ngân\s+hàng(?:\s+tmcp)?"
        r")"
    )

    with questions_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(
                    "Skipping invalid JSON at %s:%d",
                    questions_path,
                    line_number,
                )
                continue

            question = str(record.get("question", "")).strip()
            if not question:
                continue

            for raw_name, ticker in pair_pattern.findall(question):
                company_name = re.split(r"[;,:]", raw_name)[-1].strip()

                prefix_matches = list(company_prefix_pattern.finditer(company_name))
                if prefix_matches:
                    company_name = company_name[prefix_matches[-1].start():].strip()

                # Remove common sentence glue accidentally captured before the name.
                company_name = re.sub(
                    r"(?i)^(?:và|hoặc|gồm|bao\s+gồm|các|trong\s+số|xét|tại)\s+",
                    "",
                    company_name,
                ).strip(" -–—,.;:")

                ticker = ticker.strip().upper()
                if company_name and ticker:
                    pairs.append((company_name, ticker))

    return pairs


def build_entity_dictionary(
    csv_path: str | Path,
    output_path: str | Path,
    questions_path: str | Path | None = None,
    parquet_dir: str | Path | None = None,
) -> dict:
    """
    Build entity_dictionary.json with strict provenance rules:

    1. code_stock.csv defines canonical tickers and company names.
    2. questions.jsonl may only add aliases from explicit "Company Name (TICKER)" pairs.
    3. parquet files confirm which tickers actually exist in the parsed dataset.
    4. No ticker found only in questions is ever created.
    """
    logger.info("Building entity dictionary from csv_path=%s", csv_path)
    df = pd.read_csv(csv_path)

    normalized_columns = {
        normalize_text(str(column)): column
        for column in df.columns
    }
    ticker_column = next(
        (
            normalized_columns[name]
            for name in ("ticker", "sticker", "ma ck")
            if name in normalized_columns
        ),
        None,
    )
    company_column = next(
        (
            normalized_columns[name]
            for name in ("company_name", "ten cong ty")
            if name in normalized_columns
        ),
        None,
    )

    missing = []
    if ticker_column is None:
        missing.append("ticker/Mã CK")
    if company_column is None:
        missing.append("company_name/Tên công ty")
    if missing:
        logger.error("Thiếu cột trong entity dictionary csv (%s): %s", csv_path, missing)
        raise ValueError(f"Thiếu cột trong code_stock.csv: {missing}")

    canonical_rows: dict[str, dict] = {}
    for _, row in df.iterrows():
        if pd.isna(row[ticker_column]) or pd.isna(row[company_column]):
            continue

        ticker = str(row[ticker_column]).strip().upper()
        company_name = str(row[company_column]).strip()
        if not ticker or not company_name:
            continue

        canonical_rows[ticker] = {
            "company_name": company_name,
            "row": row,
        }

    canonical_tickers = set(canonical_rows)

    if parquet_dir is not None:
        dataset_tickers = scan_tickers_from_parquet(parquet_dir)
        allowed_tickers = canonical_tickers & dataset_tickers
    else:
        dataset_tickers = canonical_tickers.copy()
        allowed_tickers = canonical_tickers.copy()
        logger.warning(
            "parquet_dir was not provided; entity dictionary cannot verify dataset presence"
        )

    question_aliases: dict[str, set[str]] = defaultdict(set)
    rejected_question_tickers: set[str] = set()

    if questions_path is not None:
        for company_alias, ticker in _extract_company_ticker_pairs_from_questions(
            questions_path
        ):
            # Strict rule: ticker must be canonical AND actually present in parquet.
            if ticker not in canonical_tickers or ticker not in dataset_tickers:
                rejected_question_tickers.add(ticker)
                continue

            question_aliases[ticker].add(company_alias)

    alias_column = normalized_columns.get("alias")
    entities = {}

    for ticker in sorted(allowed_tickers):
        row_info = canonical_rows[ticker]
        company_name = row_info["company_name"]
        row = row_info["row"]

        normalized_company_name = normalize_text(company_name)
        short_name = _build_short_company_name(company_name)

        aliases = {
            ticker,
            company_name,
            normalized_company_name,
            short_name,
        }

        if alias_column and pd.notna(row.get(alias_column)):
            aliases.update(
                item.strip()
                for item in str(row[alias_column]).split("|")
                if item.strip()
            )

        for question_alias in question_aliases.get(ticker, set()):
            aliases.add(question_alias)
            normalized_alias = normalize_text(question_alias)
            if normalized_alias:
                aliases.add(normalized_alias)

            short_alias = _build_short_company_name(question_alias)
            if short_alias:
                aliases.add(short_alias)

        entities[ticker] = {
            "full_name": company_name,
            "short_name": short_name or normalized_company_name,
            "aliases": sorted(alias for alias in aliases if alias),
        }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(entities, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logger.info(
        "Built entity dictionary: canonical_csv=%d, dataset_parquet=%d, output_entities=%d, question_alias_tickers=%d, rejected_question_tickers=%d, output_path=%s",
        len(canonical_tickers),
        len(dataset_tickers),
        len(entities),
        len(question_aliases),
        len(rejected_question_tickers),
        output_path,
    )

    return entities


def identify_table_type(line: str) -> str | None:
    if "<table" in line.lower():
        return None

    normalized = normalize_text(line)

    if len(normalized) > 160:
        return None

    for table_type, patterns in TABLE_PATTERNS.items():
        if any(normalized.startswith(pattern) for pattern in patterns):
            return table_type

    return None

# Nhận diện bảng 
def detect_tables(lines: list[str]) -> list[DetectedTable]:
    starts : list[tuple[int, str, str]] = []

    for index, line in enumerate(lines):
        type_table = identify_table_type(line=line)
        has_nearby_table = any(
            "<table" in candidate.lower()
            for candidate in lines[index + 1:index + 16]
        )
        if type_table and has_nearby_table:
            starts.append((index, type_table, line.strip()))

    tables = []
    for position, (start, type_table, table_name) in enumerate(starts):
        if position + 1 < len(starts):
            end = starts[position + 1][0]

        else:
            end = len(lines)

        try: 
            if start < 0 or end < 0 or start > end:
                raise ValueError(f"Dòng bắt đầu {start} hoặc dòng kết thức {end} không hợp lệ.")
            
            tables.append(DetectedTable(
                type_table=type_table,
                table_name=table_name,
                start_line=start,
                end_line=end,
                lines=lines[start:end]
            ))
        except ValueError as e:
            logger.warning("Dữ liệu không hợp lệ khi detect tables: %s", e)
            print(f"Dữ liệu không hợp lệ: {e}.")

    table_types = [t.type_table for t in tables]
    logger.info("Detected %d tables from %d lines: table_types=%s", len(tables), len(lines), table_types)

    return tables


def parse_number(value: str | int | float | None) -> Decimal:

    if value is None: 
        return None

    if isinstance(value, (int, float)):
        return Decimal(str(value))

    text = str(value).strip().lower()

    if text in NULL_VALUES:
        return None

    pattern = r"\(?-?\d{1,3}(?:[.,]\d{3})+\)?"
    matches = re.findall(pattern, text)

    if matches:
        text = matches[0]

    else:
        match = re.search(r"\(?-?\d[\d.,]*\)?", text)

        if match:
            text = match.group(0)

    negative = False

    if "(" in text or ")" in text or text.startswith("-"):
        negative = True

    text = re.sub(r"[()\-]", "", text).strip()
    # Loại bỏ khoảng trắng và ký hiệu tiền tệ
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"(vnd|vnđ|đồng|dong)$", "", text)


    # 1.234.567 hoặc 1,234,567
    if re.fullmatch(r"\d{1,3}([.,]\d{3})+", text):
        text = re.sub(r"[.,]", "", text)

    # 1234,56: dấu phẩy là dấu thập phân
    elif re.fullmatch(r"\d+,\d{1,2}", text):
        text = text.replace(",", ".")

    # Loại bỏ ký tự còn lại không hợp lệ
    text = re.sub(r"[^0-9.]", "", text)

    if not text or not re.search(r"\d", text):
        return None

    try:
        number = Decimal(text)
    except InvalidOperation as error:
        raise ValueError(f"Không parse được số: {value}") from error

    return -number if negative else number


def detect_unit(text: str) -> tuple[str, int]:
    """
    Nhận diện đơn vị
    """

    normalized = normalize_text(text)

    ordered_units = [
        ("ty dong", 1_000_000_000),
        ("trieu dong", 1_000_000),
        ("nghin dong", 1_000),
        ("ngan dong", 1_000),
        ("vnd", 1),
        ("dong", 1),
    ]

    for unit_name, multiplier in ordered_units:
        if unit_name in normalized:
            return unit_name, multiplier

    return "unknown", 1

def normalize_to_vnd(value: Decimal | None, multiplier: int,) -> int | None:

    if value is None:
        return None

    normalized = value * multiplier
    return int(normalized)

def normalize_item_name(item_name: str) -> str:
    """
    Chuẩn hóa tên chỉ tiêu
    
    """

    normalized = normalize_text(item_name)

    for canonical_name, aliases in ITEM_ALIASES.items():
        if normalized in aliases:
            return canonical_name

    return normalized.replace(" ", "_")

def parse_table_line(line: str) -> dict | None:
    line = re.sub(r"\s+", " ", line).strip()

    numbers = list(re.finditer(NUMBER_TOKEN, line))

    if len(numbers) < 2:
        return None

    # Giả định hai số cuối là hai kỳ báo cáo
    current_value_raw = numbers[-2].group()
    previous_value_raw = numbers[-1].group()

    item_part = line[:numbers[-2].start()].strip()

    # Mã số thường nằm cuối phần tên chỉ tiêu
    code_match = re.search(r"\b(\d{2,4})\b\s*$", item_part)

    item_code = None

    if code_match:
        item_code = code_match.group(1)
        item_name = item_part[:code_match.start()].strip()
    else:
        item_name = item_part

    if not item_name:
        return None

    return {
        "item_code": item_code,
        "item_name_raw": item_name,
        "current_value_raw": current_value_raw,
        "previous_value_raw": previous_value_raw,
    }


def parse_html_table_line(line: str) -> list[dict]:
    """Parse financial rows from a single embedded HTML table."""
    parser = _HTMLTableParser()
    parser.feed(line)
    parsed_rows = []

    for cells in parser.rows:
        if len(cells) < 4:
            continue

        item_code = cells[0].strip()
        if not re.fullmatch(r"\d{2,4}[a-z]?", item_code, re.IGNORECASE):
            continue

        item_name = cells[1].strip()
        if not item_name:
            continue

        parsed_rows.append({
            "item_code": item_code,
            "item_name_raw": item_name,
            "current_value_raw": cells[-2],
            "previous_value_raw": cells[-1],
        })

    return parsed_rows


def parse_table_lines(lines: list[str]) -> pd.DataFrame:
    rows = []
    html_lines = [line for line in lines if "<table" in line.lower()]

    if html_lines:
        logger.debug("Parsing table lines using HTML parser (lines_count=%d)", len(lines))
        for line in html_lines:
            rows.extend(parse_html_table_line(line))
        df = pd.DataFrame(rows)
        logger.info("Parsed %d rows from HTML table lines", len(df))
        return df

    logger.debug("Parsing table lines using text parser (lines_count=%d)", len(lines))
    pending_item_name = ""

    for line in lines:
        parsed = parse_table_line(line)

        if parsed is None:
            # Có thể đây là dòng tên chỉ tiêu bị xuống dòng
            normalized_line = re.sub(r"\s+", " ", line).strip()

            if normalized_line:
                pending_item_name = (
                    f"{pending_item_name} {normalized_line}"
                ).strip()

            continue

        if pending_item_name:
            parsed["item_name_raw"] = (
                f"{pending_item_name} {parsed['item_name_raw']}"
            ).strip()
            pending_item_name = ""

        rows.append(parsed)

    df = pd.DataFrame(rows)
    logger.info("Parsed %d rows from text table lines", len(df))
    return df

def is_header_line(line: str) -> bool:
    normalized = normalize_text(line)

    keywords = {
        "ma so",
        "thuyet minh",
        "so cuoi nam",
        "so dau nam",
        "nam nay",
        "nam truoc",
        "don vi tinh",
    }

    return any(keyword in normalized for keyword in keywords)


def is_footer_line(line: str) -> bool:
    normalized = normalize_text(line)

    keywords = {
        "nguoi lap bieu",
        "ke toan truong",
        "tong giam doc",
        "giam doc",
    }

    return any(keyword in normalized for keyword in keywords)


def validate_table(
    df: pd.DataFrame,
    minimum_table_rows: int = 3,
    maximum_null_ratio: float = 0.7,
) -> list[str]:
    errors = []

    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:
        errors.append(f"Thiếu cột: {sorted(missing)}")

    if len(df) < minimum_table_rows:
        errors.append("Bảng có quá ít dòng")

    if "value" in df.columns:
        null_ratio = df["value"].isna().mean()

        if null_ratio > maximum_null_ratio:
            errors.append(
                f"Tỷ lệ value null quá cao: {null_ratio:.2%}"
            )

    if "item_name_raw" in df.columns:
        duplicate_columns = ["item_name_raw"]
        if "period" in df.columns:
            duplicate_columns.append("period")
        duplicate_ratio = df.duplicated(subset=duplicate_columns).mean()

        if duplicate_ratio > 0.5:
            errors.append(
                f"Tỷ lệ tên chỉ tiêu trùng quá cao: {duplicate_ratio:.2%}"
            )

    if errors:
        logger.warning("Table validation found errors (shape=%s): %s", df.shape, errors)
    else:
        logger.debug("Table validation passed for dataframe shape=%s", df.shape)

    return errors


def save_parsed_table(
    df: pd.DataFrame,
    output_dir: str | Path,
    ticker: str,
    year: int,
    table_type: str,
    table_id: str | int | None = None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_{table_id}" if table_id is not None else ""
    output_path = output_dir / f"{ticker}_{year}_{table_type}{suffix}.parquet"

    df_to_save = df.copy()

    if "value" in df_to_save.columns:
        df_to_save["value"] = pd.to_numeric(
            df_to_save["value"],
            errors="coerce"
        )

    for col in df_to_save.columns:
        if pd.api.types.is_integer_dtype(df_to_save[col]):
            if (df_to_save[col] > MAX_INT64).any() or (df_to_save[col] < MIN_INT64).any():
                df_to_save[col] = df_to_save[col].astype("float64")

        elif df_to_save[col].dtype == "object":
            df_to_save[col] = df_to_save[col].astype(str)

    df_to_save.to_parquet(output_path, index=False)

    logger.info("Saved parsed table to %s (ticker=%s, year=%d, type=%s)", output_path, ticker, year, table_type)

    return output_path



def relative_difference(a: float, b: float) -> float:
    denomitor = max(abs(a), abs(b), 1)
    return abs(a - b) / denomitor

def create_dictionary_stats() -> dict:
    return {
        "indicators": defaultdict(Counter),
        "table_names": defaultdict(Counter),
        "units": Counter(),
        "tickers": Counter(),
    }


def collect_dictionary_features(
    df: pd.DataFrame,
    stats: dict,
) -> None:
    if df.empty:
        return

    indicator_columns = [
        "table_type",
        "item_code",
        "item_name_raw",
        "source_file",
    ]
    indicator_columns = [column for column in indicator_columns if column in df.columns]
    indicator_df = df.drop_duplicates(subset=indicator_columns) if indicator_columns else df

    for row in indicator_df.to_dict(orient="records"):
        table_type = row.get("table_type")
        item_code = row.get("item_code")
        item_name = row.get("item_name_raw")

        if (
            table_type
            and item_code is not None
            and str(item_code).strip()
            and item_name
        ):
            key = (str(table_type), str(item_code).strip())
            stats["indicators"][key][str(item_name).strip()] += 1

    table_columns = [
        "table_type",
        "table_name",
        "source_file",
    ]
    table_columns = [column for column in table_columns if column in df.columns]
    table_df = df.drop_duplicates(subset=table_columns) if table_columns else df

    for row in table_df.to_dict(orient="records"):
        table_type = row.get("table_type")
        table_name = row.get("table_name")
        unit = row.get("unit")
        ticker = row.get("ticker")

        if table_type and table_name:
            stats["table_names"][str(table_type)][str(table_name).strip()] += 1

        if unit:
            stats["units"][str(unit)] += 1

        if ticker:
            stats["tickers"][str(ticker).upper()] += 1


def build_indicator_aliases_from_stats(
    stats: dict,
    min_count: int = 5,
) -> dict:
    alias_targets: dict[str, Counter] = defaultdict(Counter)

    for (table_type, item_code), names in stats["indicators"].items():
        section = TABLE_TYPE_TO_SECTION.get(table_type)
        if section is None:
            continue

        schema_key = f"{section}.{item_code}"

        for raw_name, count in names.items():
            if count < min_count:
                continue

            alias = normalize_text(raw_name)
            if alias:
                alias_targets[alias][schema_key] += count

    aliases = {}
    for alias, targets in alias_targets.items():
        if len(targets) != 1:
            logger.warning(
                "Skipping conflicting indicator alias '%s': %s",
                alias,
                dict(targets),
            )
            continue

        aliases[alias] = next(iter(targets))

    return dict(sorted(aliases.items()))


def build_schema_mapping_from_stats(
    stats: dict,
    min_count: int = 5,
) -> dict:
    schema_mapping: dict[str, dict] = {}

    for (table_type, item_code), names in sorted(stats["indicators"].items()):
        section = TABLE_TYPE_TO_SECTION.get(table_type)
        if section is None:
            continue

        valid_names = {
            raw_name: count
            for raw_name, count in names.items()
            if count >= min_count
        }
        if not valid_names:
            continue

        canonical_name = max(
            valid_names.items(),
            key=lambda item: (item[1], -len(item[0])),
        )[0]

        schema_mapping.setdefault(section, {})
        schema_mapping[section][str(item_code)] = {
            "name": canonical_name
        }

    return schema_mapping


def build_dictionary_report(stats: dict) -> dict:
    indicators = {}
    for (table_type, item_code), names in stats["indicators"].items():
        section = TABLE_TYPE_TO_SECTION.get(table_type, table_type)
        schema_key = f"{section}.{item_code}"
        indicators[schema_key] = {
            "table_type": table_type,
            "item_code": item_code,
            "aliases": [
                {"name": name, "count": count}
                for name, count in names.most_common()
            ],
        }

    return {
        "indicators": dict(sorted(indicators.items())),
        "table_names": {
            table_type: [
                {"name": name, "count": count}
                for name, count in names.most_common()
            ]
            for table_type, names in sorted(stats["table_names"].items())
        },
        "units": dict(stats["units"].most_common()),
        "tickers": dict(stats["tickers"].most_common()),
    }


def save_json(
    data: dict,
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Saved JSON to %s", output_path)
    return output_path


    

    









            
