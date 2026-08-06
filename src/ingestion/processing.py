from pathlib import Path
from typing import Any
import re
import unicodedata
import json
import pandas as pd
from .schemas import DetectedTable
from decimal import Decimal, InvalidOperation
import pickle
from rank_bm25 import BM25Okapi


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
}


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
        raise FileNotFoundError(f"Không tìm thấy thư mục: {root_path}")

    return sorted(root_path.rglob("*.txt"))


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
        raise ValueError(f"Không xác định được sticker từ path: {path}")

    if year is None:
        raise ValueError(f"Không xác định được year từ path: {path}")

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

def build_entity_dictionary(
        csv_path: str | Path,
        output_path: str | Path
) -> dict:
    df = pd.read_csv(csv_path)

    required_columns = ["sticker", "company_name"]

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Thiếu cột trong code_stock.csv: {missing}")

    entities = {}

    for _, row in df.iterrows():
        sticker = str(row["sticker"]).strip().upper()
        company_name = str(row["company_name"]).strip()

        aliases = [sticker, company_name]

        if "alias" in df.columns and pd.notna(row.get("alias")):
            aliases.extend(
                item.strip()
                for item in str(row["alias"]).split("|")
                if item.strip()
            )

        aliases = sorted({
            normalize_text(alias)
            for alias in aliases
            if alias
        })

        entities["sticker"] = {
            "sticker": sticker,
            "company_name": company_name,
            "name_normalize": normalize_text(company_name),
            "aliases": aliases
        }


    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(entities, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return entities


def identify_table_type(line: str) -> str | None:
    normalized = normalize_text(line)

    for table_type, patterns in TABLE_PATTERNS.items():
        if any(pattern in normalized for pattern in patterns):
            return table_type

    return None

# Nhận diện bảng 
def detect_tables(lines: list[str]) -> list[DetectedTable]:
    starts : list[tuple[int, str, str]] = []

    for index, line in enumerate(lines):
        type_table = identify_table_type(line=line)
        if type_table:
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
            print(f"Dữ liệu không hợp lệ: {e}.")

    return tables


def parse_number(value: str | int | float | None) -> Decimal:

    if value is None: 
        return None

    if isinstance(value, (int, float)):
        return Decimal(str(value))

    text = str(value).strip().lower()
    negative = False

    if text in NULL_VALUES:
        return None

    if text.startswith("(") and text.endswith(")"):
        negative = True
        text = text[1:-1].strip()

    if text.startswith("-"):
        negative = True
        text = text[1:].strip()

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


def parse_table_lines(lines: list[str]) -> pd.DataFrame:
    rows = []

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

    return pd.DataFrame(rows)

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
        duplicate_ratio = df["item_name_raw"].duplicated().mean()

        if duplicate_ratio > 0.5:
            errors.append(
                f"Tỷ lệ tên chỉ tiêu trùng quá cao: {duplicate_ratio:.2%}"
            )

    return errors


def save_parsed_table(
    df,
    output_dir: str | Path,
    ticker: str,
    year: int,
    table_type: str,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{ticker}_{year}_{table_type}.parquet"
    df.to_parquet(output_path, index=False)

    return output_path



def relative_difference(a: float, b: float) -> float:
    denomitor = max(abs(a), abs(b), 1)
    return abs(a - b) / denomitor





    

    









            
