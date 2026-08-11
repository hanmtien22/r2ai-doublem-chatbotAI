import logging
import pandas as pd
from .schemas import DetectedTable
import json
from pathlib import Path
from .processing import (
    TABLE_TYPE_TO_SECTION,
    detect_unit,
    normalize_item_name,
    normalize_to_vnd,
    parse_number
)
# Backward-compatible imports retained for callers of the original ingestion API.
from ..indexing.bm25 import bm25_search, build_bm25_index, load_documents

logger = logging.getLogger(__name__)



def json_safe_record(record: dict) -> dict:
    return {
        key: None if pd.isna(value) else value
        for key, value in record.items()
    }


def enrich_and_normalize_table(
    dataframe: pd.DataFrame,
    document_text: str,
    metadata: dict,
    detected_table: DetectedTable,
) -> pd.DataFrame:
    unit, multiplier = detect_unit(document_text)
    ticker = metadata["sticker"]
    report_year = metadata["year"]
    report_type = metadata["report_type"]
    records = []

    period_columns = (
        ("current_value_raw", report_year),
        ("previous_value_raw", report_year - 1),
    )

    for row in dataframe.to_dict(orient="records"):
        for value_column, period in period_columns:
            raw_value = row.get(value_column)
            if raw_value is None or pd.isna(raw_value):
                continue
            value = normalize_to_vnd(parse_number(raw_value), multiplier)

            records.append({
                "item_code": row.get("item_code"),
                "item_name_raw": row["item_name_raw"],
                "item_name_normalized": normalize_item_name(
                    row["item_name_raw"]
                ),
                "value_raw": raw_value,
                "value": value,
                "unit": unit,
                "period": period,
                "ticker": ticker,
                "year": report_year,
                "report_type": report_type,
                "table_type": detected_table.type_table,
                "table_name": detected_table.table_name,
                "note_reference": row.get("note_reference"),
            })

    logger.info(
        "Enriched and normalized table for ticker=%s, year=%s: unit=%s, multiplier=%s, input_rows=%d, output_records=%d",
        ticker,
        report_year,
        unit,
        multiplier,
        len(dataframe),
        len(records),
    )

    return pd.DataFrame(records)


def write_jsonl(output_path: str | Path, records: list[dict]) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(
                json.dumps(record, ensure_ascii=False, allow_nan=False)
            )
            output_file.write("\n")

    logger.info("Wrote %d records to %s", len(records), output_path)

def build_retrieval_documents(
        dataframe: pd.DataFrame,
        detected_table: DetectedTable,
) -> list[dict]:
    documents = []

    for index, raw_row in enumerate(dataframe.to_dict(orient="records")):
        row = json_safe_record(raw_row)
        metadata = dict(row)
        metadata["section"] = TABLE_TYPE_TO_SECTION[detected_table.type_table]
        chunk_id = (
            f"{row['ticker']}:{row['year']}:{detected_table.type_table}:"
            f"{detected_table.start_line}:{index}"
        )
        value_text = "Không có dữ liệu" if row['value'] is None else f"{row['value']} VND"

        documents.append({
            "chunk_id": chunk_id,
            "text": (
                f"{row['ticker']} - {row['period']} - "
                f"{detected_table.table_name} - "
                f"{row['item_name_raw']}: {value_text}"
            ),
            "metadata": metadata,
        })

    logger.info(
        "Built %d retrieval documents for table_type=%s",
        len(documents),
        detected_table.type_table,
    )

    return documents
