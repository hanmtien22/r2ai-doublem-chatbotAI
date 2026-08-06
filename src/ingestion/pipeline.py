from pathlib import Path
import json
import traceback

import pandas as pd

from config_loader import load_config, resolve_project_path
from .processing import (
    detect_tables,
    detect_unit,
    extract_meatdata,
    normalize_item_name,
    normalize_to_vnd,
    parse_number,
    parse_table_lines,
    save_parsed_table,
    scan_financial_files,
    validate_table,
)
from .schemas import DetectedTable


def _json_safe_record(record: dict) -> dict:
    return {
        key: None if pd.isna(value) else value
        for key, value in record.items()
    }


def _enrich_and_normalize_table(
    dataframe: pd.DataFrame,
    document_text: str,
    metadata: dict,
    detected_table: DetectedTable,
) -> pd.DataFrame:
    unit, multiplier = detect_unit(document_text)
    ticker = metadata["sticker"]
    report_year = metadata["year"]
    records = []

    period_columns = (
        ("current_value_raw", report_year),
        ("previous_value_raw", report_year - 1),
    )

    for row in dataframe.to_dict(orient="records"):
        for value_column, period in period_columns:
            raw_value = row.get(value_column)
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
                "unit_multiplier": multiplier,
                "period": period,
                "ticker": ticker,
                "year": report_year,
                "table_type": detected_table.type_table,
                "table_name": detected_table.table_name,
                "source_file": metadata["source_file"],
                "source_path": metadata["source_path"],
            })

    return pd.DataFrame(records)


def _build_retrieval_documents(
    dataframe: pd.DataFrame,
    detected_table: DetectedTable,
) -> list[dict]:
    documents = []

    for index, raw_row in enumerate(dataframe.to_dict(orient="records")):
        row = _json_safe_record(raw_row)
        chunk_id = (
            f"{row['ticker']}:{row['year']}:{detected_table.type_table}:"
            f"{detected_table.start_line}:{index}"
        )
        value_text = "không có dữ liệu" if row["value"] is None else f"{row['value']} VND"

        documents.append({
            "chunk_id": chunk_id,
            "text": (
                f"{row['ticker']} - {row['period']} - "
                f"{detected_table.table_name} - "
                f"{row['item_name_raw']}: {value_text}"
            ),
            "metadata": row,
        })

    return documents


def _write_jsonl(output_path: str | Path, records: list[dict]) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(
                json.dumps(record, ensure_ascii=False, allow_nan=False)
            )
            output_file.write("\n")


def run_ingestion_pipeline(config: dict | None = None):
    config = load_config() if config is None else config

    source_dir = resolve_project_path(config["source_dir"])
    output_dir = resolve_project_path(config["output_dir"])
    parser_config = config["parser"]

    output_dir.mkdir(parents=True, exist_ok=True)

    files = scan_financial_files(source_dir)

    all_records = []
    all_documents = []
    errors = []

    for path in files:
        try:
            metadata = extract_meatdata(path)

            text = path.read_text(
                encoding="utf-8",
                errors="replace",
            )

            lines = text.splitlines()
            detected_tables = detect_tables(lines)

            for table in detected_tables:
                dataframe = parse_table_lines(table.lines)

                if dataframe.empty:
                    errors.append({
                        "source_file": str(path),
                        "table_type": table.type_table,
                        "errors": [
                            f"Không parse được bảng {table.table_name}"
                        ],
                    })
                    continue

                dataframe = _enrich_and_normalize_table(
                    dataframe=dataframe,
                    document_text="\n".join(table.lines),
                    metadata=metadata,
                    detected_table=table,
                )

                validation_errors = validate_table(
                    dataframe,
                    minimum_table_rows=parser_config["minimum_table_rows"],
                    maximum_null_ratio=parser_config["maximum_null_ratio"],
                )

                if validation_errors:
                    errors.append({
                        "source_file": str(path),
                        "table_type": table.type_table,
                        "errors": validation_errors,
                    })

                table_path = save_parsed_table(
                    df=dataframe,
                    output_dir=output_dir / "tables",
                    ticker=metadata["sticker"],
                    year=metadata["year"],
                    table_type=table.type_table,
                )

                records = [
                    _json_safe_record(record)
                    for record in dataframe.to_dict(orient="records")
                ]
                all_records.extend(records)

                documents = _build_retrieval_documents(
                    dataframe=dataframe,
                    detected_table=table,
                )
                all_documents.extend(documents)

        except Exception as error:
            errors.append({
                "source_file": str(path),
                "error": str(error),
                "traceback": traceback.format_exc(),
            })

    _write_jsonl(
        output_dir / "records.jsonl",
        all_records,
    )

    _write_jsonl(
        output_dir / "retrieval_documents.jsonl",
        all_documents,
    )

    _write_jsonl(
        output_dir / "ingestion_errors.jsonl",
        errors,
    )

    return {
        "file_count": len(files),
        "record_count": len(all_records),
        "document_count": len(all_documents),
        "error_count": len(errors),
    }
