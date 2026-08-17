"""End-to-end ingestion orchestration for statements and disclosures."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import traceback

from src.config_loader import configure_logging, load_config, resolve_project_path
from ..indexing.indexing import build_search_indexes
from .chunk_builder import (
    build_retrieval_documents, enrich_and_normalize_table, json_safe_record,
    write_jsonl,
)
from .notes import build_notes_chunks, build_notes_retrieval_documents, save_notes_chunks
from .processing import (
    build_dictionary_report, build_entity_dictionary,
    build_indicator_aliases_from_stats, build_schema_mapping_from_stats,
    collect_dictionary_features, create_dictionary_stats, detect_tables,
    extract_metadata, parse_table_lines, save_json, save_parsed_table,
    infer_report_type, scan_financial_files, validate_table,
)

logger = logging.getLogger(__name__)


def run_ingestion_pipeline(config: dict | None = None) -> dict:
    config = load_config() if config is None else config
    if config.get("logging"):
        configure_logging(config)
    source_dir = resolve_project_path(config["source_dir"])
    output_dir = resolve_project_path(config["output_dir"])
    parser_config = config.get("parser", {})
    notes_config = config.get("notes", {})
    output_dir.mkdir(parents=True, exist_ok=True)

    files = scan_financial_files(source_dir)
    all_records: list[dict] = []
    all_documents: list[dict] = []
    errors: list[dict] = []
    dictionary_stats = create_dictionary_stats()
    statement_count = 0
    notes_chunk_count = 0

    for path in files:
        file_record_start = len(all_records)
        file_document_start = len(all_documents)
        file_error_start = len(errors)
        logger.info("Processing file: %s", path)
        try:
            metadata = extract_metadata(path)
            text = path.read_text(encoding="utf-8", errors="replace")
            lines = text.splitlines()
            tables = detect_tables(lines)
            if metadata["report_type"] is None:
                metadata["report_type"] = infer_report_type(lines, tables)
                if metadata["report_type"] is None:
                    metadata["report_type"] = config.get("defaults", {}).get("report_type", "consolidated")
                    logger.warning("Could not detect report_type for %s; using %s", path, metadata["report_type"])
            counts: dict[str, int] = {}
            for table in tables:
                counts[table.type_table] = counts.get(table.type_table, 0) + 1
            sequence: dict[str, int] = {}

            for table in tables:
                raw = parse_table_lines(table.lines)
                if raw.empty:
                    errors.append({
                        "source_file": str(path), "table_type": table.type_table,
                        "table_name": table.table_name,
                        "reason_code": "no_valid_item_code_rows",
                        "errors": [
                            "Không có dòng structured hợp lệ: bảng không có cột "
                            "Mã số/item_code hoặc header OCR không đủ tin cậy"
                        ],
                    })
                    continue
                frame = enrich_and_normalize_table(
                    raw, "\n".join(table.context_lines + table.lines), metadata, table,
                )
                validation_errors = validate_table(
                    frame,
                    minimum_table_rows=int(parser_config.get("minimum_table_rows", 3)),
                    maximum_null_ratio=float(parser_config.get("maximum_null_ratio", 0.7)),
                )
                if validation_errors:
                    errors.append({
                        "source_file": str(path), "table_type": table.type_table,
                        "table_name": table.table_name, "errors": validation_errors,
                    })
                    continue
                sequence[table.type_table] = sequence.get(table.type_table, 0) + 1
                table_id = sequence[table.type_table] if counts[table.type_table] > 1 else None
                save_parsed_table(
                    frame, output_dir / "tables", metadata["sticker"], metadata["year"],
                    table.type_table, metadata["report_type"], table_id,
                )
                statement_count += 1
                collect_dictionary_features(frame, dictionary_stats)
                all_records.extend(json_safe_record(row) for row in frame.to_dict("records"))
                all_documents.extend(build_retrieval_documents(frame, table))

            if notes_config.get("enabled", True):
                notes = build_notes_chunks(
                    lines, tables, metadata,
                    max_chars=int(notes_config.get("max_chars", 3000)),
                    overlap_chars=int(notes_config.get("overlap_chars", 300)),
                    min_chars=int(notes_config.get("min_chars", 100)),
                )
                if not notes.empty:
                    notes_dir = output_dir / notes_config.get("output_dir", "notes")
                    save_notes_chunks(notes, notes_dir, metadata)
                    notes_chunk_count += len(notes)
                    all_documents.extend(build_notes_retrieval_documents(notes))
            logger.info(
                "Processed file: %s | statements=%d | records=%d | documents=%d | errors=%d",
                path,
                len(tables),
                len(all_records) - file_record_start,
                len(all_documents) - file_document_start,
                len(errors) - file_error_start,
            )
        except Exception as error:
            logger.exception("Error processing %s", path)
            errors.append({
                "source_file": str(path), "error": str(error),
                "traceback": traceback.format_exc(),
            })
            logger.info(
                "Processed file with exception: %s | records=%d | documents=%d | errors=%d",
                path,
                len(all_records) - file_record_start,
                len(all_documents) - file_document_start,
                len(errors) - file_error_start,
            )

    dictionaries = config.get("dictionaries", {})
    dictionary_root = (
        resolve_project_path(dictionaries["root_dir"])
        if dictionaries.get("root_dir")
        else output_dir / "dictionaries"
    )
    builder = config.get("dictionary_builder", {})
    if builder.get("enabled", True):
        try:
            min_count = int(builder.get("min_count", 5))
            generated_dir = resolve_project_path(builder["output_dir"]) if builder.get("output_dir") else output_dir
            aliases = build_indicator_aliases_from_stats(dictionary_stats, min_count)
            curated_path = dictionary_root / dictionaries.get("indicator_aliases", "indicator_aliases.json")
            if curated_path.exists():
                curated = json.loads(curated_path.read_text(encoding="utf-8"))
                if isinstance(curated, dict):
                    aliases.update(curated)
            save_json(aliases, generated_dir / builder.get("indicator_aliases_file", "indicator_aliases.json"))
            save_json(build_schema_mapping_from_stats(dictionary_stats, min_count), generated_dir / builder.get("schema_mapping_file", "schema_mapping.json"))
            save_json(build_dictionary_report(dictionary_stats), output_dir / builder.get("report_file", "dictionary_report.json"))
        except Exception as error:
            logger.exception("Dictionary build failed")
            errors.append({"stage": "dictionary_builder", "error": str(error), "traceback": traceback.format_exc()})

    entity = config.get("entity_dictionary", {})
    if entity.get("rebuild", False):
        try:
            if entity.get("output_path"):
                entity_output = resolve_project_path(entity["output_path"])
            else:
                entity_output = dictionary_root / entity.get("output_file", "entity_dictionary.json")
            logger.info("Writing entity dictionary to: %s", entity_output)
            build_entity_dictionary(
                resolve_project_path(entity.get("source_csv", "data/code_stock.csv")),
                entity_output,
                questions_path=resolve_project_path(entity.get("questions_file", "data/questions/questions.jsonl")),
                structured_dir=output_dir / "tables",
            )
        except Exception as error:
            logger.exception("Entity dictionary build failed")
            errors.append({"stage": "entity_dictionary", "error": str(error), "traceback": traceback.format_exc()})

    indexing = config.get("indexing", {})
    if indexing.get("enabled", False):
        try:
            build_search_indexes(
                all_documents, output_dir / indexing.get("output_dir", "indexes"),
                indexing, config.get("embedding", {}),
            )
        except Exception as error:
            logger.exception("Indexing failed")
            errors.append({"stage": "indexing", "error": str(error), "traceback": traceback.format_exc()})
            if indexing.get("fail_on_error", False):
                raise

    write_jsonl(output_dir / "records.jsonl", all_records)
    write_jsonl(output_dir / "retrieval_documents.jsonl", all_documents)
    write_jsonl(output_dir / "ingestion_errors.jsonl", errors)
    return {
        "file_count": len(files), "record_count": len(all_records),
        "document_count": len(all_documents), "error_count": len(errors),
    }
