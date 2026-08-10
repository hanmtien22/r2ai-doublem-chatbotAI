# import logging
# from pathlib import Path
# import traceback
# from src.config_loader import load_config, resolve_project_path
# from ..indexing.indexing import build_search_indexes
# from .chunk_builder import (
#     json_safe_record,
#     build_retrieval_documents,
#     enrich_and_normalize_table,
#     write_jsonl
# )

# from .processing import (
#     scan_financial_files,
#     extract_meatdata,
#     detect_tables,
#     parse_table_lines,
#     validate_table,
#     save_parsed_table,
#     build_entity_dictionary
# )

# logger = logging.getLogger(__name__)


# def run_ingestion_pipeline(config: dict | None = None):
#     config = load_config() if config is None else config

#     source_dir = resolve_project_path(config["source_dir"])
#     output_dir = resolve_project_path(config["output_dir"])
#     parser_config = config["parser"]

#     logger.info(
#         "Starting ingestion pipeline with source_dir=%s, output_dir=%s",
#         source_dir,
#         output_dir,
#     )

#     output_dir.mkdir(parents=True, exist_ok=True)

#     files = scan_financial_files(source_dir)
#     logger.info("Found %d financial files to process", len(files))

#     all_records = []
#     all_documents = []
#     errors = []

#     for path in files:
#         logger.info("Processing file: %s", path)
#         try:
#             metadata = extract_meatdata(path)

#             text = path.read_text(
#                 encoding="utf-8",
#                 errors="replace",
#             )

#             lines = text.splitlines()
#             detected_tables = detect_tables(lines)
#             logger.info(
#                 "Detected %d tables in %s", len(detected_tables), path.name
#             )

#             for table in detected_tables:
#                 logger.debug(
#                     "Parsing table '%s' (type=%s) in %s",
#                     table.table_name,
#                     table.type_table,
#                     path.name,
#                 )
#                 dataframe = parse_table_lines(table.lines)

#                 if dataframe.empty:
#                     logger.warning(
#                         "Empty dataframe for table '%s' in %s",
#                         table.table_name,
#                         path.name,
#                     )
#                     errors.append({
#                         "source_file": str(path),
#                         "table_type": table.type_table,
#                         "errors": [
#                             f"Không parse được bảng {table.table_name}"
#                         ],
#                     })
#                     continue

#                 logger.debug(
#                     "Enriching table '%s' (input rows=%d)",
#                     table.table_name,
#                     len(dataframe),
#                 )
#                 dataframe = enrich_and_normalize_table(
#                     dataframe=dataframe,
#                     document_text="\n".join(table.lines),
#                     metadata=metadata,
#                     detected_table=table,
#                 )

#                 validation_errors = validate_table(
#                     dataframe,
#                     minimum_table_rows=parser_config["minimum_table_rows"],
#                     maximum_null_ratio=parser_config["maximum_null_ratio"],
#                 )

#                 if validation_errors:
#                     logger.warning(
#                         "Validation errors for table '%s' in %s: %s",
#                         table.type_table,
#                         path.name,
#                         validation_errors,
#                     )
#                     errors.append({
#                         "source_file": str(path),
#                         "table_type": table.type_table,
#                         "errors": validation_errors,
#                     })

#                 save_parsed_table(
#                     df=dataframe,
#                     output_dir=output_dir / "tables",
#                     ticker=metadata["sticker"],
#                     year=metadata["year"],
#                     table_type=table.type_table,
#                     table_id=table.start_line,
#                 )
#                 logger.debug("Saved parsed table for %s", metadata["sticker"])

#                 records = [
#                     json_safe_record(record)
#                     for record in dataframe.to_dict(orient="records")
#                 ]
#                 all_records.extend(records)

#                 documents = build_retrieval_documents(
#                     dataframe=dataframe,
#                     detected_table=table,
#                 )
#                 all_documents.extend(documents)
#                 logger.debug(
#                     "Added %d records and %d documents for table %s",
#                     len(records),
#                     len(documents),
#                     table.type_table,
#                 )

#         except Exception as error:
#             logger.error("Error processing file %s: %s", path, error, exc_info=True)
#             errors.append({
#                 "source_file": str(path),
#                 "error": str(error),
#                 "traceback": traceback.format_exc(),
#             })

#     entity_config = config.get("entity_dictionary", {})
#     if entity_config.get("rebuild", False):
#         logger.info("Rebuilding entity dictionary...")
#         try:
#             build_entity_dictionary(
#                 resolve_project_path(entity_config.get("source_csv", "data/code_stock.csv")),
#                 output_dir / entity_config.get("output_file", "entity_dictionary.json"),
#             )
#             logger.info("Entity dictionary built successfully.")
#         except Exception as error:
#             logger.error("Error building entity dictionary: %s", error, exc_info=True)
#             errors.append({
#                 "stage": "entity_dictionary",
#                 "error": str(error),
#                 "traceback": traceback.format_exc(),
#             })

#     indexing_config = config.get("indexing", {})
#     if indexing_config.get("enabled", False):
#         logger.info("Building search indexes stage...")
#         try:
#             index_output_dir = output_dir / indexing_config.get("output_dir", "indexes")
#             build_search_indexes(
#                 documents=all_documents,
#                 output_dir=index_output_dir,
#                 indexing_config=indexing_config,
#                 embedding_config=config.get("embedding", {}),
#             )
#             logger.info("Search indexing stage completed.")
#         except Exception as error:
#             logger.error("Error in indexing stage: %s", error, exc_info=True)
#             errors.append({
#                 "stage": "indexing",
#                 "error": str(error),
#                 "traceback": traceback.format_exc(),
#             })
#             if indexing_config.get("fail_on_error", False):
#                 raise

#     logger.info("Writing output jsonl files...")
#     write_jsonl(
#         output_dir / "records.jsonl",
#         all_records,
#     )

#     write_jsonl(
#         output_dir / "retrieval_documents.jsonl",
#         all_documents,
#     )

#     write_jsonl(
#         output_dir / "ingestion_errors.jsonl",
#         errors,
#     )

#     summary = {
#         "file_count": len(files),
#         "record_count": len(all_records),
#         "document_count": len(all_documents),
#         "error_count": len(errors),
#     }
#     logger.info("Ingestion pipeline completed. Summary: %s", summary)

#     return summary

import logging
from pathlib import Path
import traceback
from src.config_loader import load_config, resolve_project_path
from ..indexing.indexing import build_search_indexes
from .chunk_builder import (
    json_safe_record,
    build_retrieval_documents,
    enrich_and_normalize_table,
    write_jsonl
)

from .processing import (
    scan_financial_files,
    extract_meatdata,
    detect_tables,
    parse_table_lines,
    validate_table,
    save_parsed_table,
    build_entity_dictionary,
    create_dictionary_stats,
    collect_dictionary_features,
    build_indicator_aliases_from_stats,
    build_schema_mapping_from_stats,
    build_dictionary_report,
    save_json,
)

logger = logging.getLogger(__name__)


def run_ingestion_pipeline(config: dict | None = None):
    config = load_config() if config is None else config

    source_dir = resolve_project_path(config["source_dir"])
    output_dir = resolve_project_path(config["output_dir"])
    parser_config = config["parser"]

    logger.info(
        "Starting ingestion pipeline with source_dir=%s, output_dir=%s",
        source_dir,
        output_dir,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    files = scan_financial_files(source_dir)
    logger.info("Found %d financial files to process", len(files))

    all_records = []
    all_documents = []
    errors = []
    dictionary_stats = create_dictionary_stats()

    for path in files:
        logger.info("Processing file: %s", path)
        try:
            metadata = extract_meatdata(path)

            text = path.read_text(
                encoding="utf-8",
                errors="replace",
            )

            lines = text.splitlines()
            detected_tables = detect_tables(lines)
            logger.info(
                "Detected %d tables in %s", len(detected_tables), path.name
            )

            for table in detected_tables:
                logger.debug(
                    "Parsing table '%s' (type=%s) in %s",
                    table.table_name,
                    table.type_table,
                    path.name,
                )
                dataframe = parse_table_lines(table.lines)

                if dataframe.empty:
                    logger.warning(
                        "Empty dataframe for table '%s' in %s",
                        table.table_name,
                        path.name,
                    )
                    errors.append({
                        "source_file": str(path),
                        "table_type": table.type_table,
                        "errors": [
                            f"Không parse được bảng {table.table_name}"
                        ],
                    })
                    continue

                logger.debug(
                    "Enriching table '%s' (input rows=%d)",
                    table.table_name,
                    len(dataframe),
                )
                dataframe = enrich_and_normalize_table(
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
                    logger.warning(
                        "Validation errors for table '%s' in %s: %s",
                        table.type_table,
                        path.name,
                        validation_errors,
                    )
                    errors.append({
                        "source_file": str(path),
                        "table_type": table.type_table,
                        "errors": validation_errors,
                    })

                save_parsed_table(
                    df=dataframe,
                    output_dir=output_dir / "tables",
                    ticker=metadata["sticker"],
                    year=metadata["year"],
                    table_type=table.type_table,
                    table_id=table.start_line,
                )
                logger.debug("Saved parsed table for %s", metadata["sticker"])

                if not validation_errors:
                    collect_dictionary_features(
                        dataframe,
                        dictionary_stats,
                    )

                records = [
                    json_safe_record(record)
                    for record in dataframe.to_dict(orient="records")
                ]
                all_records.extend(records)

                documents = build_retrieval_documents(
                    dataframe=dataframe,
                    detected_table=table,
                )
                all_documents.extend(documents)
                logger.debug(
                    "Added %d records and %d documents for table %s",
                    len(records),
                    len(documents),
                    table.type_table,
                )

        except Exception as error:
            logger.error("Error processing file %s: %s", path, error, exc_info=True)
            errors.append({
                "source_file": str(path),
                "error": str(error),
                "traceback": traceback.format_exc(),
            })

    dictionaries_config = config.get("dictionaries", {})
    dictionary_root = resolve_project_path(
        dictionaries_config.get("root_dir", "data/dictionaries")
    )
    dictionary_root.mkdir(parents=True, exist_ok=True)

    dictionary_config = config.get("dictionary_builder", {})
    if dictionary_config.get("enabled", True):
        logger.info("Building dataset dictionaries...")
        try:
            min_count = int(dictionary_config.get("min_count", 5))
            dictionary_output_dir = resolve_project_path(
                dictionary_config.get("output_dir", str(dictionary_root))
            )
            dictionary_output_dir.mkdir(parents=True, exist_ok=True)

            indicator_aliases = build_indicator_aliases_from_stats(
                dictionary_stats,
                min_count=min_count,
            )
            schema_mapping = build_schema_mapping_from_stats(
                dictionary_stats,
                min_count=min_count,
            )
            dictionary_report = build_dictionary_report(dictionary_stats)

            save_json(
                indicator_aliases,
                dictionary_output_dir / dictionary_config.get(
                    "indicator_aliases_file",
                    "indicator_aliases.json",
                ),
            )
            save_json(
                schema_mapping,
                dictionary_output_dir / dictionary_config.get(
                    "schema_mapping_file",
                    "schema_mapping.json",
                ),
            )
            save_json(
                dictionary_report,
                output_dir / dictionary_config.get(
                    "report_file",
                    "dictionary_report.json",
                ),
            )
            logger.info("Dataset dictionaries built successfully.")
        except Exception as error:
            logger.error("Error building dataset dictionaries: %s", error, exc_info=True)
            errors.append({
                "stage": "dictionary_builder",
                "error": str(error),
                "traceback": traceback.format_exc(),
            })

    entity_config = config.get("entity_dictionary", {})
    if entity_config.get("rebuild", False):
        logger.info("Rebuilding entity dictionary...")
        try:
            entity_output_path = resolve_project_path(
                entity_config.get(
                    "output_path",
                    str(dictionary_root / "entity_dictionary.json"),
                )
            )
            questions_path = resolve_project_path(
                entity_config.get(
                    "questions_file",
                    "data/questions/questions.jsonl",
                )
            )
            parquet_dir = output_dir / "tables"

            build_entity_dictionary(
                csv_path=resolve_project_path(
                    entity_config.get("source_csv", "data/code_stock.csv")
                ),
                output_path=entity_output_path,
                questions_path=questions_path,
                parquet_dir=parquet_dir,
            )
            logger.info("Entity dictionary built successfully.")
        except Exception as error:
            logger.error("Error building entity dictionary: %s", error, exc_info=True)
            errors.append({
                "stage": "entity_dictionary",
                "error": str(error),
                "traceback": traceback.format_exc(),
            })

    indexing_config = config.get("indexing", {})
    if indexing_config.get("enabled", False):
        logger.info("Building search indexes stage...")
        try:
            index_output_dir = output_dir / indexing_config.get("output_dir", "indexes")
            build_search_indexes(
                documents=all_documents,
                output_dir=index_output_dir,
                indexing_config=indexing_config,
                embedding_config=config.get("embedding", {}),
            )
            logger.info("Search indexing stage completed.")
        except Exception as error:
            logger.error("Error in indexing stage: %s", error, exc_info=True)
            errors.append({
                "stage": "indexing",
                "error": str(error),
                "traceback": traceback.format_exc(),
            })
            if indexing_config.get("fail_on_error", False):
                raise

    logger.info("Writing output jsonl files...")
    write_jsonl(
        output_dir / "records.jsonl",
        all_records,
    )

    write_jsonl(
        output_dir / "retrieval_documents.jsonl",
        all_documents,
    )

    write_jsonl(
        output_dir / "ingestion_errors.jsonl",
        errors,
    )

    summary = {
        "file_count": len(files),
        "record_count": len(all_records),
        "document_count": len(all_documents),
        "error_count": len(errors),
    }
    logger.info("Ingestion pipeline completed. Summary: %s", summary)

    return summary
