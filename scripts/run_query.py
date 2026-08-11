#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import DEFAULT_CONFIG_PATH, load_config
from src.llm.client import LLMClient
from src.query.pipeline import QueryPipeline
from src.retrieval.pipeline import QueryRetrievalPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phân tích câu hỏi và truy xuất dữ liệu tài chính đã ingestion.",
    )
    parser.add_argument("question", nargs="+", help="Câu hỏi tài chính cần xử lý.")
    parser.add_argument(
        "--documents",
        type=Path,
        default=PROJECT_ROOT / "data" / "parsed_tables" / "retrieval_documents.jsonl",
        help="Đường dẫn retrieval_documents.jsonl.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Đường dẫn file YAML cấu hình chung.",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        help="Đường dẫn thư mục chứa bm25.pkl và thư mục faiss.",
    )
    parser.add_argument(
        "--reference-year",
        type=int,
        help="Ghi đè defaults.reference_year trong cấu hình.",
    )
    parser.add_argument("--top-k", type=int, help="Ghi đè retrieval.top_k trong cấu hình.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    fuzzy_config = config.get("fuzzy_match", {})
    router_config = config.get("router", {})
    llm_config = config.get("llm", {})
    cache_config = config.get("cache", {})
    retrieval_config = config.get("retrieval", {})
    embedding_config = config.get("embedding", {})
    defaults = config.get("defaults", {})
    dictionaries_config = config.get("dictionaries", {})
    reference_year = args.reference_year or defaults.get("reference_year", 2024)
    dictionary_root = PROJECT_ROOT / dictionaries_config.get(
        "root_dir", "data/dictionaries"
    )

    def dictionary_path(config_key: str, default_name: str) -> Path:
        filename = dictionaries_config.get(config_key, default_name)
        dataset_path = args.documents.parent / filename
        return dataset_path if dataset_path.exists() else dictionary_root / filename

    entity_dictionary_path = dictionary_path(
        "entity_dictionary", "entity_dictionary.json"
    )
    indicator_aliases_path = dictionary_path(
        "indicator_aliases", "indicator_aliases.json"
    )
    schema_mapping_path = dictionary_path(
        "schema_mapping", "schema_mapping.json"
    )

    llm_client = None
    if llm_config.get("enabled", False):
        llm_client = LLMClient(
            model_name=llm_config.get("model_name", "Qwen2.5-14B-Instruct"),
            model_path=llm_config.get("model_path"),
            max_tokens=llm_config.get("max_tokens", 100),
            temperature=llm_config.get("temperature", 0.0),
            timeout=llm_config.get("timeout", 30),
            max_retries=llm_config.get("max_retries", 2),
        )

    pipeline = QueryRetrievalPipeline(
        documents_path=args.documents,
        index_dir=args.index_dir,
        retrieval_config=retrieval_config,
        embedding_config=embedding_config,
        query_pipeline=QueryPipeline(
            entity_dict_path=str(entity_dictionary_path),
            indicator_aliases_path=str(indicator_aliases_path),
            schema_mapping_path=str(schema_mapping_path),
            reference_year=reference_year,
            company_threshold=fuzzy_config.get("company_threshold", 85),
            indicator_threshold=fuzzy_config.get("indicator_threshold", 80),
            use_llm_fallback=router_config.get("use_llm_fallback", False),
            llm_client=llm_client,
            cache_enabled=cache_config.get("enabled", True),
            cache_max_size=cache_config.get("max_size", 1000),
        ),
    )
    top_k = args.top_k or retrieval_config.get("top_k", 5)
    result = pipeline.process(" ".join(args.question), top_k_per_query=top_k)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
