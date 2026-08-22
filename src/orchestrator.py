import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

from src.query.pipeline import QueryPipeline
from src.query.hybrid_fetcher import EasyHybridSolver
from src.graph.executor import LangGraphExecutor
from src.compute.code_generator import CodeGenerator
from src.compute.sandbox import Sandbox
from src.compute.result_verifier import ResultVerifier
from src.compute.retry_manager import RetryManager
from src.llm.tgi_client import GenericLLMClient

logger = logging.getLogger(__name__)


class QuestionOrchestrator:
    def __init__(self, data_dir: str, endpoint: str, api_key: str, model: str):
        self.data_dir = Path(data_dir)
        self.router_pipeline = QueryPipeline()
        self.easy_solver = EasyHybridSolver(data_dir=data_dir)

        self.llm_client = GenericLLMClient(endpoint=endpoint, api_key=api_key, model=model)
        self.graph_executor = LangGraphExecutor(endpoint=endpoint, api_key=api_key, model=model, data_dir=data_dir)

        self.code_gen = CodeGenerator(llm_client=self.llm_client)
        self.sandbox = Sandbox(timeout=15)
        self.verifier = ResultVerifier()
        self.compute_manager = RetryManager(self.code_gen, self.sandbox, self.verifier)

    def process_question(self, q_id: int, question: str) -> dict:
        logger.info(f"--- Processing Question {q_id} ---")

        query_result = self.router_pipeline.process(question)
        q_type = query_result.query_type

        entities_dict = query_result.entities.to_dict() if hasattr(query_result.entities, 'to_dict') else query_result.entities

        tickers = entities_dict.get("tickers", [])
        years = entities_dict.get("years", [])

        is_hard = False
        if len(tickers) > 1 or len(years) > 1:
            is_hard = True
        if q_type in ["multi_comparison", "derived_indicator"]:
            is_hard = True

        if not is_hard:
            logger.info("-> Route: EASY (Single Lookup)")
            return self._solve_easy(q_id, question, query_result)
        else:
            logger.info("-> Route: HARD (LangGraph + Pandas CodeGen)")
            return self._solve_hard(q_id, question, query_result)

    def _solve_easy(self, q_id: int, question: str, query_result) -> dict:
        entities_dict = query_result.entities.to_dict() if hasattr(query_result.entities, 'to_dict') else query_result.entities
        inputs = {}
        tickers = entities_dict.get("tickers", [])
        if tickers: inputs["ticker"] = tickers[0]

        years = entities_dict.get("years", [])
        if years: inputs["period"] = years[0]

        indicators = entities_dict.get("indicators", [])
        if indicators: inputs["metric"] = indicators[0]

        indicator_codes = entities_dict.get("indicator_codes", [])
        if indicator_codes:
            parts = indicator_codes[0].split('.')
            if len(parts) == 2:
                inputs["table_type"] = parts[0]
                inputs["indicator_code"] = parts[1]

        val, doc, table, evidence = self.easy_solver.fetch_data(inputs)

        if hasattr(val, "item"):
            val = val.item()
        elif val is not None and pd.isna(val):
            val = None

        return {
            "id": q_id,
            "question": question,
            "answer": val,
            "relevant_docs": [doc] if doc else [],
            "relevant_tables": [table] if table else [],
            "evidence": [evidence] if evidence else [],
            "pandas_query": ""
        }

    def _collect_tables_for_compute(self, query_result) -> List[Dict[str, Any]]:
        entities_dict = query_result.entities.to_dict() if hasattr(query_result.entities, 'to_dict') else query_result.entities
        tickers = entities_dict.get("tickers", [])
        years = entities_dict.get("years", [])
        indicator_codes = entities_dict.get("indicator_codes", [])

        tables_dir = self.data_dir / "tables"
        if not tables_dir.exists():
            tables_dir = self.data_dir / "parsed_tables" / "tables"
        if not tables_dir.exists():
            return []

        all_rows = []
        matched_files = set()

        for ticker in tickers:
            for csv_path in tables_dir.glob(f"{ticker.upper()}_*.csv"):
                if csv_path.name in matched_files:
                    continue
                try:
                    df = pd.read_csv(csv_path, dtype={"item_code": str})
                    if "period" in df.columns and "value" in df.columns:
                        if years:
                            df = df[df["period"].isin([int(y) for y in years])]
                        if not df.empty:
                            matched_files.add(csv_path.name)
                            for _, row in df.iterrows():
                                all_rows.append([
                                    row.get("ticker", ticker.upper()),
                                    row.get("period", ""),
                                    row.get("item_name_normalized", row.get("item_name_raw", "")),
                                    str(row.get("item_code", "")),
                                    row.get("value", None),
                                    row.get("section", ""),
                                    row.get("report_type", ""),
                                    row.get("unit", "vnd"),
                                ])
                except Exception as e:
                    logger.warning(f"Error reading {csv_path}: {e}")

        if not all_rows:
            return []

        return [{
            "ticker": ", ".join(tickers),
            "columns": ["ticker", "period", "item_name", "item_code", "value",
                         "section", "report_type", "unit"],
            "data": all_rows,
            "description": "Dòng số liệu từ BCTC chính (cột `value` đơn vị VND)",
        }]

    def _solve_hard(self, q_id: int, question: str, query_result) -> dict:
        try:
            result = self.graph_executor.run(question)
            if result.get("answer") is not None:
                result["id"] = q_id
                return result
            logger.warning(f"LangGraph returned None for question {q_id}, trying Pandas CodeGen")
        except Exception as e:
            logger.warning(f"LangGraph failed for question {q_id}: {e}, trying Pandas CodeGen")

        tables_for_compute = self._collect_tables_for_compute(query_result)
        if tables_for_compute:
            try:
                is_success, computed_result, final_code, error_msg = self.compute_manager.compute(
                    question, tables_for_compute
                )
                if is_success and computed_result is not None:
                    if hasattr(computed_result, "item"):
                        computed_result = computed_result.item()

                    entities_dict = query_result.entities.to_dict() if hasattr(query_result.entities, 'to_dict') else query_result.entities
                    tickers = entities_dict.get("tickers", [])
                    years = entities_dict.get("years", [])

                    return {
                        "id": q_id,
                        "question": question,
                        "answer": computed_result,
                        "relevant_docs": [f"{t}_{y}_consolidated" for t in tickers for y in years] if tickers and years else [],
                        "relevant_tables": [],
                        "evidence": [],
                        "pandas_query": final_code
                    }
                logger.warning(f"Pandas CodeGen failed for question {q_id}: {error_msg}")
            except Exception as e:
                logger.warning(f"Pandas CodeGen exception for question {q_id}: {e}")

        return self._solve_easy(q_id, question, query_result)
