import logging
import pandas as pd
from src.query.pipeline import QueryPipeline
from src.query.hybrid_fetcher import EasyHybridSolver
from src.graph.executor import LangGraphExecutor

logger = logging.getLogger(__name__)

class QuestionOrchestrator:
    def __init__(self, data_dir: str, endpoint: str, api_key: str, model: str):
        # Bộ phân loại câu hỏi (Rule-based)
        self.router_pipeline = QueryPipeline()
        # Bộ giải câu DỄ (Nhanh)
        self.easy_solver = EasyHybridSolver(data_dir=data_dir)
        # Bộ giải câu KHÓ (LLM + Graph)
        self.graph_executor = LangGraphExecutor(endpoint=endpoint, api_key=api_key, model=model, data_dir=data_dir)

    def process_question(self, q_id: int, question: str) -> dict:
        logger.info(f"--- Processing Question {q_id} ---")
        
        # 1. Phân loại câu hỏi
        query_result = self.router_pipeline.process(question)
        q_type = query_result.query_type
        
        entities_dict = query_result.entities.to_dict() if hasattr(query_result.entities, 'to_dict') else query_result.entities
        
        tickers = entities_dict.get("tickers", [])
        years = entities_dict.get("years", [])
        
        # Logic phân loại:
        # Nếu có từ khóa so sánh rõ ràng, hoặc tính công thức phức tạp (2+ công ty, hoặc 2+ năm để so sánh) -> HARD (LangGraph)
        # Nếu chỉ đơn thuần tra cứu 1 công ty và 1 năm -> EASY
        is_hard = False
        if len(tickers) > 1 or len(years) > 1:
            is_hard = True
        # derived_indicator cũng cần HARD vì cần tính toán công thức (vd: tỷ suất lợi nhuận)
        if q_type in ["multi_comparison", "derived_indicator"]:
            is_hard = True

        if not is_hard:
            logger.info("-> Route: EASY (Single Lookup)")
            return self._solve_easy(q_id, question, query_result)
        else:
            logger.info("-> Route: HARD (LangGraph)")
            return self._solve_hard(q_id, question)

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

        # Lấy số liệu
        val, doc, table, evidence = self.easy_solver.fetch_data(inputs)
        
        # Chuyển đổi an toàn các kiểu numpy int64/float64 sang Python native
        if hasattr(val, "item"):
            val = val.item()
        elif pd.isna(val):
            val = None
            
        # Trả kết quả chuẩn format
        return {
            "id": q_id,
            "question": question,
            "answer": val,
            "relevant_docs": [doc] if doc else [],
            "relevant_tables": [table] if table else [],
            "evidence": [evidence] if evidence else [],
            "pandas_query": ""
        }

    def _solve_hard(self, q_id: int, question: str) -> dict:
        result = self.graph_executor.run(question)
        result["id"] = q_id
        return result