import logging
import json
from typing import Dict, Any, List
from langgraph.graph import StateGraph, END
from src.graph.state import AgentState
from src.graph.models import ExecutionPlan, Step
from src.llm.tgi_client import GenericLLMClient
from src.query.hybrid_fetcher import EasyHybridSolver
from src.query.evaluator import PythonEvaluator

logger = logging.getLogger(__name__)

class LangGraphExecutor:
    def __init__(self, endpoint: str, api_key: str, model: str, data_dir: str = "data"):
        self.llm_client = GenericLLMClient(endpoint=endpoint, api_key=api_key, model=model)
        self.fetcher = EasyHybridSolver(data_dir=data_dir)
        self.evaluator = PythonEvaluator()
        self.graph = self._build_graph()

    def _build_planner_prompt(self, query: str) -> str:
        return f"""You are an expert financial analysis planner. Break down the user query into precise JSON execution steps based ONLY on what is asked.

User Query: "{query}"

Standard Financial Table Types:
- "IS" or "income_statement": For revenue, profit (Lợi nhuận sau thuế, Lợi nhuận gộp), expenses.
- "BS" or "balance_sheet": For assets, liabilities, equity.
- "CF" or "cash_flow": For cash flows.

Rules:
1. Extract the exact ticker(s) from the query (e.g. "CTCP Chứng khoán FPT" is ticker "FTS", Vietjet is "VJC", FPT Corporation is "FPT").
2. For "fetch_data", standard fields:
   - "ticker": string (e.g. "FTS")
   - "period": integer (e.g. 2023)
   - "metric": string (in Vietnamese as requested, e.g. "lợi nhuận sau thuế")
   - "table_type": "IS" or "BS" or "CF"
3. Every step defines an "output_variable" (e.g. "fts_profit").
4. If the query asks for a single metric of a company without multi-step arithmetic, generate only 1 "fetch_data" step and set "final_answer_variable" to its output_variable.
5. If math is needed, only use the exact variable names created in previous steps.

Produce valid JSON matching the ExecutionPlan schema.
"""

    def plan_node(self, state: AgentState) -> Dict:
        logger.info("Executing node: PLANNER")
        query = state["query"]
        
        prompt = self._build_planner_prompt(query)
        raw_response = self.llm_client.generate(prompt, schema=ExecutionPlan, temperature=0.1)
        
        if not raw_response or not isinstance(raw_response, str):
            logger.error("Planner failed: LLM returned empty or non-string response")
            return {"plan": [], "current_step": 0, "error": "LLM returned empty response"}

        try:
            text = raw_response.strip()
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                text = text[start:end]
            response = json.loads(text)
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Planner failed: cannot parse LLM response as JSON: {e}")
            return {"plan": [], "current_step": 0, "error": f"JSON parse error: {e}"}

        if not isinstance(response, dict):
            logger.error("Planner failed: parsed response is not a dict")
            return {"plan": [], "current_step": 0, "error": "Response is not a dict"}
            
        steps = response.get("steps", [])
        final_ans_var = response.get("final_answer_variable")
        
        if not isinstance(steps, list):
             steps = []
             
        logger.info(f"Plan generated ({len(steps)} steps): {json.dumps(steps, ensure_ascii=False)}")
        return {
            "plan": steps, 
            "current_step": 0, 
            "final_answer_variable": final_ans_var,
            "variables_memory": {},
            "collected_docs": [],
            "collected_tables": [],
            "collected_evidence": []
        }

    def execute_node(self, state: AgentState) -> Dict:
        step_idx = state["current_step"]
        steps = state["plan"]
        variables = state.get("variables_memory", {})
        
        if step_idx >= len(steps):
            return {}
            
        step = steps[step_idx]
        
        # Xác định action và inputs linh hoạt (hỗ trợ cả chuẩn Pydantic lẫn trường hợp LLM sinh key trực tiếp)
        action = step.get("action")
        inputs = step.get("inputs", {})
        
        if not action:
            if "fetch_data" in step:
                action = "fetch_data"
                inputs = step.get("fetch_data", {})
            elif "evaluate_condition" in step:
                action = "evaluate_condition"
                inputs = step.get("evaluate_condition", {})
                
        inputs = inputs.copy() if isinstance(inputs, dict) else {}
        logger.info(f"Executing step {step.get('step_id', step_idx)}: {action} with inputs {inputs}")
        
        for k, v in inputs.items():
            if isinstance(v, str) and v in variables:
                inputs[k] = variables[v]

        updates = {}
        if action == "fetch_data":
            val, doc, table, evidence = self.fetcher.fetch_data(inputs)
            out_var = step.get("output_variable")
            updates = {
                "variables_memory": {out_var: val} if out_var else {},
                "collected_docs": [doc] if doc else [],
                "collected_tables": [table] if table else [],
                "collected_evidence": [evidence] if evidence else []
            }
        elif action == "evaluate_condition":
            val = self.evaluator.evaluate(inputs, variables)
            out_var = step.get("output_variable")
            updates = {
                "variables_memory": {out_var: val} if out_var else {}
            }
            
        updates["current_step"] = step_idx + 1
        return updates

    def should_continue(self, state: AgentState) -> str:
        if state.get("error"):
            return "end"
        if state["current_step"] >= len(state.get("plan", [])):
            return "end"
        return "continue"
        
    def format_node(self, state: AgentState) -> Dict:
        logger.info("Executing node: FORMATTER")
        ans = None
        final_var = state.get("final_answer_variable")
        
        if final_var is not None:
            if isinstance(final_var, list):
                ans = [state["variables_memory"].get(v) for v in final_var]
            else:
                ans = state["variables_memory"].get(final_var)
        # Fallback nếu evaluate trả về None hoặc lỗi
        if ans is None and len(state.get("plan", [])) > 0:
            for step in state["plan"]:
                var_name = step.get("output_variable")
                if var_name in state["variables_memory"] and state["variables_memory"][var_name] is not None:
                    ans = state["variables_memory"][var_name]
                    break

        return {"final_answer": ans}

    def _build_graph(self):
        workflow = StateGraph(AgentState)
        
        workflow.add_node("planner", self.plan_node)
        workflow.add_node("executor", self.execute_node)
        workflow.add_node("formatter", self.format_node)
        
        workflow.set_entry_point("planner")
        workflow.add_edge("planner", "executor")
        
        workflow.add_conditional_edges(
            "executor",
            self.should_continue,
            {
                "continue": "executor",
                "end": "formatter"
            }
        )
        
        workflow.add_edge("formatter", END)
        
        return workflow.compile()
        
    def run(self, query: str) -> Dict:
        initial_state = {
            "query": query,
            "plan": [],
            "current_step": 0,
            "variables_memory": {},
            "collected_docs": [],
            "collected_tables": [],
            "collected_evidence": [],
            "final_answer": None,
            "error": None
        }
        
        result_state = self.graph.invoke(initial_state)
        
        # Chuyển đổi an toàn kiểu dữ liệu sang JSON serializable
        final_ans = result_state.get("final_answer")
        if hasattr(final_ans, "item"):
            final_ans = final_ans.item()
            
        output = {
            "id": None, # Will be set by batch runner
            "question": query,
            "answer": final_ans,
            "relevant_docs": list(set(result_state.get("collected_docs", []))),
            "relevant_tables": list(set(result_state.get("collected_tables", []))),
            "evidence": result_state.get("collected_evidence", []),
            "pandas_query": ""
        }
        return output
