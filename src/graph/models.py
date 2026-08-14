from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field

class Indicator(BaseModel):
    search_keywords: List[str] = Field(default_factory=list)
    code: Optional[str] = None
    search_in: Optional[str] = None


class Step(BaseModel):
    step_id: int
    action: str  # 'fetch_data' or 'evaluate_condition'
    target_variable: str
    tickers: Optional[List[str]] = None
    years: Optional[List[int]] = None
    indicator: Optional[Indicator] = None
    logic: Optional[str] = None
    description: Optional[str] = None
    report_type: Optional[str] = None


class ExecutionPlan(BaseModel):
    steps: List[Step]
    final_compute: str
    final_answer_unit: Optional[str] = None


class PlanResponse(BaseModel):
    execution_plan: ExecutionPlan