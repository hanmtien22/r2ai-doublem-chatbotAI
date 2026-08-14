from typing import TypedDict, List, Dict, Any, Annotated, Optional
import operator

def merge_dicts(a: dict, b: dict) -> dict:
    c = a.copy() if a else {}
    if b:
        c.update(b)
    return c

def merge_lists(a: list, b: list) -> list:
    res = list(a) if a else []
    if b:
        res.extend(b)
    return res

class AgentState(TypedDict):
    query: str
    plan: Optional[List[dict]]
    final_answer_variable: Optional[Any]
    current_step: int
    variables_memory: Annotated[Dict[str, Any], merge_dicts]
    collected_docs: Annotated[List[str], merge_lists]
    collected_tables: Annotated[List[str], merge_lists]
    collected_evidence: Annotated[List[dict], merge_lists]
    final_answer: Any
    error: Optional[str]
