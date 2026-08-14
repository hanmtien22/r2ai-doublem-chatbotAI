import logging
from typing import Dict, Any
import simpleeval

logger = logging.getLogger(__name__)

class PythonEvaluator:
    def __init__(self):
        # Thiết lập các hàm an toàn cho parser chuỗi toán học
        self.safe_functions = {
            "abs": abs,
            "max": max,
            "min": min,
            "round": round
        }

    def evaluate(self, inputs: Dict[str, Any], variables: Dict[str, Any]) -> Any:
        logger.info(f"Evaluating logic with inputs: {inputs}")
        
        operation = inputs.get("operation")
        
        try:
            if operation == "argmax":
                data = inputs.get("data", {})
                if isinstance(data, str) and data in variables:
                    data = variables[data]
                if not isinstance(data, dict) or not data:
                    return None
                return max(data, key=data.get)

            elif operation == "argmin":
                data = inputs.get("data", {})
                if isinstance(data, str) and data in variables:
                    data = variables[data]
                if not isinstance(data, dict) or not data:
                    return None
                return min(data, key=data.get)

            elif operation == "compare_greater":
                 val1 = inputs.get("val1")
                 val2 = inputs.get("val2")
                 
                 if isinstance(val1, str) and val1 in variables: val1 = variables[val1]
                 if isinstance(val2, str) and val2 in variables: val2 = variables[val2]
                     
                 if float(val1) > float(val2):
                     return inputs.get("key1", val1)
                 else:
                     return inputs.get("key2", val2)

            elif operation == "compare_less":
                 val1 = inputs.get("val1")
                 val2 = inputs.get("val2")
                 
                 if isinstance(val1, str) and val1 in variables: val1 = variables[val1]
                 if isinstance(val2, str) and val2 in variables: val2 = variables[val2]
                     
                 if float(val1) < float(val2):
                     return inputs.get("key1", val1)
                 else:
                     return inputs.get("key2", val2)

            elif operation == "add":
                val1 = self._resolve_val(inputs.get("val1"), variables)
                val2 = self._resolve_val(inputs.get("val2"), variables)
                return val1 + val2

            elif operation == "subtract":
                val1 = self._resolve_val(inputs.get("val1"), variables)
                val2 = self._resolve_val(inputs.get("val2"), variables)
                return val1 - val2

            elif operation == "divide":
                num = self._resolve_val(inputs.get("numerator"), variables)
                den = self._resolve_val(inputs.get("denominator"), variables)
                if den == 0: return None
                return num / den

            elif operation == "multiply":
                val1 = self._resolve_val(inputs.get("val1"), variables)
                val2 = self._resolve_val(inputs.get("val2"), variables)
                return val1 * val2
                
            elif operation == "abs_difference":
                val1 = self._resolve_val(inputs.get("val1"), variables)
                val2 = self._resolve_val(inputs.get("val2"), variables)
                return abs(val1 - val2)

            elif operation == "growth_rate":
                current = self._resolve_val(inputs.get("current"), variables)
                previous = self._resolve_val(inputs.get("previous"), variables)
                if previous == 0: return None
                return ((current - previous) / abs(previous)) * 100

            elif operation == "ratio":
                part = self._resolve_val(inputs.get("part"), variables)
                whole = self._resolve_val(inputs.get("whole"), variables)
                if whole == 0: return None
                return (part / whole) * 100

            # Parser linh hoạt bằng simpleeval
            elif operation == "math":
                expr = inputs.get("expression", "")
                
                # Nếu chỉ có 1 biến trong memory mà biểu thức dùng 'var_a', tự động map
                local_names = {k: v for k, v in variables.items() if isinstance(v, (int, float))}
                if "var_a" not in local_names and len(local_names) == 1:
                    local_names["var_a"] = list(local_names.values())[0]
                    
                # Tính toán an toàn
                result = simpleeval.simple_eval(
                    expr,
                    names=local_names,
                    functions=self.safe_functions
                )
                return float(result)

        except Exception as e:
            logger.error(f"Failed to evaluate {operation}: {e}")
            return None

        return None
        
    def _resolve_val(self, val: Any, variables: Dict[str, Any]) -> float:
        if isinstance(val, str) and val in variables:
            val = variables[val]
        try:
            return float(val)
        except (TypeError, ValueError):
            return 0.0