
from __future__ import annotations

import io
import logging
import multiprocessing
import pandas as pd
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 10


def _exec_worker(
    code: str,
    df_records: Dict[str, Any],
    result_queue: multiprocessing.Queue,
) -> None:
    import io, contextlib, math
    import pandas as pd
    import numpy as np

    # Code do LLM sinh hay mở đầu bằng "import pandas as pd"; nếu chặn hết
    # __import__ thì mọi lần sinh như vậy đều chết ở dòng đầu tiên.
    _ALLOWED_MODULES = {"pandas", "numpy", "math", "statistics", "decimal", "datetime", "re"}

    def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        root = name.split(".")[0]
        if root not in _ALLOWED_MODULES:
            raise ImportError(f"Module '{name}' không được phép trong sandbox")
        return __import__(name, globals, locals, fromlist, level)

    env_globals: Dict[str, Any] = {
        "__builtins__": {
            "abs": abs, "round": round, "len": len, "range": range,
            "list": list, "dict": dict, "tuple": tuple, "set": set,
            "int": int, "float": float, "str": str, "bool": bool,
            "min": min, "max": max, "sum": sum, "sorted": sorted,
            "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
            "isinstance": isinstance, "type": type, "print": print,
            "None": None, "True": True, "False": False,
            "any": any, "all": all, "abs": abs, "divmod": divmod,
            "__import__": _safe_import,
        },
        "pd": pd,
        "np": np,
        "math": math,
    }

    env_locals: Dict[str, Any] = {}
    for df_name, df_info in df_records.items():
        try:
            env_locals[df_name] = pd.DataFrame(
                df_info["data"], columns=df_info["columns"]
            ) if df_info["data"] else pd.DataFrame(columns=df_info.get("columns", []))
        except Exception:
            env_locals[df_name] = pd.DataFrame()

    env_locals["final_result"] = None
    captured = io.StringIO()

    try:
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            exec(code, env_globals, env_locals)  # noqa: S102
        result_queue.put(("ok", env_locals.get("final_result"), captured.getvalue()))
    except Exception as e:
        result_queue.put(("error", None, captured.getvalue() + "\n" + str(e)))


class Sandbox:
    def __init__(self, timeout: float = _DEFAULT_TIMEOUT):
        self.timeout = timeout

    def execute(self, code: str, dfs: Dict[str, pd.DataFrame]) -> Tuple[bool, Any, str]:
        """Trả về: (success, result, error_or_stdout)"""
        if not code or not code.strip():
            return False, None, "Empty code"

        df_records: Dict[str, Any] = {}
        for name, df in dfs.items():
            try:
                df_records[name] = {
                    "columns": list(df.columns),
                    "data": df.where(pd.notnull(df), None).values.tolist(),
                }
            except Exception as e:
                logger.warning("Cannot serialize df '%s': %s", name, e)
                df_records[name] = {"columns": [], "data": []}

        result_queue: multiprocessing.Queue = multiprocessing.Queue()
        proc = multiprocessing.Process(
            target=_exec_worker,
            args=(code, df_records, result_queue),
            daemon=True,
        )

        try:
            proc.start()
            proc.join(timeout=self.timeout)

            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=2)
                if proc.is_alive():
                    proc.kill()
                msg = f"Timeout: code execution exceeded {self.timeout}s"
                logger.warning(msg)
                return False, None, msg

            if result_queue.empty():
                return False, None, "Process exited without result"

            status, result, output = result_queue.get_nowait()
            if status == "ok":
                return True, result, output
            logger.warning("Sandbox exec error: %s", output[:200])
            return False, None, output

        except Exception as e:
            logger.error("Sandbox error: %s", e)
            if proc.is_alive():
                proc.terminate()
            return False, None, str(e)
