import io
import sys
import contextlib
import logging
import pandas as pd
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)


class Sandbox:
    """
    Môi trường thực thi mã Python an toàn (cơ bản).
    Chạy mã do LLM sinh ra để xử lý các bảng dữ liệu Pandas và trả về kết quả.
    """
    def __init__(self):
        pass

    def execute(self, code: str, dfs: Dict[str, pd.DataFrame]) -> Tuple[bool, Any, str]:
        """
        Thực thi đoạn mã.
        Trả về: (success, result, error_message_or_stdout)
        """
        # Chuẩn bị môi trường (chỉ cho phép pandas và các DataFrames)
        env_globals = {
            "__builtins__": __builtins__,
            "pd": pd,
        }
        env_locals = dfs.copy()
        env_locals["final_result"] = None

        output_capture = io.StringIO()
        try:
            with contextlib.redirect_stdout(output_capture), contextlib.redirect_stderr(output_capture):
                # Thực thi mã trong môi trường cô lập cơ bản
                exec(code, env_globals, env_locals)
            
            result = env_locals.get("final_result")
            stdout = output_capture.getvalue()
            return True, result, stdout
            
        except Exception as e:
            error_trace = output_capture.getvalue() + "\n" + str(e)
            logger.warning(f"Lỗi khi chạy code trong Sandbox: {e}")
            return False, None, error_trace
