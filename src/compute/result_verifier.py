import logging
import math
import pandas as pd
from typing import Any, Tuple

logger = logging.getLogger(__name__)

class ResultVerifier:
    """
    Xác thực kết quả thu được từ Sandbox.
    """
    def __init__(self):
        pass

    def verify(self, result: Any) -> Tuple[bool, str]:
        """
        Kiểm tra xem kết quả có hợp lệ không.
        Trả về: (is_valid, error_reason)
        """
        if result is None:
            return False, "Kết quả (final_result) trả về None. Có thể dữ liệu không đủ hoặc code logic sai."

        if isinstance(result, float):
            if math.isnan(result):
                return False, "Kết quả trả về NaN (không có dữ liệu số hợp lệ trong DataFrame)."
            if math.isinf(result):
                return False, "Kết quả trả về Inf (phép chia cho 0 hoặc tràn số)."

        if isinstance(result, (pd.DataFrame, pd.Series)):
            if result.empty:
                return False, "Kết quả trả về DataFrame/Series rỗng."

        return True, ""
