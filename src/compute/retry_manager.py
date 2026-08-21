import logging
import pandas as pd
from typing import Dict, Any, Tuple, List

from src.compute.code_generator import CodeGenerator
from src.compute.sandbox import Sandbox
from src.compute.result_verifier import ResultVerifier

logger = logging.getLogger(__name__)


class RetryManager:
    def __init__(self, code_gen: CodeGenerator, sandbox: Sandbox, verifier: ResultVerifier):
        self.code_gen = code_gen
        self.sandbox = sandbox
        self.verifier = verifier
        # Mỗi lần sinh lại code là một lượt LLM sinh vài trăm token (~20s).
        # 3x2 lượt khiến câu hỏi thất bại mất hơn 4 phút mà hiếm khi cứu được kết quả.
        self.max_tier1_retries = 2  # runtime errors
        self.max_tier2_retries = 1  # logic errors

    def _build_df_context(self, dfs: Dict[str, pd.DataFrame]) -> str:
        """Tóm tắt dtypes + 3 dòng đầu của mỗi df để nhúng vào feedback."""
        lines = []
        for df_name, df in dfs.items():
            lines.append(f"{df_name} (shape={df.shape}):")
            lines.append(f"  dtypes: {df.dtypes.to_dict()}")
            try:
                lines.append(f"  first 3 rows: {df.head(3).to_dict(orient='records')}")
            except Exception:
                pass
            lines.append("")
        return "\n".join(lines)[:800]

    def compute(
        self, question: str, retrieved_tables: List[Dict[str, Any]]
    ) -> Tuple[bool, Any, str, str]:
        """Trả về: (success, result, code, error_message)."""
        dfs = {}
        for i, table in enumerate(retrieved_tables):
            data = table.get("data", [])
            columns = table.get("columns", [])
            dfs[f"df_{i}"] = pd.DataFrame(data, columns=columns) if data and columns else pd.DataFrame()

        df_context = self._build_df_context(dfs)
        logic_attempts = 0
        feedback = ""
        code = ""

        while logic_attempts <= self.max_tier2_retries:
            runtime_attempts = 0
            code = self.code_gen.generate_code(question, retrieved_tables, feedback)

            while runtime_attempts <= self.max_tier1_retries:
                logger.info(
                    "Chạy code (Tier2: %d/%d, Tier1: %d/%d)",
                    logic_attempts, self.max_tier2_retries,
                    runtime_attempts, self.max_tier1_retries,
                )
                is_success, result, exec_output = self.sandbox.execute(code, dfs)

                if is_success:
                    is_valid, verify_err = self.verifier.verify(result)
                    if is_valid:
                        logger.info("Tính toán thành công và xác thực hợp lệ.")
                        return True, result, code, ""
                    logger.warning("Xác thực thất bại: %s", verify_err)
                    feedback = (
                        f"Code chạy thành công nhưng kết quả bị lỗi logic: {verify_err}\n"
                        f"Output console: {exec_output}\n\n"
                        f"Thông tin DataFrame:\n{df_context}"
                    )
                    break
                else:
                    logger.warning("Lỗi runtime: %s", exec_output)
                    feedback = f"Lỗi khi chạy mã:\n{exec_output[:600]}\n\nThông tin DataFrame:\n{df_context}"
                    runtime_attempts += 1
                    if runtime_attempts <= self.max_tier1_retries:
                        code = self.code_gen.generate_code(question, retrieved_tables, feedback)

            logic_attempts += 1

        logger.error("Hết số lần thử lại. Tính toán thất bại.")
        return False, None, code, feedback
