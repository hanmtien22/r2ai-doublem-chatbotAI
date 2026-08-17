import logging
logging.basicConfig(level=logging.INFO)
from src.llm.client import LLMClient
from src.compute.code_generator import CodeGenerator

llm_client = LLMClient(model_name="qwen2.5:3b")
cg = CodeGenerator(llm_client)

q = "Lãi tiền gửi năm 2018 của công ty mẹ CTCP Hàng không Vietjet (VJC) là bao nhiêu triệu đồng?"
tables = [
    {
        "ticker": "VJC",
        "columns": ["Chỉ tiêu", "Năm 2018"],
        "data": [["Lãi tiền gửi", 1234]]
    }
]

raw = cg.llm_client.generate(cg.prompt_template.format(question=q, schemas="df_0 with columns ['Chỉ tiêu', 'Năm 2018']"), max_tokens=300)
print("=== RAW ===")
print(raw)
print("=== EXTRACTED ===")
print(cg.generate_code(q, tables))
