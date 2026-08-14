import argparse
import json
import logging
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.graph.executor import LangGraphExecutor

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Run LangGraph based Execution Plan")
    parser.add_argument("query", type=str, nargs="?", default="Trong năm 2023, giữa FPT và VNM, công ty nào có chi phí bán hàng cao hơn? Hãy tính Hệ số thanh toán hiện hành (Current Ratio) của công ty đó.")
    parser.add_argument("--endpoint", type=str, default="http://localhost:11434/v1", help="API endpoint URL (Ollama/vLLM/OpenAI)")
    parser.add_argument("--api-key", type=str, default="EMPTY", help="API Key (if required)")
    parser.add_argument("--model", type=str, default="qwen2.5", help="Model name to use")
    
    args = parser.parse_args()
    
    executor = LangGraphExecutor(endpoint=args.endpoint, api_key=args.api_key, model=args.model)
    print("\n--- RUNNING GRAPH ---")
    result = executor.run(args.query)
    
    print("\n--- FINAL OUTPUT ---")
    json_output = json.dumps(result, indent=2, ensure_ascii=False)
    # Use standard print but handle unicode on Windows console
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    print(json_output)

if __name__ == "__main__":
    main()
