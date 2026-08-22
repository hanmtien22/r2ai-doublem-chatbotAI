import argparse
import json
import logging
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.orchestrator import QuestionOrchestrator

def setup_logging():
    log_dir = Path('data')
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=str(log_dir / 'batch_run.log'),
        filemode='a'
    )
    # Log errors to console as well
    console = logging.StreamHandler()
    console.setLevel(logging.ERROR)
    logging.getLogger('').addHandler(console)

def main():
    parser = argparse.ArgumentParser(description="Run full dataset batch")
    parser.add_argument("--input", type=str, default="data/easy_questions.json", help="Input questions JSON or JSONL file")
    parser.add_argument("--output", type=str, default="data/submission.jsonl", help="Output submission JSONL file")
    parser.add_argument("--endpoint", type=str, default="https://openrouter.ai/api/v1", help="API endpoint")
    parser.add_argument("--api-key", type=str, default="EMPTY", help="API Key")
    parser.add_argument("--model", type=str, default="qwen/qwen-2.5-7b-instruct", help="Model ID")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory containing tables and indexes")
    
    args = parser.parse_args()
    setup_logging()

    # Khởi tạo Hệ thống
    print("Initializing Orchestrator and loading indexes...")
    orchestrator = QuestionOrchestrator(
        data_dir=args.data_dir,
        endpoint=args.endpoint,
        api_key=args.api_key,
        model=args.model
    )
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    # Đọc danh sách câu hỏi
    questions = []
    if input_path.suffix == '.jsonl':
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    questions.append(json.loads(line))
    else: # .json array
        with open(input_path, 'r', encoding='utf-8') as f:
            questions = json.load(f)

    # Resume cơ chế: Kiểm tra file output đã chạy đến đâu
    processed_ids = set()
    if output_path.exists():
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        res = json.loads(line)
                        if "id" in res:
                            processed_ids.add(res["id"])
                    except:
                        pass
    
    to_process = [q for q in questions if q.get("id") not in processed_ids]
    print(f"Total questions: {len(questions)}. Already processed: {len(processed_ids)}. Remaining: {len(to_process)}")

    # Chạy vòng lặp với thanh tiến độ
    with open(output_path, 'a', encoding='utf-8') as out_f:
        for q in tqdm(to_process, desc="Processing"):
            q_id = q.get("id")
            q_text = q.get("question")
            
            try:
                # Xử lý
                result = orchestrator.process_question(q_id, q_text)
                
                # Ghi ra file ngay lập tức
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_f.flush()
                
            except Exception as e:
                logging.error(f"Error processing question {q_id}: {e}")
                # Ghi tạm lỗi để chạy tiếp
                error_res = {
                    "id": q_id,
                    "question": q_text,
                    "answer": None,
                    "error": str(e)
                }
                out_f.write(json.dumps(error_res, ensure_ascii=False) + "\n")
                out_f.flush()

    print(f"\nDone! Results saved to {args.output}")

if __name__ == "__main__":
    main()
