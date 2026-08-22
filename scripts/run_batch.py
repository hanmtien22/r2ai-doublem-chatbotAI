import argparse
import json
import logging
import sys
import time
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
    parser.add_argument("--endpoint", type=str, default="https://openrouter.ai/api/v1/chat/completions", help="API endpoint (e.g. OpenRouter or Groq)")
    parser.add_argument("--api-key", type=str, default="EMPTY", help="API Key")
    parser.add_argument("--model", type=str, default="qwen/qwen-2.5-72b-instruct:free", help="Model ID")
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
    jsonl_output = output_path.with_suffix('.jsonl')
    
    # Kiểm tra file jsonl tạm thời
    if jsonl_output.exists():
        with open(jsonl_output, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        res = json.loads(line)
                        if "id" in res:
                            processed_ids.add(res["id"])
                    except:
                        pass
                        
    # Kiểm tra file json chính thức (nếu đã từng chạy xong)
    elif output_path.exists() and output_path.suffix == '.json':
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                for res in existing_data:
                    if "id" in res:
                        processed_ids.add(res["id"])
        except:
            pass
    
    to_process = [q for q in questions if q.get("id") not in processed_ids]
    print(f"Total questions: {len(questions)}. Already processed: {len(processed_ids)}. Remaining: {len(to_process)}")

    # Chạy vòng lặp với thanh tiến độ (Ghi ra file tạm JSONL)
    if to_process:
        with open(jsonl_output, 'a', encoding='utf-8') as out_f:
            for q in tqdm(to_process, desc="Processing"):
                q_id = q.get("id")
                q_text = q.get("question")
                
                try:
                    # Xử lý
                    result = orchestrator.process_question(q_id, q_text)
                    
                    # Ghi ra file ngay lập tức
                    out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    out_f.flush()
                    
                    # Tạm dừng 1 giây giữa các câu hỏi để tránh Rate Limit API (429)
                    time.sleep(1.0)
                    
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

    # Convert file JSONL sang JSON chuẩn ở cuối
    if jsonl_output.exists():
        print("Converting temporary JSONL to final JSON array...")
        results = []
        with open(jsonl_output, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        results.append(json.loads(line))
                    except:
                        pass
                        
        final_output = output_path.with_suffix('.json')
        with open(final_output, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
            
        print(f"\nDone! Final results saved to {final_output}")
    else:
        print(f"\nDone! No new processing needed.")

if __name__ == "__main__":
    main()
