"""
Batch runner: chạy toàn bộ file questions.jsonl qua FullQAPipeline.

Cách dùng:
    python run_all_questions.py                        # chạy hết
    python run_all_questions.py --limit 50             # chỉ chạy 50 câu đầu
    python run_all_questions.py --resume               # tiếp tục từ câu bị dừng
    python run_all_questions.py --output my_results.jsonl
"""

import argparse
import json
import logging
import time
from pathlib import Path

# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,  # Tắt spam INFO từ pipeline để dễ đọc progress
    format="%(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("batch_runner")
logger.setLevel(logging.INFO)

# ─── Paths ─────────────────────────────────────────────────────────────────────────────
# Dò cả /kaggle/input lẫn ./data để cùng một lệnh chạy được ở local và Kaggle.
from src.paths import find_documents, find_index_dir, find_questions, writable_dir

ROOT = Path(__file__).parent
QUESTIONS_PATH = find_questions() or ROOT / "data" / "questions" / "questions.jsonl"
DOCUMENTS_PATH = find_documents() or ROOT / "data" / "parsed_tables" / "retrieval_documents.jsonl"
INDEX_DIR = find_index_dir() or ROOT / "data" / "parsed_tables" / "indexes"
OUTPUT_PATH = writable_dir() / "results_all.jsonl"


def parse_args():
    p = argparse.ArgumentParser(description="Run all questions through FullQAPipeline")
    p.add_argument("--questions", default=str(QUESTIONS_PATH), help="Đường dẫn file questions.jsonl")
    p.add_argument("--documents", default=str(DOCUMENTS_PATH), help="Đường dẫn file retrieval_documents.jsonl")
    p.add_argument("--index-dir", default=str(INDEX_DIR), help="Thư mục chứa bm25.pkl")
    p.add_argument("--output", default=str(OUTPUT_PATH), help="File lưu kết quả (JSONL)")
    p.add_argument("--limit", type=int, default=None, help="Chỉ chạy N câu đầu (debug)")
    p.add_argument("--resume", action="store_true", help="Bỏ qua các câu đã có trong output file")
    p.add_argument("--model", default=None, help="Tên model LLM (mặc định theo backend)")
    p.add_argument("--llm-backend", default="auto", choices=["auto", "ollama", "hf", "none"],
                   help="auto=tự dò | ollama=server local | hf=transformers trong tiến trình | none=chỉ tra cứu tất định")
    p.add_argument("--host", default="http://localhost:11434", help="Ollama host")
    p.add_argument("--timeout", type=float, default=10.0, help="Sandbox timeout (giây)")
    p.add_argument("--no-router", action="store_true", help="Tắt LLM router")
    return p.parse_args()


def load_done_ids(output_path: Path) -> set[int]:
    """Đọc các id câu đã chạy xong từ output file (để resume)."""
    done = set()
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if "id" in obj:
                        done.add(obj["id"])
                except json.JSONDecodeError:
                    pass
    return done


def load_questions(questions_path: Path, limit: int | None = None) -> list[dict]:
    questions = []
    with open(questions_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            questions.append(json.loads(line))
    return questions


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    return f"{m}m{s:02d}s"


def print_progress(idx: int, total: int, successes: int, failures: int, elapsed: float):
    pct = (idx / total) * 100
    rate = idx / elapsed if elapsed > 0 else 0
    remaining = (total - idx) / rate if rate > 0 else 0
    bar_len = 30
    filled = int(bar_len * idx / total)
    bar = "█" * filled + "░" * (bar_len - filled)
    print(
        f"\r[{bar}] {pct:5.1f}%  {idx}/{total}  "
        f"✓{successes} ✗{failures}  "
        f"~{rate:.2f} q/s  ETA: {format_duration(remaining)}",
        end="",
        flush=True,
    )


def main():
    args = parse_args()

    questions_path = Path(args.questions)
    documents_path = Path(args.documents)
    index_dir = Path(args.index_dir)
    output_path = Path(args.output)

    if not questions_path.exists():
        logger.error(f"Không tìm thấy file câu hỏi: {questions_path}")
        return

    if not documents_path.exists():
        logger.error(f"Không tìm thấy documents: {documents_path}")
        return

    # Load questions
    logger.info(f"Đang load câu hỏi từ {questions_path}...")
    all_questions = load_questions(questions_path, limit=args.limit)
    logger.info(f"Tổng số câu hỏi: {len(all_questions)}")

    # Resume: lọc bỏ câu đã chạy
    done_ids: set[int] = set()
    if args.resume and output_path.exists():
        done_ids = load_done_ids(output_path)
        logger.info(f"Resume mode: đã có {len(done_ids)} câu, bỏ qua.")

    todo = [q for q in all_questions if q.get("id") not in done_ids]
    logger.info(f"Số câu cần chạy: {len(todo)}")

    if not todo:
        logger.info("Tất cả câu đã chạy xong!")
        return

    # Khởi tạo pipeline (1 lần duy nhất)
    logger.info("Đang khởi tạo FullQAPipeline...")
    try:
        from src.qa_pipeline import FullQAPipeline
        pipeline = FullQAPipeline(
            documents_path=documents_path,
            index_dir=index_dir,
            llm_model=args.model,
            ollama_host=args.host,
            llm_backend=args.llm_backend,
            use_llm_router=not args.no_router,
            sandbox_timeout=args.timeout,
        )
    except Exception as e:
        logger.error(f"Không thể khởi tạo pipeline: {e}")
        raise

    logger.info("Pipeline sẵn sàng. Bắt đầu chạy...\n")

    total = len(todo)
    successes = 0
    failures = 0
    start_time = time.time()

    # Mở output file ở chế độ append
    with open(output_path, "a", encoding="utf-8") as out_f:
        for idx, item in enumerate(todo, start=1):
            q_id = item.get("id")
            question = item.get("question", "")

            t0 = time.time()
            try:
                result = pipeline.run(question)
                duration = time.time() - t0

                record = {
                    "id": q_id,
                    "question": question,
                    "answer": result.get("answer", ""),
                    "citations": result.get("citations", ""),
                    "success": result.get("success", False),
                    "computed_result": result.get("computed_result"),
                    "duration_s": round(duration, 3),
                }

                if result.get("success"):
                    successes += 1
                else:
                    failures += 1
                    record["error"] = result.get("error", "")

            except Exception as e:
                duration = time.time() - t0
                failures += 1
                record = {
                    "id": q_id,
                    "question": question,
                    "answer": "",
                    "citations": "",
                    "success": False,
                    "error": str(e),
                    "duration_s": round(duration, 3),
                }
                logger.debug(f"Exception câu {q_id}: {e}")

            # Ghi ngay sau mỗi câu (safe nếu bị kill)
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()

            elapsed = time.time() - start_time
            print_progress(idx, total, successes, failures, elapsed)

    # Summary
    total_time = time.time() - start_time
    print()  # newline sau progress bar
    print("\n" + "=" * 60)
    print(f"  ✅ Thành công : {successes}/{total}")
    print(f"  ❌ Thất bại  : {failures}/{total}")
    print(f"  ⏱  Tổng thời gian : {format_duration(total_time)}")
    print(f"  ⚡ Tốc độ TB : {total / total_time:.2f} câu/giây")
    print(f"  📄 Kết quả lưu tại: {output_path.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
