# Chạy R2AI Financial QA trên Kaggle

## 1. Chuẩn bị dataset

Pipeline hỏi–đáp chỉ cần 3 thứ (**không cần** `data/financial_statements/` 379 MB
và **không cần** `data/parsed_tables/indexes/` 2.4 GB):

| Đưa lên dataset | Kích thước | Vì sao cần |
|---|---|---|
| `data/parsed_tables/retrieval_documents.jsonl` | 428 MB | Toàn bộ số liệu đã parse |
| `data/dictionaries/*.json` | 140 KB | Từ điển công ty, chỉ tiêu, công thức |
| `data/questions/questions.jsonl` | 224 KB | Bộ câu hỏi |

Tổng khoảng **430 MB**. Index BM25 được build lại trên Kaggle trong 2–3 phút
(xem mục 3), nên upload index chỉ tổ tốn thời gian.

Tạo dataset bằng Kaggle CLI:

```bash
mkdir -p /tmp/r2ai-data/{parsed_tables,dictionaries,questions}
cp data/parsed_tables/retrieval_documents.jsonl /tmp/r2ai-data/parsed_tables/
cp data/dictionaries/*.json                      /tmp/r2ai-data/dictionaries/
cp data/questions/questions.jsonl                /tmp/r2ai-data/questions/

cd /tmp/r2ai-data
kaggle datasets init -p .
# sửa dataset-metadata.json: đặt "title" và "id" (dạng "<username>/r2ai-financial-qa")
kaggle datasets create -p . --dir-mode zip
```

Hoặc upload thủ công qua giao diện **Create → New Dataset**.

Cấu trúc thư mục bên trong dataset **không quan trọng** — `src/paths.py` dò theo
tên file (`retrieval_documents.jsonl`, `entity_dictionary.json`, …) chứ không
đoán tên thư mục, vì tên thư mục trong `/kaggle/input/` do người upload đặt.

## 2. Cấu hình notebook

Trong panel bên phải của Kaggle notebook:

- **Add Input** → chọn dataset vừa tạo
- **Settings → Internet: On** (cần để `pip install` và tải model từ HuggingFace)
- **Settings → Accelerator: GPU T4 x2** — chỉ cần nếu dùng `llm_backend='hf'`

Rồi mở `notebooks/kaggle_r2ai.ipynb` và chạy lần lượt.

## 3. Ba chế độ chạy

`build_pipeline(llm_backend=...)`:

| Backend | Cần GPU | Cần Internet | Đặc điểm |
|---|---|---|---|
| `none` | không | không¹ | Chỉ đường tra cứu tất định. Nhanh (~1–2 s/câu), kết quả **lặp lại được** |
| `hf` | có | có | Nạp `Qwen/Qwen2.5-3B-Instruct` bằng transformers, không cần server |
| `ollama` | — | — | Dùng khi máy đã có `ollama serve` (chủ yếu cho máy cá nhân) |
| `auto` | — | — | Dò Ollama → `hf` → `none` |

¹ vẫn cần Internet lần đầu nếu Kaggle thiếu `rank-bm25` / `rapidfuzz`.

Đổi model mà không sửa code:

```python
import os
os.environ["R2AI_LLM_BACKEND"] = "hf"
os.environ["R2AI_LLM_MODEL"]   = "Qwen/Qwen2.5-7B-Instruct"
```

## 4. Về BM25 index

`setup()` tự lo phần này:

- Không tìm thấy `bm25.pkl` → build từ `retrieval_documents.jsonl`, lưu vào
  `/kaggle/working/indexes/` (~560 MB, khoảng 2–3 phút cho 382 k tài liệu).
- Có sẵn `bm25.pkl` → đối chiếu `bm25.meta.json` với số dòng của file documents.
  Lệch nhau thì build lại. Kiểm tra này quan trọng: pipeline đọc tài liệu **từ
  bên trong** file pickle, nên một index cũ sẽ bị dùng im lặng và cho câu trả lời sai.

Muốn giữ index giữa các phiên: Save Version notebook, `/kaggle/working/` sẽ
thành output và có thể add lại làm input cho lần sau.

## 5. Giới hạn của Kaggle cần biết

- Phiên notebook tối đa 9 tiếng (12 tiếng với một số loại) → dùng `--resume` của
  `run_all_questions.py` để chạy tiếp từ chỗ dừng.
- `/kaggle/input/` là **read-only** → mọi thứ ghi ra phải nằm trong `/kaggle/working/`.
- RAM ~30 GB, pipeline dùng khoảng 3–4 GB nên thoải mái.

## 6. Chạy bằng dòng lệnh

```bash
python run_all_questions.py --llm-backend none --no-router \
    --output /kaggle/working/results_all.jsonl --resume
```

Đường dẫn dữ liệu tự dò; chỉ truyền `--documents` / `--index-dir` khi muốn ép cụ thể.

## 7. Kiểm tra nhanh khi có sự cố

```python
from src.paths import resolve_all, is_kaggle
print(is_kaggle()); resolve_all()          # in ra từng đường dẫn tìm được
```

- `documents: KHÔNG TÌM THẤY` → chưa Add Input, hoặc dataset thiếu
  `retrieval_documents.jsonl`. Có thể ép bằng `os.environ["R2AI_DATA_DIR"] = "/kaggle/input/<ten>"`.
- Cài thư viện thất bại → chưa bật Internet.
- `llm_backend='hf'` báo hết VRAM → đổi sang model nhỏ hơn
  (`Qwen/Qwen2.5-1.5B-Instruct`) hoặc dùng `llm_backend='none'`.
