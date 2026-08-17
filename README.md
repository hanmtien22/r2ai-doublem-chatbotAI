# R2AI Financial QA

Pipeline xử lý báo cáo tài chính và truy xuất số liệu từ câu hỏi tiếng Việt.
Phạm vi hiện tại bao gồm ingestion, indexing, query understanding và
retrieval/validation. Các phase tính toán, xác minh và sinh câu trả lời cuối
trong `plan.md` chưa được triển khai.

## Luồng hoạt động

```text
code_stock.csv
    └─> Entity Dictionary (ticker, tên công ty, alias rút gọn)

financial_statements/**/*_extracted.txt
    ├─> Phát hiện bảng BS / IS / CF / EQ
    │   └─> Parse strict, chuẩn hóa, validate ─> tables/*.csv
    └─> Loại primary table, nhận diện section notes
        └─> Chunk theo section ─> notes/*_notes.csv
            └─> records/retrieval documents + BM25/FAISS

Câu hỏi tiếng Việt
    └─> Chuẩn hóa Unicode, viết tắt, typo và năm tương đối
        └─> Exact + fuzzy entity resolution
            └─> Query Router
                ├─> single_lookup
                ├─> multi_comparison
                ├─> derived_indicator + Formula Library
                └─> out_of_scope
                    └─> Structured RetrievalQuery
                        ├─> Exact match
                        └─> BM25 / FAISS fallback
                            └─> Rank fusion / reranker tùy chọn
                                └─> Schema validation, confidence, re-query
                                    └─> Query metadata + nguồn số liệu
```

Retriever ưu tiên khớp chính xác theo:

```text
ticker + period + section + item_code
```

Nếu không có exact match, hệ thống tìm kiếm bằng BM25 và FAISS, hợp nhất thứ
hạng, lọc theo schema/metadata và reformulate query khi confidence thấp.

## Cài đặt

Chạy từ thư mục gốc dự án:

```bash
uv venv
uv pip install --python .venv/bin/python -r requirements.txt
```

Không cần `source .venv/bin/activate`; các lệnh dưới đây gọi Python trong
`.venv` trực tiếp.

## Chạy từ đầu đến cuối

### 1. Chạy thử ingestion trên một báo cáo

```bash
.venv/bin/python -m src.pipeline \
  --config configs/config.yaml \
  --source-dir data/financial_statements/AAA/2025/AAA_financial_statements_2025_consolidated \
  --output-dir /tmp/r2ai-aaa-2025
```

Kết quả kiểm chứng hiện tại cho báo cáo này:

```json
{
  "file_count": 1,
  "record_count": 272,
  "document_count": 272,
  "error_count": 0
}
```

### 2. Query dữ liệu vừa tạo

```bash
.venv/bin/python scripts/run_query.py \
  --config configs/config.yaml \
  --documents /tmp/r2ai-aaa-2025/retrieval_documents.jsonl \
  --reference-year 2025 \
  "ROE cua AAA nam 2025"
```

CLI tự tìm các artifact liên quan tại output và `data/dictionaries`:

```text
data/dictionaries/entity_dictionary.json
/tmp/r2ai-aaa-2025/indexes/bm25.pkl
/tmp/r2ai-aaa-2025/indexes/faiss/index.faiss  # nếu đã bật dense
```

Đầu ra JSON gồm:

- `query`: câu hỏi đã chuẩn hóa, entity, loại query, công thức và các
  `retrieval_queries`.
- `hits`: score, nguồn tài liệu và metadata của từng số liệu tìm thấy.

Ví dụ ROE sinh hai yêu cầu:

```text
IS.60  → Lợi nhuận sau thuế
BS.400 → Vốn chủ sở hữu
```

### 3. Chạy ingestion toàn bộ dữ liệu

```bash
uv run -m src.pipeline --config configs/config.yaml
```

Theo cấu hình mặc định, đầu vào là `data/financial_statements` và đầu ra là
`data/parsed_tables`.

### 4. Query dữ liệu toàn bộ

```bash
.venv/bin/python scripts/run_query.py \
  --config configs/config.yaml \
  --documents data/parsed_tables/retrieval_documents.jsonl \
  "So sanh doanh thu thuan cua AAA va HPG nam 2025"
```

Có thể ghi đè tham số retrieval:

```bash
.venv/bin/python scripts/run_query.py \
  --documents data/parsed_tables/retrieval_documents.jsonl \
  --index-dir data/parsed_tables/indexes \
  --reference-year 2025 \
  --top-k 10 \
  "Doanh thu thuan cua AAA nam 2025"
```

## Cấu trúc đầu ra ingestion

```text
data/parsed_tables/
├── tables/                       # BS/IS/CF/EQ dạng CSV UTF-8 BOM
├── notes/                        # Notes/disclosures chunk theo section dạng CSV
├── indexes/
│   ├── bm25.pkl                 # Sparse index, bật mặc định
│   └── faiss/
│       ├── index.faiss          # Dense index khi dense_enabled=true
│       └── documents.json
├── records.jsonl                # Các bản ghi đã chuẩn hóa
├── retrieval_documents.jsonl    # Document dùng cho retrieval
├── ingestion_errors.jsonl       # Lỗi và cảnh báo validation
└── dictionary_report.json

data/dictionaries/
├── entity_dictionary.json
├── indicator_aliases.json
├── schema_mapping.json
├── abbreviations.json
└── formula_library.json
```

`error_count` có thể bao gồm cảnh báo validation và không luôn đồng nghĩa với
việc pipeline bị dừng.

## Cấu hình

Toàn bộ ingestion và query dùng chung `configs/config.yaml`.

Các section chính:

| Section | Chức năng |
|---|---|
| `parser` | Ngưỡng kiểm tra bảng |
| `notes` | Bật/tắt notes, thư mục và kích thước chunk |
| `dictionaries` | Thư mục dictionary dùng chung |
| `entity_dictionary` | Nguồn CSV và việc rebuild dictionary |
| `indexing` | Bật/tắt BM25, dense index và thư mục artifact |
| `embedding` | Model, batch size và normalization |
| `fuzzy_match` | Ngưỡng fuzzy company/indicator |
| `router` | LLM fallback và confidence |
| `retrieval` | Dense search, reranker, RRF, top-k và confidence |
| `llm` | Backend/model LLM tùy chọn |
| `cache` | Query cache |
| `defaults` | Năm tham chiếu và mặc định nghiệp vụ |

### Bật embedding và FAISS

Dense retrieval mặc định tắt để lần chạy đầu không tự tải model lớn. Bật cả
hai cờ sau:

```yaml
indexing:
  dense_enabled: true

retrieval:
  dense_enabled: true
```

Model được lấy từ:

```yaml
embedding:
  model_name: bkai-foundation-models/vietnamese-bi-encoder
```

### Bật reranker

```yaml
retrieval:
  reranker_model_name: cross-encoder/ms-marco-MiniLM-L-6-v2
```

### Bật LLM Router fallback

```yaml
router:
  use_llm_fallback: true

llm:
  enabled: true
  model_name: Qwen2.5-14B-Instruct
  model_path: null
```

Nếu không có `ollama` hoặc `llama_cpp`, `LLMClient` dùng mock backend; các rule
deterministic vẫn là đường phân loại chính.

## Chạy kiểm thử

```bash
PYTHONDONTWRITEBYTECODE=1 \
.venv/bin/python -m pytest -p no:cacheprovider
```

Kết quả hiện tại:

```text
95 passed
```
## Luồng hoạt động từ Phase1 đến Phase2

```text
[ Câu hỏi từ User / Batch ]
          |
          v
[ Question Orchestrator ] ---> Quyết định: DỄ hay KHÓ? (Dựa trên Query Router)
          |
          +---> [ DỄ (Single Lookup) ]
          |        |
          |        +-> Gọi thẳng tới [ EasyHybridSolver ]
          |               - Lọc chính xác bằng Ticker, Period, Item Code
          |               - Fallback: Keyword Match hoặc FAISS Vector Search
          |        +-> Trả kết quả (Value + Evidence)
          |
          +---> [ KHÓ (So sánh, Tính toán - LangGraph) ]
                   |
                   +-> [ Node Planner ] 
                   |      Gửi câu hỏi lên OpenRouter / Groq / HuggingFace API
                   |      sinh ra Bản thiết kế JSON (Execution Plan).
                   |
                   +-> [ Node Executor (Vòng lặp) ]
                   |      - action: fetch_data -> Gọi lại [ EasyHybridSolver ]
                   |      - action: evaluate   -> Gọi [ PythonEvaluator ] (Toán học)
                   |
                   +-> [ Node Formatter ]
                          Gom kết quả -> Xuất định dạng JSON submit.
```

## Yêu cầu hệ thống và Cài đặt

Cài đặt thư viện Python (yêu cầu Python 3.10+):

```bash
uv venv
uv pip install --python .venv/bin/python -r requirements.txt
uv pip install simpleeval tqdm huggingface_hub langgraph langchain-core pydantic
```

*(Lưu ý: Nếu không sử dụng `uv`, bạn có thể dùng `pip install -r requirements.txt` tiêu chuẩn).*

## Chuẩn bị Dữ liệu 

Hệ thống yêu cầu các tệp dữ liệu phải được Ingest đầy đủ từ Phase 0 và đặt đúng vị trí trong thư mục `data/`:

1.  **Thư mục chứa Bảng đã parse (CSV):** `data/tables/*.csv`
2.  **Bộ Index tìm kiếm (Vector & BM25):**
    *   `data/indexes/bm25.pkl` (Dành cho Keyword Search)
    *   `data/indexes/faiss/index.faiss` (Lưu trữ Vector Dense)
    *   `data/indexes/faiss/documents.json` (Ánh xạ Metadata cho Vector)
3.  **Tệp câu hỏi:** `data/questions.jsonl`

## Chạy Hệ thống 

Sử dụng script `run_batch.py` để xử lý hàng loạt 10,000 câu hỏi từ tệp `questions.jsonl`.
Hệ thống có cơ chế **Resume**, nếu bị ngắt giữa chừng (do sập mạng/hết RAM), lần chạy sau sẽ tự động đọc file output và tiếp tục xử lý từ câu bị lỗi.

### Tùy chọn 1: Chạy bằng OpenRouter 
Đây là cách tốt nhất vì OpenRouter cung cấp các model open-source mạnh mẽ với mức giá rất rẻ hoặc miễn phí.
```bash
python scripts/run_batch.py \
  --input data/questions.jsonl \
  --output data/submission.jsonl \
  --endpoint "https://openrouter.ai/api/v1" \
  --api-key "sk-or-v1-YOUR-KEY" \
  --model "qwen/qwen-2.5-7b-instruct"
```
#### Chạy trên terminal: 
python scripts/run_batch.py --input data/questions.jsonl --output data/submission.jsonl --endpoint "https://openrouter.ai/api/v1" --api-key "your_api_key" --model "qwen/qwen-2.5-7b-instruct"

### Tùy chọn 2: Chạy bằng Groq 
```bash
python scripts/run_batch.py \
  --input data/questions.jsonl \
  --output data/submission.jsonl \
  --endpoint "https://api.groq.com/openai/v1" \
  --api-key "gsk_YOUR-KEY" \
  --model "llama-3.1-8b-instant"
```
#### Chạy trên terminal: 
python scripts/run_batch.py --input data/questions.jsonl --output data/submission.jsonl --endpoint "https://api.groq.com/openai/v1" --api-key "gsk_KEY_CỦA_BẠN_Ở_ĐÂY" --model "llama-3.1-8b-instant"
### Tùy chọn 3: Chạy Local bằng vLLM / Ollama 
Nếu máy bạn có GPU đủ mạnh (VRAM >= 12GB), bạn có thể tải model về máy qua Ollama và chạy 100% offline, không tốn phí API:
1. Mở Ollama và chạy model: `ollama run qwen2.5:7b-instruct`
2. Chạy pipeline chỉ định đến cổng của Ollama (mặc định là 11434):
```bash
python scripts/run_batch.py \
  --input data/questions.jsonl \
  --output data/submission.jsonl \
  --endpoint "http://localhost:11434/v1" \
  --api-key "EMPTY" \
  --model "qwen2.5:7b-instruct"
```
#### Chạy trên terminal: 
python scripts/run_batch.py --input data/questions.jsonl --output data/submission.jsonl --endpoint "http://localhost:11434/v1" --api-key "EMPTY" --model "qwen2.5:7b-instruct"
### Các tham số tùy chỉnh:
*   `--input`: Đường dẫn file câu hỏi (Hỗ trợ `.jsonl` hoặc `.json`).
*   `--output`: File kết quả (Đúng chuẩn format của Ban tổ chức, trường `pandas_query` để trống `""`).
*   `--data-dir`: Thư mục gốc chứa dữ liệu `tables` và `indexes` (Mặc định là `data`).

## Cấu trúc Đầu ra (JSONL Format)

Mỗi dòng trong file `submission.jsonl` sẽ có cấu trúc như sau:

```json
{
  "id": 16,
  "question": "Số dư vay ngắn hạn của công ty mẹ CEO cuối năm 2025 là bao nhiêu tỷ đồng?",
  "answer": 30907778008628,
  "relevant_docs": [
    "CEO_2025_consolidated"
  ],
  "relevant_tables": [
    "CEO_2025_consolidated|0"
  ],
  "evidence": [
    {
      "variable": "df_CEO_2025",
      "csv_path": "data/tables/CEO_2025_consolidated.csv"
    }
  ],
  "pandas_query": ""
}
```

*Ghi chú: Giá trị sau dấu `|` trong `relevant_tables` là vị trí `start_line` trích xuất từ dữ liệu thực tế. Nếu dữ liệu không cung cấp `start_line`, hệ thống mặc định gán là `0`.*

## Kiến trúc Mã Nguồn Mới

*   `src/orchestrator.py`: Điều phối luồng xử lý câu hỏi (Easy -> Hybrid Fetcher, Hard -> LangGraph).
*   `src/query/hybrid_fetcher.py`: Engine tìm kiếm đa năng. Lọc chính xác bằng Pandas DataFrame + Cứu cánh bằng Text Search (FAISS & BM25) + Trọng tài `rapidfuzz` chống trùng lặp dữ liệu. Tự động tìm trong Text Chunks (Thuyết minh) nếu không có bảng.
*   `src/query/evaluator.py`: Máy tính toán học độc lập, thực thi các phép toán tài chính thông qua thư viện an toàn `simpleeval`.
*   `src/graph/executor.py`: Quản lý quy trình Agent (Lập kế hoạch, Vòng lặp lấy số, Tính toán, Trả kết quả).
*   `src/llm/tgi_client.py`: API Client có hỗ trợ tự động Retry Exponential Backoff khi bị Rate Limit (Lỗi 429). Mặc dù tên là tgi_client, class bên trong `GenericLLMClient` hỗ trợ chuẩn gọi API OpenAI (OpenRouter, Groq, vLLM, Ollama).

### Xem trợ giúp CLI:

```bash
.venv/bin/python -m src.pipeline --help
.venv/bin/python scripts/run_query.py --help
```
