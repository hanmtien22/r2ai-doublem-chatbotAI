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

Nếu không có `vllm` hoặc `llama_cpp`, `LLMClient` dùng mock backend; các rule
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

Xem trợ giúp CLI:

```bash
.venv/bin/python -m src.pipeline --help
.venv/bin/python scripts/run_query.py --help
```
