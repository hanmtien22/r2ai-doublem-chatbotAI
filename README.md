# r2ai-doublem-chatbotAI

## Chạy ingestion pipeline

Chạy các lệnh sau từ thư mục gốc của dự án.

### Chạy toàn bộ dữ liệu

Pipeline mặc định đọc cấu hình từ `configs/config.yaml`:

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
uv run src/pipeline.py
```

### Chạy thử một thư mục dữ liệu

Có thể ghi đè thư mục đầu vào và đầu ra mà không cần sửa YAML:

```bash
uv run src/pipeline.py \
  --source-dir data/financial_statements/AAA/2025/AAA_financial_statements_2025_consolidated \
  --output-dir /tmp/r2ai-pipeline-test
```

### Dùng file cấu hình khác

```bash
uv run src/pipeline.py --config configs/config.yaml
```

Xem tất cả tham số CLI:

```bash
uv run src/pipeline.py --help
```

### Kết quả đầu ra

Với cấu hình mặc định, kết quả được ghi vào `data/parsed_tables`:

```text
data/parsed_tables/
├── tables/                       # Các bảng Parquet đã chuẩn hóa
├── records.jsonl                 # Các dòng dữ liệu đã parse
├── retrieval_documents.jsonl     # Dữ liệu phục vụ retrieval
└── ingestion_errors.jsonl         # Cảnh báo validation và lỗi ingestion
```

Sau khi chạy xong, CLI in ra thống kê:

```json
{
  "file_count": 1,
  "record_count": 458,
  "document_count": 458,
  "error_count": 29
}
```

`error_count` bao gồm cả cảnh báo validation; giá trị lớn hơn `0` không nhất thiết có nghĩa pipeline đã bị dừng.
