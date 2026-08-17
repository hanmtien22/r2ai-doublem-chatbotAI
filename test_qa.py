import sys
from pathlib import Path
from src.qa_pipeline import FullQAPipeline
from src.config_loader import configure_logging, load_config
import logging

logging.basicConfig(level=logging.INFO)

def main():
    print("Testing FullQAPipeline...")
    
    # We will use dummy values for documents_path since we are mocking LLM anyway if no model is loaded
    # Create a dummy jsonl file
    dummy_doc_path = Path("dummy_docs.jsonl")
    if not dummy_doc_path.exists():
        with open(dummy_doc_path, "w") as f:
            f.write('{"id": "1", "content": "VNM Doanh thu 2024 la 1000 ty dong", "metadata": {"ticker": "VNM", "year": 2024}}\n')
            
    try:
        pipeline = FullQAPipeline(documents_path=dummy_doc_path)
        
        question = "Doanh thu VNM năm 2024 là bao nhiêu?"
        print(f"\nQuestion: {question}")
        
        result = pipeline.run(question)
        
        # Nếu pipeline mặc định không ra hits, thử test trực tiếp Bước 3 và 4
        print("\n--- TEST TRỰC TIẾP PHASE 3 & 4 (COMPUTE + ANSWER) ---")
        mock_hits = [
            {
                "content": "Doanh thu năm 2024 của VNM là 12000 tỷ đồng", 
                "metadata": {"ticker": "VNM", "year": 2024, "document_type": "BCTC", "page": 10, "table_name": "KQKD"}
            }
        ]
        tables = [{"ticker": "VNM", "columns": ["content"], "data": [["Doanh thu năm 2024 của VNM là 12000 tỷ đồng"]]}]
        is_success, computed_result, code, error_msg = pipeline.compute_manager.compute(question, tables)
        if is_success:
            ans = pipeline.answer_formatter.format_answer(question, computed_result)
            cites = pipeline.citation_builder.build_citation(mock_hits)
            print("\nKết quả tính toán:")
            print(f"Code sinh ra:\n{code}")
            print(f"Giá trị cuối: {computed_result}")
            print(f"Câu trả lời:\n{ans}")
            print(f"Trích dẫn:\n{cites}")
        else:
            print("Lỗi tính toán:", error_msg)
        
        print("\n=== KẾT QUẢ TỪ PIPELINE CHÍNH ===")
        import json
        # Remove functions or un-jsonable items if any, but result should be a dict of strings/bools
        print(json.dumps({k: v for k, v in result.items() if isinstance(v, (str, bool, int, float, list, dict))}, ensure_ascii=False, indent=2))
        
    except Exception as e:
        print(f"Error during testing: {e}")

if __name__ == "__main__":
    main()
