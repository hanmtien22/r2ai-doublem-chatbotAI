import json
import sys
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.qa_pipeline import FullQAPipeline

def safe_float(val: Any) -> float:
    try:
        import math
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (ValueError, TypeError):
        return None

def generate_results(input_jsonl: Path, output_json: Path, documents_path: Path, limit: int = None):
    print(f"Loading pipeline with documents from {documents_path}...")
    pipeline = FullQAPipeline(documents_path=documents_path)
    
    results = []
    
    print(f"Reading questions from {input_jsonl}...")
    with open(input_jsonl, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            if not line.strip():
                continue
            item = json.loads(line)
            q_id = item.get("id")
            question = item.get("question")
            
            print(f"Processing question {q_id}: {question}")
            try:
                res = pipeline.run(question)
                
                computed_val = res.get("computed_result")
                ans_float = safe_float(computed_val)
                
                pandas_query = res.get("code", "")
                hits = res.get("hits", [])
                
                relevant_docs = []
                relevant_tables = []
                evidence = []
                
                for i, hit in enumerate(hits[:10]):  # chỉ lấy 10 hits đầu làm evidence
                    # RetrievalHit.to_dict() trả về {document: {metadata:...}, score:...}
                    doc = hit.get("document", hit)
                    meta = doc.get("metadata", hit.get("metadata", {}))
                    doc_id = meta.get("document_id") or meta.get("report_id") or f"{meta.get('ticker', 'unknown')}_{meta.get('period', meta.get('year', 'unknown'))}"
                    if doc_id not in relevant_docs:
                        relevant_docs.append(doc_id)

                    table_pos = meta.get("table_id") or meta.get("table_type") or f"table_{i}"
                    table_ref = f"{doc_id}|{table_pos}"
                    if table_ref not in relevant_tables:
                        relevant_tables.append(table_ref)

                    csv_path = meta.get("csv_path") or meta.get("file_path", "")
                    evidence.append({
                        "variable": f"df_{i}",
                        "csv_path": csv_path
                    })


                    
                result_entry = {
                    "id": q_id,
                    "question": question,
                    "answer": ans_float,
                    "relevant_docs": relevant_docs,
                    "relevant_tables": relevant_tables,
                    "evidence": evidence,
                    "pandas_query": pandas_query
                }
                results.append(result_entry)
                
            except Exception as e:
                print(f"Error processing question {q_id}: {e}")
                results.append({
                    "id": q_id,
                    "question": question,
                    "answer": None,
                    "relevant_docs": [],
                    "relevant_tables": [],
                    "evidence": [],
                    "pandas_query": ""
                })
                
    print(f"Saving {len(results)} results to {output_json}...")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    print("Done!")

if __name__ == "__main__":
    input_file = Path("data/questions/questions.jsonl")
    output_file = Path("results.json")
    
    docs_path = Path("data/parsed_tables/retrieval_documents.jsonl")
        
    generate_results(input_file, output_file, docs_path, limit=5)
