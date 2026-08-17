import pandas as pd
import pickle
import faiss
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import json
import re

from src.utils.text import remove_diacritics

logger = logging.getLogger(__name__)

class EasyHybridSolver:
    """Công cụ tìm kiếm bốc số từ CSV (Exact Match) và Fallback bằng BM25 / FAISS Vector Search."""
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.tables_dir = self.data_dir / "tables"
        self.parsed_tables_dir = self.data_dir / "parsed_tables" / "tables"
        self.indexes_dir = self.data_dir / "indexes"
        self.parsed_indexes_dir = self.data_dir / "parsed_tables" / "indexes"
        
        # Biến chứa dữ liệu trong RAM
        self.bm25_index = None
        self.faiss_index = None
        self.documents_map: List[dict] = [] # Lưu metadata của các vector từ documents.json
        self.embed_model = None
        self._load_indexes()

    def _get_active_tables_dir(self) -> Path:
        """Tìm thư mục chứa các bảng csv đã parse."""
        if self.tables_dir.exists():
            return self.tables_dir
        if self.parsed_tables_dir.exists():
            return self.parsed_tables_dir
        return self.tables_dir

    def _load_indexes(self):
        """Load BM25 và FAISS 1 lần duy nhất khi khởi động hệ thống."""
        idx_dirs = [self.indexes_dir, self.parsed_indexes_dir]
        
        # Load BM25 (lazy load)
        self.bm25_path = None
        for d in idx_dirs:
            p = d / "bm25.pkl"
            if p.exists():
                self.bm25_path = p
                break
                
        if self.bm25_path:
            logger.info(f"BM25 index located at {self.bm25_path}. Will lazy-load when fallback is needed.")
        else:
            logger.warning("BM25 index not found in standard paths.")

        # Load FAISS và documents.json (lazy load)
        self.faiss_path = None
        self.docs_path = None
        for d in idx_dirs:
            p1 = d / "faiss" / "index.faiss"
            p2 = d / "faiss" / "documents.json"
            if p1.exists() and p2.exists():
                self.faiss_path = p1
                self.docs_path = p2
                break
                
        if self.faiss_path and self.docs_path:
            logger.info(f"FAISS index located at {self.faiss_path}. Will lazy-load when fallback is needed.")
        else:
            logger.warning("FAISS index or documents.json not found in standard paths.")

    def _ensure_bm25_loaded(self):
        """Lazy load BM25 index khi thực sự cần fallback."""
        if self.bm25_index is None and self.bm25_path:
            try:
                logger.info(f"Lazy loading BM25 index from {self.bm25_path} (~800MB)...")
                with open(self.bm25_path, "rb") as f:
                    self.bm25_index = pickle.load(f)
                logger.info("Loaded BM25 index.")
            except Exception as e:
                logger.error(f"Error loading BM25 index: {e}")

    def _ensure_faiss_loaded(self):
        """Lazy load FAISS index và documents.json khi thực sự cần fallback."""
        if self.faiss_index is None and self.faiss_path and self.docs_path:
            try:
                logger.info(f"Lazy loading FAISS index and documents.json (~500MB)...")
                self.faiss_index = faiss.read_index(str(self.faiss_path))
                with open(self.docs_path, "r", encoding="utf-8") as f:
                    self.documents_map = json.load(f)
                logger.info(f"Loaded FAISS index ({self.faiss_index.ntotal} vectors) and {len(self.documents_map)} documents.")
            except Exception as e:
                logger.error(f"Error loading FAISS index or documents.json: {e}")

    def _get_embedding_model(self):
        """Lazy load mô hình embedding (local cache)."""
        if self.embed_model is None:
            try:
                import os
                os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
                os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
                from sentence_transformers import SentenceTransformer
                # Dùng model đã tải sẵn trong cache local
                model_name = "bkai-foundation-models/vietnamese-bi-encoder"
                self.embed_model = SentenceTransformer(model_name, local_files_only=True)
                logger.info(f"Loaded embedding model from local cache: {model_name}")
            except Exception:
                try:
                    from sentence_transformers import SentenceTransformer
                    self.embed_model = SentenceTransformer("bkai-foundation-models/vietnamese-bi-encoder")
                except Exception as e:
                    logger.warning(f"Could not load SentenceTransformer: {e}")
                    self.embed_model = False
        return self.embed_model if self.embed_model is not False else None

    def fetch_data(self, inputs: Dict[str, Any]) -> Tuple[Any, str, str, dict]:
        """
        Tìm kiếm giá trị từ dữ liệu (CSV -> Fallback FAISS / BM25).
        Inputs: ticker, year, period, metric (tên chỉ tiêu), indicator_code (mã số), table_type
        """
        logger.info(f"HybridSolver fetching data: {inputs}")
        ticker = inputs.get("ticker", "").upper()
        year = inputs.get("year")
        period = inputs.get("period", year)
        item_code = inputs.get("indicator_code")
        table_type = inputs.get("table_type")
        metric_name = inputs.get("metric", "")

        # Ưu tiên 1: Exact match trên các file bảng CSV đã chuẩn hoá
        if ticker and period:
            val, doc, table, evidence = self._exact_match_csv(ticker, period, item_code, table_type, metric_name)
            if val is not None:
                return val, doc, table, evidence

        # Ưu tiên 2: Fallback bằng chuỗi tên chuẩn hoá trên bảng CSV
        if ticker and period and metric_name:
            val, doc, table, evidence = self._fallback_name_match_csv(ticker, period, table_type, metric_name)
            if val is not None:
                return val, doc, table, evidence

        # Ưu tiên 3: Fallback bằng Dense Search (FAISS + documents.json)
        self._ensure_faiss_loaded()
        if self.faiss_index is not None and len(self.documents_map) > 0:
            logger.info("Triggering FAISS dense search fallback...")
            val, doc, table, evidence = self._faiss_search(ticker, period, metric_name)
            if val is not None:
                return val, doc, table, evidence

        logger.warning(f"No match found for inputs: {inputs}")
        return None, "", "", {}

    def _exact_match_csv(self, ticker: str, period: int, item_code: Optional[str], table_type: Optional[str], metric_name: str) -> Tuple[Any, str, str, dict]:
        """Tìm file CSV khớp Ticker, Period và lọc đúng dòng chứa item_code với trọng tài rapidfuzz."""
        tables_dir = self._get_active_tables_dir()
        if not tables_dir.exists():
            return None, "", "", {}

        search_pattern = f"{ticker}_*.csv"
        matching_files = list(tables_dir.glob(search_pattern))
        
        # Nếu có tiền tố IS, BS, CF thì map sang full name
        type_alias_map = {
            "is": "income_statement",
            "bs": "balance_sheet",
            "cf": "cash_flow",
            "eq": "equity_statement"
        }
        filter_type = type_alias_map.get(str(table_type).lower(), str(table_type).lower()) if table_type else None

        for file_path in matching_files:
            if filter_type and filter_type not in file_path.name.lower():
                continue

            try:
                df = pd.read_csv(file_path, dtype={"item_code": str})
                if 'period' in df.columns and 'value' in df.columns:
                    match = pd.DataFrame()
                    if item_code and 'item_code' in df.columns:
                        # Clean item_code (vd '21a' -> '21' hoặc '21a', '110' -> '110')
                        code_str = str(item_code).strip()
                        clean_num_code = re.sub(r'[^0-9]', '', code_str)
                        
                        # Match cả int và float cho period
                        match = df[(df['item_code'].isin([code_str, clean_num_code])) & (df['period'].isin([int(period), float(period)]))]

                    if not match.empty:
                        # Tie-breaker nếu có nhiều dòng trùng item_code
                        if len(match) == 1:
                            best_row = match.iloc[0]
                        else:
                            try:
                                from rapidfuzz import process, fuzz
                                names = match['item_name_normalized'].fillna("").tolist()
                                best_match = process.extractOne(metric_name.lower(), names, scorer=fuzz.partial_ratio)
                                best_row = match.iloc[best_match[2]] if best_match else match.iloc[0]
                            except Exception:
                                best_row = match.iloc[0]

                        val = best_row['value']
                        if pd.isna(val):
                            continue

                        df_var = f"df_{ticker}_{int(period)}"
                        evidence = {
                            "variable": df_var,
                            "csv_path": f"data/tables/{file_path.name}",
                        }
                        
                        doc_name = file_path.stem
                        start_line = best_row.get("start_line", 0) if "start_line" in best_row else 0
                        table_name = f"{doc_name}|{int(start_line) if pd.notna(start_line) else 0}"

                        logger.info(f"Exact CSV match: {val} from {file_path.name}")
                        return val, doc_name, table_name, evidence
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")
                continue

        return None, "", "", {}

    def _faiss_search(self, ticker: str, period: Optional[int], metric_name: str) -> Tuple[Any, str, str, dict]:
        """Truy vấn FAISS vector store kết hợp documents.json để tìm kiếm theo ngữ nghĩa từ cả Tables lẫn Notes."""
        model = self._get_embedding_model()
        if model is None or self.faiss_index is None:
            return None, "", "", {}

        try:
            # Nếu có ticker và period, ta lọc trước các documents thuộc ticker và period đó
            filtered_docs = [
                d for d in self.documents_map 
                if (not ticker or str(d.get("metadata", {}).get("ticker", "")).upper() == ticker)
                and (not period or float(d.get("metadata", {}).get("period") or d.get("metadata", {}).get("year") or 0) == float(period))
            ]
            
            if filtered_docs and metric_name:
                from rapidfuzz import process, fuzz
                
                # 1. Ưu tiên tìm trong các Table documents trước
                table_docs = [d for d in filtered_docs if d.get("metadata", {}).get("document_type") != "notes" and d.get("metadata", {}).get("value") is not None]
                if table_docs:
                    table_texts = [d.get("text", "") for d in table_docs]
                    best_match = process.extractOne(metric_name.lower(), table_texts, scorer=fuzz.partial_ratio)
                    if best_match and best_match[1] >= 65:
                        best_doc = table_docs[best_match[2]]
                        meta = best_doc.get("metadata", {})
                        doc_name = meta.get("source_file", "").replace(".txt", "") or f"{ticker}_{period}_consolidated"
                        start_line = meta.get("start_line", 0)
                        table_name = f"{doc_name}|{start_line}"
                        csv_path = meta.get("csv_path", f"data/tables/{doc_name}.csv")
                        val = meta.get("value")
                        return val, doc_name, table_name, {"variable": f"df_{ticker}_{period}", "csv_path": csv_path}

                # 2. Nếu không có bảng chính, tìm trong Notes Chunks
                notes_docs = [d for d in filtered_docs if d.get("metadata", {}).get("document_type") == "notes" or d.get("metadata", {}).get("value") is None]
                if notes_docs:
                    # Lọc trước các chunks có title hoặc text liên quan đến tiền gửi / TCTD
                    relevant_notes = [d for d in notes_docs if any(kw in f"{d.get('metadata', {}).get('section_title', '')} {d.get('text', '')}".lower() for kw in ["tiền gửi", "tctd", "tổ chức tín dụng", "ngân hàng"])]
                    target_notes = relevant_notes if relevant_notes else notes_docs
                    
                    notes_texts = [f"{d.get('metadata', {}).get('section_title', '')}\n{d.get('text', '')}" for d in target_notes]
                    best_match = process.extractOne(metric_name.lower(), notes_texts, scorer=fuzz.token_set_ratio)
                    
                    if best_match and best_match[1] >= 50:
                        best_doc = target_notes[best_match[2]]
                        meta = best_doc.get("metadata", {})
                        doc_name = meta.get("source_file", "").replace(".txt", "") or f"{ticker}_{period}_consolidated"
                        start_line = meta.get("start_line", 0)
                        table_name = f"{doc_name}|{start_line}"
                        csv_path = meta.get("csv_path", f"data/tables/{doc_name}.csv")

                        text = best_doc.get("text", "")
                        lines = text.splitlines()
                        for line in lines:
                            line_lower = line.lower()
                            # Tìm dòng chứa đúng cụm từ chỉ tiêu (vd: 'tiền gửi tại các tctd khác')
                            if any(term in line_lower for term in ["tiền gửi tại các tctd khác", "tiền gửi tại tctd", "tctd khác"]) and "|" in line:
                                nums = re.findall(r"(\d{1,3}(?:\.\d{3})+|\d+)", line)
                                if nums:
                                    clean_val = int(nums[0].replace(".", ""))
                                    logger.info(f"Notes Specific Match ({best_match[1]}%): {clean_val} from line: {line}")
                                    return clean_val, doc_name, table_name, {"variable": f"df_{ticker}_{period}", "csv_path": csv_path}
                                    
                        # Nếu không có dòng khớp cụ thể, lấy số đầu tiên từ bảng trong chunk
                        for line in lines:
                            if "|" in line:
                                nums = re.findall(r"(\d{1,3}(?:\.\d{3})+|\d+)", line)
                                if nums:
                                    clean_val = int(nums[0].replace(".", ""))
                                    return clean_val, doc_name, table_name, {"variable": f"df_{ticker}_{period}", "csv_path": csv_path}

            # Fallback sang vector search toàn cục nếu lọc cục bộ không ra
            query_text = f"{ticker} {period if period else ''} {metric_name}".strip()
            query_vector = model.encode([query_text], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
            
            scores, indices = self.faiss_index.search(query_vector, 30)
            for idx, score in zip(indices[0], scores[0]):
                if idx < 0 or idx >= len(self.documents_map):
                    continue
                    
                doc = self.documents_map[idx]
                meta = doc.get("metadata", {})
                val = meta.get("value")
                doc_name = meta.get("source_file", "").replace(".txt", "") or f"{ticker}_{period}_consolidated"
                start_line = meta.get("start_line", 0)
                table_name = f"{doc_name}|{start_line}"
                csv_path = meta.get("csv_path", f"data/tables/{doc_name}.csv")

                if val is not None and not pd.isna(val):
                    return val, doc_name, table_name, {"variable": f"df_{ticker}_{period}", "csv_path": csv_path}

        except Exception as e:
            logger.error(f"Error in FAISS search: {e}")

        return None, "", "", {}

    def _fallback_name_match_csv(self, ticker: str, period: int, table_type: Optional[str], metric_name: str) -> Tuple[Any, str, str, dict]:
        """Dò tìm theo tên đã chuẩn hoá trong tất cả các bảng CSV của ticker."""
        tables_dir = self._get_active_tables_dir()
        if not tables_dir.exists():
            return None, "", "", {}

        normalized_query = re.sub(r'[^a-z0-9]', '_', remove_diacritics(metric_name.lower())).strip('_')
        # Tách các từ quan trọng (vd: ['loi_nhuan', 'sau_thue'])
        keywords = [k for k in normalized_query.split('_') if len(k) > 2]
        
        search_pattern = f"{ticker}_*.csv"

        for file_path in tables_dir.glob(search_pattern):
            if table_type and table_type.lower() not in file_path.name.lower():
                continue

            try:
                df = pd.read_csv(file_path)
                if 'period' in df.columns and 'value' in df.columns and 'item_name_normalized' in df.columns:
                    period_df = df[df['period'] == float(period)]
                    if period_df.empty:
                        continue

                    for _, row in period_df.iterrows():
                        row_name = str(row.get('item_name_normalized', '')).lower()
                        # Khớp nếu toàn bộ query hoặc các keywords chính cùng xuất hiện
                        match_direct = normalized_query in row_name or row_name in normalized_query
                        match_keywords = all(kw in row_name for kw in keywords) if keywords else False
                        
                        if match_direct or match_keywords:
                            val = row['value']
                            if pd.isna(val):
                                continue

                            df_var = f"df_{ticker}_{int(period)}"
                            evidence = {
                                "variable": df_var,
                                "csv_path": f"data/tables/{file_path.name}",
                            }
                            doc_name = file_path.stem
                            start_line = row.get("start_line", 0)
                            table_name = f"{doc_name}|{int(start_line) if pd.notna(start_line) else 0}"
                            return val, doc_name, table_name, evidence
            except Exception:
                continue

        return None, "", "", {}
