import pandas as pd
import pickle
import faiss
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import re
import numpy as np
from src.indexing.bm25 import bm25_search
from src.indexing.embedding import dense_search
from src.indexing.hybrid_search import reciprocal_rank_fusion

logger = logging.getLogger(__name__)

# Từ khoá dùng để lọc notes chunk — đưa ra ngoài để dễ tuỳ chỉnh khi mở rộng domain
NOTES_FILTER_KEYWORDS: List[str] = [
    "tiền gửi", "tctd", "tổ chức tín dụng", "ngân hàng"
]
# Từ khoá của chỉ tiêu cụ thể trong notes (có thể override từ config sau này)
NOTES_SPECIFIC_TERMS: List[str] = [
    "tiền gửi tại các tctd khác", "tiền gửi tại tctd", "tctd khác"
]

class EasyHybridSolver:
    """Công cụ tìm kiếm bốc số từ CSV (Exact) và Fallback bằng BM25/FAISS."""
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.tables_dir = self.data_dir / "tables"
        self.indexes_dir = self.data_dir / "indexes"
        self.parsed_indexes_dir = self.data_dir / "parsed_tables" / "indexes"

        # Cache CSV DataFrame vào RAM: tránh đọc lại file mỗi lần query
        self._csv_cache: dict[str, pd.DataFrame] = {}

        # Đọc config để biết có dùng dense search (FAISS) không
        self._dense_enabled = self._read_dense_enabled_config()

        # Biến chứa dữ liệu trong RAM
        self.bm25_index = None
        self.faiss_index = None
        self.documents_map: List[dict] = []
        self.embed_model = None
        self._load_indexes()

    @staticmethod
    def _read_dense_enabled_config() -> bool:
        """Đọc cờ dense_enabled từ config.yaml, mặc định True nếu không đọc được."""
        try:
            from src.config_loader import load_config
            return bool(load_config().get("retrieval", {}).get("dense_enabled", True))
        except Exception:
            return True

    def _get_active_tables_dir(self) -> Path:
        """Tìm thư mục chứa các bảng csv đã parse."""
        if self.tables_dir.exists():
            return self.tables_dir
        if self.parsed_tables_dir.exists():
            return self.parsed_tables_dir
        return self.tables_dir

    def _get_csv(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Lấy DataFrame từ cache hoặc đọc từ đĩa nếu chưa có."""
        path_str = str(file_path)
        if path_str not in self._csv_cache:
            self._csv_cache[path_str] = pd.read_csv(file_path, **kwargs)
        return self._csv_cache[path_str]

    def _load_indexes(self):
        """Load BM25 và FAISS 1 lần duy nhất khi khởi động hệ thống."""
        bm25_path = self.indexes_dir / "bm25.pkl"
        faiss_path = self.indexes_dir / "faiss" / "index.faiss"     
        docs_path = self.indexes_dir / "faiss" / "documents.json"
        
        # Load BM25 (lazy load)
        self.bm25_path = None
        idx_dirs = [self.indexes_dir, self.parsed_indexes_dir]
        for d in idx_dirs:
            p = d / "bm25.pkl"
            if p.exists():
                self.bm25_path = p
                break
                
        if self.bm25_path:
            logger.info(f"BM25 index located at {self.bm25_path}. Will lazy-load when fallback is needed.")
        else:
            logger.warning("BM25 index not found in standard paths.")

        # Load FAISS và documents.json (lazy load, chỉ khi dense_enabled=true trong config)
        self.faiss_path = None
        self.docs_path = None
        if not self._dense_enabled:
            logger.info("FAISS dense search disabled by config (dense_enabled=false). Skipping FAISS load.")
        else:
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
        try:
            if self.bm25_index is None and self.bm25_path:
                logger.info(f"Lazy loading BM25 index from {self.bm25_path} (~800MB)...")
                with open(self.bm25_path, "rb") as f:
                    self.bm25_index = pickle.load(f)
                logger.info("Loaded BM25 index.")
            else:
                logger.warning(f"BM25 index not found at {self.bm25_path}")
        except Exception as e:
            logger.error(f"Error load bm25 index: {e}")
            
    def _ensure_faiss_loaded(self):
     try:
         if self.faiss_index is None and self.faiss_path and self.docs_path and self.faiss_path.exists() and self.docs_path.exists():
             logger.info(f"Lazy loading FAISS index from {self.faiss_path}")
             self.faiss_index = faiss.read_index(str(self.faiss_path))
             import json
             with open(self.docs_path, "r", encoding="utf-8") as f:
                 self.documents_map = json.load(f)
             logger.info(f"Loaded FAISS index and documents.json from {self.docs_path}")
         elif not self.faiss_path:
             logger.warning("FAISS path is None (dense_enabled is false).")
         else:
             logger.warning(f"FAISS index or documents not found at {self.faiss_path}")
     except Exception as e:
         logger.error(f"Error loading FAISS index: {e}")
    
                
    def fetch_data(self, inputs: Dict[str, Any]) -> Tuple[Any, str, str, dict]:
        """
        Tìm kiếm giá trị từ dữ liệu.
        Inputs: ticker, year, metric (tên chỉ tiêu), indicator_code (mã số), table_type (BS/IS/CF/EQ)
        """
        logger.info(f"HybridSolver fetching data: {inputs}")
        ticker = inputs.get("ticker", "").upper()
        # Trong BCTC, year của file có thể là 2019, nhưng period chứa số thực tế là 2018
        year = inputs.get("year") 
        period = inputs.get("period", year) # Nếu không truyền period, mặc định bằng year
        item_code = inputs.get("indicator_code")
        table_type = inputs.get("table_type")
        metric_name = inputs.get("metric", "")

        # Lọc bằng Code chính xác trên thư mục Tables (Quy tắc lọc bằng Pandas)
        if ticker and period and item_code:
            val, doc, table, evidence = self._exact_match_csv(ticker, period, item_code, table_type, metric_name)
            if val is not None:
                return val, doc, table, evidence

        # Dung hybrid search de fallback
        self._ensure_bm25_loaded()
        self._ensure_faiss_loaded()
        if self.bm25_index is not None:
            logger.info("falling back to bm25 + dense")
            val, doc, table, evidence = self._hybrid_search(ticker, period, metric_name)
            
            if val is not None:
                return val, doc , table, evidence
            
    def _get_embedding_model(self):
        # Neu model da duoc load vao RAM
        if getattr(self, "embed_model", None) is not None:
            return self.embed_model
        try:
            import os 
            os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] =  "1"
            from sentence_transformers import SentenceTransformer
            from src.config_loader import load_config
            # Lay ten mnodel tu file config
            config = load_config()
            model_name = config.get("embedding",{}).get("model_name", "bkai-foundation-models/vietnamese-bi-encoder")
            logger.info(f"loading embedding model : {model_name}")
            # Uu tien load tu local file truoc de tang toc 
            try:
                self.embed_model = SentenceTransformer(model_name, local_files_only=True)
                logger.info("Successfully load model from local cache")
            except Exception as e:
                logger.warning(f"Local cache not found for {model_name}, attempting to download...")
                self.embed_model = SentenceTransformer(model_name)
                logger.info("Successfully downloaded and loaded model.")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self.embed_model = None
        return self.embed_model    
        
    def _hybrid_search(self, ticker : str, period : int, metric_name: str) -> Tuple[Any, str, str, dict]:
        if self.bm25_index is None:
            return None, "", "", {}
        model = self._get_embedding_model()
        try:
            query_text = f"{ticker} {period if period else ''} {metric_name}".strip()
            bm25_object = self.bm25_index["bm25"]
            bm25_res = bm25_search(
                query= query_text,
                bm25= bm25_object,
                documents= self.bm25_index["documents"],
                top_k= 7
            )
                
            dense_res = dense_search(
                query=query_text,
                model= model,
                index= self.faiss_index,
                documents= self.documents_map,
                top_k = 7,
                normalize_embeddings= True
            ) if self.faiss_index is not None else []
                
            hybrid_res = reciprocal_rank_fusion(
                bm25_results= bm25_res,
                dense_results= dense_res,
                rrf_k= 60,
                top_k= 7
            )
            for res in hybrid_res:
                doc = res["document"]
                meta = doc.get("metadata", {})
                val = meta.get("value")
                if val is not None and not pd.isna(val):
                    doc_name = meta.get("source_file", "").replace(".txt", "")
                    start_line = meta.get("start_line",0)
                    table_name = f"{doc_name}|{start_line}"
                    csv_path = meta.get("csv_path", f"data/{doc_name}.csv")
                    return val, doc_name, table_name, {"variable":f"df_{ticker}_{period}", "csv_path": f"{csv_path}"}
        except Exception as e:
            logger.error(f"Error: {e}")
                
        return None, "","",{}
                    
        
    def _exact_match_csv(self, ticker: str, period: int, item_code: str, table_type: str, metric_name: str) -> Tuple[Any, str, str, dict]:
        """Tìm file CSV khớp Ticker, Year và lọc đúng dòng chứa item_code."""
        if not self.tables_dir.exists():
            return None, "", "", {}
            
        # Giả sử convention file là: {ticker}_{year}_consolidated_{table_type}.csv
        # Dò tìm tất cả file bắt đầu bằng ticker
        for file_path in self.tables_dir.glob(f"{ticker}_*.csv"):
            if table_type and table_type not in file_path.name:
                continue
                
            try:
                # Đọc DataFrame
                df = pd.read_csv(file_path, dtype={"item_code": str})
                
                # Cấu trúc file: item_code, period, value...
                if 'item_code' in df.columns and 'period' in df.columns and 'value' in df.columns:
                    # Lọc dữ liệu
                    match = df[(df['item_code'] == str(item_code)) & (df['period'] == int(period))]
                    if not match.empty:
                        if len(match) == 1:
                            best_row = match.iloc[0]
                        else:
                            from rapidfuzz import process, fuzz
                            # Lay danh sach ten cua cac dong bi trung
                            danh_sach_ten = match['item_name_normalized'].fillna('').tolist()
                            best_match = process.extractOne(
                                metric_name.lower(),
                                danh_sach_ten,
                                scorer = fuzz.partial_ratio
                            )
                            
                            if best_match:
                                best_idx = best_match[2]
                                best_row = match.iloc[best_idx]
                            else:
                                best_row = match.iloc[0] #Fallback
                        val = best_row['value']
                        evidence = {
                            "variable": f"df_{ticker}_{period}",
                            "csv_path": f"data/{file_path.name}",
                        }
                        # Đóng gói Document và Table
                        doc_name = file_path.stem
                        table_name = f"{doc_name}|0"
                        
                        logger.info(f"Exact match found: {val}")
                        return val, doc_name, table_name, evidence
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")
                continue

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
                # Dùng cache để tránh đọc lại file CSV mỗi lần query
                df = self._get_csv(file_path)
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
