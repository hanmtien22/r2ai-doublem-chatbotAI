from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

from src.utils.text import remove_diacritics

logger = logging.getLogger(__name__)

_TICKER_PATTERN = re.compile(r"\b([A-Z]{3})\b")
_YEAR_PATTERN = re.compile(r"\b(20[0-2]\d)\b")


def _word_boundary_match(keyword: str, text: str) -> bool:
    pattern = r"(?<![a-zA-ZÀ-ỹ0-9])" + re.escape(keyword) + r"(?![a-zA-ZÀ-ỹ0-9])"
    return bool(re.search(pattern, text, re.IGNORECASE))

def _clean_indicator_name(name: str) -> str:
    # Bỏ các prefix đánh số La Mã, chữ cái, số thứ tự ở đầu
    # vd: "I.", "II.", "1.", "A.", "1", "a", "i"
    name = re.sub(r'^(?:[ivxlcdm]+|[a-z]|\d+)\s*[.\-:]*\s+', '', name.strip(), flags=re.IGNORECASE)
    # Loại bỏ các dấu câu
    name = re.sub(r'[^\w\s]', ' ', name)
    # Rút gọn khoảng trắng
    return re.sub(r'\s+', ' ', name).strip()


class EntityExtractor:
    def __init__(
        self,
        entity_dict: dict, #Tu dien cac thuc the
        indicator_aliases: dict[str, str], #tu dien alias cua cac chi tieu, mapping alias -> indicator_code
        schema_mapping: dict, #tu dien mapping schema, mapping indicator_code -> (section, code)
    ):
        #Gan state co ban
        self._entity_dict = entity_dict
        self._indicator_aliases = indicator_aliases
        self._schema_mapping = schema_mapping
        self._alias_to_ticker = self._build_alias_to_ticker(entity_dict)
        self._all_indicator_names = self._load_schema_mapping(schema_mapping)

        # Pre-sort 1 lần trong __init__ để tránh sort lại mỗi lần extract
        self._sorted_ticker_aliases = sorted(self._alias_to_ticker.keys(), key=len, reverse=True)
        self._sorted_indicator_aliases = sorted(self._indicator_aliases.keys(), key=len, reverse=True)

    # Ham xay dung mapping alias -> ticker tu entity_dict    
    def _build_alias_to_ticker(self, entity_dict: dict) -> dict[str,str]:
        alias_map = {} 
        
        # Danh sách các từ khóa quá chung chung, không được phép làm alias cho 1 công ty cụ thể
        stop_aliases = {
            "ctcp", "cong ty co phan", "cong ty", "tap doan", "group", "jsc", 
            "joint stock company", "corporation", "corp", "co", "ltd", "tnhh"
        }
        
        for ticker, info in entity_dict.items():
            raw_names = [ticker]
            raw_names.extend(info.get('aliases', [])) #them aliases vao danh sach raw_names, neu aliases rong them chuoi rong []
            #Neu full_name va short_name co trong info thi them vao raw_names
            if 'full_name' in info:
                raw_names.append(info['full_name'])
            if 'short_name' in info:
                raw_names.append(info['short_name'])
            
            for name in raw_names:
                # Neu name khong phai la string hoac name rong thi bo qua
                if not isinstance(name,str) or name.strip() == "":
                    continue
                # Chuyen name sang dang thuong va bo khoang trang dau/cuoi
                clean_key = name.strip().lower()
                clean_key_no_dia = remove_diacritics(clean_key)
                
                if clean_key in stop_aliases or clean_key_no_dia in stop_aliases:
                    continue
                    
                # Neu clean_key da ton tai trong alias_map va dang map voi ticker khac thi log canh bao roi ghi de
                if clean_key in alias_map and alias_map[clean_key] != ticker:
                    logger.warning("Xung dot alias: %s da tro toi %s, bi ghi de boi %s",
                                   clean_key, alias_map[clean_key], ticker)
                    
                alias_map[clean_key] = ticker
        return alias_map
        
    # Ham xay dung danh sach cac chi tieu tu indicator_aliases va schema_mapping    
    def _load_schema_mapping(self, schema_mapping: dict) -> list[dict]:
        indicator_list = [] #Khoi tao list cac chi tieu rong

        #Duyet qua cac section va cac items trong schema_mapping
        for section, items in schema_mapping.items():
            if not isinstance(items, dict): # Neu items khong phai la dict thi bo qua
                continue
            
            for code, info in items.items():
                if not isinstance(info, dict):
                    continue
                # Lay name tu info, neu name khong phai la string thi bo qua
                raw_name = info.get('name', '')
                if not isinstance(raw_name, str):
                    continue
                name_lower = raw_name.strip().lower()
                name_no_diacritics = remove_diacritics(name_lower) # Bo dau tu name_lower
                
                indicator_list.append({
                    'section' : section,
                    'code' : code,
                    'name' : raw_name,
                    'name_lower' : name_lower,
                    'name_no_diacritics' : name_no_diacritics
                })
                
        return indicator_list 

    # Load file indicator_aliases.json thanh mot dict 
    def _load_indicator_aliases(self, path: str) -> None:
        try:
            with open(path, 'r', encoding = 'utf-8') as f:
                raw_aliases = json.load(f)
                
                self._indicator_aliases = {}
                for alias, mapped_indicator in raw_aliases.items():
                    if not isinstance(alias, str) or not isinstance(mapped_indicator,str):
                        logger.warning("Alias khong hop le: %s -> %s", alias, mapped_indicator)
                        continue
                    self._indicator_aliases[alias.strip().lower()] = mapped_indicator
                logger.info("Da load indicator aliases tu %s", path)
        except FileNotFoundError:
            logger.error("File not found: %s", path)
        except json.JSONDecodeError:
            logger.error("Invalid JSON in file: %s", path)

    @property
    def entity_dict(self) -> dict:
        return self._entity_dict

    def extract_tickers(self, question: str) -> list[str]:
        tickers: list[str] = []
        # Tim kiem cac ticker co trong cau hoi bang regex
        direct_matches = _TICKER_PATTERN.findall(question)
        for t in direct_matches:
            if t in self._entity_dict: # Neu ticker co trong entity_dict thi them vao danh sach tickers
                tickers.append(t)

        q_lower = question.lower()
        # Dùng danh sách đã sort sẵn từ __init__, không cần sort lại
        for alias in self._sorted_ticker_aliases:
            if alias in q_lower:
                ticker = self._alias_to_ticker[alias] # Lay ticker tu alias_to_ticker
                if ticker not in tickers:
                    tickers.append(ticker)

        return tickers

    def extract_years(self, question: str) -> list[int]:
        matches = _YEAR_PATTERN.findall(question)
        return sorted(set(int(y) for y in matches))

    def extract_indicators(self, question: str) -> list[dict]:
        indicators: list[dict] = []
        q_lower = question.lower()
        # Loại bỏ các dấu câu phổ biến khỏi q_lower để dễ match
        q_clean = re.sub(r'[^\w\s]', ' ', q_lower)
        q_clean = re.sub(r'\s+', ' ', q_clean).strip()
        q_no_diacritics = remove_diacritics(q_clean)

        sorted_aliases = self._sorted_indicator_aliases  # Dùng list đã pre-sort từ __init__
        for alias in sorted_aliases:
            if alias.isnumeric() or len(alias) <= 2:
                continue
            
            alias_clean = _clean_indicator_name(remove_diacritics(alias.lower()))
            
            if len(alias_clean) <= 2:
                continue
                
            if _word_boundary_match(alias_clean, q_no_diacritics):
                code_str = self._indicator_aliases[alias]
                section, code = code_str.split(".")
                already = any(
                    ind["section"] == section and ind["code"] == code
                    for ind in indicators
                )
                if not already:
                    indicators.append({
                        "name": alias,
                        "section": section,
                        "code": code,
                        "indicator_code": code_str,
                    })
                pattern = r"(?<![a-zA-Z0-9])" + re.escape(alias_clean) + r"(?![a-zA-Z0-9])"
                q_no_diacritics = re.sub(pattern, " ", q_no_diacritics, flags=re.IGNORECASE)

        if not indicators:
            for ind_info in self._all_indicator_names:
                name_no_dia = ind_info["name_no_diacritics"]
                
                if name_no_dia.isnumeric() or len(name_no_dia) <= 2:
                    continue
                    
                name_no_dia_clean = _clean_indicator_name(name_no_dia)
                
                if len(name_no_dia_clean) <= 2:
                    continue
                    
                if _word_boundary_match(name_no_dia_clean, q_no_diacritics):
                    already = any(
                        ind["section"] == ind_info["section"] and ind["code"] == ind_info["code"]
                        for ind in indicators
                    )
                    if not already:
                        indicators.append({
                            "name": ind_info["name"],
                            "section": ind_info["section"],
                            "code": ind_info["code"],
                            "indicator_code": f"{ind_info['section']}.{ind_info['code']}",
                        })
                    pattern2 = r"(?<![a-zA-Z0-9])" + re.escape(name_no_dia_clean) + r"(?![a-zA-Z0-9])"
                    q_no_diacritics = re.sub(pattern2, " ", q_no_diacritics, flags=re.IGNORECASE)

        # Neu van khong co indicator nao, tra ve mot "Unresolved Indicator" voi ten la tu khoa con lai trong cau hoi
        if not indicators:
            # Thu lay keyword tu q_clean (da bo ten cong ty va nam ra ngoai - tam thoi lay luon q_clean)
            # Phan nay chu yeu de Phase 2 nhan dien va tim kiem trong thuyet minh
            indicators.append({
                "name": question.strip(),
                "section": "NOTES",
                "code": "UNKNOWN",
                "indicator_code": "NOTES.UNKNOWN",
            })
            
        return indicators

    def extract_all(self, question: str) -> dict:
        tickers = self.extract_tickers(question)
        years = self.extract_years(question)
        indicators = self.extract_indicators(question)

        return {
            "tickers": tickers,
            "years": years,
            "indicators": [ind["name"] for ind in indicators],
            "indicator_codes": [ind["indicator_code"] for ind in indicators],
            "indicator_details": indicators,
        }
