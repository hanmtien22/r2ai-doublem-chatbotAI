from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

from src.utils.text import remove_diacritics

logger = logging.getLogger(__name__)

_TICKER_PATTERN = re.compile(r"\b([A-Z]{3})\b")
_YEAR_PATTERN = re.compile(r"\b(20[0-2]\d)\b")


@lru_cache(maxsize=4096)
def _compile_boundary_pattern(keyword: str) -> re.Pattern:
    """Cache compiled regex để tránh recursion depth khi compile lặp lại nhiều lần."""
    escaped = re.escape(keyword)
    # Dùng word boundary ASCII thay vì Unicode lookahead/lookbehind để tránh stack overflow
    return re.compile(r"(?<![\w])" + escaped + r"(?![\w])", re.IGNORECASE)


def _word_boundary_match(keyword: str, text: str) -> bool:
    try:
        pattern = _compile_boundary_pattern(keyword)
        return bool(pattern.search(text))
    except re.error:
        # Fallback: simple substring check nếu regex lỗi
        return keyword.lower() in text.lower()

def _clean_indicator_name(name: str) -> str:
    # Bỏ các prefix đánh số La Mã, chữ cái, số thứ tự ở đầu
    # vd: "I.", "II.", "1.", "A.", "1", "a", "i"
    name = re.sub(r'^(?:[ivxlcdm]+|[a-z]|\d+)\s*[.\-:]*\s+', '', name.strip(), flags=re.IGNORECASE)
    # Loại bỏ các dấu câu
    name = re.sub(r'[^\w\s]', ' ', name)
    # Rút gọn khoảng trắng
    return re.sub(r'\s+', ' ', name).strip()


# Cụm từ khung của câu hỏi — bỏ đi để còn lại phần mô tả chỉ tiêu
_BOILERPLATE_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bla bao nhieu\b", r"\bbao nhieu\b", r"\bcho biet\b", r"\bhay cho biet\b",
        r"\bcua cong ty me\b", r"\bcong ty me\b", r"\bcong ty\b", r"\bctcp\b",
        r"\bbao cao tai chinh (hop nhat|rieng)\b", r"\bhop nhat\b", r"\brieng le\b",
        # Chỉ bỏ "dong" khi là đơn vị tiền: "\bdong\b" trần sẽ cắt mất
        # "hoat dong", "co dong", "lao dong".
        r"\b(nghin ty|ty|trieu|nghin)\s+dong\b", r"\bbao nhieu dong\b",
        r"\bngay 31 thang 12\b", r"\bthoi diem\b",
        r"\bla\b", r"\bcua\b", r"\btrong\b", r"\bvao\b",
    )
]

# "nam" chỉ là từ khung khi đi kèm năm cụ thể — nếu không sẽ nuốt mất "Viet Nam"
_YEAR_PHRASE_PATTERN = re.compile(
    r"\b(?:trong|vao|tai|cuoi|dau|ket thuc ngay)?\s*nam\s+20[0-2]\d\b", re.IGNORECASE
)

# Alias 1 từ ("tien", "lai", "i"…) khớp gần như mọi câu hỏi → không đủ đặc trưng
_MIN_INDICATOR_TOKENS = 2
_MIN_INDICATOR_CHARS = 5


def _is_specific_indicator(cleaned: str) -> bool:
    """Alias chỉ được chấp nhận khi đủ đặc trưng để không khớp bừa."""
    return len(cleaned) >= _MIN_INDICATOR_CHARS and len(cleaned.split()) >= _MIN_INDICATOR_TOKENS


# "công ty mẹ" / "riêng" -> báo cáo riêng (separate); "hợp nhất" -> consolidated.
_SEPARATE_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (r"\bcong ty me\b", r"\bcty me\b", r"\brieng le\b", r"\bbao cao\s+\w*\s*rieng\b", r"\brieng\b")
]
_CONSOLIDATED_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in (r"\bhop nhat\b", r"\btoan tap doan\b", r"\bca tap doan\b")
]


def extract_report_type(question: str) -> Optional[str]:
    """'separate' khi hỏi công ty mẹ, 'consolidated' khi hỏi hợp nhất, None nếu không nói rõ."""
    text = remove_diacritics(question.lower())
    if any(p.search(text) for p in _SEPARATE_PATTERNS):
        return "separate"
    if any(p.search(text) for p in _CONSOLIDATED_PATTERNS):
        return "consolidated"
    return None


def extract_core_phrase(text: str) -> str:
    """Bóc phần mô tả chỉ tiêu ra khỏi câu hỏi (bỏ năm và khung câu).

    Tên công ty phải được loại trước khi gọi hàm này (xem `_strip_company_names`),
    vì bước bỏ năm sẽ phá vỡ các cụm như "Viet Nam".
    """
    text = remove_diacritics(text.lower())
    text = _YEAR_PHRASE_PATTERN.sub(" ", text)
    text = re.sub(r"\b20[0-2]\d\b", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    for pattern in _BOILERPLATE_PATTERNS:
        text = pattern.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


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
        """Khớp alias theo span dài nhất.

        "CTCP Chứng khoán FPT" phải ra FTS, không được ra thêm FPT chỉ vì
        chuỗi "fpt" nằm lọt bên trong tên công ty dài hơn.
        """
        q_norm = remove_diacritics(question.lower())

        # (start, end, ticker) của mọi alias khớp trong câu hỏi
        spans: list[tuple[int, int, str]] = []

        for alias in self._sorted_ticker_aliases:
            alias_norm = remove_diacritics(alias.strip().lower())
            if not alias_norm:
                continue
            for m in _compile_boundary_pattern(alias_norm).finditer(q_norm):
                spans.append((m.start(), m.end(), self._alias_to_ticker[alias]))

        # Ticker viết hoa xuất hiện trực tiếp trong câu hỏi gốc
        for m in _TICKER_PATTERN.finditer(question):
            if m.group(1) in self._entity_dict:
                spans.append((m.start(), m.end(), m.group(1)))

        # Loại các span bị bao trọn bởi một span khác dài hơn (và của ticker khác)
        kept: list[tuple[int, int, str]] = []
        for s, e, ticker in spans:
            covered = any(
                other_t != ticker and os_ <= s and e <= oe and (oe - os_) > (e - s)
                for os_, oe, other_t in spans
            )
            if not covered:
                kept.append((s, e, ticker))

        tickers: list[str] = []
        for _, _, ticker in sorted(kept):
            if ticker not in tickers:
                tickers.append(ticker)
        return tickers

    def extract_years(self, question: str) -> list[int]:
        matches = _YEAR_PATTERN.findall(question)
        return sorted(set(int(y) for y in matches))

    def _match_candidates(self, q_no_diacritics: str) -> list[dict]:
        """Tìm mọi alias/tên chỉ tiêu xuất hiện trong câu hỏi, kèm vị trí và độ dài."""
        candidates: list[dict] = []

        def _add(source_name: str, section: str, code: str, cleaned: str, priority: int) -> None:
            if not _is_specific_indicator(cleaned):
                return
            for m in _compile_boundary_pattern(cleaned).finditer(q_no_diacritics):
                candidates.append({
                    "name": source_name,
                    "section": section,
                    "code": code,
                    "indicator_code": f"{section}.{code}",
                    "start": m.start(),
                    "end": m.end(),
                    "matched_len": len(cleaned),
                    "priority": priority,
                })

        for alias in self._sorted_indicator_aliases:
            code_str = self._indicator_aliases[alias]
            if "." not in code_str:
                continue
            section, code = code_str.split(".", 1)
            _add(alias, section, code, _clean_indicator_name(remove_diacritics(alias.lower())), 0)

        for ind_info in self._all_indicator_names:
            _add(
                ind_info["name"],
                ind_info["section"],
                ind_info["code"],
                _clean_indicator_name(ind_info["name_no_diacritics"]),
                1,
            )

        return candidates

    @staticmethod
    def _select_best_candidates(candidates: list[dict]) -> list[dict]:
        """Giữ các match dài nhất, bỏ match nằm lọt trong một match dài hơn."""
        # Dài trước, alias (priority 0) trước tên schema khi bằng độ dài
        candidates.sort(key=lambda c: (-c["matched_len"], c["priority"], c["start"]))

        selected: list[dict] = []
        covered: list[tuple[int, int]] = []
        seen_codes: set[str] = set()

        for cand in candidates:
            span = (cand["start"], cand["end"])
            if any(s <= span[0] and span[1] <= e for s, e in covered):
                continue
            if cand["indicator_code"] in seen_codes:
                continue
            seen_codes.add(cand["indicator_code"])
            covered.append(span)
            selected.append(cand)

        return selected

    def _fuzzy_match_indicator(self, core_phrase: str, threshold: int = 80) -> Optional[dict]:
        """Khớp mờ khi câu hỏi diễn đạt khác từ điển (vd: thiếu/thừa một vài từ)."""
        if not core_phrase or len(core_phrase.split()) < 2:
            return None
        try:
            from rapidfuzz import fuzz
        except ImportError:
            return None

        core_tokens = set(core_phrase.split())
        best = None
        best_score = 0.0

        for alias, code_str in self._indicator_aliases.items():
            if "." not in code_str:
                continue
            cleaned = _clean_indicator_name(remove_diacritics(alias.lower()))
            if not _is_specific_indicator(cleaned):
                continue

            alias_tokens = set(cleaned.split())
            overlap = len(alias_tokens & core_tokens)
            # Hai chiều: alias phải nằm gần trọn trong câu hỏi, VÀ phải giải thích
            # được phần lớn cụm chỉ tiêu. Chỉ kiểm tra một chiều thì "chi phi khac"
            # sẽ khớp 100% với "chi phi luong va cac khoan khac theo luong".
            if overlap / len(alias_tokens) < 0.75:
                continue
            if overlap / len(core_tokens) < 0.6:
                continue

            score = fuzz.token_sort_ratio(core_phrase, cleaned)
            # Ưu tiên alias dài hơn khi điểm ngang nhau
            score_adj = score + min(len(alias_tokens), 10) * 0.5
            if score >= threshold and score_adj > best_score:
                best_score = score_adj
                section, code = code_str.split(".", 1)
                best = {
                    "name": alias,
                    "section": section,
                    "code": code,
                    "indicator_code": code_str,
                }

        if best:
            logger.debug("Fuzzy indicator match: '%s' -> %s", core_phrase, best["indicator_code"])
        return best

    def _strip_company_names(self, text: str, tickers: list[str]) -> str:
        """Bỏ mã và tên công ty khỏi cụm chỉ tiêu để khớp mờ không bị nhiễu."""
        for ticker in tickers:
            names = [ticker]
            info = self._entity_dict.get(ticker, {})
            names.extend(info.get("aliases", []))
            names.append(info.get("full_name", ""))
            names.append(info.get("short_name", ""))
            for name in sorted((n for n in names if n), key=len, reverse=True):
                cleaned = remove_diacritics(name.lower())
                cleaned = re.sub(r"[^\w\s]", " ", cleaned)
                cleaned = re.sub(r"\s+", " ", cleaned).strip()
                if cleaned:
                    text = _compile_boundary_pattern(cleaned).sub(" ", text)
        return re.sub(r"\s+", " ", text).strip()

    def extract_indicators(self, question: str, tickers: Optional[list[str]] = None) -> list[dict]:
        q_clean = re.sub(r"[^\w\s]", " ", question.lower())
        q_clean = re.sub(r"\s+", " ", q_clean).strip()
        q_no_diacritics = remove_diacritics(q_clean)

        candidates = self._match_candidates(q_no_diacritics)
        selected = self._select_best_candidates(candidates)

        core_phrase = extract_core_phrase(self._strip_company_names(q_no_diacritics, tickers or []))

        # Không có match chính xác nào -> thử khớp mờ trên phần mô tả chỉ tiêu
        if not selected:
            fuzzy = self._fuzzy_match_indicator(core_phrase)
            if fuzzy:
                return [fuzzy]
            return [{
                "name": core_phrase or question.strip(),
                "section": "NOTES",
                "code": "UNKNOWN",
                "indicator_code": "NOTES.UNKNOWN",
            }]

        # Có match nhưng ngắn hơn hẳn phần mô tả chỉ tiêu (vd: chỉ khớp "tien mat"
        # trong "tien mat va cac khoan tuong duong tien") -> ưu tiên khớp mờ dài hơn
        best_len = max(len(c["name"].split()) for c in selected)
        if core_phrase and len(core_phrase.split()) >= best_len + 3:
            fuzzy = self._fuzzy_match_indicator(core_phrase)
            if fuzzy and len(fuzzy["name"].split()) > best_len:
                return [fuzzy]

        return [
            {k: c[k] for k in ("name", "section", "code", "indicator_code")}
            for c in selected
        ]

    def extract_all(self, question: str) -> dict:
        tickers = self.extract_tickers(question)
        years = self.extract_years(question)
        indicators = self.extract_indicators(question, tickers)
        q_clean = re.sub(r"[^\w\s]", " ", question.lower())
        q_clean = re.sub(r"\s+", " ", q_clean).strip()
        core_phrase = extract_core_phrase(
            self._strip_company_names(remove_diacritics(q_clean), tickers)
        )

        return {
            "tickers": tickers,
            "years": years,
            "report_type": extract_report_type(question),
            "core_phrase": core_phrase,
            "indicators": [ind["name"] for ind in indicators],
            "indicator_codes": [ind["indicator_code"] for ind in indicators],
            "indicator_details": indicators,
        }
