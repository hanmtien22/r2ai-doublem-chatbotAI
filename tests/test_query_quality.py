"""Test cho các bước quyết định độ chính xác của câu trả lời.

Toàn bộ test ở đây chạy offline: không cần BM25 index (~800MB) cũng không cần LLM,
nên có thể chạy nhanh sau mỗi lần sửa.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.answer.answer_formatter import detect_requested_unit, _to_vnd
from src.compute.notes_table import find_value_by_label, parse_notes_table, parse_number
from src.compute.sandbox import Sandbox
from src.query.entity_extractor import EntityExtractor, extract_report_type
from src.query.preprocessor import QueryPreprocessor

DICT_DIR = Path(__file__).resolve().parents[1] / "data" / "dictionaries"


@pytest.fixture(scope="module")
def extractor() -> EntityExtractor:
    def load(name: str) -> dict:
        return json.loads((DICT_DIR / name).read_text(encoding="utf-8"))

    return EntityExtractor(
        entity_dict=load("entity_dictionary.json"),
        indicator_aliases=load("indicator_aliases.json"),
        schema_mapping=load("schema_mapping.json"),
    )


@pytest.fixture(scope="module")
def preprocessor() -> QueryPreprocessor:
    return QueryPreprocessor()


# --------------------------------------------------------------------------- #
#  Preprocessor                                                                #
# --------------------------------------------------------------------------- #
def test_khong_pha_hong_cum_cong_ty_me(preprocessor):
    """Viết tắt 'ty' -> 'ty dong' từng biến "công ty mẹ" thành "công ty dong mẹ"."""
    result = preprocessor.normalize("Chi phí phạt của công ty mẹ SCR năm 2017 là bao nhiêu?")
    assert "công ty mẹ" in result
    assert "dong mẹ" not in result


def test_van_mo_rong_don_vi_khi_dung_sau_so(preprocessor):
    assert "trieu dong" in preprocessor.normalize("Lợi nhuận 5 tr")


# --------------------------------------------------------------------------- #
#  Nhận diện công ty                                                           #
# --------------------------------------------------------------------------- #
def test_uu_tien_ten_cong_ty_dai_nhat(extractor):
    """'CTCP Chứng khoán FPT' là FTS, không được nhận thêm FPT."""
    assert extractor.extract_tickers("Lợi nhuận sau thuế của CTCP Chứng khoán FPT năm 2023") == ["FTS"]


def test_nhan_dien_ma_ngan_hang(extractor):
    assert "BID" in extractor.extract_tickers(
        "Số dư tiền gửi của Ngân hàng TMCP Đầu tư và Phát triển Việt Nam (BID) là bao nhiêu?"
    )


# --------------------------------------------------------------------------- #
#  Nhận diện chỉ tiêu                                                          #
# --------------------------------------------------------------------------- #
def test_khong_khop_alias_qua_chung_chung(extractor):
    """'tien' từng nuốt cả câu hỏi về lưu chuyển tiền tệ và cho ra BS.111."""
    codes = extractor.extract_all(
        "Lưu chuyển tiền thuần từ hoạt động kinh doanh của công ty mẹ VSC trong năm 2017"
    )["indicator_codes"]
    assert codes == ["CF.20"]


def test_chi_tieu_khong_co_trong_tu_dien_thi_de_cho_notes(extractor):
    """Chỉ tiêu chỉ có trong thuyết minh phải trả NOTES.UNKNOWN, không đoán bừa mã khác."""
    entities = extractor.extract_all(
        "Chi phí lương và các khoản khác theo lương của công ty mẹ CTCP Chứng khoán FPT trong năm 2021"
    )
    assert entities["indicator_codes"] == ["NOTES.UNKNOWN"]
    # core_phrase là từ khoá sạch để tìm trong thuyết minh, không phải cả câu hỏi
    assert entities["core_phrase"] == "chi phi luong va cac khoan khac theo luong"


def test_core_phrase_giu_lai_ten_quoc_gia(extractor):
    """Bỏ 'nam' vô điều kiện sẽ cắt mất 'Việt Nam' trong tên công ty."""
    entities = extractor.extract_all(
        "Số dư tiền gửi tại các TCTD khác cuối năm 2016 của Ngân hàng TMCP Đầu tư và Phát triển Việt Nam (BID)"
    )
    assert entities["core_phrase"] == "so du tien gui tai cac tctd khac"


# --------------------------------------------------------------------------- #
#  Loại báo cáo                                                                #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "question, expected",
    [
        ("Chi phí phạt của công ty mẹ SCR năm 2017", "separate"),
        ("Báo cáo tài chính riêng của HPG", "separate"),
        ("Lợi nhuận hợp nhất của VNM năm 2022", "consolidated"),
        ("Chi phí khác của SAM năm 2023", None),
    ],
)
def test_nhan_dien_loai_bao_cao(question, expected):
    assert extract_report_type(question) == expected


# --------------------------------------------------------------------------- #
#  Bảng trong thuyết minh                                                      #
# --------------------------------------------------------------------------- #
NOTES_TEXT = """8. TIỀN GỬI TẠI CÁC TCTD KHÁC VÀ CHO VAY CÁC TCTD KHÁC
Số cuối năm | Số đầu năm
Triệu VND | Triệu VND
Tiền gửi tại các TCTD khác | 39.849.011 | 47.523.973
Tiền gửi không kỳ hạn | 9.468.532 | 17.687.509
Dự phòng rủi ro | (60.295) | (1.003)
"""


@pytest.mark.parametrize(
    "token, expected",
    [("39.849.011", 39849011.0), ("(60.295)", -60295.0), ("-", None), ("1.234,56", 1234.56)],
)
def test_doc_so_kieu_viet_nam(token, expected):
    assert parse_number(token) == expected


def test_parse_bang_thuyet_minh():
    rows = parse_notes_table(NOTES_TEXT)
    first = rows[0]
    assert first["label"] == "Tiền gửi tại các TCTD khác"
    assert first["column_role"] == "current"
    assert first["value"] == 39849011.0
    # Header ghi "Triệu VND" nên phải quy về VND
    assert first["value_vnd"] == 39849011.0 * 1_000_000
    # Số trong ngoặc là số âm
    assert any(r["label"] == "Dự phòng rủi ro" and r["value"] == -60295.0 for r in rows)


def test_tra_so_theo_nhan_uu_tien_cot_cuoi_nam():
    rows = parse_notes_table(NOTES_TEXT)
    match = find_value_by_label(rows, "so du tien gui tai cac tctd khac")
    assert match is not None
    assert match["label"] == "Tiền gửi tại các TCTD khác"
    assert match["column_role"] == "current"
    assert match["value"] == 39849011.0


def test_bang_co_header_cho_cot_nhan():
    """Header dư 1 ô (ô đầu là tên cột nhãn) thì giá trị phải dịch header cho khớp."""
    rows = parse_notes_table(
        "12. ĐẦU TƯ VÀO CÔNG TY CON\n"
        "Công ty con | Tỷ lệ sở hữu | Giá gốc\n"
        "Triệu VND | Triệu VND | Triệu VND\n"
        "Công ty A | 51 | 1.000\n"
    )
    assert [(r["column"], r["value"]) for r in rows] == [
        ("Tỷ lệ sở hữu", 51.0),
        ("Giá gốc", 1000.0),
    ]


def test_header_khong_phai_don_vi_thi_khong_bi_nhan_he_so():
    """'Công ty con' chứa 'ty' nhưng không phải dòng đơn vị -> không được nhân 1 tỷ."""
    rows = parse_notes_table(
        "Chỉ tiêu | Số cuối năm\nCông ty con | 5\n"
    )
    assert rows and rows[0]["value_vnd"] == 5.0


def test_bang_phan_loai_nhan_la_ten_hang_muc():
    """Bảng "Theo ngành nghề kinh doanh" có nhãn ngắn ("Thương mại");
    nhãn ngắn nhưng nằm nguyên cụm trong câu hỏi vẫn phải khớp."""
    rows = parse_notes_table(
        "9.6 Theo ngành nghề kinh doanh\n"
        "31.12.2022Triệu VND | 31.12.2021Triệu VND\n"
        "Thương mại | 72.917.566 | 64.617.561\n"
        "Sản xuất và gia công chế biến | 25.628.170 | 24.439.499\n",
        "Theo ngành nghề kinh doanh",
    )
    match = find_value_by_label(rows, "so du cho vay khach hang nganh thuong mai")
    assert match is not None
    assert match["label"] == "Thương mại"
    # Header "31.12.2022Triệu VND": năm mới hơn là số cuối kỳ, đơn vị là triệu VND
    assert match["column_role"] == "current"
    assert match["value_vnd"] == 72_917_566 * 1_000_000


def test_nhan_trung_o_nhieu_bang_thi_khong_doan():
    """Cùng nhãn, cùng tiêu đề nhưng giá trị khác nhau -> phải nói không biết."""
    rows = parse_notes_table(
        "2021VND | 2020VND\n"
        "Chu Thị Bình | 1.150.851.285 | 1.099.739.984\n"
        "2021VND | 2020VND\n"
        "Chu Thị Bình | 150.000.000 | 150.000.000\n"
    )
    assert find_value_by_label(rows, "thu lao thanh vien hdqt chu thi binh") is None


def test_tieu_de_thuyet_minh_pha_the_hoa():
    """Nhãn giống nhau ở hai thuyết minh: tiêu đề sát câu hỏi hơn sẽ thắng."""
    rows = parse_notes_table(
        "8. TIỀN GỬI TẠI CÁC TCTD KHÁC\n"
        "Số cuối năm\nTriệu VND\n"
        "Tiền gửi tại các TCTD khác | 39.849.011\n"
    ) + parse_notes_table(
        "36. TIỀN VÀ CÁC KHOẢN TƯƠNG ĐƯƠNG TIỀN\n"
        "Số cuối năm\nTriệu VND\n"
        "Tiền gửi tại các TCTD khác | 11.111.111\n"
    )
    match = find_value_by_label(rows, "so du tien gui tai cac tctd khac")
    assert match is not None and match["value"] == 39849011.0


SSH_NOTES_TEXT = """30. CHI PHÍ TÀI CHÍNH
Năm nay | Năm trước
VND | VND
Lãi tiền vay và trái phiếu | 104.777.542.153 | 133.862.100.456
Chi phí lãi từ hợp đồng hợp tác đầu tư(Hoàn nhập) dự phòng khoản đầu tư tại Công ty | 9.277.775.342(1.265.429.612) | 2.222.597.877(201.179.439)
112.789.887.883 | 135.883.518.894
"""


def test_dong_hong_khong_bi_hieu_la_dong_don_vi():
    """Dòng dữ liệu lỗi ("...hợp đồng...Công ty") từng khớp 'dong'/'ty' và
    làm mọi giá trị phía sau bị nhân sai 1 tỷ lần."""
    rows = parse_notes_table(SSH_NOTES_TEXT)
    assert rows, "phải parse được bảng"
    # Header ghi "VND" nên hệ số quy đổi luôn là 1
    for row in rows:
        assert row["value_vnd"] == row["value"]


def test_dong_tong_duoc_dat_ten_theo_tieu_de_thuyet_minh():
    """Dòng tổng không có nhãn; tổng của "30. CHI PHÍ TÀI CHÍNH" chính là chi phí tài chính."""
    match = find_value_by_label(parse_notes_table(SSH_NOTES_TEXT), "chi phi tai chinh")
    assert match is not None
    assert match["value"] == 112_789_887_883


def test_khong_tra_bua_khi_nhan_khong_lien_quan():
    assert find_value_by_label(parse_notes_table(NOTES_TEXT), "loi nhuan sau thue") is None


# --------------------------------------------------------------------------- #
#  Đơn vị câu trả lời                                                          #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "question, expected_label",
    [
        ("... là bao nhiêu triệu đồng?", "triệu đồng"),
        ("... là bao nhiêu tỷ đồng?", "tỷ đồng"),
        ("... bao nhiêu nghìn tỷ đồng vào cuối năm 2024?", "nghìn tỷ đồng"),
    ],
)
def test_nhan_dien_don_vi_duoc_hoi(question, expected_label):
    requested = detect_requested_unit(question)
    assert requested is not None and requested[1] == expected_label


def test_cong_ty_khong_bi_hieu_thanh_don_vi_ty():
    """'cong ty' (bỏ dấu) từng khớp luật đơn vị 'ty' và ép mọi câu trả lời về tỷ đồng."""
    requested = detect_requested_unit(
        "Lãi tiền gửi của công ty mẹ Vietjet là bao nhiêu triệu đồng?"
    )
    assert requested is not None and requested[1] == "triệu đồng"
    assert detect_requested_unit("Quỹ khen thưởng của công ty mẹ HT1 là bao nhiêu?") is None


def test_quy_doi_don_vi_goc_ve_vnd():
    assert _to_vnd(39849011, "Triệu VND") == 39849011 * 1_000_000
    assert _to_vnd(1000, "vnd") == 1000


# --------------------------------------------------------------------------- #
#  Sandbox                                                                     #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "question, value, expected",
    [
        # 300.000đ cho câu hỏi tính bằng triệu đồng -> gần như chắc chắn sai đơn vị
        ("Chi phí dự phòng ... là bao nhiêu triệu đồng?", 300_000, False),
        ("Chi phí khác ... là bao nhiêu triệu đồng?", 1_457_476, True),
        ("Tỷ lệ sở hữu ... là bao nhiêu phần trăm?", 51, True),
        ("Quỹ khen thưởng là bao nhiêu?", 5, True),
        ("Lợi nhuận sau thuế ... là bao nhiêu tỷ đồng?", 0, False),
    ],
)
def test_chan_ket_qua_vo_ly_tu_code_llm(question, value, expected):
    from src.qa_pipeline import FullQAPipeline

    assert FullQAPipeline._plausible_amount(question, value) is expected


def test_sandbox_cho_phep_import_pandas():
    """Code do LLM sinh thường mở đầu bằng 'import pandas as pd'."""
    ok, result, _ = Sandbox(timeout=15).execute(
        "import pandas as pd\nfinal_result = float(df_0['value'].max())",
        {"df_0": pd.DataFrame({"value": [1.0, 5.0]})},
    )
    assert ok and result == 5.0


def test_sandbox_chan_module_nguy_hiem():
    ok, _, error = Sandbox(timeout=15).execute(
        "import os\nfinal_result = 1", {"df_0": pd.DataFrame()}
    )
    assert not ok and "os" in error
