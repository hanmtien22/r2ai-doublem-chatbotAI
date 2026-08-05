from __future__ import annotations

import re
import unicodedata


def normalize_unicode(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    replacements = {
        "\u2013": "-",
        "\u2014": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2026": "...",
        "\u00a0": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"\s+", " ", text).strip()
    return text


_DIACRITICS_MAP = str.maketrans(
    "àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ"
    "ÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴĐ",
    "a" * 17 + "e" * 11 + "i" * 5 + "o" * 17 + "u" * 11 + "y" * 5 + "d"
    + "A" * 17 + "E" * 11 + "I" * 5 + "O" * 17 + "U" * 11 + "Y" * 5 + "D",
)


def remove_diacritics(text: str) -> str:
    return text.translate(_DIACRITICS_MAP)


def clean_number(text: str) -> float | None:
    text = text.strip()
    text = text.replace(" ", "")

    if re.fullmatch(r"-?[\d.]+,\d+", text):
        text = text.replace(".", "").replace(",", ".")
    elif re.fullmatch(r"-?[\d,]+\.\d+", text):
        text = text.replace(",", "")
    else:
        text = text.replace(".", "").replace(",", ".")

    text = text.strip("()")
    if text.startswith("(") or text.endswith(")"):
        text = "-" + text.strip("()")

    try:
        return float(text)
    except ValueError:
        return None
