"""Text normalization for scoring ASR output and sanity-checking MT output."""
import re
import unicodedata

_WS = re.compile(r"\s+")
# Letters that appear in Vietnamese but essentially never in plain English.
_VI_CHARS = set(
    "ăâđêôơưàáạảãằắặẳẵầấậẩẫèéẹẻẽềếệểễìíịỉĩòóọỏõồốộổỗờớợởỡùúụủũừứựửữỳýỵỷỹ"
)


def normalize(text: str, lang: str) -> str:
    """NFKC + lowercase + strip punctuation/symbols. Japanese drops all spaces (scored per character)."""
    t = unicodedata.normalize("NFKC", text).lower()
    t = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in t)
    if lang == "ja":
        return _WS.sub("", t)
    return _WS.sub(" ", t).strip()


def asr_text_for_mt(text: str) -> str:
    """Some ASR models (e.g. the Vietnamese zipformers) emit ALL CAPS; MT models read that as shouting or
    acronyms and start inventing content. Convert to sentence case before translating."""
    t = text.strip()
    letters = [c for c in t if c.isalpha()]
    if letters and sum(c.isupper() for c in letters) / len(letters) > 0.8:
        t = t.lower()
        t = t[:1].upper() + t[1:]
    return t


def is_japanese_char(c: str) -> bool:
    o = ord(c)
    return (
        0x3040 <= o <= 0x30FF  # hiragana + katakana
        or 0x3400 <= o <= 0x4DBF  # CJK ext A
        or 0x4E00 <= o <= 0x9FFF  # CJK unified
        or 0xFF66 <= o <= 0xFF9F  # half-width katakana
        or c == "々"
    )


def script_ok(text: str, lang: str) -> bool:
    """Is `text` plausibly written in `lang`? Catches outputs in the wrong language."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    ja_share = sum(is_japanese_char(c) for c in letters) / len(letters)
    if lang == "ja":
        return ja_share >= 0.5
    if ja_share > 0.1:
        return False
    lower = text.lower()
    vi_count = sum(c in _VI_CHARS for c in lower)
    if lang == "vi":
        return len(letters) < 15 or vi_count > 0
    if lang == "en":
        return vi_count / len(letters) < 0.05
    return True
