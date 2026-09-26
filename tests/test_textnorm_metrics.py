import math

from bench.metrics import asr_error_rate, translation_flags
from transvoice.textnorm import asr_text_for_mt, normalize, script_ok


def test_normalize_japanese_drops_punct_and_spaces():
    assert normalize("技術決定論の、ほとんど。 「影響」", "ja") == "技術決定論のほとんど影響"


def test_normalize_fullwidth_digits_to_ascii():
    assert normalize("２つ", "ja") == "2つ"


def test_normalize_vietnamese_keeps_diacritics():
    assert normalize("Xin chào, Việt Nam!", "vi") == "xin chào việt nam"


def test_asr_text_for_mt_fixes_all_caps_only():
    assert asr_text_for_mt("THỰC VẬT TẠO RA THỨC ĂN") == "Thực vật tạo ra thức ăn"
    assert asr_text_for_mt("He works at NASA and IBM.") == "He works at NASA and IBM."
    assert asr_text_for_mt("これはペンです") == "これはペンです"


def test_script_ok():
    assert script_ok("これはペンです", "ja")
    assert not script_ok("This is a pen", "ja")
    assert script_ok("Đây là cây bút", "vi")
    assert not script_ok("This is a pen and it is quite long", "vi")
    assert script_ok("This is a pen", "en")
    assert not script_ok("これはペンです", "en")


def test_asr_error_rate_cer_and_wer():
    assert asr_error_rate(["あいうえお"], ["あいうえか"], "ja") == 20.0
    assert asr_error_rate(["a b c d"], ["a b x d"], "en") == 25.0
    assert asr_error_rate(["a b"], [""], "en") == 100.0
    assert math.isnan(asr_error_rate([""], ["x"], "en"))


def test_translation_flags():
    ref = "Tôi thích ăn phở vào buổi sáng."
    assert translation_flags("src", "Tôi thích ăn phở buổi sáng.", ref, "vi") == []
    assert translation_flags("src", "", ref, "vi") == ["empty"]
    assert "wrong_language" in translation_flags("src", "I like eating pho in the morning.", ref, "vi")
    assert "too_short" in translation_flags("src", "Tôi.", ref, "vi")
    assert "repetition" in translation_flags("src", "tôi thích tôi thích tôi thích tôi thích", ref, "vi")
    assert "commentary" in translation_flags("src", "Bản dịch: Tôi thích ăn phở vào buổi sáng.", ref, "vi")
    assert "unknown_token" in translation_flags("src", "昆虫, ⁇ 類,鳥などの獲物を食べます", "昆虫、げっ歯類、鳥などの獲物を食べます", "ja")
