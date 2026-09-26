from transvoice.mt import glossary_terms


def test_weekday_after_time_word():
    assert glossary_terms("cuộc họp dời sang thứ năm nhé anh", "vi", "en") == [("thứ năm", "Thursday")]
    assert glossary_terms("Thứ bảy này tôi bận, chủ nhật thì được.", "vi", "ja") == [("Thứ bảy", "土曜日"), ("chủ nhật", "日曜日")]
    assert glossary_terms("hẹn gặp vào thứ 6", "vi", "en") == [("thứ 6", "Friday")]


def test_ordinal_is_not_a_weekday():
    assert glossary_terms("đây là lần thứ hai tôi đến Nhật", "vi", "en") == []
    assert glossary_terms("người thứ ba trong hàng", "vi", "ja") == []
    assert glossary_terms("Thứ hai, chúng ta cần thêm thời gian.", "vi", "en") == []


def test_only_vietnamese_source():
    assert glossary_terms("thứ năm", "ja", "en") == []
    assert glossary_terms("sang thứ năm", "vi", "vi") == []
