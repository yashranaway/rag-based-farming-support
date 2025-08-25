from app.services.lang import detect_language, choose_response_language, DetectedLanguage


def test_detect_language_english():
    text = "What crop is best for Kharif season?"
    det = detect_language(text)
    assert det is not None
    assert det.code in {"en"}
    assert det.confidence >= 0.5


def test_detect_language_hindi_devanagari():
    text = "आज के मौसम में कौन सी फसल उचित है?"
    det = detect_language(text)
    assert det is not None
    assert det.code == "hi"
    assert det.confidence >= 0.2


def test_choose_response_language_priority():
    # prefs override
    lang = choose_response_language(DetectedLanguage("hi", 0.9), prefs_language="mr", locale="hi-IN")
    assert lang == "mr"
    # then locale
    lang = choose_response_language(DetectedLanguage("hi", 0.9), prefs_language=None, locale="ta-IN")
    assert lang == "ta-IN"
    # then detected
    lang = choose_response_language(DetectedLanguage("hi", 0.9), prefs_language=None, locale=None)
    assert lang == "hi"
    # default
    lang = choose_response_language(None, None, None)
    assert lang == "auto"
