import os
from importlib import reload


def test_parse_text_basic():
    from app.services.parsing import parse_text

    out = parse_text("  Hello world  ")
    assert out.text == "Hello world"
    assert out.images == []


def test_parse_html_basic_without_vision(monkeypatch):
    monkeypatch.setenv("FEATURE_VISION", "0")
    from app.services import parsing

    reload(parsing)
    html = """
    <html><head><style>.x{}</style><script>1</script></head>
    <body><h1>Title</h1><p>Para 1</p><p>Para 2</p></body></html>
    """
    out = parsing.parse_html(html)
    assert "Title" in out.text and "Para 1" in out.text and "Para 2" in out.text
    assert out.images == [] and out.media_notes is None


def test_parse_html_with_vision(monkeypatch):
    monkeypatch.setenv("FEATURE_VISION", "1")
    from app.services import parsing

    reload(parsing)
    html = "<p>Text only</p>"
    out = parsing.parse_html(html)
    assert out.images and isinstance(out.images, list)
    assert out.media_notes and "Vision feature enabled" in out.media_notes
