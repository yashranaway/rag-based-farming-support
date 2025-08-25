from app.api.models import QueryRequest, AnswerResponse, Citation, Diagnostics, Location, UserPreferences
from datetime import datetime, UTC


def test_user_preferences_verbosity_validation():
    prefs = UserPreferences(verbosity="basic")
    assert prefs.verbosity == "basic"


def test_user_preferences_invalid_verbosity():
    import pytest

    with pytest.raises(ValueError):
        UserPreferences(verbosity="verbose")


def test_query_request_serialization():
    req = QueryRequest(
        text="What crop should I plant?",
        locale="hi-IN",
        location=Location(gps=(19.0760, 72.8777), pincode="400001", district="Mumbai"),
        preferences=UserPreferences(language="hi", verbosity="detailed"),
    )
    data = req.model_dump()
    assert data["text"] == "What crop should I plant?"
    assert data["locale"] == "hi-IN"
    assert data["location"]["pincode"] == "400001"


def test_answer_response_serialization():
    resp = AnswerResponse(
        answer="Plant tomatoes in early October.",
        language="hi",
        citations=[Citation(title="IMD Advisory", url="https://example.com", timestamp=datetime.now(UTC))],
        warnings=["Heavy rain expected"],
        diagnostics=Diagnostics(latency_ms=1200, tokens_prompt=500, tokens_output=120, retrieval_k=6),
    )
    data = resp.model_dump()
    assert data["answer"].startswith("Plant")
    assert data["language"] == "hi"
    assert isinstance(data["citations"], list)
