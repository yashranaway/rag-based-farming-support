from app.services.llm import GraniteAdapter, FakeAdapter, LLMResponse


def test_fake_adapter_generates_deterministic_text():
    llm = FakeAdapter(response="hello world", model="fake")
    out = llm.generate("prompt here", max_tokens=10)
    assert isinstance(out, LLMResponse)
    assert out.text == "hello world"
    assert out.model == "fake"
    assert out.tokens_prompt >= 1
    assert out.tokens_output >= 1


def test_granite_adapter_stub_behavior_no_key_required():
    ga = GraniteAdapter(model="granite-13b-chat")
    prompt = "Advise farmer on tomato irrigation schedule based on forecast."
    out = ga.generate(prompt, max_tokens=32)
    assert isinstance(out, LLMResponse)
    assert out.model == "granite-13b-chat"
    assert out.text.startswith("[granite-stub:")
    assert "tomato" in out.text or len(out.text) > 0
