import os
from contextlib import contextmanager

import pytest

from app.services.llm import GraniteAdapter, FakeAdapter


@contextmanager
def env(**kwargs):
    old = {k: os.environ.get(k) for k in kwargs}
    try:
        for k, v in kwargs.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_stop_sequences_truncate_output():
    llm = FakeAdapter(response="hello STOP there", model="fake")
    out = llm.generate("ignored", stop=["STOP"])
    assert out.text == "hello "


def test_stream_respects_stop_sequence():
    llm = GraniteAdapter()
    chunks = list(llm.stream_generate("abc STOP def", stop=["STOP"]))
    # After stop, nothing further should appear once merged
    merged = "".join(chunks)
    assert "STOP" not in merged


def test_simulated_quota_error_in_generate():
    with env(LLM_SIMULATE_ERROR="quota"):
        llm = GraniteAdapter()
        with pytest.raises(RuntimeError) as exc:
            llm.generate("x")
        assert "quota_exceeded" in str(exc.value)


def test_simulated_credit_error_in_stream():
    with env(LLM_SIMULATE_ERROR="credit"):
        llm = GraniteAdapter()
        with pytest.raises(RuntimeError) as exc:
            list(llm.stream_generate("x"))
        assert "insufficient_credit" in str(exc.value)
