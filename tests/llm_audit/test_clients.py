"""Tests for error classification, retry/backoff, and the circuit-breaker."""

import pytest

from llm_audit.clients import BaseClient, CircuitBreakerTrip, RetryPolicy, classify_status
from llm_audit.config import ModelSpec

SPEC = ModelSpec("test", "anthropic", "test-model")


class _Boom(Exception):
    def __init__(self, status_code):
        super().__init__(f"status {status_code}")
        self.status_code = status_code


class _FlakyClient(BaseClient):
    """Raises a scripted sequence of errors, then succeeds."""

    def __init__(self, errors, **kw):
        super().__init__(SPEC, sleep=lambda s: None, **kw)
        self._errors = list(errors)
        self.attempts = 0

    def _call(self, system, user, params):
        self.attempts += 1
        if self._errors:
            raise self._errors.pop(0)
        return "ok", {"input_tokens": 1, "output_tokens": 1}


def test_classify_status():
    assert classify_status(429) == "retry"
    assert classify_status(503) == "retry"
    assert classify_status(None) == "retry"
    assert classify_status(401) == "circuit"
    assert classify_status(403) == "circuit"
    assert classify_status(400) == "fatal"


def test_retry_then_success_on_429():
    c = _FlakyClient([_Boom(429), _Boom(429)], retry=RetryPolicy(max_attempts=5))
    text, _ = c.complete("s", "u", {})
    assert text == "ok"
    assert c.attempts == 3


def test_circuit_breaker_on_403():
    c = _FlakyClient([_Boom(403)])
    with pytest.raises(CircuitBreakerTrip) as ei:
        c.complete("s", "u", {})
    assert ei.value.provider == "anthropic"
    assert ei.value.status_code == 403


class _Credit400(Exception):
    status_code = 400
    def __str__(self):
        return "Error code: 400 - Your credit balance is too low to access the Anthropic API."


def test_credit_exhaustion_400_trips_circuit_breaker():
    c = _FlakyClient([_Credit400()])
    with pytest.raises(CircuitBreakerTrip):
        c.complete("s", "u", {})


def test_fatal_error_propagates():
    c = _FlakyClient([_Boom(400)])
    with pytest.raises(_Boom):
        c.complete("s", "u", {})


def test_ollama_transient_retry_and_partial_recovery():
    import httpx
    from llm_audit.clients import OllamaClient, RetryPolicy
    from llm_audit.config import ModelSpec

    spec = ModelSpec("pagoda", "ollama", "llama3.3:70b")
    c = OllamaClient(spec, host="http://localhost:11434", api_key="tok",
                     retry=RetryPolicy(max_attempts=3), sleep=lambda s: None)

    # (a) transient error then success -> retried, returns the good result.
    calls = {"n": 0}
    def stream_then_ok(system, user, params):
        calls["n"] += 1
        if calls["n"] < 2:
            raise httpx.ConnectError("proxy disconnect")
        return "sexist", {"partial": False}
    c._stream = stream_then_ok
    assert c.complete("s", "u", {})[0] == "sexist"
    assert calls["n"] == 2

    # (b) always transient with partial content -> returns partial on exhaustion.
    def always_transient(system, user, params):
        exc = httpx.ReadError("connection reset")
        exc._partial = "not_sex"
        raise exc
    c._stream = always_transient
    text, meta = c.complete("s", "u", {})
    assert text == "not_sex" and meta["partial"] is True

    # (c) non-transient error propagates immediately.
    def fatal(system, user, params):
        raise ValueError("bad request")
    c._stream = fatal
    with pytest.raises(ValueError):
        c.complete("s", "u", {})


def test_retries_exhausted_raises_runtime():
    c = _FlakyClient([_Boom(429)] * 10, retry=RetryPolicy(max_attempts=3))
    with pytest.raises(RuntimeError):
        c.complete("s", "u", {})
    assert c.attempts == 3
