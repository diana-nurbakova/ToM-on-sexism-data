"""Backend clients: a unified ``complete()`` with retry/backoff + circuit-breaker.

Per master spec §6.2, API failures are separated by type:
  * HTTP 429 / 5xx  -> transient: exponential backoff + retry.
  * HTTP 401/403 / policy flag -> dangerous: raise ``CircuitBreakerTrip`` so the
    runner halts *all* calls to that provider and waits for a human decision.

``AnthropicClient`` (direct) is wired for the step-zero smoke; the OpenAI-compatible
client covers OpenAI / DeepInfra / OpenRouter via a swapped ``base_url``. ``FakeClient``
backs the offline tests. Every client returns ``(raw_text, meta)`` where ``meta``
carries token usage and stop reason for logging.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from . import config
from .config import ModelSpec


class CircuitBreakerTrip(RuntimeError):
    """Raised on an auth/permission/policy failure — halt the provider lane."""

    def __init__(self, provider: str, message: str, status_code: int | None = None):
        super().__init__(f"[{provider}] circuit-breaker: {message}")
        self.provider = provider
        self.status_code = status_code


@dataclass
class RetryPolicy:
    max_attempts: int = 6
    base_delay: float = 1.0
    max_delay: float = 60.0

    def delay(self, attempt: int) -> float:
        return min(self.max_delay, self.base_delay * (2 ** attempt))


def classify_status(status_code: int | None) -> str:
    """Map an HTTP status to a handling class: retry / circuit / fatal."""
    if status_code is None:
        return "retry"  # network/timeout with no status — treat as transient
    if status_code == 429 or status_code >= 500:
        return "retry"
    if status_code in (401, 403):
        return "circuit"
    return "fatal"


# Account-level failures that arrive as a 400 but are pointless to retry per-call —
# the whole provider lane should halt (e.g. credit/billing exhaustion).
_HALT_MESSAGE_CUES = ("credit balance", "billing", "insufficient", "quota exceeded",
                      "payment required")


def is_halt_message(message: str) -> bool:
    low = message.lower()
    return any(cue in low for cue in _HALT_MESSAGE_CUES)


def _status_of(exc: Exception) -> int | None:
    return getattr(exc, "status_code", None)


class BaseClient:
    """Wraps a concrete ``_call`` with retry/backoff and circuit-breaker logic."""

    def __init__(self, spec: ModelSpec, retry: RetryPolicy | None = None, sleep=time.sleep):
        self.spec = spec
        self.retry = retry or RetryPolicy()
        self._sleep = sleep

    # Concrete clients override this.
    def _call(self, system: str, user: str, params: dict) -> tuple[str, dict]:
        raise NotImplementedError

    def complete(self, system: str, user: str, params: dict) -> tuple[str, dict]:
        last_exc: Exception | None = None
        for attempt in range(self.retry.max_attempts):
            try:
                return self._call(system, user, params)
            except CircuitBreakerTrip:
                raise
            except Exception as exc:  # noqa: BLE001 — classified below
                status = _status_of(exc)
                kind = classify_status(status)
                # A 400 that means the account/lane is unusable (credit/billing) is
                # halt-worthy, not a per-call fatal — trip the breaker so we don't
                # burn the rest of the run pointlessly.
                if kind == "fatal" and is_halt_message(str(exc)):
                    kind = "circuit"
                if kind == "circuit":
                    raise CircuitBreakerTrip(self.spec.backend, str(exc), status) from exc
                if kind == "fatal":
                    raise
                last_exc = exc
                if attempt < self.retry.max_attempts - 1:
                    self._sleep(self.retry.delay(attempt))
        raise RuntimeError(
            f"exhausted {self.retry.max_attempts} retries for {self.spec.name}"
        ) from last_exc


class AnthropicClient(BaseClient):
    def __init__(self, spec: ModelSpec, api_key: str, **kw):
        super().__init__(spec, **kw)
        import anthropic

        self._sdk = anthropic
        self._client = anthropic.Anthropic(api_key=api_key)

    def _call(self, system: str, user: str, params: dict) -> tuple[str, dict]:
        resp = self._client.messages.create(
            model=self.spec.model_id,
            max_tokens=params.get("max_tokens", 512),
            temperature=params.get("temperature", config.TEMPERATURE),
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        meta = {
            "stop_reason": resp.stop_reason,
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
            "response_model": resp.model,
        }
        return text, meta


class OpenAICompatibleClient(BaseClient):
    """OpenAI / DeepInfra / OpenRouter via the openai SDK with a base_url swap."""

    def __init__(self, spec: ModelSpec, api_key: str, base_url: str | None = None, **kw):
        super().__init__(spec, **kw)
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def _call(self, system: str, user: str, params: dict) -> tuple[str, dict]:
        resp = self._client.chat.completions.create(
            model=self.spec.model_id,
            temperature=params.get("temperature", config.TEMPERATURE),
            max_tokens=params.get("max_tokens", 512),
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        choice = resp.choices[0]
        meta = {
            "stop_reason": choice.finish_reason,
            "input_tokens": getattr(resp.usage, "prompt_tokens", None),
            "output_tokens": getattr(resp.usage, "completion_tokens", None),
            "response_model": resp.model,
        }
        return choice.message.content or "", meta


class FakeClient(BaseClient):
    """Offline test double: returns scripted responses by call index or callable."""

    def __init__(self, spec: ModelSpec, responses, **kw):
        super().__init__(spec, **kw)
        self._responses = responses
        self.calls: list[tuple[str, str, dict]] = []

    def _call(self, system: str, user: str, params: dict) -> tuple[str, dict]:
        idx = len(self.calls)
        self.calls.append((system, user, params))
        r = self._responses(system, user, params) if callable(self._responses) else \
            self._responses[idx % len(self._responses)]
        text = r if isinstance(r, str) else r[0]
        meta = {} if isinstance(r, str) else r[1]
        return text, {"input_tokens": 0, "output_tokens": 0, **meta}


# Pagoda's reverse proxy drops connections under concurrency, so cap in-flight
# Ollama requests to 2 across the whole process (shared module-level semaphore).
_OLLAMA_SEM = threading.Semaphore(2)


class OllamaClient(BaseClient):
    """Pagoda Ollama via the ollama SDK + bearer token, tuned for a slow proxied 70B.

    Per the lab setup: official ollama client (not plain HTTP), bearer auth, generous
    timeouts (read=600s), ``keep_alive='30m'`` to keep the model GPU-resident, a
    2-request concurrency cap, and extended retries on transient proxy disconnects
    with partial-streamed-content recovery on final exhaustion.
    """

    def __init__(self, spec: ModelSpec, host: str, api_key: str | None = None,
                 retry: RetryPolicy | None = None, **kw):
        super().__init__(spec, retry=retry or RetryPolicy(max_attempts=12), **kw)
        import httpx
        import ollama

        self._httpx = httpx
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = ollama.Client(
            host=host, headers=headers,
            timeout=httpx.Timeout(connect=30.0, read=600.0, write=30.0, pool=600.0),
        )

    def _is_transient(self, exc: Exception) -> bool:
        httpx = self._httpx
        if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError,
                            httpx.RemoteProtocolError, httpx.ReadError, ConnectionError)):
            return True
        low = str(exc).lower()
        return any(c in low for c in ("disconnect", "timeout", "connection reset",
                                      "eof", "broken pipe"))

    def _stream(self, system: str, user: str, params: dict) -> tuple[str, dict]:
        acc: list[str] = []
        try:
            for chunk in self._client.chat(
                model=self.spec.model_id,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                options={"temperature": params.get("temperature", config.TEMPERATURE),
                         "num_predict": params.get("max_tokens", 512)},
                keep_alive="30m", stream=True,
            ):
                acc.append(chunk.get("message", {}).get("content", ""))
        except Exception as exc:  # attach partial so the retry loop can recover it
            exc._partial = "".join(acc)  # type: ignore[attr-defined]
            raise
        return "".join(acc), {"partial": False, "response_model": self.spec.model_id}

    def complete(self, system: str, user: str, params: dict) -> tuple[str, dict]:
        last_partial, last_exc = "", None
        for attempt in range(self.retry.max_attempts):
            try:
                with _OLLAMA_SEM:
                    return self._stream(system, user, params)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                last_partial = getattr(exc, "_partial", "") or last_partial
                if not self._is_transient(exc):
                    raise
                if attempt < self.retry.max_attempts - 1:
                    self._sleep(self.retry.delay(attempt))
        if last_partial:  # final exhaustion: keep the streamed work rather than discard it
            return last_partial, {"partial": True, "error": repr(last_exc)}
        raise RuntimeError(
            f"exhausted {self.retry.max_attempts} retries for {self.spec.name}"
        ) from last_exc


def make_client(spec: ModelSpec, env: dict[str, str] | None = None, **kw) -> BaseClient:
    """Construct the right client for a model spec from its backend + credentials."""
    env = env if env is not None else config.load_env()
    if spec.backend == config.BACKEND_OLLAMA:
        host = env.get(config.OLLAMA_BASE_URL_ENV)
        if not host:
            raise RuntimeError(
                f"Missing {config.OLLAMA_BASE_URL_ENV} for Pagoda backend (model "
                f"'{spec.name}'). Set it in .env and bring up the lab VPN.")
        return OllamaClient(spec, host=host, api_key=env.get("OLLAMA_API_KEY"), **kw)
    api_key = config.api_key_for(spec, env)
    if spec.backend == config.BACKEND_ANTHROPIC:
        return AnthropicClient(spec, api_key, **kw)
    base_url = config.BACKEND_BASE_URL.get(spec.backend)
    return OpenAICompatibleClient(spec, api_key, base_url=base_url, **kw)
