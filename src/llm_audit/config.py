"""Model registry, backend routing, sampling defaults, and ``.env`` loading.

The registry maps a stable *logical* model name (used everywhere in the logs and
results tree) to a backend + provider snapshot id + quantisation, per master spec
§6 (model panel) and §6.1 (serving routes). Only the entries needed for a given
run must have working credentials; the smoke test uses ``claude-4-sonnet`` only.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import dotenv_values

# Repo root = three levels up from this file (src/llm_audit/config.py).
REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = REPO_ROOT / ".env"
RESULTS_ROOT = REPO_ROOT / "results"

# ── Sampling defaults (master spec §5.1) ──────────────────────────────────────
TEMPERATURE = 0.7
K_SAMPLES = 6  # EXIST six-annotator denominator; Format C draws K, A/B draw 1.

# ── Backends (serving routes, §6.1) ───────────────────────────────────────────
BACKEND_ANTHROPIC = "anthropic"        # direct provider API (proprietary)
BACKEND_OPENAI = "openai"              # direct provider API (proprietary)
BACKEND_DEEPINFRA = "deepinfra"        # hosted open-weight, OpenAI-compatible
BACKEND_OPENROUTER = "openrouter"      # hosted, OpenAI-compatible
BACKEND_OLLAMA = "ollama"              # Pagoda local via ollama client + bearer token

# Which .env key and base_url each backend uses. OpenAI-compatible backends are
# driven through the openai SDK with a swapped base_url.
BACKEND_ENV_KEY = {
    BACKEND_ANTHROPIC: "ANTHROPIC_API_KEY",
    BACKEND_OPENAI: "OPENAI_API_KEY",
    BACKEND_DEEPINFRA: "DEEPINFRA_API_KEY",
    BACKEND_OPENROUTER: "OPENROUTER_API_KEY",
    BACKEND_OLLAMA: "OLLAMA_API_KEY",
}
BACKEND_BASE_URL = {
    BACKEND_DEEPINFRA: "https://api.deepinfra.com/v1/openai",
    BACKEND_OPENROUTER: "https://openrouter.ai/api/v1",
}
# Pagoda Ollama host comes from this env var (set behind the lab VPN).
OLLAMA_BASE_URL_ENV = "OLLAMA_BASE_URL"

# Backends that need account-suspension protection (the circuit-breaker lane,
# §6.2). Open-weight hosts have no account to suspend in the same way, but the
# breaker treats 401/403 on any backend as halt-worthy.
PROPRIETARY_BACKENDS = frozenset({BACKEND_ANTHROPIC, BACKEND_OPENAI})


@dataclass(frozen=True)
class ModelSpec:
    """A single panel entry.

    Attributes:
        name: stable logical name used in logs/paths (e.g. ``claude-4-sonnet``).
        backend: one of the BACKEND_* constants.
        model_id: the provider-side snapshot string sent on the wire.
        quantisation: quant level for open-weight hosts; ``None`` for proprietary.
        role: short human label of the panel role (master spec §6 table).
    """

    name: str
    backend: str
    model_id: str
    quantisation: str | None = None
    role: str = ""

    @property
    def is_proprietary(self) -> bool:
        return self.backend in PROPRIETARY_BACKENDS


# ── Panel registry ────────────────────────────────────────────────────────────
# Only the smoke-test entry is fully wired now; the rest are placeholders kept in
# one place so adding a backend later is a registry edit, not a code change. The
# default backend recorded here is the *intended* route; routing can be overridden
# per run. Snapshot ids for the open-weight panel are filled in when those cells
# are scheduled (master spec §6.1 one-backend-per-experiment constraint).
MODEL_REGISTRY: dict[str, ModelSpec] = {
    # Smoke-test / step-zero reference model. Pinned to the original Claude 4
    # Sonnet snapshot, matching the literal "Claude 4 Sonnet" in the spec panel.
    "claude-4-sonnet": ModelSpec(
        name="claude-4-sonnet",
        backend=BACKEND_ANTHROPIC,
        model_id="claude-sonnet-4-20250514",
        role="Anthropic frontier (step-zero reference)",
    ),
    # ── proprietary panel (filled as scheduled) ──
    "gpt-4.1": ModelSpec("gpt-4.1", BACKEND_OPENAI, "gpt-4.1", role="OpenAI representative"),
    "gpt-4o": ModelSpec("gpt-4o", BACKEND_OPENAI, "gpt-4o", role="OpenAI version contrast"),
    # ── open-weight panel via DeepInfra (ids verified live) ──
    "llama-3.3-70b-instruct": ModelSpec(
        "llama-3.3-70b-instruct", BACKEND_DEEPINFRA,
        "meta-llama/Llama-3.3-70B-Instruct", role="Meta representative; regression secondary",
    ),
    "qwen-2.5-72b-instruct": ModelSpec(
        "qwen-2.5-72b-instruct", BACKEND_DEEPINFRA,
        "Qwen/Qwen2.5-72B-Instruct", role="Alibaba representative",
    ),
    "deepseek-v3": ModelSpec(
        "deepseek-v3", BACKEND_DEEPINFRA,
        "deepseek-ai/DeepSeek-V3", role="Chinese-origin frontier; cost-efficient",
    ),
    "llama-3.1-8b-instruct": ModelSpec(
        "llama-3.1-8b-instruct", BACKEND_DEEPINFRA,
        "meta-llama/Meta-Llama-3.1-8B-Instruct", role="Capability-tier reference",
    ),
    "mistral-7b-instruct": ModelSpec(
        "mistral-7b-instruct", BACKEND_DEEPINFRA,
        "mistralai/Mistral-7B-Instruct-v0.3", role="Mistral family",
    ),
    # Gemma 4 31B (the spec-panel model), served on OpenRouter. The H7 heterogeneity
    # candidate. (gemma-3-27b-it kept below as the earlier stand-in / contrast.)
    "gemma-4-31b": ModelSpec(
        "gemma-4-31b", BACKEND_OPENROUTER,
        "google/gemma-4-31b-it", role="Heterogeneity test (H7); Gemma 4 31B",
    ),
    "gemma-3-27b-it": ModelSpec(
        "gemma-3-27b-it", BACKEND_OPENROUTER,
        "google/gemma-3-27b-it", role="Gemma 3 27B (earlier stand-in / contrast)",
    ),
    # ── Strachan reproduction models (E3 §5.4 "same ruler" calibration) ──
    # Run the released Strachan battery on the paper's original models under our
    # harness, to compare against their published scores. IDs to verify at run time.
    "gpt-4-strachan": ModelSpec(
        "gpt-4-strachan", BACKEND_OPENAI, "gpt-4", role="Strachan reproduction (GPT-4)",
    ),
    "gpt-3.5-turbo-strachan": ModelSpec(
        "gpt-3.5-turbo-strachan", BACKEND_OPENAI, "gpt-3.5-turbo",
        role="Strachan reproduction (GPT-3.5)",
    ),
    "llama-2-70b-strachan": ModelSpec(
        "llama-2-70b-strachan", BACKEND_DEEPINFRA, "meta-llama/Llama-2-70b-chat-hf",
        role="Strachan reproduction (LLaMA2-70B)",
    ),
    # ── Pagoda / Ollama (Lane D, behind lab VPN; reverse proxy + bearer token) ──
    # Separate logical names so a model is never split across backends mid-comparison
    # (§6.1 one-backend-per-experiment).
    "llama-3.3-70b-pagoda": ModelSpec(
        "llama-3.3-70b-pagoda", BACKEND_OLLAMA, "llama3.3:70b",
        quantisation="ollama-default", role="Meta representative (Pagoda local)",
    ),
    "mistral-7b-pagoda": ModelSpec(
        "mistral-7b-pagoda", BACKEND_OLLAMA, "mistral:7b",
        quantisation="ollama-default", role="Mistral family (Pagoda local)",
    ),
}


def get_model(name: str) -> ModelSpec:
    if name not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model '{name}'. Known: {sorted(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[name]


def load_env() -> dict[str, str]:
    """Read ``.env`` from the repo root, falling back to the process env.

    Returns a plain dict of the credentials present. Values from ``.env`` take
    precedence over the ambient environment so a run is reproducible from the
    file the repo ships.
    """
    values: dict[str, str] = {}
    if ENV_PATH.exists():
        values.update({k: v for k, v in dotenv_values(ENV_PATH).items() if v is not None})
    for key in BACKEND_ENV_KEY.values():
        if key not in values and os.environ.get(key):
            values[key] = os.environ[key]
    return values


# Isolated PyEvALL environment (master spec §7.1). ICM-Soft is computed
# out-of-process because PyEvALL pins heavy deps; this points at the interpreter
# of the env where PyEvALL is installed.
PYEVALL_PYTHON_ENV = "PYEVALL_PYTHON"


def pyevall_python() -> str | None:
    """Path to the python executable of the isolated PyEvALL env, or ``None``.

    Read from ``PYEVALL_PYTHON`` in ``.env`` or the ambient environment. ICM-Soft
    (``llm_audit.icm_soft``) shells out to this interpreter so PyEvALL's
    dependencies never enter the main environment (master spec §7.1).
    """
    val = load_env().get(PYEVALL_PYTHON_ENV) or os.environ.get(PYEVALL_PYTHON_ENV)
    return val or None


def api_key_for(spec: ModelSpec, env: dict[str, str] | None = None) -> str:
    """Return the API key for a model's backend, raising if it is missing."""
    env = env if env is not None else load_env()
    key_name = BACKEND_ENV_KEY[spec.backend]
    key = env.get(key_name)
    if not key:
        raise RuntimeError(
            f"Missing credential {key_name} for backend '{spec.backend}' "
            f"(model '{spec.name}'). Add it to {ENV_PATH}."
        )
    return key
