"""LLM Sexism / ToM audit harness for EXIST 2025.

Run harness for the experiment series specified in
``specs/llm-sexism-bias/`` (master spec ``specification-llm-sexism-tom.md``).
Implements the §6.2 run-safety design: complete per-call logging, parse-status
taxonomy, defensive parsing, idempotent resume, and a per-provider
circuit-breaker. E0 prompts are sourced verbatim from
``prompts-e0-baseline.md``.

This package is intentionally separate from ``nlpercep`` (the prior paper's
analysis pipeline); it reuses ``nlpercep.data.load_dataset`` for data loading.
"""

__all__ = []
