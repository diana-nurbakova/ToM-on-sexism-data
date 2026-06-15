"""E5 — Cialdini persuasion probe (H9), the integrative sub-study.

Thirteen conditions (master spec §5.6, ``prompts-e5-cialdini.md``): a neutral
control plus three Cialdini principles — authority (4 cells: stance × verdict),
social proof of approval (6 cells: 3 intensities × 2 directions), and
commitment/consistency (2 cells: prior direction). Per the author's decision, E5 is
harmonised onto the E0 *bare* base: it shares the E0 system prompt, bare body, and
elicitation suffixes with E0/E4, and only the Cialdini framing content (third-party
expert verdicts, third-party-crowd majorities, labelled precedents) differs — keeping
E5 structurally parallel to E4's first-person counter-primes without a base-prompt
confound. Runs via the E0 ``Runner`` through its prompt hook; the commitment cells
draw a per-item exemplar from a fixed pairing schedule (``exemplars`` module).

Open-weight-first for the high account-suspension risk (§5.6/§6.2); proprietary cells
stay serial behind the runner's circuit-breaker.

  * ``sampling`` — the 200-item sub-sample stratified by category × E1 invariance
    class, overlapping with the E1 items (§5.6).
  * ``exemplars`` — the commitment/consistency exemplar pools + pairing schedule.
"""

from . import exemplars, sampling  # noqa: F401

__all__ = ["exemplars", "sampling"]
