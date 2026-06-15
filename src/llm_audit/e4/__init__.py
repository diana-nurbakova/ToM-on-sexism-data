"""E4 — base-vs-instruct and counter-priming (H6).

Separates the safety-RLHF prior (H6a) from sycophancy (H6b) across open-weight
base/instruct pairs. Counter-prime framings (``prompts-e4-counter-priming.md``) wrap
the E0 *bare* 1.1 body as user-turn prefixes: three relaxing (free-speech,
banter/harm-minimising, men's-rights) and one tightening (safety), with free-speech
and safety forming the matched bidirectional minimal pair (§3.5). User-stated mode is
primary (sycophancy); persona-adoption mode is a robustness slice (role-steering).
E4 is open-weight-only, so the high-pressure framings carry no provider-suspension
risk. H6a's per-category gap and the swap contrast reuse the E0 prompts directly.

Runs via the E0 ``Runner`` through its prompt hook.

  * ``sampling`` — the 500-item sub-sample, the same items as E3 (full overlap,
    §5.5); delegates to ``e3.sampling`` so the two cannot drift.
"""

from . import sampling  # noqa: F401

__all__ = ["sampling"]
