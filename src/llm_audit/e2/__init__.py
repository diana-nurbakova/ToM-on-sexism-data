"""E2 — counterfactual perturbation (gender swap).

Measures LLM flip-rate / soft-label divergence under the frozen woman↔man token
swap (master spec §5.3). Items are drawn from the full train+dev pool screened for
swappability (the post-fix E1 check), targeting equal-N 120 per category with the
take-all shortfall rule. Reuses the E1 ``swap`` engine, the E0 ``runner`` (via its
prompt hook), and the E0 prompts. A construct-rule first-pass tags every E2 item
with an invariance class (INVARIANT / SHIFTING / AMBIGUOUS / UNSWAPPABLE) for the
per-partition reporting, flagged for author adjudication.
"""

__all__ = []
