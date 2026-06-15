"""E1 — label-invariance audit (construct-derivation study).

Implements the deterministic core of the OSF pre-registration
(``specs/llm-sexism-bias/OSF/E1-preregistration-OSF.md``):

  * ``sampling`` — bin-first stratified 200-item sample (5 categories × 3 detection
    agreement bins, seed 42), disjoint from the regression set.
  * ``swap`` — the frozen woman↔man token swap (Appendix A): pronouns with the
    POS-disambiguated "her" rule, gendered nouns, honorifics, curated names with a
    name-swap flag, and UNSWAPPABLE / author-review flagging.
  * ``rule`` — the frozen construct-derived first-pass rule (§6 step 1), shared with
    E2 (``e2.tagging`` re-imports it) so the two cannot drift.
  * ``worksheet`` — the adjudication workflow: build the worksheet (swap + rule
    first-pass + blank author cells), then finalise the author-completed worksheet
    into the pre-/post-adjudication partitions E2/E5 condition on.
  * ``judge`` — the GPT-4.1 / Claude 4 Sonnet LLM-judge cross-check (E1-H3).
  * ``analysis`` — E1-H1/H2/H3 (SHIFTING proportion, category association, judge
    convergence) over the author-resolved partition.

Author adjudication of the AMBIGUOUS/SEXIST residual is the one expert step the
workflow cannot automate; everything around it is deterministic.
"""

from . import analysis, judge, rule, sampling, swap, worksheet  # noqa: F401

__all__ = ["analysis", "judge", "rule", "sampling", "swap", "worksheet"]
