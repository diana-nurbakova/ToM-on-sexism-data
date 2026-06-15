"""E3 — ToM-scaffolding heterogeneity (H7).

Five scaffold conditions (neutral, generic-deliberation control, cognitive ToM,
affective ToM, combined) wrapping the E0 bases — 1.1 on the bare base, 1.3 on the
defined base — reusing the E0 system prompt and elicitation suffixes
(``prompts-e3-tom-scaffolding.md``). Directive mode (primary). The scaffold clauses
are verbatim from the spec; the affective clause is deliberately target-neutral
(PE10). Runs via the E0 ``runner`` through its prompt hook.

NOTE: the Strachan ToM-capacity covariate (H7b predictor) and the MentalRiskES
control are *external published benchmarks* (spec §7), not built here — they need
the released items.

  * ``sampling`` — bin-first stratified 500-item sub-sample (5 categories × 3
    detection-agreement bins, seed 42), reused by E4 and H1c.
"""

from . import sampling  # noqa: F401

__all__ = ["sampling"]
