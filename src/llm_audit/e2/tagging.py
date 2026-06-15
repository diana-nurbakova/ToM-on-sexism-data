"""Construct-rule first-pass invariance tagging for E2 items (OSF pre-reg §6).

Applies the transparent woman-focused-construct rule to assign each E2 item a
first-pass invariance class. This is a *first pass*, not the final partition: every
SEXIST-derived tag is flagged ``needs_review`` for author adjudication (the
construct judgement is the author's expert task). The decision to tag all E2 items
(not just E1's 200) is per the run-plan choice, so E2 can report per invariance
partition over its whole item set.

The rule itself is the single frozen one registered for E1 and lives in
``e1.rule.construct_rule_tag`` — imported here (not re-implemented) so E1 and E2
share one definition, per master spec §5.3.
"""

from __future__ import annotations

from ..e1.rule import construct_rule_tag  # frozen, shared E1/E2 rule (pre-reg §6)

__all__ = ["construct_rule_tag", "tag_e2_sample"]


def tag_e2_sample(sample):
    """Add first-pass ``invariance_class`` / ``needs_review`` / ``justification`` columns."""
    sample = sample.copy()
    tags = sample.apply(
        lambda r: construct_rule_tag(
            r["gold_1_1"]["sexist"], r["gendered_swapped"],
            r["name_swapped"], not r["swappable"]),
        axis=1)
    sample["invariance_class"] = [t[0] for t in tags]
    sample["needs_review"] = [t[1] for t in tags]
    sample["justification"] = [t[2] for t in tags]
    return sample
