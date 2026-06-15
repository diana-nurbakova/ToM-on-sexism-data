"""Frozen construct-derived labelling rule (OSF pre-reg §6 step 1).

This is the single, canonical first-pass rule that applies the official EXIST
woman-focused construct to a *swapped* item and proposes an invariance class. It is
**shared, frozen, and identical for E1 and E2** (master spec §5.3: "the same frozen
construct-derived rule from E1 is applied first-pass to all ~600 E2 items"). E1 is
its registered home; ``e2.tagging`` re-imports it so the two experiments cannot
drift apart.

The rule is deliberately conservative. It resolves *cleanly* only the two cases the
construct settles without semantic reading:

  * UNSWAPPABLE — no gendered referent to swap (excluded from the partition).
  * INVARIANT  — the original is majority NOT-SEXIST, so the label stays not-sexist
                 under swap (the woman-focused construct is not engaged either way).

Every SEXIST-derived item is sent to author adjudication (``needs_adjudication``):

  * SEXIST + a gendered (non-name) token swapped -> SHIFTING first-pass — the
    woman-focused construct most likely no longer engages the man-targeted swap, but
    the author confirms (some sexism does not hinge on the target being a woman, e.g.
    items that *describe* or *criticise* sexism, which stay INVARIANT).
  * SEXIST + only a name swapped -> AMBIGUOUS — the gender signal is weak.
  * SEXIST + swappable but neither flag -> AMBIGUOUS.

The SHIFTING / AMBIGUOUS first-pass on SEXIST items is a *proposal that speeds
adjudication*, not a definitional label: ``needs_adjudication`` is True for all of
them, and the author's ``invariance_class`` is what the final partition uses. The
pre-registered token-swap caveat applies — the swap can empty a stereotype rather
than mirror it, so a SHIFTING verdict partly reflects stereotype removal.
"""

from __future__ import annotations

INVARIANCE_CLASSES = ("INVARIANT", "SHIFTING", "AMBIGUOUS")
EXCLUSION = "UNSWAPPABLE"
SEXIST_THRESHOLD = 0.5  # majority detection cut on the gold P(sexist)


def construct_rule_tag(p_sexist: float, gendered_swapped: bool,
                       name_swapped: bool, unswappable: bool) -> tuple[str, bool, str]:
    """Apply the frozen construct rule to one swapped item.

    Args:
        p_sexist: gold detection mass P(sexist) over the 6 annotators (``gold_1_1``).
        gendered_swapped: a non-name gendered token (pronoun/noun/honorific) was swapped.
        name_swapped: a curated name was swapped.
        unswappable: no swappable gendered token at all.

    Returns:
        ``(rule_class, needs_adjudication, rule_reason)`` where ``rule_class`` is one
        of INVARIANT / SHIFTING / AMBIGUOUS / UNSWAPPABLE, ``needs_adjudication`` is
        True for every item the author must confirm, and ``rule_reason`` is a frozen
        one-line justification recorded in the worksheet.
    """
    if unswappable:
        return EXCLUSION, False, "no swappable gendered token (excluded from partition)"
    if p_sexist < SEXIST_THRESHOLD:
        return "INVARIANT", False, "majority not-sexist; label stays not-sexist under swap"
    if gendered_swapped:
        return ("SHIFTING", True,
                "sexist + woman-focused gendered token swapped; construct likely shifts "
                "(author adjudication)")
    if name_swapped:
        return ("AMBIGUOUS", True,
                "sexist but only a name swapped; gender signal weak (author adjudication)")
    return "AMBIGUOUS", True, "sexist + swappable; requires author adjudication"
