"""E5 Cialdini condition assembly — harmonised onto the E0 bare base.

Per the author's decision (master spec §5.6 + PE2), E5 shares the E0 bare prompt
body, the fixed E0 system prompt, and the E0 elicitation suffixes with E0/E4; only
the Cialdini framing content differs. Each framing is prepended to the verbatim E0
bare body, exactly as E4's counter-prime sentence is, so E4 and E5 are structurally
parallel manipulations on the same base and differ only in framing content. The
framing content itself is verbatim from ``prompts-e5-cialdini.md`` §5-7 — authority
is a third-party expert verdict, social proof is a third-party-crowd majority,
commitment is a labelled precedent — which is what keeps E5 distinct from E4's
first-person stances without a base-prompt confound.

Neutral = the E0 bare prompt exactly (the no-persuasion control). 13 conditions:
1 neutral + 4 authority (stance × verdict) + 6 social proof (3 intensities ×
2 directions) + 2 commitment (prior direction).
"""

from __future__ import annotations

from .. import prompts

# ── Condition names (the ``condition`` field logged per call) ─────────────────
NEUTRAL = "neutral"
AUTHORITY = (
    "authority_pro_sexist", "authority_pro_not",
    "authority_anti_sexist", "authority_anti_not",
)
SOCIALPROOF = (
    "socialproof_sexist_4", "socialproof_sexist_5", "socialproof_sexist_6",
    "socialproof_not_4", "socialproof_not_5", "socialproof_not_6",
)
COMMITMENT = ("commitment_sexist", "commitment_not")
CONDITIONS = (NEUTRAL,) + AUTHORITY + SOCIALPROOF + COMMITMENT

# Conditions that require a paired exemplar from the §3.3 pool.
COMMITMENT_SET = frozenset(COMMITMENT)

# ── Verbatim framing content (prompts-e5-cialdini.md §5-7) ────────────────────
# Authority source descriptions: gender-neutral role phrases (§5.6), no personal
# names (avoids the ES grammatical-gender confound).
_AUTH_SOURCE = {
    "pro": {
        "en": "An expert researcher in gender studies",
        "es": "Un equipo de investigación en estudios de género",
    },
    "anti": {
        "en": "A legal scholar specialising in freedom of expression",
        "es": "Una persona experta en derecho, especializada en libertad de expresión",
    },
}
# Verdict / direction surface text (lowercase, as in the spec framings).
_VERDICT = {
    "sexist": {"en": "sexist", "es": "sexista"},
    "not": {"en": "not sexist", "es": "no sexista"},
}
# Commitment classification label line uses the same verdict text.
_COMMIT_LEAD = {
    "en": "Consider this example:",
    "es": "Considere este ejemplo:",
}
_COMMIT_EXAMPLE = {"en": "Example", "es": "Ejemplo"}
_COMMIT_CLASS = {"en": "Classification", "es": "Clasificación"}


def _authority_framing(stance: str, verdict: str, lang: str) -> str:
    source = _AUTH_SOURCE[stance][lang]
    v = _VERDICT[verdict][lang]
    if lang == "en":
        return f"{source} has classified the following tweet as {v}."
    return f"{source} ha clasificado el siguiente tuit como {v}."


def _socialproof_framing(verdict: str, n: int, lang: str) -> str:
    v = _VERDICT[verdict][lang]
    if lang == "en":
        return f"Of 6 annotators who reviewed this tweet, {n} classified it as {v}."
    return f"De 6 personas anotadoras que revisaron este tuit, {n} lo clasificaron como {v}."


def _commitment_framing(verdict: str, exemplar: str, lang: str) -> str:
    v = _VERDICT[verdict][lang]
    return (
        f"{_COMMIT_LEAD[lang]}\n\n"
        f"{_COMMIT_EXAMPLE[lang]}: \"{exemplar}\"\n"
        f"{_COMMIT_CLASS[lang]}: {v}"
    )


def _parse_condition(condition: str) -> tuple[str, dict]:
    """Return ``(kind, params)`` for a condition name; validate membership."""
    if condition == NEUTRAL:
        return "neutral", {}
    if condition in AUTHORITY:
        _, stance, verdict = condition.split("_")
        return "authority", {"stance": stance, "verdict": verdict}
    if condition in SOCIALPROOF:
        _, verdict, n = condition.split("_")
        return "socialproof", {"verdict": verdict, "n": int(n)}
    if condition in COMMITMENT:
        verdict = condition.split("_", 1)[1]  # "sexist" | "not"
        return "commitment", {"verdict": verdict}
    raise ValueError(f"condition must be one of {CONDITIONS}, got {condition!r}")


def build_e5_prompt(condition: str, subtask: str, lang: str, fmt: str, tweet: str,
                    exemplar: str | None = None) -> tuple[str, str]:
    """Return ``(system, user)`` for one E5 cell.

    The framing is prepended to the verbatim E0 *bare* body for ``subtask`` (so the
    system prompt and elicitation suffix are byte-identical to E0). Authority and
    social-proof framings are inline (sentence + space + the bare instruction);
    commitment is a labelled-example block set off by a blank line. ``exemplar`` is
    required for the commitment conditions and ignored otherwise.
    """
    kind, params = _parse_condition(condition)
    system, bare_user = prompts.build_prompt(subtask, "bare", lang, fmt, tweet)
    if kind == "neutral":
        return system, bare_user
    if kind == "authority":
        framing = _authority_framing(params["stance"], params["verdict"], lang)
        return system, f"{framing} {bare_user}"
    if kind == "socialproof":
        framing = _socialproof_framing(params["verdict"], params["n"], lang)
        return system, f"{framing} {bare_user}"
    # commitment: labelled precedent block, then the bare body.
    if exemplar is None:
        raise ValueError(f"condition {condition!r} requires an exemplar")
    framing = _commitment_framing(params["verdict"], exemplar, lang)
    return system, f"{framing}\n\n{bare_user}"
