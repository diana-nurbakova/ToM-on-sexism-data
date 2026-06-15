"""E3 scaffold prompt assembly (verbatim from prompts-e3-tom-scaffolding.md).

Five conditions. Subtask 1.1 uses the bare base with the scaffold clause folded into
the instruction line (§4); subtask 1.3 prepends the scaffold clause to the E0
*defined* 1.3 body (§5.1 — reasoning about the categories presupposes the rubric;
we reuse the verbatim pinned E0 rubric and prepend the clause as a lead sentence
rather than the spec's [VERIFY] inline paraphrase). System prompt and elicitation
suffixes are the unchanged E0 ones (PE7).
"""

from __future__ import annotations

from .. import prompts

CONDITIONS = ("neutral", "generic", "cognitive", "affective", "combined")

# ── Subtask 1.1 user-prompt leads (verbatim, §4.1-4.5). {tweet}/{suffix} filled later. ─
_LEAD_11 = {
    "neutral": {
        "en": "Classify the following tweet as SEXIST or NOT SEXIST.",
        "es": "Clasifique el siguiente tuit como SEXISTA o NO SEXISTA.",
    },
    "generic": {
        "en": "Read the tweet carefully and consider it before deciding. Then classify "
              "the following tweet as SEXIST or NOT SEXIST.",
        "es": "Lea el tuit con atención y considérelo antes de decidir. Después, "
              "clasifique el siguiente tuit como SEXISTA o NO SEXISTA.",
    },
    "cognitive": {
        "en": "Consider the author's intent: what are they trying to communicate, and "
              "what belief does the message express? Taking this into account, classify "
              "the following tweet as SEXIST or NOT SEXIST.",
        "es": "Considere la intención de quien escribe: ¿qué trata de comunicar y qué "
              "creencia expresa el mensaje? Teniendo esto en cuenta, clasifique el "
              "siguiente tuit como SEXISTA o NO SEXISTA.",
    },
    "affective": {
        "en": "Consider the experience of the person or group that the message is about "
              "or directed at: how would they feel reading it? Taking this into account, "
              "classify the following tweet as SEXIST or NOT SEXIST.",
        "es": "Considere la experiencia de la persona o el grupo al que se refiere o se "
              "dirige el mensaje: ¿cómo se sentiría al leerlo? Teniendo esto en cuenta, "
              "clasifique el siguiente tuit como SEXISTA o NO SEXISTA.",
    },
    "combined": {
        "en": "Consider both: (1) the author's intent, namely what they are trying to "
              "communicate and what belief the message expresses; and (2) the experience "
              "of the person or group the message is about or directed at, namely how "
              "they would feel reading it. Taking both into account, classify the "
              "following tweet as SEXIST or NOT SEXIST.",
        "es": "Considere ambos aspectos: (1) la intención de quien escribe, es decir, qué "
              "trata de comunicar y qué creencia expresa el mensaje; y (2) la experiencia "
              "de la persona o el grupo al que se refiere o se dirige el mensaje, es "
              "decir, cómo se sentiría al leerlo. Teniendo ambos en cuenta, clasifique el "
              "siguiente tuit como SEXISTA o NO SEXISTA.",
    },
}

# ── Standalone scaffold clauses for 1.3 (prepended to the E0 defined 1.3 body, §3) ─
_CLAUSE_13 = {
    "neutral": {"en": "", "es": ""},
    "generic": {
        "en": "Read the tweet carefully and consider it before deciding.",
        "es": "Lea el tuit con atención y considérelo antes de decidir.",
    },
    "cognitive": {
        "en": "Consider the author's intent: what are they trying to communicate, and "
              "what belief does the message express?",
        "es": "Considere la intención de quien escribe: ¿qué trata de comunicar y qué "
              "creencia expresa el mensaje?",
    },
    "affective": {
        "en": "Consider the experience of the person or group that the message is about "
              "or directed at: how would they feel reading it?",
        "es": "Considere la experiencia de la persona o el grupo al que se refiere o se "
              "dirige el mensaje: ¿cómo se sentiría al leerlo?",
    },
    "combined": {
        "en": "Consider both: (1) the author's intent, namely what they are trying to "
              "communicate and what belief the message expresses; and (2) the experience "
              "of the person or group the message is about or directed at, namely how "
              "they would feel reading it.",
        "es": "Considere ambos aspectos: (1) la intención de quien escribe, es decir, qué "
              "trata de comunicar y qué creencia expresa el mensaje; y (2) la experiencia "
              "de la persona o el grupo al que se refiere o se dirige el mensaje, es "
              "decir, cómo se sentiría al leerlo.",
    },
}


def build_scaffold_prompt(condition: str, subtask: str, lang: str, fmt: str,
                          tweet: str) -> tuple[str, str]:
    """Return ``(system, user)`` for one E3 cell. 1.1=bare base, 1.3=defined base."""
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}, got {condition!r}")
    if subtask not in ("1.1", "1.3"):
        raise ValueError("E3 runs on subtasks 1.1 and 1.3 only")
    system = prompts.SYSTEM_PROMPT[lang]
    suffix = prompts._SUFFIXES[subtask][fmt][lang]
    if subtask == "1.1":
        user = f"{_LEAD_11[condition][lang]}\n\nTweet: {tweet}\n\n{suffix}" if lang == "en" \
            else f"{_LEAD_11[condition][lang]}\n\nTuit: {tweet}\n\n{suffix}"
        return system, user
    # 1.3: prepend the clause (if any) to the verbatim E0 defined 1.3 body.
    _, defined_body = prompts.build_prompt("1.3", "defined", lang, fmt, tweet)
    clause = _CLAUSE_13[condition][lang]
    user = f"{clause}\n\n{defined_body}" if clause else defined_body
    return system, user
