"""H1c persona system-prompt assembly (verbatim from prompts-h1c-persona.md).

The full gender × age grid mirroring the EXIST annotator stratification (§2.1): six
personas, female/male × {18-22, 23-45, 46+}. The persona is a role, so it lives in
the **system prompt** (§2.2), replacing the fixed E0 system prompt; the user turn is
the *unchanged* E0 prompt — bare for 1.1, defined for 1.3 (§2.4, §3.4). Only the
stratified attributes (gender, age) are named (§2.1, PE18) and the persona names *who
the perceiver is*, never *how they should judge* (§2.3, PE19). The format-instruction
sentence is byte-identical to E0 across all six personas (§5 checklist), so only the
identity sentence varies. The no-persona reference is the bare-prompt E0 result (§2.2).
"""

from __future__ import annotations

from .. import prompts

PERSONAS = ("F-young", "F-mid", "F-older", "M-young", "M-mid", "M-older")
H1C_SUBTASKS = ("1.1", "1.3")

# The shared format-instruction sentence, byte-identical to the tail of the E0
# system prompt (§5 checklist guarantee).
_FORMAT_SENTENCE = {
    "en": "Follow the response format exactly as specified, and respond only in that "
          "format with no additional commentary.",
    "es": "Siga exactamente el formato de respuesta indicado y responda únicamente en "
          "ese formato, sin comentarios adicionales.",
}

# System-prompt template (§3.1); the format sentence is appended verbatim.
_TEMPLATE = {
    "en": "You are a {gender} aged {age_band}, annotating tweets for a research study. "
          "Label each tweet as you would. " + _FORMAT_SENTENCE["en"],
    "es": "Usted es {gender} de {age_band} años, y está anotando tuits para un estudio "
          "de investigación. Etiquete cada tuit como usted lo haría. " + _FORMAT_SENTENCE["es"],
}

# The six fills (§3.2). EN/ES gender + age-band surface forms.
_FILLS = {
    "F-young": {"en": ("woman", "18-22"), "es": ("una mujer", "18 a 22")},
    "F-mid": {"en": ("woman", "23-45"), "es": ("una mujer", "23 a 45")},
    "F-older": {"en": ("woman", "46 or older"), "es": ("una mujer", "46 o más")},
    "M-young": {"en": ("man", "18-22"), "es": ("un hombre", "18 a 22")},
    "M-mid": {"en": ("man", "23-45"), "es": ("un hombre", "23 a 45")},
    "M-older": {"en": ("man", "46 or older"), "es": ("un hombre", "46 o más")},
}

# Subtask → E0 base (bare 1.1 / defined 1.3, §2.4).
_BASE_SPEC = {"1.1": "bare", "1.3": "defined"}


def persona_system_prompt(persona: str, lang: str) -> str:
    """Return the persona system prompt for one cell."""
    if persona not in PERSONAS:
        raise ValueError(f"persona must be one of {PERSONAS}, got {persona!r}")
    gender, age_band = _FILLS[persona][lang]
    return _TEMPLATE[lang].format(gender=gender, age_band=age_band)


def build_persona_prompt(persona: str, subtask: str, lang: str, fmt: str,
                         tweet: str) -> tuple[str, str]:
    """Return ``(system, user)`` for one H1c cell.

    The system prompt is the persona role; the user turn is the unchanged E0 base
    (bare 1.1 / defined 1.3). 1.2 is out of H1c scope (§2.4).
    """
    if subtask not in _BASE_SPEC:
        raise ValueError("H1c runs on subtasks 1.1 and 1.3 only")
    _, user = prompts.build_prompt(subtask, _BASE_SPEC[subtask], lang, fmt, tweet)
    return persona_system_prompt(persona, lang), user
