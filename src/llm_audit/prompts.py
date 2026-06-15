"""E0 prompt assembly — verbatim from ``specs/llm-sexism-bias/prompts-e0-baseline.md``.

Single source of truth for the E0 classification prompts. Bodies and the fixed
content-neutral system prompt (§9.1) are copied verbatim; the bare/defined contrast
and the three elicitation formats (A=strict JSON, B=plain, C=K=6 categorical) are
encoded as data. ``build_prompt`` returns ``(system, user)`` ready to send.

Any wording change here is a "prompt change" and must trigger a regression re-run
(``prompt-regression-set.md`` §5).
"""

from __future__ import annotations

SUBTASKS = ("1.1", "1.2", "1.3")
SPECS = ("bare", "defined")
LANGS = ("en", "es")
FORMATS = ("A", "B", "C")

# ── System prompt (E0 §9.1, fixed across all of E0) ───────────────────────────
SYSTEM_PROMPT = {
    "en": (
        "You are a text classification assistant. You will be shown a tweet and "
        "asked to classify it. Follow the response format exactly as specified, and "
        "respond only in that format with no additional commentary."
    ),
    "es": (
        "Usted es un asistente de clasificación de textos. Se le mostrará un tuit y "
        "se le pedirá que lo clasifique. Siga exactamente el formato de respuesta "
        "indicado y responda únicamente en ese formato, sin comentarios adicionales."
    ),
}

# ── Bodies (verbatim, prompts §3-5). {tweet} and {suffix} are filled at build. ─
_BODIES: dict[str, dict[str, dict[str, str]]] = {
    "1.1": {
        "bare": {
            "en": (
                "Classify the following tweet as SEXIST or NOT SEXIST.\n\n"
                "Tweet: {tweet}\n\n{suffix}"
            ),
            "es": (
                "Clasifique el siguiente tuit como SEXISTA o NO SEXISTA.\n\n"
                "Tuit: {tweet}\n\n{suffix}"
            ),
        },
        "defined": {
            "en": (
                "A tweet is sexist if it expresses sexist ideas, because it is sexist "
                "itself, it describes a sexist situation, or it criticizes a sexist "
                "behaviour.\n\n"
                "According to this definition, classify the following tweet as SEXIST "
                "or NOT SEXIST.\n\n"
                "Tweet: {tweet}\n\n{suffix}"
            ),
            "es": (
                "Un tuit es sexista si expresa ideas sexistas, ya sea porque es sexista "
                "en sí mismo, porque describe una situación sexista, o porque critica un "
                "comportamiento sexista.\n\n"
                "De acuerdo con esta definición, clasifique el siguiente tuit como "
                "SEXISTA o NO SEXISTA.\n\n"
                "Tuit: {tweet}\n\n{suffix}"
            ),
        },
    },
    "1.2": {
        "bare": {
            "en": (
                "If the following tweet is sexist, identify the author's intention. "
                "The options are:\n\n"
                "- not sexist: the tweet is not sexist\n"
                "- direct: the author expresses a sexist message in their own voice\n"
                "- reported: the author describes or shares a sexist situation "
                "experienced or witnessed\n"
                "- judgemental: the author describes a sexist situation in order to "
                "criticise or condemn it\n\n"
                "Tweet: {tweet}\n\n{suffix}"
            ),
            "es": (
                "Si el siguiente tuit es sexista, identifique la intención de quien lo "
                "escribe. Las opciones son:\n\n"
                "- no sexista: el tuit no es sexista\n"
                "- directo: quien escribe expresa un mensaje sexista en su propia voz\n"
                "- reportado: quien escribe describe o comparte una situación sexista "
                "vivida o presenciada\n"
                "- crítico: quien escribe describe una situación sexista con el fin de "
                "criticarla o condenarla\n\n"
                "Tuit: {tweet}\n\n{suffix}"
            ),
        },
        "defined": {
            "en": (
                "Tweets are classified by the author's intention into one of the "
                "following, following the EXIST 2025 guidelines:\n\n"
                "- NOT SEXIST: the tweet does not convey sexism.\n"
                "- DIRECT: the intention is to write a message that is sexist by itself "
                "or incites sexism.\n"
                "- REPORTED: the intention is to report and share a sexist situation "
                "suffered by a woman or women, in first or third person.\n"
                "- JUDGEMENTAL: the intention is to condemn sexist situations or "
                "behaviours.\n\n"
                "Classify the following tweet according to these definitions.\n\n"
                "Tweet: {tweet}\n\n{suffix}"
            ),
            "es": (
                "Los tuits se clasifican según la intención de quien los escribe en una "
                "de las siguientes categorías, según las directrices de EXIST 2025:\n\n"
                "- NO SEXISTA: el tuit no transmite sexismo.\n"
                "- DIRECTO: la intención es escribir un mensaje que es sexista en sí "
                "mismo o que incita al sexismo.\n"
                "- REPORTADO: la intención es informar y compartir una situación sexista "
                "sufrida por una o varias mujeres, en primera o tercera persona.\n"
                "- CRÍTICO: la intención es condenar situaciones o comportamientos "
                "sexistas.\n\n"
                "Clasifique el siguiente tuit de acuerdo con estas definiciones.\n\n"
                "Tuit: {tweet}\n\n{suffix}"
            ),
        },
    },
    "1.3": {
        "bare": {
            "en": (
                "If the following tweet is sexist, indicate which of these categories "
                "apply. More than one may apply.\n\n"
                "- ideological and inequality\n"
                "- stereotyping and dominance\n"
                "- objectification\n"
                "- sexual violence\n"
                "- misogyny and non-sexual violence\n\n"
                "Tweet: {tweet}\n\n{suffix}"
            ),
            "es": (
                "Si el siguiente tuit es sexista, indique cuáles de estas categorías se "
                "aplican. Puede aplicarse más de una.\n\n"
                "- ideológica y de desigualdad\n"
                "- estereotipos y dominación\n"
                "- cosificación\n"
                "- violencia sexual\n"
                "- misoginia y violencia no sexual\n\n"
                "Tuit: {tweet}\n\n{suffix}"
            ),
        },
        "defined": {
            "en": (
                "Sexist tweets are categorised into one or more of the following, "
                "following the EXIST 2025 guidelines. Many facets of a woman's life may "
                "be the focus of sexist attitudes, including domestic role, career "
                "opportunities, and sexual image.\n\n"
                "- IDEOLOGICAL AND INEQUALITY: includes messages that discredit the "
                "feminist movement, reject inequality between men and women, or present "
                "men as victims of gender-based oppression.\n"
                "- STEREOTYPING AND DOMINANCE: includes messages that suggest women are "
                "more suitable or inappropriate for certain tasks, and somehow inferior "
                "to men.\n"
                "- OBJECTIFICATION: includes messages where women are presented as "
                "objects apart from their dignity and personal aspects, or that assume "
                "or describe certain physical qualities that women must have in order to "
                "fulfil traditional gender roles.\n"
                "- SEXUAL VIOLENCE: includes messages where sexual suggestions, "
                "requests, or harassment of a sexual nature (rape or sexual assault) are "
                "made.\n"
                "- MISOGYNY AND NON-SEXUAL VIOLENCE: includes expressions of hatred and "
                "violence towards women.\n\n"
                "Each sexist tweet must be assigned one or more of these categories. "
                "Categorise the following tweet according to these definitions.\n\n"
                "Tweet: {tweet}\n\n{suffix}"
            ),
            "es": (
                "Los tuits sexistas se clasifican en una o varias de las siguientes "
                "categorías, según las directrices de EXIST 2025. Muchas facetas de la "
                "vida de una mujer pueden ser el foco de actitudes sexistas, incluidos "
                "el rol doméstico, las oportunidades laborales y la imagen sexual.\n\n"
                "- IDEOLÓGICA Y DE DESIGUALDAD: incluye mensajes que desacreditan el "
                "movimiento feminista, niegan la desigualdad entre hombres y mujeres, o "
                "presentan a los hombres como víctimas de la opresión de género.\n"
                "- ESTEREOTIPOS Y DOMINACIÓN: incluye mensajes que sugieren que las "
                "mujeres son más adecuadas o inadecuadas para ciertas tareas, y de algún "
                "modo inferiores a los hombres.\n"
                "- COSIFICACIÓN: incluye mensajes en los que se presenta a las mujeres "
                "como objetos al margen de su dignidad y de sus aspectos personales, o "
                "que asumen o describen ciertas cualidades físicas que las mujeres deben "
                "tener para cumplir los roles de género tradicionales.\n"
                "- VIOLENCIA SEXUAL: incluye mensajes en los que se hacen insinuaciones, "
                "peticiones o acoso de naturaleza sexual (violación o agresión sexual).\n"
                "- MISOGINIA Y VIOLENCIA NO SEXUAL: incluye expresiones de odio y "
                "violencia hacia las mujeres.\n\n"
                "A cada tuit sexista se le debe asignar una o varias de estas "
                "categorías. Clasifique el siguiente tuit de acuerdo con estas "
                "definiciones.\n\n"
                "Tuit: {tweet}\n\n{suffix}"
            ),
        },
    },
}

# ── Elicitation suffixes (prompts §6). [subtask][fmt][lang] ───────────────────
_SUFFIXES: dict[str, dict[str, dict[str, str]]] = {
    "1.1": {
        "A": {
            "en": 'Respond with a JSON object: {"sexist": X, "not_sexist": Y}, X and Y in [0,1] summing to 1. No other text.',
            "es": 'Responda con un objeto JSON: {"sexista": X, "no_sexista": Y}, X e Y en [0,1] cuya suma es 1. Ningún otro texto.',
        },
        "B": {
            "en": "Respond in the format: sexist: X | not_sexist: Y, with X and Y in [0,1] summing to 1. No other text.",
            "es": "Responda en el formato: sexista: X | no_sexista: Y, con X e Y en [0,1] cuya suma es 1. Ningún otro texto.",
        },
        "C": {
            "en": "Respond with exactly one word: sexist or not_sexist.",
            "es": "Responda con exactamente una palabra: sexista o no_sexista.",
        },
    },
    "1.2": {
        "A": {
            "en": 'Respond with a JSON object: {"not_sexist": X0, "direct": X1, "reported": X2, "judgemental": X3}, each in [0,1] summing to 1. No other text.',
            "es": 'Responda con un objeto JSON: {"no_sexista": X0, "directo": X1, "reportado": X2, "critico": X3}, cada uno en [0,1] cuya suma es 1. Ningún otro texto.',
        },
        "B": {
            "en": "Respond in the format: not_sexist: X0 | direct: X1 | reported: X2 | judgemental: X3, each in [0,1] summing to 1. No other text.",
            "es": "Responda en el formato: no_sexista: X0 | directo: X1 | reportado: X2 | critico: X3, cada uno en [0,1] cuya suma es 1. Ningún otro texto.",
        },
        "C": {
            "en": "Respond with exactly one of: not_sexist, direct, reported, judgemental.",
            "es": "Responda con exactamente una de: no_sexista, directo, reportado, critico.",
        },
    },
    "1.3": {
        "A": {
            "en": 'Respond with a JSON object giving a probability in [0,1] for each category: {"ideological_inequality": X1, "stereotyping_dominance": X2, "objectification": X3, "sexual_violence": X4, "misogyny_non_sexual_violence": X5}. Values need not sum to 1. If the tweet is not sexist, all values are 0. No other text.',
            "es": 'Responda con un objeto JSON que indique una probabilidad en [0,1] para cada categoría: {"ideologica_desigualdad": X1, "estereotipos_dominacion": X2, "cosificacion": X3, "violencia_sexual": X4, "misoginia_violencia_no_sexual": X5}. Los valores no tienen que sumar 1. Si el tuit no es sexista, todos los valores son 0. Ningún otro texto.',
        },
        "B": {
            "en": "Respond in the format: ideological_inequality: X1 | stereotyping_dominance: X2 | objectification: X3 | sexual_violence: X4 | misogyny_non_sexual_violence: X5, each in [0,1]. If the tweet is not sexist, all values are 0. No other text.",
            "es": "Responda en el formato: ideologica_desigualdad: X1 | estereotipos_dominacion: X2 | cosificacion: X3 | violencia_sexual: X4 | misoginia_violencia_no_sexual: X5, cada uno en [0,1]. Si el tuit no es sexista, todos los valores son 0. Ningún otro texto.",
        },
        "C": {
            "en": "Respond with the categories that apply, separated by commas, choosing from: ideological_inequality, stereotyping_dominance, objectification, sexual_violence, misogyny_non_sexual_violence. If none apply, respond with: none. No other text.",
            "es": "Responda con las categorías que se aplican, separadas por comas, eligiendo entre: ideologica_desigualdad, estereotipos_dominacion, cosificacion, violencia_sexual, misoginia_violencia_no_sexual. Si ninguna se aplica, responda: ninguna. Ningún otro texto.",
        },
    },
}


def _validate(subtask: str, spec: str, lang: str, fmt: str) -> None:
    if subtask not in SUBTASKS:
        raise ValueError(f"subtask must be one of {SUBTASKS}, got {subtask!r}")
    if spec not in SPECS:
        raise ValueError(f"spec must be one of {SPECS}, got {spec!r}")
    if lang not in LANGS:
        raise ValueError(f"lang must be one of {LANGS}, got {lang!r}")
    if fmt not in FORMATS:
        raise ValueError(f"fmt must be one of {FORMATS}, got {fmt!r}")


def build_prompt(subtask: str, spec: str, lang: str, fmt: str, tweet: str) -> tuple[str, str]:
    """Return ``(system, user)`` for one E0 cell.

    ``condition`` for logging is conventionally ``f"{spec}"`` (the bare/defined
    factor); subtask/lang/format are logged as their own fields.
    """
    _validate(subtask, spec, lang, fmt)
    suffix = _SUFFIXES[subtask][fmt][lang]
    user = _BODIES[subtask][spec][lang].format(tweet=tweet, suffix=suffix)
    return SYSTEM_PROMPT[lang], user
