"""Defensive response parsing and the parse-status taxonomy (master spec §6.2).

Every model response is classified into exactly one ``ParseStatus`` with the raw
text always retained upstream. Parsing is layered: strict → tolerant recovery →
label fuzzy matching against a per-class accepted-variants map. A response that
still does not resolve is ``format_violation`` (logged, never guessed); a decline
is ``refusal`` (a first-class datapoint, excluded from soft metrics).

Parsed value shapes:
  * Formats A/B  -> dict[class_key -> float]  (1.1/1.2 a distribution; 1.3 per-cat)
  * Format C 1.1/1.2 -> str   (one canonical class, a single categorical draw)
  * Format C 1.3     -> list[str]  (canonical categories; [] for "none")
"""

from __future__ import annotations

import json
import re
import unicodedata

from .data import T11_CLASSES, T12_CLASSES, T13_CATEGORIES

# ── Parse-status taxonomy ─────────────────────────────────────────────────────
PARSED_OK = "parsed_ok"
REFUSAL = "refusal"
DISCLAIMER_PREFIXED = "disclaimer_prefixed"
FORMAT_VIOLATION = "format_violation"
EMPTY = "empty"
ERROR = "error"

# ── Refusal / disclaimer surface cues (EN + ES) ───────────────────────────────
_REFUSAL_CUES = (
    "i can't", "i cannot", "i can not", "i won't", "i will not", "i'm unable",
    "i am unable", "i'm sorry", "i am sorry", "cannot assist", "can't help",
    "not comfortable", "i'm not able", "i am not able", "as an ai",
    "no puedo", "lo siento", "no me es posible", "no estoy en condiciones",
)
_DISCLAIMER_CUES = (
    "please note", "it's important to note", "it is important to note",
    "i should note", "disclaimer", "keep in mind", "i'd like to note",
    "cabe señalar", "es importante señalar", "tenga en cuenta",
)

# ── Accepted-variant maps: normalized surface form -> canonical key ───────────
# Built once; the matcher normalises (lowercase, strip accents/punctuation) before
# lookup, so only the distinct lexical variants need listing here.
_T11_VARIANTS = {
    "sexist": "sexist", "sexista": "sexist", "is sexist": "sexist",
    "es sexista": "sexist", "yes": "sexist",
    "not_sexist": "not_sexist", "not sexist": "not_sexist", "notsexist": "not_sexist",
    "no sexista": "not_sexist", "no_sexista": "not_sexist", "nosexista": "not_sexist",
    "no": "not_sexist", "non sexist": "not_sexist",
}
_T12_VARIANTS = {
    "not_sexist": "not_sexist", "not sexist": "not_sexist", "no sexista": "not_sexist",
    "no_sexista": "not_sexist",
    "direct": "direct", "directo": "direct", "directa": "direct",
    "reported": "reported", "reportado": "reported", "reportada": "reported",
    "judgemental": "judgemental", "judgmental": "judgemental", "critico": "judgemental",
    "crítico": "judgemental", "critica": "judgemental", "enjuiciador": "judgemental",
    "de juicio": "judgemental", "de juicio critico": "judgemental",
}
_T13_VARIANTS = {
    # ideological_inequality
    "ideological_inequality": "ideological_inequality",
    "ideological and inequality": "ideological_inequality",
    "ideological inequality": "ideological_inequality",
    "ideologica_desigualdad": "ideological_inequality",
    "ideologica y de desigualdad": "ideological_inequality",
    "ideologica desigualdad": "ideological_inequality",
    "ideological": "ideological_inequality",
    # stereotyping_dominance
    "stereotyping_dominance": "stereotyping_dominance",
    "stereotyping and dominance": "stereotyping_dominance",
    "stereotyping dominance": "stereotyping_dominance",
    "estereotipos_dominacion": "stereotyping_dominance",
    "estereotipos y dominacion": "stereotyping_dominance",
    "estereotipos dominacion": "stereotyping_dominance",
    "stereotyping": "stereotyping_dominance",
    # objectification
    "objectification": "objectification", "cosificacion": "objectification",
    "objetificacion": "objectification",
    # sexual_violence
    "sexual_violence": "sexual_violence", "sexual violence": "sexual_violence",
    "violencia_sexual": "sexual_violence", "violencia sexual": "sexual_violence",
    # misogyny_non_sexual_violence
    "misogyny_non_sexual_violence": "misogyny_non_sexual_violence",
    "misogyny and non sexual violence": "misogyny_non_sexual_violence",
    "misogyny non sexual violence": "misogyny_non_sexual_violence",
    "misoginia_violencia_no_sexual": "misogyny_non_sexual_violence",
    "misoginia y violencia no sexual": "misogyny_non_sexual_violence",
    "misoginia violencia no sexual": "misogyny_non_sexual_violence",
    "misogyny": "misogyny_non_sexual_violence", "misoginia": "misogyny_non_sexual_violence",
}
_VARIANTS = {"1.1": _T11_VARIANTS, "1.2": _T12_VARIANTS, "1.3": _T13_VARIANTS}
_CANON = {"1.1": T11_CLASSES, "1.2": T12_CLASSES, "1.3": T13_CATEGORIES}
# JSON/plain keys we expect per subtask for distribution formats.
_DIST_KEYS = {"1.1": T11_CLASSES, "1.2": T12_CLASSES, "1.3": T13_CATEGORIES}


def normalize(s: str) -> str:
    """Lowercase, strip accents/punctuation, collapse whitespace."""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().replace("_", " ")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a or not b:
        return max(len(a), len(b))
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def match_label(surface: str, subtask: str, max_edit: int = 2) -> str | None:
    """Resolve a surface form to a canonical class key, or ``None``.

    Order: normalised exact in the variant map -> tight-threshold fuzzy match
    against the variant keys. The fuzzy step only fires for near-misses
    (misspellings) and never merges two distinct canonical classes (an ambiguous
    nearest match returns ``None``).
    """
    variants = _VARIANTS[subtask]
    norm = normalize(surface)
    if not norm:
        return None
    if norm in variants:
        return variants[norm]
    # Fuzzy only for reasonably long surfaces — short tokens (e.g. "i", "no")
    # are too close to many variants and produce false positives in prose.
    if len(norm) < 4:
        return None
    # Fuzzy: nearest variant key within threshold, must be unambiguous.
    best, best_d, ambiguous = None, max_edit + 1, False
    for key, canon in variants.items():
        d = _levenshtein(norm, key)
        if d < best_d:
            best, best_d, ambiguous = canon, d, False
        elif d == best_d and canon != best:
            ambiguous = True
    if best is not None and best_d <= max_edit and not ambiguous:
        return best
    return None


# ── JSON / number extraction ──────────────────────────────────────────────────
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(raw: str) -> dict | None:
    """Strict parse, then tolerant recovery (strip fences, first {...} object)."""
    for candidate in (raw, *_FENCE_RE.findall(raw)):
        candidate = candidate.strip()
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, TypeError):
            pass
    m = _OBJ_RE.search(raw)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            return None
    return None


def _coerce_float(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f


def _dist_from_items(items: dict, subtask: str) -> dict[str, float] | None:
    """Map raw {key: value} to canonical {class: float}; None if nothing resolves."""
    out: dict[str, float] = {}
    for raw_key, raw_val in items.items():
        canon = match_label(raw_key, subtask)
        f = _coerce_float(raw_val)
        if canon is not None and f is not None:
            out[canon] = max(0.0, min(1.0, f))
    return out or None


def _finalize_dist(parsed: dict[str, float], subtask: str) -> dict[str, float]:
    """Fill missing classes with 0; for 1.1/1.2 renormalise to sum 1 if possible."""
    full = {c: parsed.get(c, 0.0) for c in _CANON[subtask]}
    if subtask in ("1.1", "1.2"):
        total = sum(full.values())
        if total > 0:
            full = {c: v / total for c, v in full.items()}
    return full


_PAIR_RE = re.compile(r"([A-Za-z_ ]+?)\s*[:=]\s*(-?\d*\.?\d+)")


def _parse_plain_pairs(raw: str, subtask: str) -> dict[str, float] | None:
    items = {k.strip(): v for k, v in _PAIR_RE.findall(raw)}
    return _dist_from_items(items, subtask) if items else None


# ── Refusal / disclaimer detection ────────────────────────────────────────────
def _has_cue(raw: str, cues) -> bool:
    low = raw.lower()
    return any(cue in low for cue in cues)


# ── Top-level parse ───────────────────────────────────────────────────────────
def parse_response(raw: str | None, subtask: str, fmt: str) -> tuple[object, str]:
    """Classify and parse a single response → ``(parsed_value, ParseStatus)``."""
    if subtask not in _CANON:
        raise ValueError(f"unknown subtask {subtask!r}")
    if raw is None or not str(raw).strip():
        return None, EMPTY

    parsed = _parse_value(raw, subtask, fmt)
    if parsed is not None:
        if _has_cue(raw, _REFUSAL_CUES) or _has_cue(raw, _DISCLAIMER_CUES):
            return parsed, DISCLAIMER_PREFIXED
        return parsed, PARSED_OK

    if _has_cue(raw, _REFUSAL_CUES):
        return None, REFUSAL
    return None, FORMAT_VIOLATION


def _parse_value(raw: str, subtask: str, fmt: str) -> object:
    """Format-specific parse to the canonical value shape, or ``None`` on failure."""
    if fmt == "A":
        obj = _extract_json(raw)
        if obj is None:
            return None
        dist = _dist_from_items(obj, subtask)
        return _finalize_dist(dist, subtask) if dist else None
    if fmt == "B":
        dist = _parse_plain_pairs(raw, subtask)
        return _finalize_dist(dist, subtask) if dist else None
    # Format C — categorical draw(s).
    if subtask == "1.3":
        return _parse_categorical_multi(raw)
    return _parse_categorical_single(raw, subtask)


def _parse_categorical_single(raw: str, subtask: str) -> str | None:
    text = raw.strip()
    direct = match_label(text, subtask)
    if direct is not None:
        return direct
    # Token scan: first token that resolves to a canonical class.
    for tok in re.split(r"[\s,.;:!?\"'()]+", text):
        canon = match_label(tok, subtask)
        if canon is not None:
            return canon
    return None


def _parse_categorical_multi(raw: str) -> list[str] | None:
    text = raw.strip()
    if normalize(text) in ("none", "ninguna", "ninguno", "n a"):
        return []
    found: list[str] = []
    for piece in re.split(r"[,\n;]+", text):
        canon = match_label(piece, "1.3")
        if canon is not None and canon not in found:
            found.append(canon)
    if found:
        return found
    # Fall back to token scan for category names embedded in prose.
    if _has_cue(raw, _REFUSAL_CUES):
        return None
    return [] if "none" in text.lower() else None
