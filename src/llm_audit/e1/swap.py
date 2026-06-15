"""Frozen woman↔man token swap for E1 English (OSF pre-reg Appendix A).

Deterministic, bidirectional, case-preserving substitution of gendered pronouns,
nouns, honorifics, and curated names. The two interpretive cases are handled as
registered: the ambiguous possessive/object "her" (and its mirror "his") is
disambiguated by a lightweight next-token POS heuristic and **flagged for author
review**; names are swapped only via the curated list (others left unchanged).
Items with no swappable gendered token are flagged UNSWAPPABLE (excluded from the
partition). No semantic repair beyond the substitution itself.

NOTE: the curated name-pair list (Appendix A.6) is not yet frozen in the spec; the
seed list below is a starting point, and any name not on it is left unchanged. The
"her"/"his" heuristic is a documented stand-in for a full POS tagger — every such
decision is flagged so the author verifies it (a safe superset of "uncertain only").
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ── Unambiguous pairs (female, male); applied in both directions ──────────────
_PAIRS = [
    ("she", "he"), ("herself", "himself"),
    ("woman", "man"), ("women", "men"), ("girl", "boy"), ("girls", "boys"),
    ("lady", "gentleman"), ("ladies", "gentlemen"),
    ("mother", "father"), ("wife", "husband"), ("sister", "brother"),
    ("daughter", "son"), ("aunt", "uncle"), ("niece", "nephew"),
    ("grandmother", "grandfather"), ("queen", "king"),
    ("actress", "actor"), ("waitress", "waiter"),
    ("girlfriend", "boyfriend"), ("widow", "widower"), ("bride", "groom"),
]
# Many-to-one (female variants -> one male form); reverse picks the first female.
_MANY_TO_ONE = [
    (["mom", "mum"], "dad"),
    (["mrs", "ms", "miss"], "mr"),
    (["madam", "maam"], "sir"),
]
# Curated names (seed; Appendix A.6 to be frozen). Bidirectional.
_NAME_PAIRS = [
    ("mary", "john"), ("sarah", "david"), ("susan", "michael"), ("linda", "james"),
    ("karen", "robert"), ("emma", "william"), ("laura", "thomas"), ("anna", "george"),
]

# Build the symmetric lookup: lowercase token -> (replacement, is_name).
_SWAP: dict[str, str] = {}
for f, m in _PAIRS:
    _SWAP[f] = m
    _SWAP[m] = f
for fems, male in _MANY_TO_ONE:
    for fem in fems:
        _SWAP[fem] = male
    _SWAP[male] = fems[0]  # reverse to the canonical female form
_NAMES: dict[str, str] = {}
for f, m in _NAME_PAIRS:
    _NAMES[f] = m
    _NAMES[m] = f

# Female / male side membership, for E2 swap-direction (F->M vs M->F) tagging.
_FEMALE = ({f for f, _ in _PAIRS} | {fem for fems, _ in _MANY_TO_ONE for fem in fems}
           | {"her", "hers", "herself"})
_MALE = ({m for _, m in _PAIRS} | {male for _, male in _MANY_TO_ONE}
         | {"his", "him", "himself"})
_FEMALE_NAMES = {f for f, _ in _NAME_PAIRS}
_MALE_NAMES = {m for _, m in _NAME_PAIRS}

# Tokens requiring the possessive/object lookahead (flagged for author review).
_AMBIGUOUS = {"her", "his"}
# Crude stoplist: if "her"/"his" is followed by one of these, treat as OBJECT
# (pronoun), else POSSESSIVE (determiner before a noun phrase).
_NON_NOUN_NEXT = {
    "and", "or", "but", "so", "then", "to", "of", "in", "on", "at", "for", "with",
    "is", "was", "are", "were", "be", "been", "did", "does", "do", "will", "would",
    "the", "a", "an", "i", "you", "we", "they", "he", "she", "it",
}

_WORD_RE = re.compile(r"[A-Za-z']+")
_POSSESSIVE_RE = re.compile(r"^(.*?)('s|’s|')$", re.IGNORECASE)
# @mentions are platform handles, not gendered tokens (e1-fix-mention-exclusion.md).
# Tokens inside a handle are never swapped (handles stay intact in the output) and
# never count toward swappability. Negative lookbehind avoids email-like 'a@b'.
_MENTION_RE = re.compile(r"(?<![A-Za-z0-9_])@[A-Za-z0-9_]+")


def _split_possessive(token: str) -> tuple[str, str]:
    """Return (base, suffix) for a possessive like Women's -> (Women, 's)."""
    m = _POSSESSIVE_RE.match(token)
    if m and m.group(1):
        return m.group(1), token[len(m.group(1)):]
    return token, ""


def _match_case(src: str, repl: str) -> str:
    if src.isupper():
        return repl.upper()
    if src[:1].isupper():
        return repl.capitalize()
    return repl


def _resolve_ambiguous(token_lower: str, next_word: str | None) -> str:
    """Resolve 'her'/'his' to object vs possessive forms via a next-token heuristic."""
    possessive = next_word is not None and next_word.lower() not in _NON_NOUN_NEXT
    if token_lower == "her":
        return "his" if possessive else "him"
    # token_lower == "his"
    return "her" if possessive else "hers"


@dataclass
class SwapResult:
    original: str
    swapped: str
    swapped_tokens: list[tuple[str, str]] = field(default_factory=list)  # (src, repl)
    name_swapped: bool = False
    gendered_swapped: bool = False  # a non-name gendered token (pronoun/noun/honorific)
    female_source: bool = False     # a female-side token was swapped (F->M direction)
    male_source: bool = False       # a male-side token was swapped (M->F direction)
    unswappable: bool = False
    author_review: bool = False
    review_reasons: list[str] = field(default_factory=list)


def swap_text(text: str) -> SwapResult:
    """Apply the frozen woman↔man token swap to ``text``."""
    words = list(_WORD_RE.finditer(text))
    mention_spans = [(m.start(), m.end()) for m in _MENTION_RE.finditer(text)]
    out = []
    last = 0
    swapped_tokens: list[tuple[str, str]] = []
    res = SwapResult(original=text, swapped=text)

    def _in_mention(pos: int) -> bool:
        return any(a <= pos < b for a, b in mention_spans)

    for i, m in enumerate(words):
        token = m.group(0)
        if _in_mention(m.start()):
            # Inside an @handle: leave intact, do not swap, do not count.
            out.append(text[last:m.end()])
            last = m.end()
            continue
        base, suffix = _split_possessive(token)  # e.g. Women's -> (Women, 's)
        low = base.lower()
        repl = None
        if low in _AMBIGUOUS:
            nxt = words[i + 1].group(0) if i + 1 < len(words) else None
            repl_low = _resolve_ambiguous(low, nxt)
            repl = _match_case(base, repl_low) + suffix
            res.gendered_swapped = True
            res.female_source = res.female_source or low == "her"
            res.male_source = res.male_source or low == "his"
            res.author_review = True
            res.review_reasons.append(f"ambiguous '{low}' -> '{repl_low}' (POS heuristic)")
        elif low in _SWAP:
            repl = _match_case(base, _SWAP[low]) + suffix
            res.gendered_swapped = True
            res.female_source = res.female_source or low in _FEMALE
            res.male_source = res.male_source or low in _MALE
            if low in ("ms", "mrs", "miss", "mr", "madam", "maam"):
                # honorific many-to-one / reverse default is review-worthy
                res.review_reasons.append(f"honorific '{low}' -> '{repl}'")
                res.author_review = True
        elif low in _NAMES:
            repl = _match_case(base, _NAMES[low]) + suffix
            res.name_swapped = True
            res.female_source = res.female_source or low in _FEMALE_NAMES
            res.male_source = res.male_source or low in _MALE_NAMES

        out.append(text[last:m.start()])
        if repl is not None and repl != token:
            out.append(repl)
            swapped_tokens.append((token, repl))
        else:
            out.append(token)
        last = m.end()
    out.append(text[last:])

    res.swapped = "".join(out)
    res.swapped_tokens = swapped_tokens
    # UNSWAPPABLE: no gendered token (incl. names) was swapped.
    if not swapped_tokens:
        res.unswappable = True
    return res
