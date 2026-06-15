"""Data access + gold soft-label derivation for the LLM audit.

Wraps ``nlpercep.data.load_dataset`` (the canonical loader; do not re-implement)
and derives the EXIST gold *soft* labels — the annotator-proportion distributions
the calibration metrics compare model outputs against (master spec §5.1, §7.1).

The soft-label functions are pure (they take the raw per-annotator label lists) so
they can be unit-checked against hand-computed fixtures without a DataFrame.
Internal class keys match the prompt JSON keys in ``prompts-e0-baseline.md`` so a
parsed model distribution and the gold distribution share one schema.
"""

from __future__ import annotations

import pandas as pd

from nlpercep.data import load_dataset

# ── Canonical class keys (match the elicitation JSON keys, prompts §6) ─────────
T11_CLASSES = ("not_sexist", "sexist")
T12_CLASSES = ("not_sexist", "direct", "reported", "judgemental")
T13_CATEGORIES = (
    "ideological_inequality",
    "stereotyping_dominance",
    "objectification",
    "sexual_violence",
    "misogyny_non_sexual_violence",
)

# EXIST dataset identifier -> internal key.
_INTENT_MAP = {"DIRECT": "direct", "REPORTED": "reported", "JUDGEMENTAL": "judgemental"}
_CATEGORY_MAP = {
    "IDEOLOGICAL-INEQUALITY": "ideological_inequality",
    "STEREOTYPING-DOMINANCE": "stereotyping_dominance",
    "OBJECTIFICATION": "objectification",
    "SEXUAL-VIOLENCE": "sexual_violence",
    "MISOGYNY-NON-SEXUAL-VIOLENCE": "misogyny_non_sexual_violence",
}
# Tie-break order for the E1 stratification category (master spec §5.2).
_STRAT_TIEBREAK = T13_CATEGORIES


# ── Pure soft-label functions ─────────────────────────────────────────────────
def soft_label_1_1(t1_1: list[str]) -> dict[str, float]:
    """Detection soft label: P(sexist) = YES proportion over the annotators."""
    n = len(t1_1)
    if n == 0:
        raise ValueError("empty annotator list")
    yes = sum(1 for v in t1_1 if v == "YES")
    p = yes / n
    return {"not_sexist": 1.0 - p, "sexist": p}


def soft_label_1_2(t1_1: list[str], t1_2: list[str]) -> dict[str, float]:
    """Source-intention soft label over {not_sexist, direct, reported, judgemental}.

    Each annotator falls in exactly one bin: a valid intention if they judged the
    item sexist and named one, otherwise ``not_sexist``. So the four masses sum to 1.
    """
    n = len(t1_1)
    if n == 0 or len(t1_2) != n:
        raise ValueError("annotator lists must be equal, non-empty length")
    counts = {c: 0 for c in T12_CLASSES}
    for det, intent in zip(t1_1, t1_2):
        cls = _INTENT_MAP.get(intent) if det == "YES" else None
        counts[cls if cls else "not_sexist"] += 1
    return {c: counts[c] / n for c in T12_CLASSES}


def soft_label_1_3(t1_1: list[str], t1_3: list[list[str]]) -> dict[str, float]:
    """Multi-label categorisation soft label: per-category annotator proportion.

    Denominator is the full annotator count (an annotator who judged the item not
    sexist contributes 0 to every category), matching the EXIST hierarchical
    convention. Categories do not sum to 1.
    """
    n = len(t1_1)
    if n == 0 or len(t1_3) != n:
        raise ValueError("annotator lists must be equal, non-empty length")
    counts = {c: 0 for c in T13_CATEGORIES}
    for det, cats in zip(t1_1, t1_3):
        if det != "YES" or not isinstance(cats, list):
            continue
        for raw in cats:
            key = _CATEGORY_MAP.get(raw)
            if key is not None:
                counts[key] += 1
    return {c: counts[c] / n for c in T13_CATEGORIES}


# ── Stratification helpers (E1/regression sampling, master spec §5.2) ──────────
def agreement_bin(t1_1: list[str]) -> str:
    """Detection-agreement bin: high {6-0,5-1}, mixed {4-2}, low {3-3}."""
    yes = sum(1 for v in t1_1 if v == "YES")
    no = len(t1_1) - yes
    disagreement = min(yes, no)
    if disagreement <= 1:
        return "high"
    if disagreement == 2:
        return "mixed"
    return "low"


def stratification_category(t1_1: list[str], t1_3: list[list[str]]) -> str | None:
    """Single 1.3 category for stratification: highest annotator agreement,
    fixed tie-break order. ``None`` if no category mass (e.g. not-sexist item)."""
    soft = soft_label_1_3(t1_1, t1_3)
    top = max(soft.values())
    if top == 0.0:
        return None
    for cat in _STRAT_TIEBREAK:  # tie-break by canonical order
        if soft[cat] == top:
            return cat
    return None


# ── Row/DataFrame plumbing ────────────────────────────────────────────────────
def _ann_lists(row: pd.Series) -> tuple[list[str], list[str], list[list[str]]]:
    n = int(row["n_annotators"])
    t1_1 = [row[f"ann_{i}_task1_1"] for i in range(n)]
    t1_2 = [row[f"ann_{i}_task1_2"] for i in range(n)]
    t1_3 = [row[f"ann_{i}_task1_3"] for i in range(n)]
    return t1_1, t1_2, t1_3


def add_gold_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Attach gold soft-label dicts and stratification keys to each row."""
    df = df.copy()
    records = df.apply(lambda r: _ann_lists(r), axis=1)
    df["gold_1_1"] = [soft_label_1_1(t1) for t1, _, _ in records]
    df["gold_1_2"] = [soft_label_1_2(t1, t2) for t1, t2, _ in records]
    df["gold_1_3"] = [soft_label_1_3(t1, t3) for t1, _, t3 in records]
    df["agreement_bin"] = [agreement_bin(t1) for t1, _, _ in records]
    df["strat_category"] = [stratification_category(t1, t3) for t1, _, t3 in records]
    return df


def load_pool(lang: str | None = None) -> pd.DataFrame:
    """Load the train+dev pool with gold soft labels and stratification keys.

    Args:
        lang: optional ``'en'`` / ``'es'`` filter. ``None`` returns both.
    """
    df = load_dataset()
    if lang is not None:
        df = df[df["lang"] == lang].reset_index(drop=True)
    df = df.rename(columns={"id": "item_id"})
    df["item_id"] = df["item_id"].astype(str)
    return add_gold_columns(df)


def stratified_sample(
    df: pd.DataFrame, n: int, by: list[str], seed: int = 42
) -> pd.DataFrame:
    """Proportional stratified sample of ``n`` rows over the ``by`` columns.

    Allocates per-stratum counts proportional to stratum size (largest-remainder),
    samples within each stratum under a fixed seed. Used for the pilot and smoke
    slices; E1's exact bin-first allocation is handled separately in ``regression``.
    """
    if n >= len(df):
        return df.reset_index(drop=True)
    groups = list(df.groupby(by, dropna=False))
    sizes = {key: len(g) for key, g in groups}
    total = len(df)
    # Largest-remainder apportionment.
    raw = {key: n * sz / total for key, sz in sizes.items()}
    alloc = {key: int(v) for key, v in raw.items()}
    remainder = n - sum(alloc.values())
    for key, _ in sorted(raw.items(), key=lambda kv: kv[1] - alloc[kv[0]], reverse=True)[:remainder]:
        alloc[key] += 1
    parts = []
    for key, g in groups:
        take = min(alloc.get(key, 0), len(g))
        if take > 0:
            parts.append(g.sample(n=take, random_state=seed))
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
