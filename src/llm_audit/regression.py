"""Shared prompt-regression set: builder + noise-band estimation.

Implements ``prompt-regression-set.md``: ~15 real train+dev items per language in
stratified slots (clear-sexist / clear-not / hard / category-spanning /
definition-sensitive), each carrying its gold soft labels. The set is rerun at
K=6 after every prompt change; the noise band (run-to-run variation of the
unchanged prompt) is the threshold an edit must exceed (§5). Definition-sensitive
items are auto-nominated by criterion but flagged ``needs_review`` for the author's
expert sign-off (the construct-vs-colloquial divergence is not auto-decidable).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import T13_CATEGORIES
from .metrics import aggregate_format_c_single, aggregate_format_c_multi, jsd, multilabel_jsd

_INTENTS = ("direct", "reported", "judgemental")


def _take(df: pd.DataFrame, mask, used: set, n: int, seed: int) -> pd.DataFrame:
    pool = df[mask & ~df["item_id"].isin(used)]
    if pool.empty:
        return pool
    picked = pool.sample(n=min(n, len(pool)), random_state=seed)
    used.update(picked["item_id"])
    return picked


def build_regression_set(df: pd.DataFrame, lang: str, seed: int = 42) -> list[dict]:
    """Select the regression items for one language from a gold-annotated pool.

    ``df`` must carry the gold columns from ``data.add_gold_columns`` and be
    filtered to one language. Returns a list of item dicts (schema in §3).
    """
    df = df[df["lang"] == lang].copy()
    df["p_sexist"] = df["gold_1_1"].apply(lambda d: d["sexist"])
    used: set = set()
    items: list[dict] = []

    def emit(rows: pd.DataFrame, slot: str, **extra):
        for _, r in rows.iterrows():
            items.append({
                "regression_id": f"REG-{lang.upper()}-{len(items) + 1:02d}",
                "exist_item_id": r["item_id"],
                "language": lang,
                "slot_type": slot,
                "text": r["text"],
                "gold_detect_dist": r["gold_1_1"],
                "gold_intent_dist": r["gold_1_2"],
                "gold_category_dist": r["gold_1_3"],
                **extra,
            })

    # Clear-sexist ×3, spread across intent where possible.
    for intent in _INTENTS:
        sel = _take(df, (df["p_sexist"] >= 5 / 6) &
                    (df["gold_1_2"].apply(lambda d, i=intent: d[i] == max(d[k] for k in _INTENTS))),
                    used, 1, seed)
        emit(sel, "clear_sexist", notes=f"high-agreement sexist, {intent}-leaning intent")

    # Clear-not-sexist ×3.
    emit(_take(df, df["p_sexist"] <= 1 / 6, used, 3, seed), "clear_not",
         notes="high-agreement not sexist")

    # Hard / 3-3 detection split ×3.
    emit(_take(df, df["agreement_bin"] == "low", used, 3, seed), "hard",
         notes="near-maximal detection disagreement (3-3)")

    # Category-spanning ×5: one high-agreement item per 1.3 category.
    for cat in T13_CATEGORIES:
        sel = _take(df, df["gold_1_3"].apply(lambda d, c=cat: d[c] >= 0.5), used, 1, seed)
        emit(sel, f"category_{cat}", notes=f"high agreement on {cat}")

    # Definition-sensitive ×2 (nominated, flagged for review).
    desc = _take(df, (df["p_sexist"] >= 4 / 6) &
                 (df["gold_1_2"].apply(lambda d: d["judgemental"] == max(d[k] for k in _INTENTS))),
                 used, 2, seed)
    emit(desc, "definition_sensitive", needs_review=True,
         notes="describes/criticises sexism: defined should pull SEXIST, bare may not")

    return items


def regression_dataframe(df: pd.DataFrame, langs=("en", "es"), seed: int = 42) -> pd.DataFrame:
    rows = []
    for lang in langs:
        rows.extend(build_regression_set(df, lang, seed))
    return pd.DataFrame(rows)


# ── Noise band ────────────────────────────────────────────────────────────────
def _chunk(draws: list, size: int) -> list[list]:
    return [draws[i:i + size] for i in range(0, len(draws) - size + 1, size)]


def soft_labels_per_repeat(draws: list, subtask: str, k: int = 6) -> list[dict]:
    """Split an ordered draw list (length R*k) into R independent K=k soft labels."""
    out = []
    for chunk in _chunk(draws, k):
        if subtask == "1.3":
            out.append(aggregate_format_c_multi(chunk))
        else:
            out.append(aggregate_format_c_single(chunk, subtask))
    return out


def estimate_noise_band(repeat_softs_by_item: dict, subtask: str,
                        B: int = 1000, seed: int = 42, pct: float = 95.0) -> dict:
    """Bootstrap the unchanged-prompt noise band from repeated soft labels.

    Args:
        repeat_softs_by_item: {item_id: [soft_dist, ...]} — R independent soft
            labels per item from rerunning the unchanged prompt.
    Returns per-item mean run-to-run divergence and a bootstrapped set-level upper
    band (the ``pct`` percentile of the resampled set mean).
    """
    from .data import T11_CLASSES, T12_CLASSES
    canon = {"1.1": T11_CLASSES, "1.2": T12_CLASSES, "1.3": T13_CATEGORIES}[subtask]
    div = multilabel_jsd if subtask == "1.3" else jsd

    per_item = {}
    for item, softs in repeat_softs_by_item.items():
        vecs = [np.array([s.get(c, 0.0) for c in canon], float) for s in softs]
        pair_divs = [div(vecs[i], vecs[j])
                     for i in range(len(vecs)) for j in range(i + 1, len(vecs))]
        if pair_divs:
            per_item[item] = float(np.mean(pair_divs))

    vals = np.array(list(per_item.values()), float)
    rng = np.random.default_rng(seed)
    if len(vals) == 0:
        return {"per_item": per_item, "set_mean": None, "upper_band": None}
    boot = [rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(B)]
    return {
        "per_item": per_item,
        "set_mean": float(vals.mean()),
        "upper_band": float(np.percentile(boot, pct)),
        "subtask": subtask,
    }
