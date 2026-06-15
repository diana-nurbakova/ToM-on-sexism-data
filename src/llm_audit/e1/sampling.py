"""E1 bin-first stratified sampling (OSF pre-reg §4, Appendix C 'Indices').

Forms the 15 (category × detection-agreement-bin) cells over the train+dev pool,
targets ~13-14 items per cell to total ~200, samples per cell under fixed seed 42,
and applies the pre-registered sparse-cell contingency (take all available, record
the shortfall, no backfill). Items are drawn disjoint from the prompt-regression
set so nothing is both tuned-on and reported-on.

Stratification uses one category per item (highest-agreement, fixed tie-break) from
``data.stratification_category``; items with no category mass (unanimous not-sexist)
fall outside the 5×3 grid by design and are not sampled.
"""

from __future__ import annotations

import pandas as pd

from ..data import T13_CATEGORIES, load_pool

BINS = ("high", "mixed", "low")
TARGET_TOTAL = 200
SEED = 42


def build_sample(df: pd.DataFrame | None = None, lang: str = "en",
                 exclude_ids: set | None = None, seed: int = SEED) -> pd.DataFrame:
    """Return the E1 stratified sample with realised per-cell counts recorded.

    Args:
        df: a gold-annotated pool (``data.load_pool``); loaded for ``lang`` if None.
        exclude_ids: item_ids to hold out (e.g. the regression set), enforcing
            disjointness.
        seed: sampling seed (pre-registered 42).
    """
    if df is None:
        df = load_pool(lang)
    exclude_ids = {str(i) for i in (exclude_ids or set())}
    pool = df[(df["lang"] == lang) & (~df["item_id"].isin(exclude_ids))
              & (df["strat_category"].notna())].copy()

    per_cell = TARGET_TOTAL // (len(T13_CATEGORIES) * len(BINS))  # 200//15 = 13
    parts = []
    shortfalls = []
    for cat in T13_CATEGORIES:
        for b in BINS:
            cell = pool[(pool["strat_category"] == cat) & (pool["agreement_bin"] == b)]
            take = min(per_cell, len(cell))
            if len(cell) < per_cell:
                shortfalls.append({"category": cat, "bin": b,
                                   "available": len(cell), "target": per_cell})
            if take > 0:
                parts.append(cell.sample(n=take, random_state=seed))
    sample = pd.concat(parts).reset_index(drop=True) if parts else pool.iloc[0:0].copy()
    sample.attrs["shortfalls"] = shortfalls
    sample.attrs["per_cell_target"] = per_cell
    return sample


def cell_report(sample: pd.DataFrame) -> pd.DataFrame:
    """Realised count per (category × bin) cell, for the pre-reg's reporting."""
    counts = (sample.groupby(["strat_category", "agreement_bin"]).size()
              .reset_index(name="n"))
    return counts.sort_values(["strat_category", "agreement_bin"]).reset_index(drop=True)
