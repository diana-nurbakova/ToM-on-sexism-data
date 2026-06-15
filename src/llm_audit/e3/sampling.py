"""E3 bin-first stratified sampling (master spec §5.4).

The 500-item ToM-scaffolding sub-sample of the train+dev pool, stratified by the
same 5 (category) × 3 (detection-agreement bin) grid as E1, targeting ~33 items per
cell. Identical bin-first machinery to ``e1.sampling`` (compute each item's cell,
sample per cell under fixed seed 42, take-all-and-record on sparse cells, no
backfill) — only the total changes (500 vs 200). E4 (``e4.sampling``) and H1c
(``h1c.sampling``) reuse this exact set, so E3 is the single source of truth for the
mid-size sub-sample.

Disjointness (spec §5.2: E1 is drawn disjoint from the E3 and E5 samples and the
regression set) is enforced by the caller through ``exclude_ids`` — the run script
passes the regression set plus the E1 (and E5) item_ids so nothing is both
sampled here and reported on elsewhere.
"""

from __future__ import annotations

import pandas as pd

from ..data import T13_CATEGORIES, load_pool

BINS = ("high", "mixed", "low")
TARGET_TOTAL = 500
SEED = 42


def build_e3_sample(df: pd.DataFrame | None = None, lang: str = "en",
                    exclude_ids: set | None = None, target: int = TARGET_TOTAL,
                    seed: int = SEED) -> pd.DataFrame:
    """Return the E3 stratified sample with realised per-cell counts recorded.

    Args:
        df: a gold-annotated pool (``data.load_pool``); loaded for ``lang`` if None.
        exclude_ids: item_ids to hold out (regression set, E1/E5 samples) so the
            E3 set stays disjoint from the other reported sub-samples (§5.2).
        target: total sample size (pre-registered 500).
        seed: sampling seed (42).
    """
    if df is None:
        df = load_pool(lang)
    exclude_ids = {str(i) for i in (exclude_ids or set())}
    pool = df[(df["lang"] == lang) & (~df["item_id"].isin(exclude_ids))
              & (df["strat_category"].notna())].copy()

    per_cell = target // (len(T13_CATEGORIES) * len(BINS))  # 500//15 = 33
    parts, shortfalls = [], []
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
    """Realised count per (category × bin) cell, for the spec's reporting."""
    counts = (sample.groupby(["strat_category", "agreement_bin"]).size()
              .reset_index(name="n"))
    return counts.sort_values(["strat_category", "agreement_bin"]).reset_index(drop=True)
