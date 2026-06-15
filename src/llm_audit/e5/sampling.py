"""E5 sampling: the 200-item Cialdini sub-sample (master spec §5.6).

E5 is "stratified by category and by E1 invariance partition (overlapping with E1
items)". So the grid is the 5 (stratification category) × 3 (invariance class:
INVARIANT / SHIFTING / AMBIGUOUS) cells, targeting ~13 items per cell to total ~200,
with UNSWAPPABLE items excluded (they carry no swap behaviour for the persuasion ×
asymmetry slice).

Two design points mirror the rest of the audit:

* **Invariance class** is taken from the *same frozen construct rule* E1/E2 use
  (``e1.rule.construct_rule_tag`` via ``e2`` screening), so E5 runs standalone before
  the E1 worksheet is finalised. When the finalised E1 partition is available it is
  passed as ``e1_partition`` and overrides the first-pass class per item (the
  author-adjudicated label is authoritative).
* **Overlap with E1** is realised by preferring E1's own sampled items in each cell
  (``e1_ids``): the cell is filled from the E1 items first, then topped up from the
  rest of the swappability-screened pool. With ``e1_ids`` unset the overlap still
  arises naturally (E1 items sit in the same pool), just without the preference.

Disjointness from the regression set (and any other held-out ids) flows through
``exclude_ids``, matching ``e1``/``e3`` sampling.
"""

from __future__ import annotations

import pandas as pd

from ..data import T13_CATEGORIES, load_pool
from ..e1.rule import EXCLUSION, INVARIANCE_CLASSES, construct_rule_tag
from ..e2.sampling import screen_swappable

TARGET_TOTAL = 200
SEED = 42


def _invariance_class(row, e1_partition: dict | None) -> str:
    """Finalised E1 class if available for this item, else the frozen rule first-pass."""
    if e1_partition is not None:
        final = e1_partition.get(str(row["item_id"]))
        if final:
            return str(final).strip().upper()
    cls, _needs_adj, _reason = construct_rule_tag(
        row["gold_1_1"]["sexist"], row["gendered_swapped"],
        row["name_swapped"], not row["swappable"])
    return cls


def build_e5_sample(df: pd.DataFrame | None = None, lang: str = "en",
                    e1_ids: set | None = None, e1_partition: dict | None = None,
                    exclude_ids: set | None = None, target: int = TARGET_TOTAL,
                    seed: int = SEED) -> pd.DataFrame:
    """Return the E5 stratified sample (category × invariance class), overlapping E1.

    Args:
        df: a gold-annotated pool (``data.load_pool``); loaded for ``lang`` if None.
        e1_ids: E1 sample item_ids, preferred in each cell to maximise E1 overlap.
        e1_partition: ``{item_id: invariance_class}`` from the finalised E1 partition
            (``e1_partition_<lang>.json``); overrides the first-pass class per item.
        exclude_ids: item_ids to hold out (regression set, etc.).
        target: total sample size (200).
        seed: sampling seed (42).

    The result carries an ``invariance_class`` column and an ``in_e1`` overlap flag;
    per-cell shortfalls are recorded in ``.attrs['shortfalls']``.
    """
    if df is None:
        df = load_pool(lang)
    exclude_ids = {str(i) for i in (exclude_ids or set())}
    e1_ids = {str(i) for i in (e1_ids or set())}

    scr = screen_swappable(df[df["lang"] == lang])
    pool = scr[scr["swappable"] & scr["strat_category"].notna()
               & ~scr["item_id"].isin(exclude_ids)].copy()
    pool["invariance_class"] = pool.apply(lambda r: _invariance_class(r, e1_partition), axis=1)
    pool = pool[pool["invariance_class"] != EXCLUSION]
    pool["in_e1"] = pool["item_id"].isin(e1_ids)

    per_cell = target // (len(T13_CATEGORIES) * len(INVARIANCE_CLASSES))  # 200//15 = 13
    parts, shortfalls = [], []
    for cat in T13_CATEGORIES:
        for cls in INVARIANCE_CLASSES:
            cell = pool[(pool["strat_category"] == cat) & (pool["invariance_class"] == cls)]
            take = min(per_cell, len(cell))
            if len(cell) < per_cell:
                shortfalls.append({"category": cat, "invariance_class": cls,
                                   "available": len(cell), "target": per_cell})
            if take > 0:
                parts.append(_take_prefer_e1(cell, take, seed))
    sample = pd.concat(parts).reset_index(drop=True) if parts else pool.iloc[0:0].copy()
    sample.attrs["shortfalls"] = shortfalls
    sample.attrs["per_cell_target"] = per_cell
    return sample


def _take_prefer_e1(cell: pd.DataFrame, take: int, seed: int) -> pd.DataFrame:
    """Fill the cell from its E1-overlap items first, then top up from the rest."""
    in_e1 = cell[cell["in_e1"]]
    if len(in_e1) >= take:
        return in_e1.sample(n=take, random_state=seed)
    rest = cell[~cell["in_e1"]]
    top_up = rest.sample(n=take - len(in_e1), random_state=seed)
    return pd.concat([in_e1, top_up])


def cell_report(sample: pd.DataFrame) -> pd.DataFrame:
    """Realised count per (category × invariance class) cell, for the spec's reporting."""
    counts = (sample.groupby(["strat_category", "invariance_class"]).size()
              .reset_index(name="n"))
    return counts.sort_values(["strat_category", "invariance_class"]).reset_index(drop=True)
