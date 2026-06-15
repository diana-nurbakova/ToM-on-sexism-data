"""E2 sampling: full-pool swappability screen, equal-N per category (master spec §5.3).

Screens the full train+dev pool with the post-fix E1 swappability check, then takes
``target`` (120) swappable items per stratification-category under a fixed seed, with
the take-all shortfall rule (no padding across categories). Returns the items with
their original + swapped text and swap flags ready for the driver.
"""

from __future__ import annotations

import pandas as pd

from ..data import T13_CATEGORIES, load_pool
from ..e1 import swap

TARGET_PER_CATEGORY = 120
SEED = 42


def screen_swappable(df: pd.DataFrame) -> pd.DataFrame:
    """Attach swap result, swapped text, and swap flags; mark swappability."""
    df = df.copy()
    res = df["text"].apply(swap.swap_text)
    df["swapped"] = res.apply(lambda r: r.swapped)
    df["swappable"] = res.apply(lambda r: not r.unswappable)
    df["name_swapped"] = res.apply(lambda r: r.name_swapped)
    df["gendered_swapped"] = res.apply(lambda r: r.gendered_swapped)
    df["female_source"] = res.apply(lambda r: r.female_source)
    df["male_source"] = res.apply(lambda r: r.male_source)
    return df


def swap_direction(row) -> str:
    """F2M (woman→man), M2F (man→woman), MIXED, or NONE — for the asymmetry metric."""
    f, m = bool(row["female_source"]), bool(row["male_source"])
    if f and m:
        return "MIXED"
    if f:
        return "F2M"
    if m:
        return "M2F"
    return "NONE"


def build_e2_sample(df: pd.DataFrame | None = None, lang: str = "en",
                    target: int = TARGET_PER_CATEGORY, seed: int = SEED) -> pd.DataFrame:
    """Equal-N (``target``) swappable items per stratification-category.

    Returns the selected items with ``swapped``, swap flags, and ``swap_direction``.
    Records per-category shortfalls in ``.attrs['shortfalls']``.
    """
    if df is None:
        df = load_pool(lang)
    df = screen_swappable(df[df["lang"] == lang])
    pool = df[df["strat_category"].notna() & df["swappable"]]
    parts, shortfalls = [], []
    for cat in T13_CATEGORIES:
        cell = pool[pool["strat_category"] == cat]
        take = min(target, len(cell))
        if len(cell) < target:
            shortfalls.append({"category": cat, "available": len(cell), "target": target})
        if take > 0:
            parts.append(cell.sample(n=take, random_state=seed))
    sample = pd.concat(parts).reset_index(drop=True) if parts else pool.iloc[0:0].copy()
    sample["swap_direction"] = sample.apply(swap_direction, axis=1)
    sample.attrs["shortfalls"] = shortfalls
    sample.attrs["target"] = target
    return sample
