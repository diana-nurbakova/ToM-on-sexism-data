"""E4 sampling: the 500-item base-vs-instruct sub-sample, overlapping with E3 (§5.5).

The spec sets E4 on a "500-item sub-sample, overlapping with E3". The base-vs-instruct
gap (H6a) and the counter-prime shift (H6b) are read *per item* against the same
items E3 scaffolds, so E4 takes the **same** 500 items as E3 rather than redrawing —
full overlap is the cleanest form of "overlapping with E3" and keeps the H6/H7
analyses on a shared item set. This module therefore delegates to ``e3.sampling``
(the single source of truth for the mid-size sub-sample) so the two cannot drift; the
seam stays here so an E4-specific draw could be substituted later without touching E3.
"""

from __future__ import annotations

import pandas as pd

from ..e3.sampling import SEED, TARGET_TOTAL, build_e3_sample, cell_report

__all__ = ["SEED", "TARGET_TOTAL", "build_e4_sample", "cell_report"]


def build_e4_sample(df: pd.DataFrame | None = None, lang: str = "en",
                    exclude_ids: set | None = None, target: int = TARGET_TOTAL,
                    seed: int = SEED) -> pd.DataFrame:
    """Return the E4 sub-sample — the E3 500-item set (full overlap, §5.5).

    Arguments mirror ``e3.sampling.build_e3_sample`` so the same disjointness
    (``exclude_ids``: regression set, E1/E5 samples) and seed flow through.
    """
    return build_e3_sample(df, lang=lang, exclude_ids=exclude_ids,
                           target=target, seed=seed)
