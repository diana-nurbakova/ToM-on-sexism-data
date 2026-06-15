"""H1c sampling: the persona-grid sub-sample (``prompts-h1c-persona.md`` §2.5).

H1c runs on a sub-sample rather than the full pool, and the prompt spec fixes that
sub-sample as "the E3 500-item set ... since H1c and the scaffolding analysis share
the per-category logic". Reusing E3's items means the no-persona reference
(bare-prompt E0) and the per-category female-minus-male shift are computed on the
same items as the scaffolding analysis. This module therefore delegates to
``e3.sampling`` (the single source of truth) so H1c and E3 cannot drift apart.
"""

from __future__ import annotations

import pandas as pd

from ..e3.sampling import SEED, TARGET_TOTAL, build_e3_sample, cell_report

__all__ = ["SEED", "TARGET_TOTAL", "build_h1c_sample", "cell_report"]


def build_h1c_sample(df: pd.DataFrame | None = None, lang: str = "en",
                     exclude_ids: set | None = None, target: int = TARGET_TOTAL,
                     seed: int = SEED) -> pd.DataFrame:
    """Return the H1c sub-sample — the E3 500-item set (§2.5).

    Arguments mirror ``e3.sampling.build_e3_sample`` so the same disjointness and
    seed flow through; the persona conditions and the bare-prompt E0 reference run
    on these items.
    """
    return build_e3_sample(df, lang=lang, exclude_ids=exclude_ids,
                           target=target, seed=seed)
