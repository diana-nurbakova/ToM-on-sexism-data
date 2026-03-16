"""Multiple-comparison correction and effect-size utilities."""

from __future__ import annotations

import numpy as np


# ── Effect sizes ──────────────────────────────────────────────────────────────


def rank_biserial_signed(diffs: np.ndarray, w_stat: float) -> float:
    """Matched-pairs rank-biserial r for a Wilcoxon signed-rank test.

    r = 1 - (2T) / (n(n+1)/2), with sign from the direction of differences.
    |r| interpretation: <0.1 negligible, 0.1-0.3 small, 0.3-0.5 medium, >0.5 large.
    """
    nonzero = diffs[diffs != 0]
    n = len(nonzero)
    if n == 0:
        return np.nan
    total_rank_sum = n * (n + 1) / 2
    r = 1 - (2 * w_stat) / total_rank_sum
    # Sign: positive when F > M on average
    if np.mean(nonzero) < 0:
        r = -r
    return r


def rank_biserial_mannwhitney(u_stat: float, n1: int, n2: int) -> float:
    """Rank-biserial r for a Mann-Whitney U test.

    r = 1 - (2U) / (n1 * n2).
    """
    if n1 == 0 or n2 == 0:
        return np.nan
    return 1 - (2 * u_stat) / (n1 * n2)


def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h for comparing two proportions.

    h = 2 arcsin(sqrt(p1)) - 2 arcsin(sqrt(p2)).
    |h| interpretation: 0.2 small, 0.5 medium, 0.8 large.
    """
    return 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))


# ── Multiple-comparison correction ────────────────────────────────────────────


def holm_bonferroni(p_values: list[float] | np.ndarray) -> np.ndarray:
    """Apply Holm-Bonferroni step-down correction to a vector of p-values.

    Returns adjusted p-values (capped at 1.0) in the *original* order.
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    if m == 0:
        return p.copy()

    # Sort ascending; keep track of original indices
    order = np.argsort(p)
    sorted_p = p[order]

    # Step-down: adjusted_i = p_i * (m - i)  (i is 0-based rank)
    adjusted = sorted_p * np.arange(m, 0, -1)

    # Enforce monotonicity (each adjusted value >= previous)
    for i in range(1, m):
        adjusted[i] = max(adjusted[i], adjusted[i - 1])

    # Cap at 1
    adjusted = np.minimum(adjusted, 1.0)

    # Restore original order
    result = np.empty(m)
    result[order] = adjusted
    return result
