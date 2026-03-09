"""Section 2 (Analysis A): Intent ambiguity predicts disagreement.

Primary operationalization: Task 1.2 labels as implicit/explicit proxy.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from scipy import stats


def classify_ambiguity(df: pd.DataFrame) -> pd.DataFrame:
    """Classify instances as explicit vs implicit/ambiguous using Task 1.2 labels.

    - Explicit: all YES annotators agree on DIRECT
    - Implicit/ambiguous: YES annotators disagree on Task 1.2, or majority is REPORTED/JUDGEMENTAL
    - Excluded: instances with no YES annotators
    """
    df = df.copy()

    def _classify(row):
        labels = row["task1_2_yes_labels"]
        if len(labels) == 0:
            return "no_yes"
        unique = set(labels)
        if unique == {"DIRECT"}:
            return "explicit"
        else:
            return "implicit"

    df["ambiguity_group"] = df.apply(_classify, axis=1)
    return df


def compare_disagreement(df: pd.DataFrame) -> dict:
    """Compare Task 1.1 entropy between explicit and implicit groups."""
    explicit = df[df["ambiguity_group"] == "explicit"]
    implicit = df[df["ambiguity_group"] == "implicit"]

    if len(explicit) == 0 or len(implicit) == 0:
        return {"error": "One or both groups are empty"}

    # Mann-Whitney U test
    u_stat, p_value = stats.mannwhitneyu(
        explicit["entropy_1_1"], implicit["entropy_1_1"],
        alternative="less",  # explicit < implicit expected
    )

    # Effect size: rank-biserial correlation
    n1, n2 = len(explicit), len(implicit)
    rank_biserial = 1 - (2 * u_stat) / (n1 * n2)

    return {
        "n_explicit": n1,
        "n_implicit": n2,
        "entropy_explicit_mean": explicit["entropy_1_1"].mean(),
        "entropy_explicit_median": explicit["entropy_1_1"].median(),
        "entropy_implicit_mean": implicit["entropy_1_1"].mean(),
        "entropy_implicit_median": implicit["entropy_1_1"].median(),
        "mann_whitney_U": u_stat,
        "p_value": p_value,
        "rank_biserial_r": rank_biserial,
    }


def by_language(df: pd.DataFrame) -> dict:
    """Run the comparison split by language."""
    results = {}
    for lang in ["en", "es"]:
        sub = df[df["lang"] == lang]
        sub = classify_ambiguity(sub)
        results[lang] = compare_disagreement(sub)
    return results


def run(df: pd.DataFrame) -> dict:
    """Run all Analysis A."""
    df = classify_ambiguity(df)
    overall = compare_disagreement(df)
    lang_results = by_language(df)

    # Group size summary
    group_counts = df["ambiguity_group"].value_counts()

    print("\n" + "=" * 60)
    print("SECTION 2 (ANALYSIS A): INTENT AMBIGUITY PREDICTS DISAGREEMENT")
    print("=" * 60)

    print("\n--- Ambiguity group sizes ---")
    for group, count in group_counts.items():
        print(f"  {group:15s}: {count}")

    print("\n--- Entropy comparison ---")
    print(f"  Explicit  — mean: {overall['entropy_explicit_mean']:.4f}, median: {overall['entropy_explicit_median']:.4f} (n={overall['n_explicit']})")
    print(f"  Implicit  — mean: {overall['entropy_implicit_mean']:.4f}, median: {overall['entropy_implicit_median']:.4f} (n={overall['n_implicit']})")
    print(f"\n  Mann-Whitney U = {overall['mann_whitney_U']:.1f}")
    print(f"  p-value = {overall['p_value']:.6f}")
    print(f"  Rank-biserial r = {overall['rank_biserial_r']:.4f}")

    sig = "***" if overall["p_value"] < 0.001 else "**" if overall["p_value"] < 0.01 else "*" if overall["p_value"] < 0.05 else "n.s."
    print(f"  Significance: {sig}")

    print("\n--- By Language ---")
    for lang, res in lang_results.items():
        if "error" in res:
            print(f"  [{lang.upper()}] {res['error']}")
            continue
        print(f"  [{lang.upper()}] explicit={res['n_explicit']}, implicit={res['n_implicit']}")
        print(f"    Entropy explicit: {res['entropy_explicit_mean']:.4f}, implicit: {res['entropy_implicit_mean']:.4f}")
        print(f"    p={res['p_value']:.6f}, r={res['rank_biserial_r']:.4f}")

    return {
        "ambiguity_groups": group_counts.to_dict(),
        "overall": overall,
        "by_language": lang_results,
        "df_with_ambiguity": df,
    }
