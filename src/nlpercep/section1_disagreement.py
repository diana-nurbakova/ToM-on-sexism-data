"""Section 1: Disagreement measurement — foundation for all analyses."""

from __future__ import annotations

import pandas as pd
import numpy as np


def agreement_category_distribution(df: pd.DataFrame) -> pd.Series:
    """Distribution of agreement categories for Task 1.1."""
    order = [
        "unanimous_yes", "strong_yes", "weak_yes",
        "split",
        "weak_no", "strong_no", "unanimous_no",
    ]
    counts = df["agreement_cat"].value_counts()
    return counts.reindex(order).fillna(0).astype(int)


def entropy_stats(df: pd.DataFrame) -> dict:
    """Mean/median entropy for Task 1.1."""
    return {
        "mean": df["entropy_1_1"].mean(),
        "median": df["entropy_1_1"].median(),
        "std": df["entropy_1_1"].std(),
    }


def agreement_proportions(df: pd.DataFrame) -> dict:
    """% full agreement vs any disagreement."""
    full_agree = ((df["yes_count"] == 6) | (df["yes_count"] == 0)).sum()
    total = len(df)
    return {
        "full_agreement_n": int(full_agree),
        "full_agreement_pct": full_agree / total * 100,
        "any_disagreement_n": int(total - full_agree),
        "any_disagreement_pct": (total - full_agree) / total * 100,
    }


def conditional_disagreement(df: pd.DataFrame) -> dict:
    """Among majority-YES instances, entropy on Task 1.2 and 1.3."""
    majority_yes = df[df["yes_count"] >= 4].copy()
    return {
        "n_majority_yes": len(majority_yes),
        "entropy_1_2_mean": majority_yes["entropy_1_2"].mean(),
        "entropy_1_2_median": majority_yes["entropy_1_2"].median(),
        "jaccard_1_3_mean": majority_yes["jaccard_1_3"].mean(),
        "jaccard_1_3_median": majority_yes["jaccard_1_3"].median(),
    }


def by_language(df: pd.DataFrame) -> dict:
    """Break all metrics by language."""
    results = {}
    for lang in ["en", "es"]:
        sub = df[df["lang"] == lang]
        results[lang] = {
            "n": len(sub),
            "agreement_dist": agreement_category_distribution(sub),
            "entropy_stats": entropy_stats(sub),
            "agreement_props": agreement_proportions(sub),
            "conditional": conditional_disagreement(sub),
        }
    return results


def run(df: pd.DataFrame) -> dict:
    """Run all Section 1 analyses."""
    agree_dist = agreement_category_distribution(df)
    e_stats = entropy_stats(df)
    agree_props = agreement_proportions(df)
    cond = conditional_disagreement(df)
    lang_results = by_language(df)

    print("\n" + "=" * 60)
    print("SECTION 1: DISAGREEMENT MEASUREMENT")
    print("=" * 60)

    print("\n--- Task 1.1 Agreement Categories ---")
    for cat, count in agree_dist.items():
        pct = count / len(df) * 100
        print(f"  {cat:20s}: {count:5d} ({pct:5.1f}%)")

    print(f"\n--- Task 1.1 Entropy ---")
    print(f"  Mean:   {e_stats['mean']:.4f}")
    print(f"  Median: {e_stats['median']:.4f}")
    print(f"  Std:    {e_stats['std']:.4f}")

    print(f"\n--- Full Agreement vs Disagreement ---")
    print(f"  Full agreement:   {agree_props['full_agreement_n']:5d} ({agree_props['full_agreement_pct']:.1f}%)")
    print(f"  Any disagreement: {agree_props['any_disagreement_n']:5d} ({agree_props['any_disagreement_pct']:.1f}%)")

    print(f"\n--- Conditional Disagreement (majority YES, n={cond['n_majority_yes']}) ---")
    print(f"  Task 1.2 entropy — mean: {cond['entropy_1_2_mean']:.4f}, median: {cond['entropy_1_2_median']:.4f}")
    print(f"  Task 1.3 Jaccard  — mean: {cond['jaccard_1_3_mean']:.4f}, median: {cond['jaccard_1_3_median']:.4f}")

    print(f"\n--- By Language ---")
    for lang, res in lang_results.items():
        print(f"\n  [{lang.upper()}] n={res['n']}")
        print(f"    Entropy mean: {res['entropy_stats']['mean']:.4f}")
        print(f"    Full agreement: {res['agreement_props']['full_agreement_pct']:.1f}%")
        print(f"    Conditional Task 1.2 entropy: {res['conditional']['entropy_1_2_mean']:.4f}")

    return {
        "agreement_distribution": agree_dist,
        "entropy_stats": e_stats,
        "agreement_proportions": agree_props,
        "conditional_disagreement": cond,
        "by_language": lang_results,
    }
