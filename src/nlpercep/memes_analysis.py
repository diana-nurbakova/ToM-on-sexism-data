"""Memes dataset analysis — parallel pipeline to tweets.

Runs Sections 0, 1, 3a, and 3d using the same computed columns
(the data loader normalizes task2_* fields to task1_* column names).
"""

from __future__ import annotations

from collections import Counter

import pandas as pd
import numpy as np
from scipy import stats


HEADER = "MEMES"


# ── Section 0: Descriptive ─────────────────────────────────────────────────────

def section0(df: pd.DataFrame) -> dict:
    counts = df.groupby(["lang", "split"]).size().reset_index(name="count")
    ann_cols = [f"ann_{i}_id" for i in range(6)]
    n_unique = len(set(v for col in ann_cols for v in df[col].unique()))

    # Label distributions
    task1_cols = [f"ann_{i}_task1_1" for i in range(6)]
    all_1_1 = [v for col in task1_cols for v in df[col]]
    dist_1_1 = pd.Series(Counter(all_1_1)).sort_values(ascending=False)

    task2_cols = [f"ann_{i}_task1_2" for i in range(6)]
    all_1_2 = [v for col in task2_cols for v in df[col] if v != "-"]
    dist_1_2 = pd.Series(Counter(all_1_2)).sort_values(ascending=False)

    task3_cols = [f"ann_{i}_task1_3" for i in range(6)]
    all_1_3: list[str] = []
    for col in task3_cols:
        for cats in df[col]:
            if isinstance(cats, list):
                all_1_3.extend(c for c in cats if c != "-")
            elif cats != "-":
                all_1_3.append(cats)
    dist_1_3 = pd.Series(Counter(all_1_3)).sort_values(ascending=False)

    # Demographics
    demographics = {}
    for attr in ["gender", "age", "ethnicity", "education", "country"]:
        cols = [f"ann_{i}_{attr}" for i in range(6)]
        all_vals = [v for col in cols for v in df[col]]
        demographics[attr] = pd.Series(Counter(all_vals)).sort_values(ascending=False)

    print(f"\n{'=' * 60}")
    print(f"{HEADER} — SECTION 0: DESCRIPTIVE STATISTICS")
    print(f"{'=' * 60}")
    print(f"\nTotal instances: {len(df)}")
    print(f"\nInstances by language and split:")
    print(counts.to_string(index=False))
    print(f"\nUnique annotators: {n_unique}")
    print(f"\n--- Label distributions ---")
    print(f"\nTask 2.1 (sexism detection):\n{dist_1_1.to_string()}")
    print(f"\nTask 2.2 (source intention):\n{dist_1_2.to_string()}")
    print(f"\nTask 2.3 (categorization):\n{dist_1_3.to_string()}")

    return {
        "instance_counts": counts,
        "n_unique_annotators": n_unique,
        "label_distributions": {"task2_1": dist_1_1, "task2_2": dist_1_2, "task2_3": dist_1_3},
        "demographics": demographics,
    }


# ── Section 1: Disagreement Measurement ────────────────────────────────────────

def section1(df: pd.DataFrame) -> dict:
    order = [
        "unanimous_yes", "strong_yes", "weak_yes",
        "split",
        "weak_no", "strong_no", "unanimous_no",
    ]
    agree_dist = df["agreement_cat"].value_counts().reindex(order).fillna(0).astype(int)

    e_stats = {
        "mean": df["entropy_1_1"].mean(),
        "median": df["entropy_1_1"].median(),
        "std": df["entropy_1_1"].std(),
    }

    full_agree = ((df["yes_count"] == 6) | (df["yes_count"] == 0)).sum()
    total = len(df)
    agree_props = {
        "full_agreement_n": int(full_agree),
        "full_agreement_pct": full_agree / total * 100,
        "any_disagreement_n": int(total - full_agree),
        "any_disagreement_pct": (total - full_agree) / total * 100,
    }

    # Conditional disagreement among majority YES
    majority_yes = df[df["yes_count"] >= 4]
    cond = {
        "n_majority_yes": len(majority_yes),
        "entropy_1_2_mean": majority_yes["entropy_1_2"].mean(),
        "entropy_1_2_median": majority_yes["entropy_1_2"].median(),
        "jaccard_1_3_mean": majority_yes["jaccard_1_3"].mean(),
        "jaccard_1_3_median": majority_yes["jaccard_1_3"].median(),
    }

    # By language
    lang_results = {}
    for lang in sorted(df["lang"].unique()):
        sub = df[df["lang"] == lang]
        sub_full = ((sub["yes_count"] == 6) | (sub["yes_count"] == 0)).sum()
        sub_maj = sub[sub["yes_count"] >= 4]
        lang_results[lang] = {
            "n": len(sub),
            "entropy_mean": sub["entropy_1_1"].mean(),
            "full_agreement_pct": sub_full / len(sub) * 100,
            "cond_entropy_1_2": sub_maj["entropy_1_2"].mean(),
        }

    print(f"\n{'=' * 60}")
    print(f"{HEADER} — SECTION 1: DISAGREEMENT MEASUREMENT")
    print(f"{'=' * 60}")

    print("\n--- Task 2.1 Agreement Categories ---")
    for cat, count in agree_dist.items():
        pct = count / total * 100
        print(f"  {cat:20s}: {count:5d} ({pct:5.1f}%)")

    print(f"\n--- Task 2.1 Entropy ---")
    print(f"  Mean:   {e_stats['mean']:.4f}")
    print(f"  Median: {e_stats['median']:.4f}")
    print(f"  Std:    {e_stats['std']:.4f}")

    print(f"\n--- Full Agreement vs Disagreement ---")
    print(f"  Full agreement:   {agree_props['full_agreement_n']:5d} ({agree_props['full_agreement_pct']:.1f}%)")
    print(f"  Any disagreement: {agree_props['any_disagreement_n']:5d} ({agree_props['any_disagreement_pct']:.1f}%)")

    print(f"\n--- Conditional Disagreement (majority YES, n={cond['n_majority_yes']}) ---")
    print(f"  Task 2.2 entropy — mean: {cond['entropy_1_2_mean']:.4f}, median: {cond['entropy_1_2_median']:.4f}")
    print(f"  Task 2.3 Jaccard  — mean: {cond['jaccard_1_3_mean']:.4f}, median: {cond['jaccard_1_3_median']:.4f}")

    print(f"\n--- By Language ---")
    for lang, res in lang_results.items():
        print(f"  [{lang.upper()}] n={res['n']}, entropy={res['entropy_mean']:.4f}, "
              f"full_agree={res['full_agreement_pct']:.1f}%, "
              f"cond_entropy_2.2={res['cond_entropy_1_2']:.4f}")

    return {
        "agreement_distribution": agree_dist,
        "entropy_stats": e_stats,
        "agreement_proportions": agree_props,
        "conditional_disagreement": cond,
        "by_language": lang_results,
    }


# ── Section 3a: Overall gender comparison ──────────────────────────────────────

def section3a(df: pd.DataFrame) -> dict:
    f_mean = df["f_yes_ratio"].mean()
    m_mean = df["m_yes_ratio"].mean()

    diffs = df["f_yes_ratio"] - df["m_yes_ratio"]
    nonzero = diffs[diffs != 0]
    if len(nonzero) > 0:
        w_stat, p_value = stats.wilcoxon(nonzero)
    else:
        w_stat, p_value = np.nan, np.nan

    result = {
        "f_yes_mean": f_mean,
        "m_yes_mean": m_mean,
        "difference": f_mean - m_mean,
        "wilcoxon_W": w_stat,
        "p_value": p_value,
        "n": len(df),
    }

    print(f"\n{'=' * 60}")
    print(f"{HEADER} — SECTION 3a: OVERALL GENDER COMPARISON")
    print(f"{'=' * 60}")
    print(f"  Female YES rate: {f_mean:.4f}")
    print(f"  Male YES rate:   {m_mean:.4f}")
    print(f"  Difference (F-M): {result['difference']:.4f}")
    print(f"  Wilcoxon W = {w_stat}, p = {p_value:.6f}")
    sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "n.s."
    print(f"  Significance: {sig}")

    return result


# ── Section 3d: Gender category rates ──────────────────────────────────────────

def section3d(df: pd.DataFrame) -> pd.DataFrame:
    categories = [
        "IDEOLOGICAL-INEQUALITY", "STEREOTYPING-DOMINANCE",
        "MISOGYNY-NON-SEXUAL-VIOLENCE", "SEXUAL-VIOLENCE", "OBJECTIFICATION",
    ]

    rows = []
    for cat in categories:
        f_count, m_count = 0, 0
        f_total, m_total = 0, 0
        for _, row in df.iterrows():
            for i in range(3):  # F annotators
                if row[f"ann_{i}_task1_1"] == "YES":
                    f_total += 1
                    cats = row[f"ann_{i}_task1_3"]
                    if isinstance(cats, list) and cat in cats:
                        f_count += 1
            for i in range(3, 6):  # M annotators
                if row[f"ann_{i}_task1_1"] == "YES":
                    m_total += 1
                    cats = row[f"ann_{i}_task1_3"]
                    if isinstance(cats, list) and cat in cats:
                        m_count += 1

        f_rate = f_count / f_total if f_total > 0 else 0
        m_rate = m_count / m_total if m_total > 0 else 0

        table = np.array([[f_count, f_total - f_count], [m_count, m_total - m_count]])
        odds_ratio, p_val = stats.fisher_exact(table)

        rows.append({
            "category": cat,
            "f_rate": f_rate,
            "m_rate": m_rate,
            "f_count": f_count,
            "m_count": m_count,
            "diff": f_rate - m_rate,
            "odds_ratio": odds_ratio,
            "p_value": p_val,
        })

    cat_rates = pd.DataFrame(rows)

    print(f"\n{'=' * 60}")
    print(f"{HEADER} — SECTION 3d: GENDER CATEGORY RATES")
    print(f"{'=' * 60}")
    print(cat_rates.to_string(index=False))

    return cat_rates


# ── Runner ─────────────────────────────────────────────────────────────────────

def run(df: pd.DataFrame) -> dict:
    """Run the full memes analysis pipeline."""
    results = {
        "s0": section0(df),
        "s1": section1(df),
        "s3a": section3a(df),
        "s3d": section3d(df),
    }
    return results
