"""Section 3 (Analysis B): Annotator gender moderates perception.

Tests whether M/F annotators differ in sexism detection and whether
the effect is concentrated on intent-ambiguous cases.
"""

from __future__ import annotations

from collections import Counter

import pandas as pd
import numpy as np
from scipy import stats

from nlpercep.correction import (
    holm_bonferroni,
    rank_biserial_signed,
    rank_biserial_mannwhitney,
    cohens_h,
)


# ── 3a: Overall gender comparison ──────────────────────────────────────────────

def overall_gender_comparison(df: pd.DataFrame) -> dict:
    """Compare F vs M YES ratios across all instances."""
    f_mean = df["f_yes_ratio"].mean()
    m_mean = df["m_yes_ratio"].mean()

    # Wilcoxon signed-rank (paired)
    diffs = (df["f_yes_ratio"] - df["m_yes_ratio"]).values
    nonzero = diffs[diffs != 0]
    if len(nonzero) > 0:
        w_stat, p_value = stats.wilcoxon(nonzero)
        r_rb = rank_biserial_signed(diffs, w_stat)
    else:
        w_stat, p_value, r_rb = np.nan, np.nan, np.nan

    return {
        "f_yes_mean": f_mean,
        "m_yes_mean": m_mean,
        "difference": f_mean - m_mean,
        "wilcoxon_W": w_stat,
        "p_value": p_value,
        "rank_biserial_r": r_rb,
        "n": len(df),
    }


# ── 3b: Gender disagreement by agreement level ────────────────────────────────

def gender_split_analysis(df: pd.DataFrame) -> dict:
    """Analyze gender patterns within 3-3 split cases."""
    splits = df[df["yes_count"] == 3].copy()
    n_splits = len(splits)
    if n_splits == 0:
        return {"n_splits": 0}

    # Check if 3-3 splits align with gender
    gender_aligned = splits[
        ((splits["f_yes_count"] == 3) & (splits["m_yes_count"] == 0)) |
        ((splits["f_yes_count"] == 0) & (splits["m_yes_count"] == 3))
    ]
    n_aligned = len(gender_aligned)

    f_yes_m_no = splits[(splits["f_yes_count"] == 3) & (splits["m_yes_count"] == 0)]
    f_no_m_yes = splits[(splits["f_yes_count"] == 0) & (splits["m_yes_count"] == 3)]

    return {
        "n_splits": n_splits,
        "n_gender_aligned": n_aligned,
        "pct_gender_aligned": n_aligned / n_splits * 100 if n_splits > 0 else 0,
        "n_3F_yes_3M_no": len(f_yes_m_no),
        "n_3F_no_3M_yes": len(f_no_m_yes),
        "n_mixed": n_splits - n_aligned,
    }


def gender_by_agreement_level(df: pd.DataFrame) -> pd.DataFrame:
    """For each agreement level, compute mean gender difference."""
    rows = []
    for cat in ["unanimous_yes", "strong_yes", "weak_yes", "split", "weak_no", "strong_no", "unanimous_no"]:
        sub = df[df["agreement_cat"] == cat]
        if len(sub) == 0:
            continue
        rows.append({
            "agreement_cat": cat,
            "n": len(sub),
            "f_yes_mean": sub["f_yes_ratio"].mean(),
            "m_yes_mean": sub["m_yes_ratio"].mean(),
            "gender_diff": (sub["f_yes_ratio"] - sub["m_yes_ratio"]).mean(),
        })
    return pd.DataFrame(rows)


# ── 3c: Gender × ambiguity interaction ─────────────────────────────────────────

def gender_ambiguity_interaction(df: pd.DataFrame) -> dict:
    """Test interaction between gender effect and ambiguity.

    Requires 'ambiguity_group' column from section2.
    """
    if "ambiguity_group" not in df.columns:
        return {"error": "ambiguity_group not computed — run Section 2 first"}

    results = {}
    for group in ["explicit", "implicit"]:
        sub = df[df["ambiguity_group"] == group]
        if len(sub) == 0:
            results[group] = {"n": 0}
            continue
        diffs = (sub["f_yes_ratio"] - sub["m_yes_ratio"]).values
        nonzero = diffs[diffs != 0]
        if len(nonzero) > 0:
            w_stat, p_val = stats.wilcoxon(nonzero)
            r_rb = rank_biserial_signed(diffs, w_stat)
        else:
            w_stat, p_val, r_rb = np.nan, np.nan, np.nan
        results[group] = {
            "n": len(sub),
            "f_yes_mean": sub["f_yes_ratio"].mean(),
            "m_yes_mean": sub["m_yes_ratio"].mean(),
            "gender_diff": diffs.mean(),
            "wilcoxon_W": w_stat,
            "p_value": p_val,
            "rank_biserial_r": r_rb,
        }

    # Compare gender difference magnitude between groups
    explicit_diffs = (df[df["ambiguity_group"] == "explicit"]["f_yes_ratio"] -
                      df[df["ambiguity_group"] == "explicit"]["m_yes_ratio"])
    implicit_diffs = (df[df["ambiguity_group"] == "implicit"]["f_yes_ratio"] -
                      df[df["ambiguity_group"] == "implicit"]["m_yes_ratio"])

    if len(explicit_diffs) > 0 and len(implicit_diffs) > 0:
        u_stat, p_interaction = stats.mannwhitneyu(
            explicit_diffs.abs(), implicit_diffs.abs(),
            alternative="less",
        )
        r_rb = rank_biserial_mannwhitney(u_stat, len(explicit_diffs), len(implicit_diffs))
        results["interaction_test"] = {
            "mann_whitney_U": u_stat,
            "p_value": p_interaction,
            "rank_biserial_r": r_rb,
        }

    return results


# ── 3d: Gender effect on categorization ────────────────────────────────────────

def gender_category_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Per-category detection rates by gender among YES annotators."""
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

        # Fisher's exact test on 2x2: (assigned cat / not assigned cat) × (F / M)
        table = np.array([
            [f_count, f_total - f_count],
            [m_count, m_total - m_count],
        ])
        if table.min() >= 0:
            odds_ratio, p_val = stats.fisher_exact(table)
        else:
            odds_ratio, p_val = np.nan, np.nan

        rows.append({
            "category": cat,
            "f_rate": f_rate,
            "m_rate": m_rate,
            "f_count": f_count,
            "m_count": m_count,
            "diff": f_rate - m_rate,
            "cohens_h": cohens_h(f_rate, m_rate),
            "odds_ratio": odds_ratio,
            "p_value": p_val,
        })

    cat_rates = pd.DataFrame(rows)

    # Holm-Bonferroni correction across the 5 category tests
    cat_rates["p_adjusted"] = holm_bonferroni(cat_rates["p_value"].values)

    return cat_rates


# ── Runner ─────────────────────────────────────────────────────────────────────

def run(df: pd.DataFrame) -> dict:
    """Run all Analysis B."""
    overall = overall_gender_comparison(df)
    split_analysis = gender_split_analysis(df)
    by_level = gender_by_agreement_level(df)
    interaction = gender_ambiguity_interaction(df)
    cat_rates = gender_category_rates(df)

    print("\n" + "=" * 60)
    print("SECTION 3 (ANALYSIS B): GENDER MODERATES PERCEPTION")
    print("=" * 60)

    # 3a
    print("\n--- 3a: Overall Gender Comparison ---")
    print(f"  Female YES rate: {overall['f_yes_mean']:.4f}")
    print(f"  Male YES rate:   {overall['m_yes_mean']:.4f}")
    print(f"  Difference (F-M): {overall['difference']:.4f}")
    print(f"  Wilcoxon W = {overall['wilcoxon_W']}, p = {overall['p_value']:.6f}, r = {overall['rank_biserial_r']:.4f}")
    sig = "***" if overall["p_value"] < 0.001 else "**" if overall["p_value"] < 0.01 else "*" if overall["p_value"] < 0.05 else "n.s."
    print(f"  Significance: {sig}")

    # 3b
    print("\n--- 3b: Gender Split Analysis (3-3 cases) ---")
    print(f"  Total 3-3 splits: {split_analysis['n_splits']}")
    if split_analysis["n_splits"] > 0:
        print(f"  Gender-aligned:   {split_analysis['n_gender_aligned']} ({split_analysis['pct_gender_aligned']:.1f}%)")
        print(f"    3F=YES, 3M=NO:  {split_analysis['n_3F_yes_3M_no']}")
        print(f"    3F=NO, 3M=YES:  {split_analysis['n_3F_no_3M_yes']}")
        print(f"    Mixed:          {split_analysis['n_mixed']}")

    print("\n--- Gender difference by agreement level ---")
    print(by_level.to_string(index=False))

    # 3c
    print("\n--- 3c: Gender × Ambiguity Interaction ---")
    for group in ["explicit", "implicit"]:
        if group in interaction and interaction[group].get("n", 0) > 0:
            r = interaction[group]
            print(f"  [{group.upper()}] n={r['n']}, F={r['f_yes_mean']:.4f}, M={r['m_yes_mean']:.4f}, diff={r['gender_diff']:.4f}, p={r.get('p_value', 'N/A')}, r={r.get('rank_biserial_r', 'N/A')}")
    if "interaction_test" in interaction:
        it = interaction["interaction_test"]
        print(f"  Interaction: U={it['mann_whitney_U']:.1f}, p={it['p_value']:.6f}, r={it['rank_biserial_r']:.4f}")

    # 3d
    print("\n--- 3d: Gender Category Rates (Holm-corrected) ---")
    print(cat_rates.to_string(index=False))

    return {
        "overall": overall,
        "split_analysis": split_analysis,
        "by_agreement_level": by_level,
        "interaction": interaction,
        "category_rates": cat_rates,
    }
