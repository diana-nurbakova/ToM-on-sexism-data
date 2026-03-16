"""TikTok videos dataset analysis — adapted pipeline for 2-3 annotators.

Unlike tweets/memes (6 annotators, always 3F+3M), TikTok videos have:
- 2 or 3 annotators per instance
- Variable gender composition
- No demographic fields (age, ethnicity, education, country)
- Task prefix task3 (normalized to task1 internally)

Runs Sections 0, 1, 3a, and 3d.
"""

from __future__ import annotations

from collections import Counter

import pandas as pd
import numpy as np
from scipy import stats

from nlpercep.correction import holm_bonferroni, rank_biserial_signed, cohens_h


HEADER = "VIDEOS (TikTok)"


# ── Section 0: Descriptive ─────────────────────────────────────────────────────

def section0(df: pd.DataFrame) -> dict:
    counts = df.groupby(["lang", "split"]).size().reset_index(name="count")

    # Annotator info
    ann_counts = df["n_annotators"].value_counts().sort_index()
    all_ids = set()
    for _, row in df.iterrows():
        for i in range(row["n_annotators"]):
            aid = row.get(f"ann_{i}_id")
            if aid is not None:
                all_ids.add(aid)
    n_unique = len(all_ids)

    # Gender pattern distribution
    def _gender_pattern(row):
        genders = []
        for i in range(row["n_annotators"]):
            g = row.get(f"ann_{i}_gender")
            if g is not None:
                genders.append(g)
        return tuple(sorted(genders))

    gender_patterns = df.apply(_gender_pattern, axis=1).value_counts()

    # Label distributions
    all_1_1 = []
    all_1_2 = []
    all_1_3: list[str] = []
    for _, row in df.iterrows():
        for i in range(row["n_annotators"]):
            v = row.get(f"ann_{i}_task1_1")
            if v is not None:
                all_1_1.append(v)
            v = row.get(f"ann_{i}_task1_2")
            if v is not None and v != "-":
                all_1_2.append(v)
            cats = row.get(f"ann_{i}_task1_3")
            if isinstance(cats, list):
                all_1_3.extend(c for c in cats if c != "-")
            elif cats is not None and cats != "-":
                all_1_3.append(cats)

    dist_1_1 = pd.Series(Counter(all_1_1)).sort_values(ascending=False)
    dist_1_2 = pd.Series(Counter(all_1_2)).sort_values(ascending=False)
    dist_1_3 = pd.Series(Counter(all_1_3)).sort_values(ascending=False)

    print(f"\n{'=' * 60}")
    print(f"{HEADER} — SECTION 0: DESCRIPTIVE STATISTICS")
    print(f"{'=' * 60}")
    print(f"\nTotal instances: {len(df)}")
    print(f"\nInstances by language and split:")
    print(counts.to_string(index=False))
    print(f"\nUnique annotators: {n_unique}")
    print(f"\nAnnotators per instance:")
    for n_ann, c in ann_counts.items():
        print(f"  {n_ann} annotators: {c} instances ({c / len(df) * 100:.1f}%)")
    print(f"\nGender composition:")
    for pattern, c in gender_patterns.items():
        print(f"  {pattern}: {c} ({c / len(df) * 100:.1f}%)")
    print(f"\n--- Label distributions ---")
    print(f"\nTask 3.1 (sexism detection):\n{dist_1_1.to_string()}")
    print(f"\nTask 3.2 (source intention):\n{dist_1_2.to_string()}")
    print(f"\nTask 3.3 (categorization):\n{dist_1_3.to_string()}")

    return {
        "instance_counts": counts,
        "n_unique_annotators": n_unique,
        "annotator_counts": ann_counts,
        "gender_patterns": gender_patterns,
        "label_distributions": {"task3_1": dist_1_1, "task3_2": dist_1_2, "task3_3": dist_1_3},
    }


# ── Section 1: Disagreement Measurement ────────────────────────────────────────

def section1(df: pd.DataFrame) -> dict:
    order = ["unanimous_yes", "majority_yes", "split", "majority_no", "unanimous_no"]
    agree_dist = df["agreement_cat"].value_counts().reindex(order).fillna(0).astype(int)

    e_stats = {
        "mean": df["entropy_1_1"].mean(),
        "median": df["entropy_1_1"].median(),
        "std": df["entropy_1_1"].std(),
    }

    full_agree = ((df["yes_count"] == df["n_annotators"]) | (df["yes_count"] == 0)).sum()
    total = len(df)
    agree_props = {
        "full_agreement_n": int(full_agree),
        "full_agreement_pct": full_agree / total * 100,
        "any_disagreement_n": int(total - full_agree),
        "any_disagreement_pct": (total - full_agree) / total * 100,
    }

    # Conditional disagreement among majority YES (yes_ratio > 0.5)
    majority_yes = df[df["yes_ratio"] > 0.5]
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
        sub_full = ((sub["yes_count"] == sub["n_annotators"]) | (sub["yes_count"] == 0)).sum()
        sub_maj = sub[sub["yes_ratio"] > 0.5]
        lang_results[lang] = {
            "n": len(sub),
            "entropy_mean": sub["entropy_1_1"].mean(),
            "full_agreement_pct": sub_full / len(sub) * 100,
            "cond_entropy_1_2": sub_maj["entropy_1_2"].mean(),
        }

    print(f"\n{'=' * 60}")
    print(f"{HEADER} — SECTION 1: DISAGREEMENT MEASUREMENT")
    print(f"{'=' * 60}")

    print("\n--- Task 3.1 Agreement Categories ---")
    for cat, count in agree_dist.items():
        pct = count / total * 100
        print(f"  {cat:20s}: {count:5d} ({pct:5.1f}%)")

    print(f"\n--- Task 3.1 Entropy ---")
    print(f"  Mean:   {e_stats['mean']:.4f}")
    print(f"  Median: {e_stats['median']:.4f}")
    print(f"  Std:    {e_stats['std']:.4f}")

    print(f"\n--- Full Agreement vs Disagreement ---")
    print(f"  Full agreement:   {agree_props['full_agreement_n']:5d} ({agree_props['full_agreement_pct']:.1f}%)")
    print(f"  Any disagreement: {agree_props['any_disagreement_n']:5d} ({agree_props['any_disagreement_pct']:.1f}%)")

    print(f"\n--- Conditional Disagreement (majority YES, n={cond['n_majority_yes']}) ---")
    print(f"  Task 3.2 entropy — mean: {cond['entropy_1_2_mean']:.4f}, median: {cond['entropy_1_2_median']:.4f}")
    print(f"  Task 3.3 Jaccard  — mean: {cond['jaccard_1_3_mean']:.4f}, median: {cond['jaccard_1_3_median']:.4f}")

    print(f"\n--- By Language ---")
    for lang, res in lang_results.items():
        print(f"  [{lang.upper()}] n={res['n']}, entropy={res['entropy_mean']:.4f}, "
              f"full_agree={res['full_agreement_pct']:.1f}%, "
              f"cond_entropy_3.2={res['cond_entropy_1_2']:.4f}")

    return {
        "agreement_distribution": agree_dist,
        "entropy_stats": e_stats,
        "agreement_proportions": agree_props,
        "conditional_disagreement": cond,
        "by_language": lang_results,
    }


# ── Section 3a: Overall gender comparison ──────────────────────────────────────

def section3a(df: pd.DataFrame) -> dict:
    # Only instances with both genders represented
    has_both = df["f_yes_ratio"].notna() & df["m_yes_ratio"].notna()
    sub = df[has_both].copy()

    f_mean = sub["f_yes_ratio"].mean()
    m_mean = sub["m_yes_ratio"].mean()

    diffs = (sub["f_yes_ratio"] - sub["m_yes_ratio"]).values
    nonzero = diffs[diffs != 0]
    if len(nonzero) > 0:
        w_stat, p_value = stats.wilcoxon(nonzero)
        r_rb = rank_biserial_signed(diffs, w_stat)
    else:
        w_stat, p_value, r_rb = np.nan, np.nan, np.nan

    result = {
        "f_yes_mean": f_mean,
        "m_yes_mean": m_mean,
        "difference": f_mean - m_mean,
        "wilcoxon_W": w_stat,
        "p_value": p_value,
        "rank_biserial_r": r_rb,
        "n_both_genders": len(sub),
        "n_total": len(df),
    }

    print(f"\n{'=' * 60}")
    print(f"{HEADER} — SECTION 3a: OVERALL GENDER COMPARISON")
    print(f"{'=' * 60}")
    print(f"  Instances with both genders: {len(sub)} / {len(df)}")
    print(f"  Female YES rate: {f_mean:.4f}")
    print(f"  Male YES rate:   {m_mean:.4f}")
    print(f"  Difference (F-M): {result['difference']:.4f}")
    print(f"  Wilcoxon W = {w_stat}, p = {p_value:.6f}, r = {r_rb:.4f}")
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
            for i in range(row["n_annotators"]):
                gender = row.get(f"ann_{i}_gender")
                if row.get(f"ann_{i}_task1_1") != "YES":
                    continue
                if gender == "F":
                    f_total += 1
                    cats = row.get(f"ann_{i}_task1_3")
                    if isinstance(cats, list) and cat in cats:
                        f_count += 1
                elif gender == "M":
                    m_total += 1
                    cats = row.get(f"ann_{i}_task1_3")
                    if isinstance(cats, list) and cat in cats:
                        m_count += 1

        f_rate = f_count / f_total if f_total > 0 else 0
        m_rate = m_count / m_total if m_total > 0 else 0

        table = np.array([[f_count, f_total - f_count], [m_count, m_total - m_count]])
        if table.min() >= 0 and (f_total + m_total) > 0:
            odds_ratio, p_val = stats.fisher_exact(table)
        else:
            odds_ratio, p_val = np.nan, np.nan

        rows.append({
            "category": cat,
            "f_rate": f_rate,
            "m_rate": m_rate,
            "f_count": f_count,
            "m_count": m_count,
            "f_total": f_total,
            "m_total": m_total,
            "diff": f_rate - m_rate,
            "cohens_h": cohens_h(f_rate, m_rate),
            "odds_ratio": odds_ratio,
            "p_value": p_val,
        })

    cat_rates = pd.DataFrame(rows)

    # Holm-Bonferroni correction across the 5 category tests
    cat_rates["p_adjusted"] = holm_bonferroni(cat_rates["p_value"].values)

    print(f"\n{'=' * 60}")
    print(f"{HEADER} — SECTION 3d: GENDER CATEGORY RATES (Holm-corrected)")
    print(f"{'=' * 60}")
    print(f"  F total YES annotations: {rows[0]['f_total'] if rows else 0}")
    print(f"  M total YES annotations: {rows[0]['m_total'] if rows else 0}")
    print(cat_rates.to_string(index=False))

    return cat_rates


# ── Runner ─────────────────────────────────────────────────────────────────────

def run(df: pd.DataFrame) -> dict:
    """Run the full videos analysis pipeline."""
    results = {
        "s0": section0(df),
        "s1": section1(df),
        "s3a": section3a(df),
        "s3d": section3d(df),
    }
    return results
