"""Qualitative example selection for Table 3 in the paper.

Selects illustrative tweets showing:
1. Detection agreement + interpretation divergence (unanimous YES, divergent 1.2/1.3)
2. Gender-differential categorization (same detection, F sees violence, M sees stereotyping)
"""

from __future__ import annotations

import pandas as pd
import numpy as np


def find_detection_agree_interpretation_diverge(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Find instances with unanimous/strong YES on Task 1.1 but high disagreement on 1.2/1.3.

    These are the paper's core examples: "agree it's sexist, disagree on everything else."
    """
    # Strong or unanimous YES
    candidates = df[df["yes_count"] >= 5].copy()

    # High Task 1.2 entropy (near-maximum interpretation disagreement)
    candidates = candidates.dropna(subset=["entropy_1_2"])
    candidates = candidates.sort_values("entropy_1_2", ascending=False)

    # Among top entropy, prefer those with low Jaccard (Task 1.3 disagreement too)
    top = candidates.head(n * 3)
    top = top.sort_values("jaccard_1_3", ascending=True)

    results = top.head(n).copy()
    return _format_examples(results, "detection_agree_interp_diverge")


def find_gender_differential_categorization(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Find instances where F and M annotators agree it's sexist but assign different categories.

    Specifically: F sees SEXUAL-VIOLENCE or MISOGYNY, M sees STEREOTYPING or IDEOLOGICAL.
    """
    # Both genders agree it's sexist (at least 2/3 each say YES)
    candidates = df[(df["f_yes_count"] >= 2) & (df["m_yes_count"] >= 2)].copy()

    harm_cats = {"SEXUAL-VIOLENCE", "MISOGYNY-NON-SEXUAL-VIOLENCE"}
    structural_cats = {"STEREOTYPING-DOMINANCE", "IDEOLOGICAL-INEQUALITY"}

    def _gender_cat_divergence(row):
        f_cats = set()
        m_cats = set()
        for i in range(3):
            if row[f"ann_{i}_task1_1"] == "YES":
                cats = row[f"ann_{i}_task1_3"]
                if isinstance(cats, list):
                    f_cats.update(c for c in cats if c != "-")
        for i in range(3, 6):
            if row[f"ann_{i}_task1_1"] == "YES":
                cats = row[f"ann_{i}_task1_3"]
                if isinstance(cats, list):
                    m_cats.update(c for c in cats if c != "-")

        # Score: F has harm cats that M doesn't, M has structural cats that F doesn't
        f_harm = len(f_cats & harm_cats - m_cats)
        m_structural = len(m_cats & structural_cats - f_cats)
        return f_harm + m_structural

    candidates["gender_cat_div"] = candidates.apply(_gender_cat_divergence, axis=1)
    candidates = candidates[candidates["gender_cat_div"] > 0]
    candidates = candidates.sort_values("gender_cat_div", ascending=False)

    results = candidates.head(n).copy()
    return _format_examples(results, "gender_differential_cat")


def _format_examples(df: pd.DataFrame, example_type: str) -> pd.DataFrame:
    """Format selected examples for display/output."""
    rows = []
    for _, row in df.iterrows():
        # Collect per-annotator info
        ann_details = []
        for i in range(6):
            gender = row[f"ann_{i}_gender"]
            t11 = row[f"ann_{i}_task1_1"]
            t12 = row[f"ann_{i}_task1_2"]
            t13 = row[f"ann_{i}_task1_3"]
            if isinstance(t13, list):
                t13 = ", ".join(t13)
            ann_details.append(f"{gender}: {t11} | {t12} | {t13}")

        rows.append({
            "id": row["id"],
            "lang": row["lang"],
            "text": row["text"][:200] + ("..." if len(row["text"]) > 200 else ""),
            "yes_count": row["yes_count"],
            "entropy_1_1": round(row["entropy_1_1"], 3),
            "entropy_1_2": round(row["entropy_1_2"], 3) if pd.notna(row["entropy_1_2"]) else None,
            "jaccard_1_3": round(row["jaccard_1_3"], 3) if pd.notna(row["jaccard_1_3"]) else None,
            "annotator_details": " || ".join(ann_details),
            "example_type": example_type,
        })
    return pd.DataFrame(rows)


def run(df: pd.DataFrame) -> dict:
    """Select all qualitative examples."""
    ex1 = find_detection_agree_interpretation_diverge(df)
    ex2 = find_gender_differential_categorization(df)

    print("\n" + "=" * 60)
    print("QUALITATIVE EXAMPLES")
    print("=" * 60)

    print("\n--- Type 1: Detection agreement + interpretation divergence ---")
    for _, row in ex1.iterrows():
        print(f"\n  [{row['id']}] ({row['lang']}) YES={row['yes_count']}/6, "
              f"entropy_1.2={row['entropy_1_2']}, jaccard_1.3={row['jaccard_1_3']}")
        print(f"  Text: {row['text']}")
        for ann in row["annotator_details"].split(" || "):
            print(f"    {ann}")

    print("\n--- Type 2: Gender-differential categorization ---")
    for _, row in ex2.iterrows():
        print(f"\n  [{row['id']}] ({row['lang']}) YES={row['yes_count']}/6")
        print(f"  Text: {row['text']}")
        for ann in row["annotator_details"].split(" || "):
            print(f"    {ann}")

    return {
        "detection_agree_interp_diverge": ex1,
        "gender_differential_cat": ex2,
    }
