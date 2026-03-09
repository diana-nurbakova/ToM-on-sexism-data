"""Section 0: Descriptive statistics of the EXIST 2025 dataset."""

from __future__ import annotations

from collections import Counter

import pandas as pd
import numpy as np


def instance_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Total instances by language and split."""
    return df.groupby(["lang", "split"]).size().reset_index(name="count")


def unique_annotators(df: pd.DataFrame) -> int:
    ann_cols = [f"ann_{i}_id" for i in range(6)]
    all_ids = set()
    for col in ann_cols:
        all_ids.update(df[col].unique())
    return len(all_ids)


def annotators_per_instance(df: pd.DataFrame) -> dict:
    """Confirm always 6 annotators, always 3F+3M."""
    n_annotators = df["n_annotators"].unique().tolist()
    gender_cols = [f"ann_{i}_gender" for i in range(6)]
    f_counts = df[gender_cols].apply(lambda r: (r == "F").sum(), axis=1)
    m_counts = df[gender_cols].apply(lambda r: (r == "M").sum(), axis=1)
    return {
        "n_annotators_values": n_annotators,
        "always_3F": bool((f_counts == 3).all()),
        "always_3M": bool((m_counts == 3).all()),
    }


def demographic_distributions(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Distribution of annotator demographics."""
    results = {}
    for attr in ["gender", "age", "ethnicity", "education", "country"]:
        cols = [f"ann_{i}_{attr}" for i in range(6)]
        all_vals = []
        for col in cols:
            all_vals.extend(df[col].tolist())
        results[attr] = pd.Series(Counter(all_vals)).sort_values(ascending=False)
    return results


def label_distributions(df: pd.DataFrame) -> dict:
    """Overall label distributions for Tasks 1.1, 1.2, 1.3."""
    # Task 1.1
    task1_1_cols = [f"ann_{i}_task1_1" for i in range(6)]
    all_1_1 = []
    for col in task1_1_cols:
        all_1_1.extend(df[col].tolist())
    dist_1_1 = pd.Series(Counter(all_1_1)).sort_values(ascending=False)

    # Task 1.2
    task1_2_cols = [f"ann_{i}_task1_2" for i in range(6)]
    all_1_2 = []
    for col in task1_2_cols:
        all_1_2.extend(df[col].tolist())
    # exclude "-" (annotators who said NO on task 1.1)
    all_1_2 = [l for l in all_1_2 if l != "-"]
    dist_1_2 = pd.Series(Counter(all_1_2)).sort_values(ascending=False)

    # Task 1.3
    task1_3_cols = [f"ann_{i}_task1_3" for i in range(6)]
    all_1_3: list[str] = []
    for col in task1_3_cols:
        for cats in df[col]:
            if isinstance(cats, list):
                all_1_3.extend(c for c in cats if c != "-")
            elif cats != "-":
                all_1_3.append(cats)
    dist_1_3 = pd.Series(Counter(all_1_3)).sort_values(ascending=False)

    return {"task1_1": dist_1_1, "task1_2": dist_1_2, "task1_3": dist_1_3}


def run(df: pd.DataFrame) -> dict:
    """Run all Section 0 analyses and return results dict."""
    counts = instance_counts(df)
    n_unique = unique_annotators(df)
    ann_check = annotators_per_instance(df)
    demographics = demographic_distributions(df)
    labels = label_distributions(df)

    print("=" * 60)
    print("SECTION 0: DESCRIPTIVE STATISTICS")
    print("=" * 60)

    print(f"\nTotal instances: {len(df)}")
    print(f"\nInstances by language and split:")
    print(counts.to_string(index=False))

    print(f"\nUnique annotators: {n_unique}")
    print(f"Always 6 annotators: {ann_check['n_annotators_values']}")
    print(f"Always 3F+3M: F={ann_check['always_3F']}, M={ann_check['always_3M']}")

    print("\n--- Demographic distributions ---")
    for attr, dist in demographics.items():
        print(f"\n{attr.upper()}:")
        print(dist.to_string())

    print("\n--- Label distributions ---")
    for task, dist in labels.items():
        print(f"\n{task}:")
        print(dist.to_string())

    return {
        "instance_counts": counts,
        "n_unique_annotators": n_unique,
        "annotator_check": ann_check,
        "demographics": demographics,
        "label_distributions": labels,
    }
