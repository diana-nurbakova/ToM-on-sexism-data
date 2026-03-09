"""Data loading and preprocessing for EXIST 2025 datasets (tweets, memes, videos)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import numpy as np

_BASE = Path(__file__).resolve().parents[2] / "EXIST 2025 Dataset V0.1"

# Tweets dataset paths
TWEETS_ROOT = _BASE / "EXIST 2025 Tweets Dataset"
TWEETS_TRAIN = TWEETS_ROOT / "training" / "EXIST2025_training.json"
TWEETS_DEV = TWEETS_ROOT / "dev" / "EXIST2025_dev.json"

# Memes dataset paths
MEMES_ROOT = _BASE / "EXIST 2025 Memes Dataset"
MEMES_TRAIN = MEMES_ROOT / "training" / "EXIST2025_training.json"

# Videos (TikTok) dataset paths
VIDEOS_ROOT = _BASE / "EXIST 2025 Videos Dataset"
VIDEOS_TRAIN = VIDEOS_ROOT / "training" / "EXIST2025_training.json"


def load_raw_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def json_to_dataframe(raw: dict, task_prefix: str = "task1", text_field: str = "tweet") -> pd.DataFrame:
    """Convert the EXIST JSON dict into a flat DataFrame with one row per instance.

    Args:
        raw: The parsed JSON dict.
        task_prefix: "task1" for tweets, "task2" for memes.
        text_field: "tweet" for tweets, "text" for memes.
    """
    rows = []
    for instance_id, d in raw.items():
        row = {
            "id": d["id_EXIST"],
            "lang": d["lang"],
            "text": d[text_field],
            "split": d["split"],
            "n_annotators": d["number_annotators"],
        }
        # Per-annotator columns — normalize to ann_*_task1_* regardless of source
        for i in range(6):
            row[f"ann_{i}_id"] = d["annotators"][i]
            row[f"ann_{i}_gender"] = d["gender_annotators"][i]
            row[f"ann_{i}_age"] = d["age_annotators"][i]
            row[f"ann_{i}_ethnicity"] = d["ethnicities_annotators"][i]
            row[f"ann_{i}_education"] = d["study_levels_annotators"][i]
            row[f"ann_{i}_country"] = d["countries_annotators"][i]
            row[f"ann_{i}_task1_1"] = d[f"labels_{task_prefix}_1"][i]
            row[f"ann_{i}_task1_2"] = d[f"labels_{task_prefix}_2"][i]
            row[f"ann_{i}_task1_3"] = d[f"labels_{task_prefix}_3"][i]
        rows.append(row)
    return pd.DataFrame(rows)


def add_computed_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add pre-computed columns useful across all analyses."""
    task1_1_cols = [f"ann_{i}_task1_1" for i in range(6)]
    task1_2_cols = [f"ann_{i}_task1_2" for i in range(6)]

    # YES/NO counts
    df["yes_count"] = df[task1_1_cols].apply(lambda r: (r == "YES").sum(), axis=1)
    df["no_count"] = 6 - df["yes_count"]

    # Female / Male YES counts (convention: first 3 annotators are F, last 3 are M)
    f_cols = [f"ann_{i}_task1_1" for i in range(3)]
    m_cols = [f"ann_{i}_task1_1" for i in range(3, 6)]
    df["f_yes_count"] = df[f_cols].apply(lambda r: (r == "YES").sum(), axis=1)
    df["m_yes_count"] = df[m_cols].apply(lambda r: (r == "YES").sum(), axis=1)
    df["f_yes_ratio"] = df["f_yes_count"] / 3
    df["m_yes_ratio"] = df["m_yes_count"] / 3

    # Shannon entropy for Task 1.1
    def binary_entropy(yes_count):
        p = yes_count / 6
        if p == 0 or p == 1:
            return 0.0
        return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

    df["entropy_1_1"] = df["yes_count"].apply(binary_entropy)

    # Agreement category
    def agreement_category(yes_count):
        mapping = {
            6: "unanimous_yes", 5: "strong_yes", 4: "weak_yes",
            3: "split",
            2: "weak_no", 1: "strong_no", 0: "unanimous_no",
        }
        return mapping[yes_count]

    df["agreement_cat"] = df["yes_count"].apply(agreement_category)

    # Task 1.2 labels among YES annotators
    def get_yes_task1_2_labels(row):
        labels = []
        for i in range(6):
            if row[f"ann_{i}_task1_1"] == "YES":
                labels.append(row[f"ann_{i}_task1_2"])
        return labels

    df["task1_2_yes_labels"] = df.apply(get_yes_task1_2_labels, axis=1)

    # Task 1.2 entropy (among YES annotators)
    def multiclass_entropy(labels):
        if len(labels) == 0:
            return np.nan
        from collections import Counter
        counts = Counter(labels)
        n = len(labels)
        h = 0.0
        for c in counts.values():
            p = c / n
            if p > 0:
                h -= p * np.log2(p)
        return h

    df["entropy_1_2"] = df["task1_2_yes_labels"].apply(multiclass_entropy)

    # Task 1.3 categories among YES annotators
    def get_yes_task1_3_sets(row):
        cat_sets = []
        for i in range(6):
            if row[f"ann_{i}_task1_1"] == "YES":
                cats = row[f"ann_{i}_task1_3"]
                if cats != ["-"]:
                    cat_sets.append(set(cats))
        return cat_sets

    df["task1_3_yes_sets"] = df.apply(get_yes_task1_3_sets, axis=1)

    # Mean pairwise Jaccard for Task 1.3
    def mean_pairwise_jaccard(cat_sets):
        if len(cat_sets) < 2:
            return np.nan
        scores = []
        for i in range(len(cat_sets)):
            for j in range(i + 1, len(cat_sets)):
                union = cat_sets[i] | cat_sets[j]
                if len(union) == 0:
                    continue
                scores.append(len(cat_sets[i] & cat_sets[j]) / len(union))
        return np.mean(scores) if scores else np.nan

    df["jaccard_1_3"] = df["task1_3_yes_sets"].apply(mean_pairwise_jaccard)

    return df


def load_dataset() -> pd.DataFrame:
    """Load and merge tweets train + dev splits, returning a fully prepared DataFrame."""
    dfs = []
    for path in [TWEETS_TRAIN, TWEETS_DEV]:
        if path.exists():
            raw = load_raw_json(path)
            dfs.append(json_to_dataframe(raw, task_prefix="task1", text_field="tweet"))
    df = pd.concat(dfs, ignore_index=True)
    df = add_computed_columns(df)
    return df


def load_memes_dataset() -> pd.DataFrame:
    """Load memes training set, returning a fully prepared DataFrame."""
    raw = load_raw_json(MEMES_TRAIN)
    df = json_to_dataframe(raw, task_prefix="task2", text_field="text")
    df = add_computed_columns(df)
    return df


# ── Videos (TikTok) dataset ──────────────────────────────────────────────────

def json_to_dataframe_videos(raw: dict) -> pd.DataFrame:
    """Convert TikTok JSON into a DataFrame handling variable annotator counts (2-3).

    Unlike tweets/memes (6 annotators, 3F+3M), videos have 2-3 annotators
    with variable gender composition and no demographic fields.
    """
    rows = []
    for instance_id, d in raw.items():
        n_ann = d["number_annotators"]
        row = {
            "id": d["id_EXIST"],
            "lang": d["lang"],
            "text": d["text"],
            "split": d["split"],
            "n_annotators": n_ann,
        }
        for i in range(n_ann):
            row[f"ann_{i}_id"] = d["annotators"][i]
            row[f"ann_{i}_gender"] = d["gender_annotators"][i]
            row[f"ann_{i}_task1_1"] = d["labels_task3_1"][i]
            row[f"ann_{i}_task1_2"] = d["labels_task3_2"][i]
            row[f"ann_{i}_task1_3"] = d["labels_task3_3"][i]
        rows.append(row)
    return pd.DataFrame(rows)


def add_computed_columns_videos(df: pd.DataFrame) -> pd.DataFrame:
    """Add computed columns for the videos dataset (variable annotator count)."""
    from collections import Counter

    def _yes_count(row):
        return sum(
            1 for i in range(row["n_annotators"])
            if row.get(f"ann_{i}_task1_1") == "YES"
        )

    df["yes_count"] = df.apply(_yes_count, axis=1)
    df["no_count"] = df["n_annotators"] - df["yes_count"]
    df["yes_ratio"] = df["yes_count"] / df["n_annotators"]

    # Gender-based YES counts (actual genders, not positional)
    def _gender_yes(row, gender):
        count, total = 0, 0
        for i in range(row["n_annotators"]):
            if row.get(f"ann_{i}_gender") == gender:
                total += 1
                if row.get(f"ann_{i}_task1_1") == "YES":
                    count += 1
        return count, total

    def _f_yes_ratio(row):
        c, t = _gender_yes(row, "F")
        return c / t if t > 0 else np.nan

    def _m_yes_ratio(row):
        c, t = _gender_yes(row, "M")
        return c / t if t > 0 else np.nan

    df["f_yes_ratio"] = df.apply(_f_yes_ratio, axis=1)
    df["m_yes_ratio"] = df.apply(_m_yes_ratio, axis=1)

    # Binary entropy (normalized by n_annotators)
    def binary_entropy(row):
        n = row["n_annotators"]
        p = row["yes_count"] / n
        if p == 0 or p == 1:
            return 0.0
        return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

    df["entropy_1_1"] = df.apply(binary_entropy, axis=1)

    # Agreement category (adapted for 2-3 annotators)
    def agreement_category(row):
        yc, n = row["yes_count"], row["n_annotators"]
        if yc == n:
            return "unanimous_yes"
        if yc == 0:
            return "unanimous_no"
        if yc / n > 0.5:
            return "majority_yes"
        if yc / n < 0.5:
            return "majority_no"
        return "split"

    df["agreement_cat"] = df.apply(agreement_category, axis=1)

    # Task 1.2 labels among YES annotators
    def get_yes_task1_2_labels(row):
        labels = []
        for i in range(row["n_annotators"]):
            if row.get(f"ann_{i}_task1_1") == "YES":
                labels.append(row.get(f"ann_{i}_task1_2"))
        return labels

    df["task1_2_yes_labels"] = df.apply(get_yes_task1_2_labels, axis=1)

    # Task 1.2 entropy
    def multiclass_entropy(labels):
        if len(labels) == 0:
            return np.nan
        counts = Counter(labels)
        n = len(labels)
        h = 0.0
        for c in counts.values():
            p = c / n
            if p > 0:
                h -= p * np.log2(p)
        return h

    df["entropy_1_2"] = df["task1_2_yes_labels"].apply(multiclass_entropy)

    # Task 1.3 category sets among YES annotators
    def get_yes_task1_3_sets(row):
        cat_sets = []
        for i in range(row["n_annotators"]):
            if row.get(f"ann_{i}_task1_1") == "YES":
                cats = row.get(f"ann_{i}_task1_3")
                if isinstance(cats, list) and cats != ["-"]:
                    cat_sets.append(set(cats))
        return cat_sets

    df["task1_3_yes_sets"] = df.apply(get_yes_task1_3_sets, axis=1)

    # Mean pairwise Jaccard
    def mean_pairwise_jaccard(cat_sets):
        if len(cat_sets) < 2:
            return np.nan
        scores = []
        for i in range(len(cat_sets)):
            for j in range(i + 1, len(cat_sets)):
                union = cat_sets[i] | cat_sets[j]
                if len(union) == 0:
                    continue
                scores.append(len(cat_sets[i] & cat_sets[j]) / len(union))
        return np.mean(scores) if scores else np.nan

    df["jaccard_1_3"] = df["task1_3_yes_sets"].apply(mean_pairwise_jaccard)

    return df


def load_videos_dataset() -> pd.DataFrame:
    """Load TikTok videos training set, returning a fully prepared DataFrame."""
    raw = load_raw_json(VIDEOS_TRAIN)
    df = json_to_dataframe_videos(raw)
    df = add_computed_columns_videos(df)
    return df
