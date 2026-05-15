"""Section 3 (extension): Gender × Intent attribution test.

Spec: specs/spec-gender-intent-test.md

Tests whether annotator gender structures intent attribution (Task 1.2 / 2.2)
disagreement, completing the ToM mapping evidence. The cognitive ToM claim
predicts no gender structuring at the intent level.

Test A: per-label Fisher's exact (parallel to per-category Task 1.3 test).
Test B: gender-alignment of intent splits (parallel to 3-3 detection split).
Test C: paired Wilcoxon on per-instance F vs M intent entropy (exploratory).
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats

from nlpercep.correction import (
    cohens_h,
    holm_bonferroni,
    rank_biserial_signed,
)


TWEET_INTENT_LABELS = ["DIRECT", "REPORTED", "JUDGEMENTAL"]
MEME_INTENT_LABELS = ["DIRECT", "JUDGEMENTAL"]


# ── Test A: Per-Label Gender Comparison ───────────────────────────────────────


def per_label_gender_comparison(
    df: pd.DataFrame,
    intent_labels: list[str],
    use_actual_gender: bool = False,
) -> pd.DataFrame:
    """Fisher's exact test per intent label with Holm correction.

    Population: annotations where Task X.1 = YES and Task X.2 != UNKNOWN.
    For tweets/memes the gender is positional (0–2 = F, 3–5 = M); pass
    ``use_actual_gender=True`` only if the dataset stores it differently.
    """
    rows = []
    for label in intent_labels:
        f_count, m_count = 0, 0
        f_total, m_total = 0, 0
        for _, row in df.iterrows():
            n_ann = int(row.get("n_annotators", 6))
            for i in range(n_ann):
                if row.get(f"ann_{i}_task1_1") != "YES":
                    continue
                intent = row.get(f"ann_{i}_task1_2")
                if intent == "UNKNOWN" or intent == "-" or intent is None:
                    continue
                gender = row.get(f"ann_{i}_gender") if use_actual_gender else ("F" if i < 3 else "M")
                if gender == "F":
                    f_total += 1
                    if intent == label:
                        f_count += 1
                elif gender == "M":
                    m_total += 1
                    if intent == label:
                        m_count += 1

        f_rate = f_count / f_total if f_total > 0 else 0.0
        m_rate = m_count / m_total if m_total > 0 else 0.0
        table = np.array([
            [f_count, f_total - f_count],
            [m_count, m_total - m_count],
        ])
        if f_total > 0 and m_total > 0:
            odds_ratio, p_val = stats.fisher_exact(table)
        else:
            odds_ratio, p_val = np.nan, np.nan

        rows.append({
            "intent_label": label,
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

    out = pd.DataFrame(rows)
    out["p_adjusted"] = holm_bonferroni(out["p_value"].values)
    return out


def unknown_by_gender(df: pd.DataFrame, use_actual_gender: bool = False) -> dict:
    """Report UNKNOWN intent counts by gender (asked in spec implementation note 2)."""
    f_unk, m_unk = 0, 0
    f_total_yes, m_total_yes = 0, 0
    for _, row in df.iterrows():
        n_ann = int(row.get("n_annotators", 6))
        for i in range(n_ann):
            if row.get(f"ann_{i}_task1_1") != "YES":
                continue
            gender = row.get(f"ann_{i}_gender") if use_actual_gender else ("F" if i < 3 else "M")
            intent = row.get(f"ann_{i}_task1_2")
            if gender == "F":
                f_total_yes += 1
                if intent == "UNKNOWN":
                    f_unk += 1
            elif gender == "M":
                m_total_yes += 1
                if intent == "UNKNOWN":
                    m_unk += 1
    return {
        "f_unknown": f_unk,
        "m_unknown": m_unk,
        "f_yes_total": f_total_yes,
        "m_yes_total": m_total_yes,
        "f_unknown_rate": f_unk / f_total_yes if f_total_yes else 0.0,
        "m_unknown_rate": m_unk / m_total_yes if m_total_yes else 0.0,
    }


# ── Test B: Gender-Alignment of Intent Splits ─────────────────────────────────


def _yes_intent_records(row, use_actual_gender: bool = False) -> list[tuple[str, str]]:
    """Return [(gender, intent), ...] for YES annotators with a non-UNKNOWN intent."""
    out = []
    n_ann = int(row.get("n_annotators", 6))
    for i in range(n_ann):
        if row.get(f"ann_{i}_task1_1") != "YES":
            continue
        intent = row.get(f"ann_{i}_task1_2")
        if intent == "UNKNOWN" or intent == "-" or intent is None:
            continue
        gender = row.get(f"ann_{i}_gender") if use_actual_gender else ("F" if i < 3 else "M")
        if gender not in {"F", "M"}:
            continue
        out.append((gender, intent))
    return out


def gender_alignment_of_intent_splits(
    df: pd.DataFrame,
    use_actual_gender: bool = False,
    alignment_threshold: float = 0.5,
) -> dict:
    """Test B: % of majority-YES instances where intent disagreement aligns with gender.

    Procedure (from spec §B):
      1. Population: majority-YES (≥4/6 YES) where YES annotators assigned
         ≥2 different (non-UNKNOWN) intent labels.
      2. Identify the two most common intent labels among those annotators.
      3. Among annotators who chose one of those two intents, compute the
         point-biserial correlation between gender (F=0, M=1) and intent
         (label A=0, label B=1).
      4. |r| > alignment_threshold → gender-aligned.
    """
    qualifying = 0
    aligned = 0
    r_values: list[float] = []
    for _, row in df.iterrows():
        if int(row.get("yes_count", 0)) < 4:
            continue
        records = _yes_intent_records(row, use_actual_gender=use_actual_gender)
        intents = [r[1] for r in records]
        if len(set(intents)) < 2:
            continue
        # Two most common labels (deterministic tie-break by alphabetical order)
        counts = Counter(intents)
        top_two = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:2]
        label_a, label_b = top_two[0][0], top_two[1][0]

        # Filter to annotators who chose one of these two labels
        sub = [(g, i) for (g, i) in records if i in (label_a, label_b)]
        if len(sub) < 2:
            continue
        gx = np.array([0 if g == "F" else 1 for g, _ in sub], dtype=float)
        ix = np.array([0 if i == label_a else 1 for _, i in sub], dtype=float)
        # Constant vector → correlation undefined; treat as not aligned
        if gx.std() == 0 or ix.std() == 0:
            qualifying += 1
            r_values.append(0.0)
            continue
        r = float(np.corrcoef(gx, ix)[0, 1])
        qualifying += 1
        r_values.append(r)
        if abs(r) > alignment_threshold:
            aligned += 1

    return {
        "n_qualifying": qualifying,
        "n_gender_aligned": aligned,
        "pct_gender_aligned": (aligned / qualifying * 100) if qualifying else 0.0,
        "alignment_threshold": alignment_threshold,
        "mean_abs_r": float(np.mean(np.abs(r_values))) if r_values else float("nan"),
    }


# ── Test C: Intent Entropy by Gender (exploratory) ────────────────────────────


def _entropy_of_labels(labels: list[str]) -> float:
    if not labels:
        return np.nan
    counts = Counter(labels)
    n = len(labels)
    h = 0.0
    for c in counts.values():
        p = c / n
        if p > 0:
            h -= p * np.log2(p)
    return h


def intent_entropy_by_gender(
    df: pd.DataFrame,
    use_actual_gender: bool = False,
) -> dict:
    """Test C: paired Wilcoxon on F vs M intent entropy among majority-YES.

    For each instance with ≥4 YES annotators, compute the per-gender intent
    entropy over YES annotators (UNKNOWN excluded). Requires at least one
    qualifying intent in both gender subgroups for that instance.
    """
    f_entropies: list[float] = []
    m_entropies: list[float] = []
    for _, row in df.iterrows():
        if int(row.get("yes_count", 0)) < 4:
            continue
        records = _yes_intent_records(row, use_actual_gender=use_actual_gender)
        f_labels = [i for (g, i) in records if g == "F"]
        m_labels = [i for (g, i) in records if g == "M"]
        if not f_labels or not m_labels:
            continue
        f_entropies.append(_entropy_of_labels(f_labels))
        m_entropies.append(_entropy_of_labels(m_labels))

    if len(f_entropies) < 2:
        return {"n": len(f_entropies), "error": "insufficient paired observations"}

    f_arr = np.asarray(f_entropies)
    m_arr = np.asarray(m_entropies)
    diffs = f_arr - m_arr
    nonzero = diffs[diffs != 0]
    if len(nonzero) > 0:
        w_stat, p_value = stats.wilcoxon(nonzero)
        r_rb = rank_biserial_signed(diffs, w_stat)
    else:
        w_stat, p_value, r_rb = np.nan, np.nan, np.nan

    return {
        "n": len(f_entropies),
        "f_entropy_mean": float(f_arr.mean()),
        "m_entropy_mean": float(m_arr.mean()),
        "mean_diff": float(diffs.mean()),
        "wilcoxon_W": float(w_stat) if not np.isnan(w_stat) else np.nan,
        "p_value": float(p_value) if not np.isnan(p_value) else np.nan,
        "rank_biserial_r": float(r_rb) if not np.isnan(r_rb) else np.nan,
    }


# ── Runner ────────────────────────────────────────────────────────────────────


def _print_per_label_table(rates: pd.DataFrame, modality: str) -> None:
    print(f"\n--- Test A: Per-Label Gender Comparison ({modality}, Holm-corrected) ---")
    print(rates.to_string(index=False))


def _print_unknown(unknown: dict, modality: str) -> None:
    print(f"\n--- UNKNOWN intent by gender ({modality}) ---")
    print(f"  F: {unknown['f_unknown']}/{unknown['f_yes_total']} "
          f"({unknown['f_unknown_rate'] * 100:.2f}%)")
    print(f"  M: {unknown['m_unknown']}/{unknown['m_yes_total']} "
          f"({unknown['m_unknown_rate'] * 100:.2f}%)")


def _print_alignment(alignment: dict, modality: str) -> None:
    print(f"\n--- Test B: Gender-Alignment of Intent Splits ({modality}) ---")
    print(f"  Qualifying majority-YES instances: {alignment['n_qualifying']}")
    print(f"  Gender-aligned (|r| > {alignment['alignment_threshold']:.2f}): "
          f"{alignment['n_gender_aligned']} "
          f"({alignment['pct_gender_aligned']:.1f}%)")
    print(f"  Mean |r| across qualifying instances: {alignment['mean_abs_r']:.4f}")


def _print_entropy(entropy: dict, modality: str) -> None:
    print(f"\n--- Test C: Intent Entropy by Gender (exploratory, {modality}) ---")
    if "error" in entropy:
        print(f"  Skipped: {entropy['error']} (n={entropy.get('n', 0)})")
        return
    print(f"  n paired instances:    {entropy['n']}")
    print(f"  F intent entropy mean: {entropy['f_entropy_mean']:.4f}")
    print(f"  M intent entropy mean: {entropy['m_entropy_mean']:.4f}")
    print(f"  Mean diff (F - M):     {entropy['mean_diff']:.4f}")
    print(f"  Wilcoxon W = {entropy['wilcoxon_W']}, "
          f"p = {entropy['p_value']:.6f}, r = {entropy['rank_biserial_r']:.4f}")


def run_tweets(df: pd.DataFrame) -> dict:
    """Run Tests A, B, C for the tweets dataset."""
    print("\n" + "=" * 60)
    print("SECTION 3 (EXTENSION): GENDER × INTENT ATTRIBUTION (TWEETS)")
    print("=" * 60)

    rates = per_label_gender_comparison(df, TWEET_INTENT_LABELS)
    unknown = unknown_by_gender(df)
    alignment = gender_alignment_of_intent_splits(df)
    entropy = intent_entropy_by_gender(df)

    _print_per_label_table(rates, "Tweets — Task 1.2")
    _print_unknown(unknown, "Tweets")
    _print_alignment(alignment, "Tweets")
    _print_entropy(entropy, "Tweets")

    return {
        "per_label_rates": rates,
        "unknown_by_gender": unknown,
        "split_alignment": alignment,
        "entropy": entropy,
    }


def run_memes(df: pd.DataFrame) -> dict:
    """Run Tests A, B, C for the memes dataset."""
    print("\n" + "=" * 60)
    print("SECTION 3 (EXTENSION): GENDER × INTENT ATTRIBUTION (MEMES)")
    print("=" * 60)

    rates = per_label_gender_comparison(df, MEME_INTENT_LABELS)
    unknown = unknown_by_gender(df)
    alignment = gender_alignment_of_intent_splits(df)
    entropy = intent_entropy_by_gender(df)

    _print_per_label_table(rates, "Memes — Task 2.2")
    _print_unknown(unknown, "Memes")
    _print_alignment(alignment, "Memes")
    _print_entropy(entropy, "Memes")

    return {
        "per_label_rates": rates,
        "unknown_by_gender": unknown,
        "split_alignment": alignment,
        "entropy": entropy,
    }
