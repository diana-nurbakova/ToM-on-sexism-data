"""Cross-modal annotator pool analysis.

Compares annotator pools across modalities (tweets, memes, videos) to assess
whether cross-modal comparisons are confounded by different annotator populations.
"""

from __future__ import annotations

from collections import Counter

import pandas as pd
import numpy as np
from scipy import stats


DEMO_ATTRS = ["gender", "age", "ethnicity", "education", "country"]


def _extract_annotator_set(df: pd.DataFrame, n_ann: int = 6) -> set[str]:
    """Return the set of unique annotator IDs from a dataset."""
    ids = set()
    for i in range(n_ann):
        col = f"ann_{i}_id"
        if col in df.columns:
            ids.update(df[col].dropna().unique())
    return ids


def _extract_annotator_demographics(
    df: pd.DataFrame, n_ann: int = 6,
) -> pd.DataFrame:
    """Build a per-annotator demographics table (one row per annotator).

    Takes the most frequent demographic value per annotator across instances
    (demographics should be constant, but mode handles any noise).
    """
    records: dict[str, dict] = {}
    for _, row in df.iterrows():
        for i in range(n_ann):
            col_id = f"ann_{i}_id"
            if col_id not in row or pd.isna(row[col_id]):
                continue
            ann_id = row[col_id]
            if ann_id not in records:
                records[ann_id] = {attr: [] for attr in DEMO_ATTRS}
            for attr in DEMO_ATTRS:
                col = f"ann_{i}_{attr}"
                if col in row and pd.notna(row[col]):
                    records[ann_id][attr].append(row[col])

    rows = []
    for ann_id, attr_lists in records.items():
        r = {"annotator_id": ann_id}
        for attr in DEMO_ATTRS:
            vals = attr_lists[attr]
            if vals:
                r[attr] = Counter(vals).most_common(1)[0][0]
            else:
                r[attr] = np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def _compare_distributions(
    demo_a: pd.DataFrame, demo_b: pd.DataFrame, label_a: str, label_b: str,
) -> dict:
    """Compare demographic distributions between two annotator groups.

    Uses chi-squared tests for categorical variables.
    """
    results = {}
    for attr in DEMO_ATTRS:
        vals_a = demo_a[attr].dropna()
        vals_b = demo_b[attr].dropna()
        if len(vals_a) == 0 or len(vals_b) == 0:
            results[attr] = {"note": "insufficient data"}
            continue

        # Build contingency table
        all_categories = sorted(set(vals_a) | set(vals_b))
        counts_a = Counter(vals_a)
        counts_b = Counter(vals_b)
        table = np.array([
            [counts_a.get(c, 0) for c in all_categories],
            [counts_b.get(c, 0) for c in all_categories],
        ])

        # Drop columns with zero total (can't test)
        col_totals = table.sum(axis=0)
        table = table[:, col_totals > 0]
        cats_used = [c for c, t in zip(all_categories, col_totals) if t > 0]

        if table.shape[1] < 2:
            results[attr] = {"note": "fewer than 2 categories with data"}
            continue

        chi2, p, dof, _ = stats.chi2_contingency(table)
        # Cramér's V
        n = table.sum()
        k = min(table.shape) - 1
        v = np.sqrt(chi2 / (n * k)) if (n * k) > 0 else 0.0

        dist_a = {c: int(counts_a.get(c, 0)) for c in cats_used}
        dist_b = {c: int(counts_b.get(c, 0)) for c in cats_used}

        results[attr] = {
            f"{label_a}_dist": dist_a,
            f"{label_b}_dist": dist_b,
            "chi2": float(chi2),
            "p_value": float(p),
            "dof": int(dof),
            "cramers_v": float(v),
        }
    return results


def run(
    tweets_df: pd.DataFrame,
    memes_df: pd.DataFrame,
    videos_df: pd.DataFrame | None = None,
) -> dict:
    """Run cross-modal annotator pool analysis."""
    print("\n" + "=" * 60)
    print("CROSS-MODAL ANNOTATOR POOL ANALYSIS")
    print("=" * 60)

    # ── 1. Pool sizes and overlap ────────────────────────────────────
    tweets_ann = _extract_annotator_set(tweets_df, n_ann=6)
    memes_ann = _extract_annotator_set(memes_df, n_ann=6)

    overlap_tm = tweets_ann & memes_ann
    only_tweets = tweets_ann - memes_ann
    only_memes = memes_ann - tweets_ann
    union_tm = tweets_ann | memes_ann

    print(f"\n--- Pool Sizes ---")
    print(f"  Tweets annotators:  {len(tweets_ann)}")
    print(f"  Memes annotators:   {len(memes_ann)}")
    print(f"  Union (T ∪ M):      {len(union_tm)}")

    print(f"\n--- Tweets ↔ Memes Overlap ---")
    print(f"  Shared annotators:  {len(overlap_tm)}")
    print(f"  Tweets-only:        {len(only_tweets)}")
    print(f"  Memes-only:         {len(only_memes)}")
    jaccard_tm = len(overlap_tm) / len(union_tm) if union_tm else 0
    print(f"  Jaccard similarity: {jaccard_tm:.4f}")
    pct_tweets_shared = len(overlap_tm) / len(tweets_ann) * 100 if tweets_ann else 0
    pct_memes_shared = len(overlap_tm) / len(memes_ann) * 100 if memes_ann else 0
    print(f"  % of tweets pool shared: {pct_tweets_shared:.1f}%")
    print(f"  % of memes pool shared:  {pct_memes_shared:.1f}%")

    result = {
        "tweets_n": len(tweets_ann),
        "memes_n": len(memes_ann),
        "overlap_tweets_memes": len(overlap_tm),
        "only_tweets": len(only_tweets),
        "only_memes": len(only_memes),
        "jaccard_tweets_memes": jaccard_tm,
        "pct_tweets_shared": pct_tweets_shared,
        "pct_memes_shared": pct_memes_shared,
    }

    # ── Videos overlap (if available) ────────────────────────────────
    if videos_df is not None:
        # Videos may have 2-3 annotators
        max_vid_ann = int(videos_df["n_annotators"].max()) if "n_annotators" in videos_df.columns else 3
        videos_ann = _extract_annotator_set(videos_df, n_ann=max_vid_ann)
        overlap_tv = tweets_ann & videos_ann
        overlap_mv = memes_ann & videos_ann
        overlap_all = tweets_ann & memes_ann & videos_ann

        print(f"\n  Videos annotators:  {len(videos_ann)}")
        print(f"  Tweets ∩ Videos:    {len(overlap_tv)}")
        print(f"  Memes ∩ Videos:     {len(overlap_mv)}")
        print(f"  All three:          {len(overlap_all)}")

        result.update({
            "videos_n": len(videos_ann),
            "overlap_tweets_videos": len(overlap_tv),
            "overlap_memes_videos": len(overlap_mv),
            "overlap_all_three": len(overlap_all),
        })

    # ── 2. Demographic comparison: shared vs modality-exclusive ──────
    # Build annotator-level demographics from tweets+memes (which have demographics)
    print(f"\n--- Demographic Comparison ---")

    tweets_demo = _extract_annotator_demographics(tweets_df, n_ann=6)
    memes_demo = _extract_annotator_demographics(memes_df, n_ann=6)

    # Shared annotators demographics (from tweets data — same person)
    shared_demo = tweets_demo[tweets_demo["annotator_id"].isin(overlap_tm)]
    tweets_only_demo = tweets_demo[tweets_demo["annotator_id"].isin(only_tweets)]
    memes_only_demo = memes_demo[memes_demo["annotator_id"].isin(only_memes)]

    print(f"\n  Shared annotators with demographics: {len(shared_demo)}")
    print(f"  Tweets-only with demographics:       {len(tweets_only_demo)}")
    print(f"  Memes-only with demographics:        {len(memes_only_demo)}")

    # Compare tweets-only vs memes-only (the populations that differ)
    demo_comparison = {}
    if len(tweets_only_demo) > 0 and len(memes_only_demo) > 0:
        print(f"\n  Comparing tweets-only vs memes-only annotator demographics:")
        demo_comparison = _compare_distributions(
            tweets_only_demo, memes_only_demo, "tweets_only", "memes_only",
        )
        for attr, res in demo_comparison.items():
            if "note" in res:
                print(f"    {attr:12s}: {res['note']}")
            else:
                sig = "***" if res["p_value"] < 0.001 else "**" if res["p_value"] < 0.01 else "*" if res["p_value"] < 0.05 else "n.s."
                print(f"    {attr:12s}: χ²={res['chi2']:.2f}, p={res['p_value']:.4f} {sig}, Cramér's V={res['cramers_v']:.3f}")

    result["demographic_comparison_exclusive"] = demo_comparison

    # Compare full tweet pool vs full meme pool
    print(f"\n  Comparing full tweets pool vs full memes pool demographics:")
    full_pool_comparison = _compare_distributions(
        tweets_demo, memes_demo, "tweets", "memes",
    )
    for attr, res in full_pool_comparison.items():
        if "note" in res:
            print(f"    {attr:12s}: {res['note']}")
        else:
            sig = "***" if res["p_value"] < 0.001 else "**" if res["p_value"] < 0.01 else "*" if res["p_value"] < 0.05 else "n.s."
            print(f"    {attr:12s}: χ²={res['chi2']:.2f}, p={res['p_value']:.4f} {sig}, Cramér's V={res['cramers_v']:.3f}")

    result["demographic_comparison_full_pools"] = full_pool_comparison

    # ── 3. Annotation volume per annotator per modality ──────────────
    print(f"\n--- Annotation Volume (instances per annotator) ---")

    def _instance_counts_per_annotator(df, n_ann=6):
        counts = Counter()
        for _, row in df.iterrows():
            for i in range(n_ann):
                col = f"ann_{i}_id"
                if col in row and pd.notna(row[col]):
                    counts[row[col]] += 1
        return counts

    tweets_counts = _instance_counts_per_annotator(tweets_df)
    memes_counts = _instance_counts_per_annotator(memes_df)

    tweets_volumes = np.array(list(tweets_counts.values()))
    memes_volumes = np.array(list(memes_counts.values()))

    print(f"  Tweets — mean: {tweets_volumes.mean():.1f}, median: {np.median(tweets_volumes):.0f}, "
          f"min: {tweets_volumes.min()}, max: {tweets_volumes.max()}")
    print(f"  Memes  — mean: {memes_volumes.mean():.1f}, median: {np.median(memes_volumes):.0f}, "
          f"min: {memes_volumes.min()}, max: {memes_volumes.max()}")

    # For shared annotators: their volume in each modality
    if overlap_tm:
        shared_tweets_vol = np.array([tweets_counts[a] for a in overlap_tm])
        shared_memes_vol = np.array([memes_counts[a] for a in overlap_tm])
        print(f"\n  Shared annotators — tweets volume:  mean={shared_tweets_vol.mean():.1f}, "
              f"median={np.median(shared_tweets_vol):.0f}")
        print(f"  Shared annotators — memes volume:   mean={shared_memes_vol.mean():.1f}, "
              f"median={np.median(shared_memes_vol):.0f}")
        result["shared_annotator_volume"] = {
            "tweets_mean": float(shared_tweets_vol.mean()),
            "tweets_median": float(np.median(shared_tweets_vol)),
            "memes_mean": float(shared_memes_vol.mean()),
            "memes_median": float(np.median(shared_memes_vol)),
        }

    result["volume"] = {
        "tweets_mean": float(tweets_volumes.mean()),
        "tweets_median": float(np.median(tweets_volumes)),
        "memes_mean": float(memes_volumes.mean()),
        "memes_median": float(np.median(memes_volumes)),
    }

    return result
