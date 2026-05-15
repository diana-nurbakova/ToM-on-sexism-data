"""Descriptive statistics about the annotator pool across datasets and languages.

For each annotator (deduplicated by ID), build a single demographic profile
using the most frequent value across their appearances. Then summarise the
pool by dataset (tweets train+dev, memes train) and by language (en, es),
and report cross-dataset / cross-language overlap.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_dataset, load_memes_dataset

DEMO_ATTRS = ["gender", "age", "ethnicity", "education", "country"]


def _per_annotator_profiles(df: pd.DataFrame, n_ann: int = 6) -> pd.DataFrame:
    """Return one row per unique annotator with their dominant demographic profile
    and counts of annotated instances (total and per language).
    """
    records: dict[str, dict] = {}
    for _, row in df.iterrows():
        lang = row["lang"]
        for i in range(n_ann):
            ann_id = row.get(f"ann_{i}_id")
            if ann_id is None or (isinstance(ann_id, float) and np.isnan(ann_id)):
                continue
            if ann_id not in records:
                records[ann_id] = {
                    "annotator_id": ann_id,
                    "n_total": 0,
                    "n_en": 0,
                    "n_es": 0,
                    **{a: [] for a in DEMO_ATTRS},
                }
            rec = records[ann_id]
            rec["n_total"] += 1
            rec[f"n_{lang}"] = rec.get(f"n_{lang}", 0) + 1
            for a in DEMO_ATTRS:
                v = row.get(f"ann_{i}_{a}")
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    rec[a].append(v)
    rows = []
    for ann_id, rec in records.items():
        r = {
            "annotator_id": ann_id,
            "n_total": rec["n_total"],
            "n_en": rec.get("n_en", 0),
            "n_es": rec.get("n_es", 0),
        }
        for a in DEMO_ATTRS:
            vals = rec[a]
            r[a] = Counter(vals).most_common(1)[0][0] if vals else None
        rows.append(r)
    return pd.DataFrame(rows)


def _language_pool(profiles: pd.DataFrame, lang: str) -> pd.DataFrame:
    """Annotators who annotated at least one instance in the given language."""
    col = f"n_{lang}"
    return profiles[profiles[col] > 0].copy()


def _distribution(profiles: pd.DataFrame, attr: str) -> pd.Series:
    return profiles[attr].value_counts(dropna=False)


def _pct(counts: pd.Series) -> pd.Series:
    total = counts.sum()
    return (counts / total * 100).round(2)


def _normalize_education(s: str | None) -> str | None:
    """Bachelor's degree appears with curly apostrophe; collapse for stable counts."""
    if s is None:
        return s
    return (
        s.replace("’", "'")
         .replace("‘", "'")
         .replace("�", "'")
    )


def _normalize_profiles(p: pd.DataFrame) -> pd.DataFrame:
    p = p.copy()
    p["education"] = p["education"].map(_normalize_education)
    return p


def run() -> dict:
    print("\n" + "=" * 70)
    print("ANNOTATOR POOL DESCRIPTIVE STATISTICS (Tweets train+dev, Memes train)")
    print("=" * 70)

    tweets_df = load_dataset()
    memes_df = load_memes_dataset()

    tweets_profiles = _normalize_profiles(_per_annotator_profiles(tweets_df))
    memes_profiles = _normalize_profiles(_per_annotator_profiles(memes_df))

    # Union pool: prefer tweets demographics for shared annotators, fall back to memes
    union_ids = sorted(set(tweets_profiles["annotator_id"]) | set(memes_profiles["annotator_id"]))
    tweets_map = tweets_profiles.set_index("annotator_id").to_dict(orient="index")
    memes_map = memes_profiles.set_index("annotator_id").to_dict(orient="index")
    union_rows = []
    for aid in union_ids:
        src = tweets_map.get(aid) or memes_map.get(aid)
        n_tweets = tweets_map.get(aid, {}).get("n_total", 0)
        n_memes = memes_map.get(aid, {}).get("n_total", 0)
        n_en = (tweets_map.get(aid, {}).get("n_en", 0)
                + memes_map.get(aid, {}).get("n_en", 0))
        n_es = (tweets_map.get(aid, {}).get("n_es", 0)
                + memes_map.get(aid, {}).get("n_es", 0))
        union_rows.append({
            "annotator_id": aid,
            "n_total": n_tweets + n_memes,
            "n_tweets": n_tweets,
            "n_memes": n_memes,
            "n_en": n_en,
            "n_es": n_es,
            **{a: src.get(a) for a in DEMO_ATTRS},
        })
    union_profiles = pd.DataFrame(union_rows)

    # ── 1. Pool sizes ────────────────────────────────────────────────
    t_ids = set(tweets_profiles["annotator_id"])
    m_ids = set(memes_profiles["annotator_id"])
    t_en = set(_language_pool(tweets_profiles, "en")["annotator_id"])
    t_es = set(_language_pool(tweets_profiles, "es")["annotator_id"])
    m_en = set(_language_pool(memes_profiles, "en")["annotator_id"])
    m_es = set(_language_pool(memes_profiles, "es")["annotator_id"])

    print(f"\n--- Pool sizes ---")
    print(f"  Tweets total unique:  {len(t_ids)}")
    print(f"  Tweets EN only:       {len(t_en - t_es)}")
    print(f"  Tweets ES only:       {len(t_es - t_en)}")
    print(f"  Tweets EN+ES:         {len(t_en & t_es)}")
    print(f"  Memes total unique:   {len(m_ids)}")
    print(f"  Memes EN only:        {len(m_en - m_es)}")
    print(f"  Memes ES only:        {len(m_es - m_en)}")
    print(f"  Memes EN+ES:          {len(m_en & m_es)}")
    print(f"  Union (T U M):        {len(t_ids | m_ids)}")
    print(f"  Tweets and Memes:     {len(t_ids & m_ids)}")

    pool_sizes = {
        "tweets_total": len(t_ids),
        "tweets_en": len(t_en),
        "tweets_es": len(t_es),
        "tweets_both_langs": len(t_en & t_es),
        "memes_total": len(m_ids),
        "memes_en": len(m_en),
        "memes_es": len(m_es),
        "memes_both_langs": len(m_en & m_es),
        "union": len(t_ids | m_ids),
        "tweets_and_memes": len(t_ids & m_ids),
        "tweets_only": len(t_ids - m_ids),
        "memes_only": len(m_ids - t_ids),
    }

    # ── 2. Per-pool demographic distributions ────────────────────────
    pools = {
        "tweets_all": tweets_profiles,
        "tweets_en": _language_pool(tweets_profiles, "en"),
        "tweets_es": _language_pool(tweets_profiles, "es"),
        "memes_all": memes_profiles,
        "memes_en": _language_pool(memes_profiles, "en"),
        "memes_es": _language_pool(memes_profiles, "es"),
        "union": union_profiles,
    }

    distributions: dict[str, dict[str, dict]] = {}
    for pool_name, pool_df in pools.items():
        distributions[pool_name] = {"n": len(pool_df)}
        for attr in DEMO_ATTRS:
            counts = _distribution(pool_df, attr)
            distributions[pool_name][attr] = {
                "counts": counts.to_dict(),
                "pct": _pct(counts).to_dict(),
            }

    # ── 3. Annotation volume per annotator (already in profiles) ──────
    volume_stats = {}
    for pool_name, pool_df in pools.items():
        if pool_name == "union":
            cols = {"all": "n_total", "tweets": "n_tweets", "memes": "n_memes"}
        elif pool_name.startswith("tweets"):
            cols = {"all": "n_total"}
            if pool_name.endswith("_en"):
                cols = {"all": "n_en"}
            elif pool_name.endswith("_es"):
                cols = {"all": "n_es"}
        else:  # memes
            cols = {"all": "n_total"}
            if pool_name.endswith("_en"):
                cols = {"all": "n_en"}
            elif pool_name.endswith("_es"):
                cols = {"all": "n_es"}
        volume_stats[pool_name] = {}
        for label, col in cols.items():
            v = pool_df[col].to_numpy()
            v = v[v > 0]
            if len(v) == 0:
                continue
            volume_stats[pool_name][label] = {
                "mean": float(np.mean(v)),
                "median": float(np.median(v)),
                "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                "min": int(np.min(v)),
                "max": int(np.max(v)),
                "n_annotators": int(len(v)),
            }

    # ── 4. Cross-pool overlap matrix ─────────────────────────────────
    overlap_matrix = {
        "tweets_en|tweets_es": {
            "intersection": len(t_en & t_es),
            "jaccard": len(t_en & t_es) / len(t_en | t_es) if (t_en | t_es) else 0,
        },
        "memes_en|memes_es": {
            "intersection": len(m_en & m_es),
            "jaccard": len(m_en & m_es) / len(m_en | m_es) if (m_en | m_es) else 0,
        },
        "tweets|memes": {
            "intersection": len(t_ids & m_ids),
            "jaccard": len(t_ids & m_ids) / len(t_ids | m_ids) if (t_ids | m_ids) else 0,
        },
        "tweets_en|memes_en": {
            "intersection": len(t_en & m_en),
            "jaccard": len(t_en & m_en) / len(t_en | m_en) if (t_en | m_en) else 0,
        },
        "tweets_es|memes_es": {
            "intersection": len(t_es & m_es),
            "jaccard": len(t_es & m_es) / len(t_es | m_es) if (t_es | m_es) else 0,
        },
    }

    print(f"\n--- Cross-pool overlap (Jaccard) ---")
    for pair, vals in overlap_matrix.items():
        print(f"  {pair:30s}  n_int={vals['intersection']:4d}  J={vals['jaccard']:.3f}")

    # Print summary table for each pool
    for pool_name, pool_data in distributions.items():
        print(f"\n--- {pool_name} (n={pool_data['n']}) ---")
        for attr in DEMO_ATTRS:
            top = sorted(
                pool_data[attr]["pct"].items(),
                key=lambda kv: kv[1], reverse=True,
            )[:6]
            top_str = ", ".join(f"{k} {v:.1f}%" for k, v in top)
            print(f"  {attr:10s}: {top_str}")

    # Save full results JSON for downstream consumption
    out = {
        "pool_sizes": pool_sizes,
        "distributions": distributions,
        "volume_stats": volume_stats,
        "overlap": overlap_matrix,
    }
    return out


if __name__ == "__main__":
    result = run()
    out_path = Path(__file__).resolve().parents[2] / "outputs" / "annotator_pool_stats.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nSaved: {out_path}")
