"""Main analysis runner for the NLPercep paper.

Usage:
    uv run python -m nlpercep.run_analysis [--sections 0 1 2 3]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from io import StringIO
from pathlib import Path

import pandas as pd

from nlpercep.data import load_dataset, load_memes_dataset, load_videos_dataset
from nlpercep import (
    section0_descriptive,
    section1_disagreement,
    section2_intent_ambiguity,
    section3_gender,
    section3_gender_intent,
    figures,
    qualitative,
    summary_table,
    memes_analysis,
    videos_analysis,
    cross_modal_annotators,
)

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs"


class Tee:
    """Write to both stdout and a string buffer."""

    def __init__(self, original):
        self.original = original
        self.buffer = StringIO()

    def write(self, text):
        try:
            self.original.write(text)
        except UnicodeEncodeError:
            encoding = getattr(self.original, "encoding", "ascii") or "ascii"
            self.original.write(text.encode(encoding, errors="replace").decode(encoding))
        self.buffer.write(text)

    def flush(self):
        self.original.flush()

    def getvalue(self):
        return self.buffer.getvalue()


def save_results(results: dict, log_text: str) -> Path:
    """Save analysis results to outputs/tables/ and the full log."""
    tables_dir = OUTPUT_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = tables_dir / f"run_{timestamp}"
    run_dir.mkdir(exist_ok=True)

    # Save full console log
    (run_dir / "full_log.txt").write_text(log_text, encoding="utf-8")

    # Section 0
    if "s0" in results:
        s0 = results["s0"]
        s0["instance_counts"].to_csv(run_dir / "s0_instance_counts.csv", index=False)
        for attr, dist in s0["demographics"].items():
            dist.to_csv(run_dir / f"s0_demo_{attr}.csv", header=["count"])
        for task, dist in s0["label_distributions"].items():
            dist.to_csv(run_dir / f"s0_labels_{task}.csv", header=["count"])

    # Section 1
    if "s1" in results:
        s1 = results["s1"]
        s1["agreement_distribution"].to_csv(run_dir / "s1_agreement_distribution.csv", header=["count"])
        with open(run_dir / "s1_stats.json", "w") as f:
            json.dump({
                "entropy_stats": s1["entropy_stats"],
                "agreement_proportions": s1["agreement_proportions"],
                "conditional_disagreement": s1["conditional_disagreement"],
            }, f, indent=2, default=_json_default)

    # Section 2
    if "s2" in results:
        s2 = results["s2"]
        with open(run_dir / "s2_analysis_a.json", "w") as f:
            json.dump({
                "ambiguity_groups": s2["ambiguity_groups"],
                "overall": s2["overall"],
                "by_language": s2["by_language"],
            }, f, indent=2, default=_json_default)

    # Section 3
    if "s3" in results:
        s3 = results["s3"]
        with open(run_dir / "s3_overall_gender.json", "w") as f:
            json.dump(s3["overall"], f, indent=2, default=_json_default)
        with open(run_dir / "s3_split_analysis.json", "w") as f:
            json.dump(s3["split_analysis"], f, indent=2, default=_json_default)
        s3["by_agreement_level"].to_csv(run_dir / "s3_gender_by_agreement.csv", index=False)
        with open(run_dir / "s3_interaction.json", "w") as f:
            json.dump(s3["interaction"], f, indent=2, default=_json_default)
        s3["category_rates"].to_csv(run_dir / "s3_category_rates.csv", index=False)

    # Section 3 (extension): Gender × Intent
    if "s3_intent" in results:
        si = results["s3_intent"]
        si["per_label_rates"].to_csv(run_dir / "s3_intent_per_label.csv", index=False)
        with open(run_dir / "s3_intent_summary.json", "w") as f:
            json.dump({
                "unknown_by_gender": si["unknown_by_gender"],
                "split_alignment": si["split_alignment"],
                "entropy": si["entropy"],
            }, f, indent=2, default=_json_default)

    # Summary Table 2
    if "table2" in results:
        results["table2"].to_csv(run_dir / "table2_summary.csv", index=False)

    # Qualitative examples
    if "qualitative" in results:
        q = results["qualitative"]
        for key, ex_df in q.items():
            if isinstance(ex_df, pd.DataFrame) and len(ex_df) > 0:
                ex_df.to_csv(run_dir / f"qualitative_{key}.csv", index=False)

    # Memes results
    if "memes" in results:
        memes_dir = run_dir / "memes"
        memes_dir.mkdir(exist_ok=True)
        m = results["memes"]
        if "s0" in m:
            m["s0"]["instance_counts"].to_csv(memes_dir / "s0_instance_counts.csv", index=False)
            for task, dist in m["s0"]["label_distributions"].items():
                dist.to_csv(memes_dir / f"s0_labels_{task}.csv", header=["count"])
        if "s1" in m:
            m["s1"]["agreement_distribution"].to_csv(memes_dir / "s1_agreement_distribution.csv", header=["count"])
            with open(memes_dir / "s1_stats.json", "w") as f:
                json.dump({
                    "entropy_stats": m["s1"]["entropy_stats"],
                    "agreement_proportions": m["s1"]["agreement_proportions"],
                    "conditional_disagreement": m["s1"]["conditional_disagreement"],
                }, f, indent=2, default=_json_default)
        if "s3a" in m:
            with open(memes_dir / "s3a_gender.json", "w") as f:
                json.dump(m["s3a"], f, indent=2, default=_json_default)
        if "s3d" in m:
            m["s3d"].to_csv(memes_dir / "s3d_category_rates.csv", index=False)
        if "s3_intent" in m:
            mi = m["s3_intent"]
            mi["per_label_rates"].to_csv(memes_dir / "s3_intent_per_label.csv", index=False)
            with open(memes_dir / "s3_intent_summary.json", "w") as f:
                json.dump({
                    "unknown_by_gender": mi["unknown_by_gender"],
                    "split_alignment": mi["split_alignment"],
                    "entropy": mi["entropy"],
                }, f, indent=2, default=_json_default)

    # Cross-modal annotator pool
    if "cross_modal" in results:
        with open(run_dir / "cross_modal_annotators.json", "w") as f:
            # Filter out non-serializable demo comparison dicts
            cm = {k: v for k, v in results["cross_modal"].items()}
            json.dump(cm, f, indent=2, default=_json_default)

    # Videos results
    if "videos" in results:
        videos_dir = run_dir / "videos"
        videos_dir.mkdir(exist_ok=True)
        v = results["videos"]
        if "s0" in v:
            v["s0"]["instance_counts"].to_csv(videos_dir / "s0_instance_counts.csv", index=False)
            for task, dist in v["s0"]["label_distributions"].items():
                dist.to_csv(videos_dir / f"s0_labels_{task}.csv", header=["count"])
        if "s1" in v:
            v["s1"]["agreement_distribution"].to_csv(videos_dir / "s1_agreement_distribution.csv", header=["count"])
            with open(videos_dir / "s1_stats.json", "w") as f:
                json.dump({
                    "entropy_stats": v["s1"]["entropy_stats"],
                    "agreement_proportions": v["s1"]["agreement_proportions"],
                    "conditional_disagreement": v["s1"]["conditional_disagreement"],
                }, f, indent=2, default=_json_default)
        if "s3a" in v:
            with open(videos_dir / "s3a_gender.json", "w") as f:
                json.dump(v["s3a"], f, indent=2, default=_json_default)
        if "s3d" in v:
            v["s3d"].to_csv(videos_dir / "s3d_category_rates.csv", index=False)

    # ── Combined all_results.json ─────────────────────────────────────────
    combined = _build_combined_results(results)
    with open(run_dir / "all_results.json", "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, default=_json_default)
    # Also stable-named copy at outputs/ root for quick consumption
    with open(OUTPUT_DIR / "all_results.json", "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, default=_json_default)

    return run_dir


def _df_to_records(obj):
    """Coerce a DataFrame to a list of dicts; pass other JSON-friendly types through."""
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    return obj


def _build_combined_results(results: dict) -> dict:
    """Aggregate per-section result blocks into one JSON-serializable dict."""
    combined: dict = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }

    # Tweets ─────────────────────────────────────────────────────────────
    tweets: dict = {}
    if "s0" in results:
        s0 = results["s0"]
        tweets["s0_descriptive"] = {
            "instance_counts": _df_to_records(s0["instance_counts"]),
            "n_unique_annotators": s0.get("n_unique_annotators"),
            "label_distributions": {
                task: dist.to_dict() for task, dist in s0["label_distributions"].items()
            },
            "demographics": {
                attr: dist.to_dict() for attr, dist in s0["demographics"].items()
            },
        }
    if "s1" in results:
        s1 = results["s1"]
        tweets["s1_disagreement"] = {
            "agreement_distribution": s1["agreement_distribution"].to_dict(),
            "entropy_stats": s1["entropy_stats"],
            "agreement_proportions": s1["agreement_proportions"],
            "conditional_disagreement": s1["conditional_disagreement"],
        }
    if "s2" in results:
        s2 = results["s2"]
        tweets["s2_intent_ambiguity"] = {
            "ambiguity_groups": s2["ambiguity_groups"],
            "overall": s2["overall"],
            "by_language": s2["by_language"],
        }
    if "s3" in results:
        s3 = results["s3"]
        tweets["s3_gender"] = {
            "overall": s3["overall"],
            "split_analysis": s3["split_analysis"],
            "by_agreement_level": _df_to_records(s3["by_agreement_level"]),
            "interaction": s3["interaction"],
            "category_rates": _df_to_records(s3["category_rates"]),
        }
    if "s3_intent" in results:
        si = results["s3_intent"]
        tweets["s3_gender_intent"] = {
            "per_label_rates": _df_to_records(si["per_label_rates"]),
            "unknown_by_gender": si["unknown_by_gender"],
            "split_alignment": si["split_alignment"],
            "entropy": si["entropy"],
        }
    if tweets:
        combined["tweets"] = tweets

    # Memes ──────────────────────────────────────────────────────────────
    if "memes" in results:
        m = results["memes"]
        memes_block: dict = {}
        if "s0" in m:
            memes_block["s0_descriptive"] = {
                "instance_counts": _df_to_records(m["s0"]["instance_counts"]),
                "n_unique_annotators": m["s0"].get("n_unique_annotators"),
                "label_distributions": {
                    task: dist.to_dict() for task, dist in m["s0"]["label_distributions"].items()
                },
                "demographics": {
                    attr: dist.to_dict() for attr, dist in m["s0"]["demographics"].items()
                },
            }
        if "s1" in m:
            memes_block["s1_disagreement"] = {
                "agreement_distribution": m["s1"]["agreement_distribution"].to_dict(),
                "entropy_stats": m["s1"]["entropy_stats"],
                "agreement_proportions": m["s1"]["agreement_proportions"],
                "conditional_disagreement": m["s1"]["conditional_disagreement"],
                "by_language": m["s1"].get("by_language"),
            }
        if "s3a" in m:
            memes_block["s3a_gender_overall"] = m["s3a"]
        if "s3d" in m:
            memes_block["s3d_category_rates"] = _df_to_records(m["s3d"])
        if "s3_intent" in m:
            mi = m["s3_intent"]
            memes_block["s3_gender_intent"] = {
                "per_label_rates": _df_to_records(mi["per_label_rates"]),
                "unknown_by_gender": mi["unknown_by_gender"],
                "split_alignment": mi["split_alignment"],
                "entropy": mi["entropy"],
            }
        combined["memes"] = memes_block

    # Videos ─────────────────────────────────────────────────────────────
    if "videos" in results:
        v = results["videos"]
        videos_block: dict = {}
        if "s0" in v:
            videos_block["s0_descriptive"] = {
                "instance_counts": _df_to_records(v["s0"]["instance_counts"]),
                "n_unique_annotators": v["s0"].get("n_unique_annotators"),
                "label_distributions": {
                    task: dist.to_dict() for task, dist in v["s0"]["label_distributions"].items()
                },
            }
        if "s1" in v:
            videos_block["s1_disagreement"] = {
                "agreement_distribution": v["s1"]["agreement_distribution"].to_dict(),
                "entropy_stats": v["s1"]["entropy_stats"],
                "agreement_proportions": v["s1"]["agreement_proportions"],
                "conditional_disagreement": v["s1"]["conditional_disagreement"],
            }
        if "s3a" in v:
            videos_block["s3a_gender_overall"] = v["s3a"]
        if "s3d" in v:
            videos_block["s3d_category_rates"] = _df_to_records(v["s3d"])
        combined["videos"] = videos_block

    # Cross-modal + summary table ────────────────────────────────────────
    if "cross_modal" in results:
        combined["cross_modal"] = results["cross_modal"]
    if "table2" in results:
        combined["table2_summary"] = _df_to_records(results["table2"])

    return combined


def _json_default(obj):
    """Handle numpy/pandas types in JSON serialization."""
    import numpy as np
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def main():
    parser = argparse.ArgumentParser(description="NLPercep analysis pipeline")
    parser.add_argument(
        "--sections", nargs="*", type=int, default=[0, 1, 2, 3],
        help="Which sections to run (default: all)",
    )
    parser.add_argument(
        "--no-memes", action="store_true",
        help="Skip memes dataset analysis",
    )
    parser.add_argument(
        "--no-videos", action="store_true",
        help="Skip TikTok videos dataset analysis",
    )
    args = parser.parse_args()

    # Tee stdout to capture full log
    tee = Tee(sys.stdout)
    sys.stdout = tee

    print("Loading EXIST 2025 tweets dataset...")
    df = load_dataset()
    print(f"Loaded {len(df)} instances.\n")

    results = {}

    if 0 in args.sections:
        results["s0"] = section0_descriptive.run(df)

    if 1 in args.sections:
        results["s1"] = section1_disagreement.run(df)

    if 2 in args.sections:
        results["s2"] = section2_intent_ambiguity.run(df)
        # Pass ambiguity column to df for section 3
        if "df_with_ambiguity" in results["s2"]:
            df = results["s2"]["df_with_ambiguity"]

    if 3 in args.sections:
        results["s3"] = section3_gender.run(df)
        results["s3_intent"] = section3_gender_intent.run_tweets(df)

    # Qualitative examples (needs full df with ambiguity)
    results["qualitative"] = qualitative.run(df)

    # Summary Table 2 (needs results from s1, s3)
    results["table2"] = summary_table.generate(results)

    # ── Memes analysis ──────────────────────────────────────────────────
    if not args.no_memes:
        print("\n\n" + "#" * 60)
        print("# MEMES DATASET ANALYSIS")
        print("#" * 60)
        print("\nLoading EXIST 2025 memes dataset...")
        memes_df = load_memes_dataset()
        print(f"Loaded {len(memes_df)} meme instances.\n")
        results["memes"] = memes_analysis.run(memes_df)

    # ── Videos (TikTok) analysis ───────────────────────────────────────
    if not args.no_videos:
        print("\n\n" + "#" * 60)
        print("# TIKTOK VIDEOS DATASET ANALYSIS")
        print("#" * 60)
        print("\nLoading EXIST 2025 videos dataset...")
        videos_df = load_videos_dataset()
        print(f"Loaded {len(videos_df)} video instances.\n")
        results["videos"] = videos_analysis.run(videos_df)

    # ── Cross-modal annotator pool analysis ──────────────────────────
    if not args.no_memes:
        print("\n\n" + "#" * 60)
        print("# CROSS-MODAL ANNOTATOR POOL ANALYSIS")
        print("#" * 60)
        videos_for_cross = videos_df if (not args.no_videos and "videos" in results) else None
        results["cross_modal"] = cross_modal_annotators.run(df, memes_df, videos_for_cross)

    # Generate figures
    fig_dir = OUTPUT_DIR / "figures"
    figures.generate_all(df, fig_dir, label="Tweets")
    if not args.no_memes and "memes" in results:
        figures.generate_all(memes_df, fig_dir, label="Memes")
        p = figures.fig_gender_categorization_side_by_side(df, memes_df, fig_dir)
        print(f"  - Gender categorization (tweets + memes): {p.name}")
        p = figures.fig_gender_categorization_side_by_side_tom(df, memes_df, fig_dir)
        print(f"  - Gender categorization (tweets + memes, ToM-grouped): {p.name}")
        p = figures.fig_gender_intent_side_by_side(df, memes_df, fig_dir)
        print(f"  - Gender intent (tweets + memes): {p.name}")
    if not args.no_videos and "videos" in results:
        figures.generate_all(videos_df, fig_dir, label="TikToks")

    # Save results
    run_dir = save_results(results, tee.getvalue())

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print(f"Results saved to: {run_dir}")
    print("=" * 60)

    # Restore stdout
    sys.stdout = tee.original

    return results


if __name__ == "__main__":
    main()
