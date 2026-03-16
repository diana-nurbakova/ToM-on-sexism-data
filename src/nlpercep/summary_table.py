"""Generate the paper's summary Table 2: key quantitative results."""

from __future__ import annotations

import pandas as pd


def generate(results: dict) -> pd.DataFrame:
    """Build Table 2 from computed analysis results.

    Expected keys in results: s0, s1, s2, s3 (from the section runners).
    """
    rows = []

    # From Section 1
    if "s1" in results:
        s1 = results["s1"]
        rows.append({
            "Measure": "Instances with disagreement",
            "Value": f"{s1['agreement_proportions']['any_disagreement_pct']:.1f}%",
            "Interpretation": "Social perception disagreement is the norm",
        })
        rows.append({
            "Measure": "Detection entropy (Task 1.1)",
            "Value": f"mean {s1['entropy_stats']['mean']:.3f}",
            "Interpretation": "Moderate disagreement at detection level",
        })
        rows.append({
            "Measure": "Intent entropy among YES (Task 1.2)",
            "Value": f"mean {s1['conditional_disagreement']['entropy_1_2_mean']:.3f}",
            "Interpretation": "Near-maximum disagreement at interpretation level",
        })
        rows.append({
            "Measure": "Categorization Jaccard among YES (Task 1.3)",
            "Value": f"mean {s1['conditional_disagreement']['jaccard_1_3_mean']:.3f}",
            "Interpretation": "Low overlap in category assignment",
        })

    # From Section 3
    if "s3" in results:
        s3 = results["s3"]
        rows.append({
            "Measure": "Gender effect on detection",
            "Value": f"p = {s3['overall']['p_value']:.3f}",
            "Interpretation": "No gender difference at detection level",
        })

        cat_rates = s3["category_rates"]
        for _, cat_row in cat_rates.iterrows():
            p_adj = cat_row.get("p_adjusted", cat_row["p_value"])
            if p_adj < 0.06:  # include borderline (Holm-corrected)
                rows.append({
                    "Measure": f"Gender effect on {cat_row['category']}",
                    "Value": f"p_adj = {p_adj:.3f}, OR = {cat_row['odds_ratio']:.2f}",
                    "Interpretation": f"Female annotators perceive more {cat_row['category'].lower().replace('-', ' ')}",
                })

        rows.append({
            "Measure": "Gender-aligned 3-3 splits",
            "Value": f"{s3['split_analysis']['pct_gender_aligned']:.1f}%",
            "Interpretation": "Gender accounts for little of the detection disagreement",
        })

    table = pd.DataFrame(rows)

    print("\n" + "=" * 60)
    print("TABLE 2: KEY QUANTITATIVE RESULTS")
    print("=" * 60)
    for _, row in table.iterrows():
        print(f"  {row['Measure']:45s} | {row['Value']:25s} | {row['Interpretation']}")

    return table
