"""Figure generation for the NLPercep paper.

Updated for v2 framing: detection-interpretation dissociation as core finding.
All figures accept a `label` parameter ("Tweets" / "Memes" / "TikToks") for clear titles.

Color palette: Wong (2011) colorblind-safe palette throughout.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats

from nlpercep.correction import holm_bonferroni, rank_biserial_signed, cohens_h

# ── Wong (2011) colorblind-safe palette ────────────────────────────────────────
# https://www.nature.com/articles/nmeth.1618
CB_ORANGE = "#E69F00"
CB_BLUE = "#0072B2"
CB_GREEN = "#009E73"
CB_VERMILLION = "#D55E00"
CB_SKYBLUE = "#56B4E9"
CB_YELLOW = "#F0E442"
CB_PURPLE = "#CC79A7"
CB_BLACK = "#000000"

# Gender: orange (F) vs blue (M) — maximally distinct under all CVD types
GENDER_F = CB_ORANGE
GENDER_M = CB_BLUE

# Agreement distribution: diverging from vermillion (YES) through purple (split) to sky-blue (NO)
AGREE_COLORS = [
    CB_VERMILLION,               # unanimous YES
    "#E8823A",                   # strong YES (blend)
    CB_YELLOW,                   # weak YES
    CB_PURPLE,                   # split
    CB_SKYBLUE,                  # weak NO
    CB_BLUE,                     # strong NO
    "#004A6F",                   # unanimous NO (darker blue)
]

# Detection vs interpretation: green (converge) vs vermillion (diverge)
DET_COLOR = CB_GREEN
INTERP_COLOR = CB_VERMILLION

# ToM grouping for categorization figures
# Base colours match the TikZ green!60!black / red!70!black
TOM_COGNITIVE_BASE = "#009900"
TOM_AFFECTIVE_BASE = "#B30000"
# Pale background fills (≈ TikZ !12 fill)
TOM_COGNITIVE_BG = "#E0F3E0"
TOM_AFFECTIVE_BG = "#F6E0E0"


def _save(fig, out_dir: Path, name: str) -> Path:
    """Save figure as both PDF and PNG."""
    pdf_path = out_dir / f"{name}.pdf"
    fig.savefig(pdf_path)
    fig.savefig(out_dir / f"{name}.png")
    plt.close(fig)
    return pdf_path


def setup_style():
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "serif",
    })


def _task_labels(label: str) -> tuple[str, str, str]:
    """Return (task_detect, task_intent, task_cat) labels for the modality."""
    if label == "Memes":
        return "Task 2.1", "Task 2.2", "Task 2.3"
    if label == "TikToks":
        return "Task 3.1", "Task 3.2", "Task 3.3"
    return "Task 1.1", "Task 1.2", "Task 1.3"


def _is_videos(label: str) -> bool:
    return label == "TikToks"


def fig_agreement_distribution(df: pd.DataFrame, out_dir: Path, label: str = "Tweets") -> Path:
    """Bar chart of detection-level agreement categories."""
    t1, _, _ = _task_labels(label)

    if _is_videos(label):
        # Videos: 2-3 annotators → 5 categories
        order = ["unanimous_yes", "majority_yes", "split", "majority_no", "unanimous_no"]
        tick_labels = [
            "Unanimous\nYES", "Majority\nYES", "Split",
            "Majority\nNO", "Unanimous\nNO",
        ]
        colors = [CB_VERMILLION, CB_YELLOW, CB_PURPLE, CB_SKYBLUE, "#004A6F"]
    else:
        # Tweets/Memes: 6 annotators → 7 categories
        order = [
            "unanimous_yes", "strong_yes", "weak_yes",
            "split",
            "weak_no", "strong_no", "unanimous_no",
        ]
        tick_labels = [
            "Unanimous\nYES (6-0)", "Strong\nYES (5-1)", "Weak\nYES (4-2)",
            "Split\n(3-3)",
            "Weak\nNO (2-4)", "Strong\nNO (1-5)", "Unanimous\nNO (0-6)",
        ]
        colors = AGREE_COLORS

    counts = df["agreement_cat"].value_counts().reindex(order).fillna(0).astype(int)
    pcts = counts / len(df) * 100

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(range(len(order)), counts.values, color=colors, edgecolor="white", linewidth=0.5)

    for bar, pct in zip(bars, pcts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                f"{pct:.1f}%", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(tick_labels, fontsize=8.5)
    ax.set_ylabel("Number of instances")
    n_ann_label = "2-3 annotators" if _is_videos(label) else "6 annotators"
    ax.set_title(f"{label} — Detection-Level Agreement ({t1}, {n_ann_label}): Is this sexist?")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    suffix = label.lower()
    return _save(fig, out_dir, f"fig_agreement_distribution_{suffix}")


def fig_detection_vs_interpretation(df: pd.DataFrame, out_dir: Path, label: str = "Tweets") -> Path:
    """KEY figure: detection converges, interpretation diverges."""
    t1, t2, t3 = _task_labels(label)
    if _is_videos(label):
        majority_yes = df[df["yes_ratio"] > 0.5].copy()
    else:
        majority_yes = df[df["yes_count"] >= 4].copy()

    det_entropy_mean = majority_yes["entropy_1_1"].mean()
    interp_entropy_mean = majority_yes["entropy_1_2"].mean()
    interp_jaccard_mean = majority_yes["jaccard_1_3"].mean()
    interp_jaccard_disagree = 1 - interp_jaccard_mean

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), gridspec_kw={"width_ratios": [3, 2]})

    measures = [f"Detection\n({t1})\nEntropy", f"Interpretation\n({t2})\nIntent Entropy",
                f"Interpretation\n({t3})\n1 - Jaccard"]
    values = [det_entropy_mean, interp_entropy_mean, interp_jaccard_disagree]
    bar_colors = [DET_COLOR, INTERP_COLOR, INTERP_COLOR]

    bars = ax1.bar(range(3), values, color=bar_colors, width=0.6, edgecolor="white", alpha=0.85)
    for bar, v in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                 f"{v:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax1.set_xticks(range(3))
    ax1.set_xticklabels(measures, fontsize=9)
    ax1.set_ylabel("Disagreement (0 = full agreement, 1 = maximum)")
    ax1.set_ylim(0, 1.1)
    ax1.set_title(f"{label} — Detection vs. Interpretation Disagreement\n(majority-YES instances, n={len(majority_yes)})")
    ax1.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax2.axis("off")
    text = (
        "Detection converges:\n"
        f"  {t1} entropy = {det_entropy_mean:.3f}\n"
        f"  (moderate, below midpoint)\n\n"
        "Interpretation diverges:\n"
        f"  {t2} entropy = {interp_entropy_mean:.3f}\n"
        f"  (near maximum of 1.0)\n\n"
        f"  {t3} Jaccard = {interp_jaccard_mean:.3f}\n"
        f"  (low category overlap)\n\n"
        "People agree something\n"
        "sexist is happening, but\n"
        "disagree on what was\n"
        "intended and what kind."
    )
    ax2.text(0.1, 0.95, text, transform=ax2.transAxes, fontsize=10,
             verticalalignment="top", fontfamily="serif",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", alpha=0.8))

    fig.tight_layout()
    suffix = label.lower()
    return _save(fig, out_dir, f"fig_detection_vs_interpretation_{suffix}")


def fig_gender_detection(df: pd.DataFrame, out_dir: Path, label: str = "Tweets") -> Path:
    """Box plot comparing F vs M YES ratios at detection level."""
    # For videos, filter to instances with both genders present
    sub = df.dropna(subset=["f_yes_ratio", "m_yes_ratio"]).copy()

    plot_data = pd.DataFrame({
        "YES ratio": pd.concat([sub["f_yes_ratio"], sub["m_yes_ratio"]], ignore_index=True),
        "Gender": ["Female"] * len(sub) + ["Male"] * len(sub),
    })

    # Compute p-value and effect size dynamically
    diffs = (sub["f_yes_ratio"] - sub["m_yes_ratio"]).values
    nonzero = diffs[diffs != 0]
    if len(nonzero) > 0:
        w_stat, p_value = stats.wilcoxon(nonzero)
        r_rb = rank_biserial_signed(diffs, w_stat)
    else:
        p_value, r_rb = np.nan, np.nan

    fig, ax = plt.subplots(figsize=(5, 5))
    sns.boxplot(
        data=plot_data, x="Gender", y="YES ratio",
        palette={"Female": GENDER_F, "Male": GENDER_M},
        width=0.5, ax=ax, showfliers=False,
    )

    f_mean = sub["f_yes_ratio"].mean()
    m_mean = sub["m_yes_ratio"].mean()
    ax.axhline(f_mean, color=GENDER_F, linestyle="--", alpha=0.7, linewidth=1)
    ax.axhline(m_mean, color=GENDER_M, linestyle="--", alpha=0.7, linewidth=1)

    ax.set_ylabel("Proportion labelling YES (sexist)")
    sig_label = f"p = {p_value:.3f}" if not np.isnan(p_value) else "p = N/A"
    if p_value < 0.001:
        sig_label = "p < .001"
    r_label = f", r = {r_rb:.3f}" if not np.isnan(r_rb) else ""
    n_label = f", n={len(sub)}" if len(sub) < len(df) else ""
    ax.set_title(f"{label} — Detection Level: Gender Comparison ({sig_label}{r_label}{n_label})")
    ax.set_ylim(-0.05, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    y_offset = 0.03 if abs(f_mean - m_mean) > 0.02 else 0.05
    ax.text(0, f_mean - y_offset, f"mean={f_mean:.3f}", ha="center", fontsize=9, color=GENDER_F)
    ax.text(1, m_mean + 0.03, f"mean={m_mean:.3f}", ha="center", fontsize=9, color=GENDER_M)

    suffix = label.lower()
    return _save(fig, out_dir, f"fig_gender_detection_{suffix}")


def _compute_gender_category_rates(df: pd.DataFrame, use_actual_gender: bool = False) -> tuple[list[float], list[float], list[float], list[float]]:
    """Compute F/M rates, Holm-corrected p-values, and Cohen's h per category.

    Args:
        use_actual_gender: If True, look up actual annotator gender (for videos).
            If False, use positional convention (0-2=F, 3-5=M) for tweets/memes.

    Returns:
        (f_rates, m_rates, p_adjusted, h_values)
    """
    cat_keys = [
        "IDEOLOGICAL-INEQUALITY", "STEREOTYPING-DOMINANCE",
        "OBJECTIFICATION", "SEXUAL-VIOLENCE", "MISOGYNY-NON-SEXUAL-VIOLENCE",
    ]
    f_rates, m_rates, p_values, h_values = [], [], [], []
    for cat in cat_keys:
        f_count, m_count = 0, 0
        f_total, m_total = 0, 0
        for _, row in df.iterrows():
            n_ann = row.get("n_annotators", 6)
            for i in range(n_ann):
                if row.get(f"ann_{i}_task1_1") != "YES":
                    continue
                if use_actual_gender:
                    gender = row.get(f"ann_{i}_gender")
                else:
                    gender = "F" if i < 3 else "M"
                cats = row.get(f"ann_{i}_task1_3")
                is_cat = isinstance(cats, list) and cat in cats
                if gender == "F":
                    f_total += 1
                    if is_cat:
                        f_count += 1
                elif gender == "M":
                    m_total += 1
                    if is_cat:
                        m_count += 1
        f_rate = f_count / f_total if f_total > 0 else 0
        m_rate = m_count / m_total if m_total > 0 else 0
        f_rates.append(f_rate)
        m_rates.append(m_rate)
        h_values.append(cohens_h(f_rate, m_rate))
        table = np.array([[f_count, f_total - f_count], [m_count, m_total - m_count]])
        if f_total + m_total > 0:
            _, p_val = stats.fisher_exact(table)
        else:
            p_val = np.nan
        p_values.append(p_val)

    # Holm-Bonferroni correction across the 5 category tests
    p_adjusted = holm_bonferroni(p_values)
    return f_rates, m_rates, list(p_adjusted), h_values


def fig_gender_categorization(df: pd.DataFrame, out_dir: Path, label: str = "Tweets") -> Path:
    """Bar chart of per-category rates by gender — interpretation-level gender effect."""
    categories = [
        "IDEOLOGICAL-\nINEQUALITY", "STEREOTYPING-\nDOMINANCE",
        "OBJECTI-\nFICATION", "SEXUAL-\nVIOLENCE", "MISOGYNY-NON-\nSEXUAL-VIOLENCE",
    ]

    f_rates, m_rates, p_values, h_values = _compute_gender_category_rates(df, use_actual_gender=_is_videos(label))

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, f_rates, width, label="Female", color=GENDER_F, alpha=0.85)
    ax.bar(x + width / 2, m_rates, width, label="Male", color=GENDER_M, alpha=0.85)

    for i, (p, h) in enumerate(zip(p_values, h_values)):
        if p < 0.001:
            marker = "***"
        elif p < 0.01:
            marker = "**"
        elif p < 0.05:
            marker = "*"
        elif p < 0.1:
            marker = "\u2020"
        else:
            marker = ""
        y_max = max(f_rates[i], m_rates[i])
        if marker:
            ax.text(x[i], y_max + 0.01, f"{marker}\nh={h:.2f}", ha="center", fontsize=9, fontweight="bold")
        elif abs(h) >= 0.05:
            ax.text(x[i], y_max + 0.01, f"h={h:.2f}", ha="center", fontsize=8, color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=8.5)
    ax.set_ylabel("Category assignment rate (among YES annotators)")
    ax.set_title(f"{label} — Interpretation Level: Gender Shapes Categorisation\n(Holm-corrected: * p<.05, ** p<.01, *** p<.001, \u2020 p<.10)")
    ax.legend(frameon=True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    suffix = label.lower()
    return _save(fig, out_dir, f"fig_gender_categorization_{suffix}")


def _draw_gender_cat_panel(ax, df: pd.DataFrame, label: str, use_actual_gender: bool = False):
    """Draw a single gender-categorization panel onto *ax*."""
    cat_labels = [
        "IDEOLOGICAL-\nINEQUALITY", "STEREOTYPING-\nDOMINANCE",
        "OBJECTI-\nFICATION", "SEXUAL-\nVIOLENCE", "MISOGYNY-NON-\nSEXUAL-VIOLENCE",
    ]
    f_rates, m_rates, p_values, h_values = _compute_gender_category_rates(df, use_actual_gender=use_actual_gender)

    x = np.arange(len(cat_labels))
    width = 0.35
    ax.bar(x - width / 2, f_rates, width, label="Female", color=GENDER_F, alpha=0.85)
    ax.bar(x + width / 2, m_rates, width, label="Male", color=GENDER_M, alpha=0.85)

    for i, (p, h) in enumerate(zip(p_values, h_values)):
        if p < 0.001:
            marker = "***"
        elif p < 0.01:
            marker = "**"
        elif p < 0.05:
            marker = "*"
        elif p < 0.1:
            marker = "\u2020"
        else:
            marker = ""
        y_max = max(f_rates[i], m_rates[i])
        if marker:
            ax.text(x[i], y_max + 0.01, f"{marker}\nh={h:.2f}", ha="center", fontsize=8, fontweight="bold")
        elif abs(h) >= 0.05:
            ax.text(x[i], y_max + 0.01, f"h={h:.2f}", ha="center", fontsize=7, color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels(cat_labels, fontsize=7.5)
    ax.set_ylim(0, max(max(f_rates), max(m_rates)) * 1.25)
    ax.set_title(f"{label}", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def fig_gender_categorization_side_by_side(
    tweets_df: pd.DataFrame,
    memes_df: pd.DataFrame,
    out_dir: Path,
) -> Path:
    """Side-by-side gender categorization: Tweets (left) and Memes (right)."""
    setup_style()
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(16, 5.5), sharey=True)

    _draw_gender_cat_panel(ax_l, tweets_df, label="Tweets")
    _draw_gender_cat_panel(ax_r, memes_df, label="Memes")

    ax_l.set_ylabel("Category assignment rate (among YES annotators)")
    ax_r.legend(frameon=True, loc="upper right")

    fig.suptitle(
        "Gender Shapes Categorisation Across Modalities\n"
        "(Holm-corrected: * p<.05, ** p<.01, *** p<.001, \u2020 p<.10; h = Cohen\u2019s h)",
        fontsize=12,
    )
    fig.tight_layout()
    return _save(fig, out_dir, "fig_gender_categorization_tweets_memes")


def _draw_gender_cat_panel_tom(
    ax,
    df: pd.DataFrame,
    label: str,
    use_actual_gender: bool = False,
    base_fs: int = 13,
) -> None:
    """Gender-categorization panel with Cognitive/Affective ToM grouping.

    Cognitive ToM: IDEOLOGICAL-INEQUALITY, STEREOTYPING-DOMINANCE (indices 0–1).
    Affective ToM: OBJECTIFICATION, SEXUAL-VIOLENCE, MISOGYNY-NON-SEXUAL-VIOLENCE (indices 2–4).
    """
    cat_labels = [
        "IDEOLOGICAL-\nINEQUALITY", "STEREOTYPING-\nDOMINANCE",
        "OBJECTI-\nFICATION", "SEXUAL-\nVIOLENCE", "MISOGYNY-NON-\nSEXUAL-VIOLENCE",
    ]
    f_rates, m_rates, p_values, h_values = _compute_gender_category_rates(df, use_actual_gender=use_actual_gender)

    n_cat = len(cat_labels)
    x = np.arange(n_cat)
    width = 0.35
    split_x = 1.5
    x_min, x_max = -0.5, n_cat - 0.5

    ax.set_xlim(x_min, x_max)
    ax.axvspan(x_min, split_x, color=TOM_COGNITIVE_BG, zorder=0)
    ax.axvspan(split_x, x_max, color=TOM_AFFECTIVE_BG, zorder=0)
    ax.axvline(split_x, linestyle="--", color="gray", linewidth=1.0, alpha=0.8, zorder=1)

    ax.bar(x - width / 2, f_rates, width, label="Female", color=GENDER_F, alpha=0.9,
           edgecolor="white", linewidth=0.5, zorder=2)
    ax.bar(x + width / 2, m_rates, width, label="Male", color=GENDER_M, alpha=0.9,
           edgecolor="white", linewidth=0.5, zorder=2)

    for i, (p, h) in enumerate(zip(p_values, h_values)):
        if p < 0.001:
            marker = "***"
        elif p < 0.01:
            marker = "**"
        elif p < 0.05:
            marker = "*"
        elif p < 0.1:
            marker = "†"
        else:
            marker = ""
        y_max = max(f_rates[i], m_rates[i])
        if marker:
            ax.text(x[i], y_max + 0.012, f"{marker}\nh={h:.2f}", ha="center",
                    fontsize=base_fs, fontweight="bold", zorder=3)
        elif abs(h) >= 0.05:
            ax.text(x[i], y_max + 0.012, f"h={h:.2f}", ha="center",
                    fontsize=base_fs - 2, color="gray", zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(cat_labels, fontsize=base_fs)
    ax.tick_params(axis="y", labelsize=base_fs)
    ax.set_ylim(0, max(max(f_rates), max(m_rates)) * 1.30)
    ax.set_title(label, fontsize=base_fs + 3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ToM-group brackets and labels underneath the x-tick labels.
    # Mixed transform: x in data coords, y in axes fraction.
    trans = ax.get_xaxis_transform()
    bracket_y = -0.20
    label_y = -0.24
    ax.plot([x_min + 0.05, split_x - 0.1], [bracket_y, bracket_y],
            color=TOM_COGNITIVE_BASE, lw=1.8, transform=trans, clip_on=False, zorder=4)
    ax.plot([split_x + 0.1, x_max - 0.05], [bracket_y, bracket_y],
            color=TOM_AFFECTIVE_BASE, lw=1.8, transform=trans, clip_on=False, zorder=4)
    ax.text((x_min + split_x) / 2, label_y, "Cognitive ToM", ha="center", va="top",
            transform=trans, fontsize=base_fs + 1, style="italic",
            color=TOM_COGNITIVE_BASE, zorder=4)
    ax.text((split_x + x_max) / 2, label_y, "Affective ToM", ha="center", va="top",
            transform=trans, fontsize=base_fs + 1, style="italic",
            color=TOM_AFFECTIVE_BASE, zorder=4)


def fig_gender_categorization_side_by_side_tom(
    tweets_df: pd.DataFrame,
    memes_df: pd.DataFrame,
    out_dir: Path,
) -> Path:
    """Side-by-side gender categorization with Cognitive/Affective ToM grouping.

    Larger fonts, pale green/red axvspan fills, dashed separator, and bracketed
    group labels underneath the x-tick labels.
    """
    setup_style()
    out_dir.mkdir(parents=True, exist_ok=True)

    base_fs = 13
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(18, 7), sharey=True)

    title_tw = "Tweets — Gender Shapes Categorisation\n(Holm-corrected: * p<.05, ** p<.01, *** p<.001, † p<.10)"
    title_me = "Memes — Gender Shapes Categorisation\n(Holm-corrected: * p<.05, ** p<.01, *** p<.001, † p<.10)"
    _draw_gender_cat_panel_tom(ax_l, tweets_df, label=title_tw, base_fs=base_fs)
    _draw_gender_cat_panel_tom(ax_r, memes_df, label=title_me, base_fs=base_fs)

    ax_l.set_ylabel("Category assignment rate\n(among YES annotators)", fontsize=base_fs + 1)
    ax_l.legend(frameon=True, loc="upper right", fontsize=base_fs)
    ax_r.legend(frameon=True, loc="upper right", fontsize=base_fs)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.20)
    return _save(fig, out_dir, "fig_gender_categorization_tweets_memes_tom")


def generate_all(df: pd.DataFrame, out_dir: Path, label: str = "Tweets") -> list[Path]:
    """Generate all figures for one dataset and return paths."""
    setup_style()
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    tag = label.lower()

    print(f"\nGenerating figures ({label})...")

    p = fig_agreement_distribution(df, out_dir, label)
    print(f"  - Agreement distribution: {p.name}")
    paths.append(p)

    p = fig_detection_vs_interpretation(df, out_dir, label)
    print(f"  - Detection vs interpretation: {p.name}")
    paths.append(p)

    p = fig_gender_detection(df, out_dir, label)
    print(f"  - Gender detection: {p.name}")
    paths.append(p)

    p = fig_gender_categorization(df, out_dir, label)
    print(f"  - Gender categorization: {p.name}")
    paths.append(p)

    return paths
