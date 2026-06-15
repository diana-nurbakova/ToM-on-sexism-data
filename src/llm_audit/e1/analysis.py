"""E1 analysis (OSF pre-reg §7, Appendix C 'Statistical models').

Operates on the author's construct-derived partition (the worksheet
``invariance_class`` column) plus the LLM-judge verdict logs:

  * E1-H1 — SHIFTING proportion with a percentile bootstrap CI (B=1000), tested
    against the 5% floor and 10% target. UNSWAPPABLE excluded; AMBIGUOUS kept in
    the denominator.
  * E1-H2 — association on the 2×2 invariance{SHIFTING,INVARIANT} ×
    grouping{cognitive,affective} table via Fisher's exact, with odds ratio and
    Cramér's V; directional prediction = SHIFTING enriched in affective.
  * E1-H3 — per-judge confusion matrix, percent agreement, and Cohen's κ vs the
    construct-derived partition (descriptive convergence check).

Inference caveat (pre-registered): annotator non-independence makes the H2 p-value
approximate; the directional effect and effect size carry the claim, not significance.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import chi2_contingency, fisher_exact

from ..data import T13_CATEGORIES
from ..metrics import iter_records

INVARIANCE_CLASSES = ("INVARIANT", "SHIFTING", "AMBIGUOUS")
COGNITIVE = {"ideological_inequality", "stereotyping_dominance"}
AFFECTIVE = {"objectification", "sexual_violence", "misogyny_non_sexual_violence"}


def category_grouping(category: str) -> str | None:
    if category in COGNITIVE:
        return "cognitive"
    if category in AFFECTIVE:
        return "affective"
    return None


# ── E1-H1: SHIFTING proportion + bootstrap CI ─────────────────────────────────
def shifting_proportion_ci(classes, B: int = 1000, seed: int = 42, pct=(2.5, 97.5)) -> dict:
    """Proportion SHIFTING with a percentile bootstrap CI.

    ``classes`` = invariance class per *classified* item (UNSWAPPABLE already
    excluded; AMBIGUOUS retained in the denominator, per the pre-reg).
    """
    arr = np.array([1 if c == "SHIFTING" else 0 for c in classes], int)
    n = len(arr)
    if n == 0:
        return {"n": 0, "p_shift": None, "ci_low": None, "ci_high": None,
                "clears_5pct": False, "clears_10pct": False}
    p = float(arr.mean())
    rng = np.random.default_rng(seed)
    boot = [rng.choice(arr, size=n, replace=True).mean() for _ in range(B)]
    lo, hi = np.percentile(boot, pct)
    return {
        "n": n, "n_shifting": int(arr.sum()), "p_shift": p,
        "ci_low": float(lo), "ci_high": float(hi),
        "clears_5pct": bool(lo > 0.05), "clears_10pct": bool(lo > 0.10),
    }


# ── E1-H2: invariance × cognitive/affective association ───────────────────────
def h2_association(pairs) -> dict:
    """Fisher's exact on the 2×2 invariance{SHIFTING,INVARIANT} × {cognitive,affective}.

    ``pairs`` = iterable of (invariance_class, category). AMBIGUOUS/UNSWAPPABLE and
    items without a cognitive/affective grouping are dropped.
    """
    # table[grouping][invariance]
    cats = {("cognitive", "SHIFTING"): 0, ("cognitive", "INVARIANT"): 0,
            ("affective", "SHIFTING"): 0, ("affective", "INVARIANT"): 0}
    for inv, category in pairs:
        if inv not in ("SHIFTING", "INVARIANT"):
            continue
        g = category_grouping(category)
        if g is None:
            continue
        cats[(g, inv)] += 1
    # rows = grouping (cognitive, affective); cols = (SHIFTING, INVARIANT)
    table = [
        [cats[("cognitive", "SHIFTING")], cats[("cognitive", "INVARIANT")]],
        [cats[("affective", "SHIFTING")], cats[("affective", "INVARIANT")]],
    ]
    result = {"table": {"cognitive": {"SHIFTING": table[0][0], "INVARIANT": table[0][1]},
                        "affective": {"SHIFTING": table[1][0], "INVARIANT": table[1][1]}},
              "n": sum(sum(r) for r in table)}
    # OR: (affective-shifting × cognitive-invariant) / (affective-invariant × cognitive-shifting)
    # Direction: SHIFTING enriched in affective -> OR for affective>cognitive on SHIFTING.
    or_table = [[table[1][0], table[1][1]], [table[0][0], table[0][1]]]  # affective on top
    try:
        odds, p_two = fisher_exact(or_table, alternative="two-sided")
        _, p_one = fisher_exact(or_table, alternative="greater")
    except ValueError:
        odds, p_two, p_one = float("nan"), float("nan"), float("nan")
    result["odds_ratio"] = float(odds)
    result["p_two_sided"] = float(p_two)
    result["p_one_sided_affective_enriched"] = float(p_one)
    result["cramers_v"] = _cramers_v(table)
    return result


def _cramers_v(table) -> float:
    arr = np.array(table, float)
    n = arr.sum()
    if n == 0 or arr.shape != (2, 2) or (arr.sum(0) == 0).any() or (arr.sum(1) == 0).any():
        return float("nan")
    chi2 = chi2_contingency(arr, correction=False)[0]
    return float(np.sqrt(chi2 / n))  # min(r-1,c-1)=1 for a 2×2


# ── E1-H3: judge convergence ──────────────────────────────────────────────────
def cohen_kappa(a, b) -> float:
    """Cohen's κ between two equal-length label sequences."""
    a, b = list(a), list(b)
    n = len(a)
    if n == 0:
        return float("nan")
    labels = sorted(set(a) | set(b))
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[l] / n) * (cb[l] / n) for l in labels)
    return float((po - pe) / (1 - pe)) if pe != 1 else 1.0


def judge_agreement(partition: dict, verdicts: dict) -> dict:
    """Confusion matrix, percent agreement, and κ between a judge and the partition.

    ``partition`` / ``verdicts`` map item_id -> invariance class. Computed over the
    intersection with a parseable verdict (available-case analysis).
    """
    items = [i for i in partition if i in verdicts and partition[i] in INVARIANCE_CLASSES]
    truth = [partition[i] for i in items]
    pred = [verdicts[i] for i in items]
    confusion = {t: {p: 0 for p in INVARIANCE_CLASSES} for t in INVARIANCE_CLASSES}
    for t, p in zip(truth, pred):
        if p in INVARIANCE_CLASSES:
            confusion[t][p] += 1
    agree = sum(1 for t, p in zip(truth, pred) if t == p)
    return {
        "n": len(items),
        "percent_agreement": (agree / len(items)) if items else None,
        "cohen_kappa": cohen_kappa(truth, pred) if items else None,
        "confusion": confusion,
    }


# ── Loaders ───────────────────────────────────────────────────────────────────
def load_partition(worksheet_csv: Path) -> dict:
    """Read the author-filled worksheet -> {item_id: invariance_class}.

    Only rows with a non-empty ``invariance_class`` are returned (so the scaffold
    runs partially as adjudication proceeds).
    """
    out = {}
    with open(worksheet_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cls = (row.get("invariance_class") or "").strip().upper()
            if cls in INVARIANCE_CLASSES:
                out[str(row["item_id"])] = cls
    return out


def load_partition_meta(worksheet_csv: Path) -> dict:
    """Read {item_id: strat_category} for the H2 grouping."""
    out = {}
    with open(worksheet_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[str(row["item_id"])] = row.get("strat_category")
    return out


def load_judge_verdicts(judge_log: Path) -> dict:
    """Read a judge log -> {item_id: verdict} for parsed_ok records."""
    out = {}
    for rec in iter_records(judge_log):
        if rec.get("parse_status") == "parsed_ok" and rec.get("parsed_value"):
            out[str(rec["item_id"])] = rec["parsed_value"]
    return out


def run_analysis(worksheet_csv: Path, judge_logs: dict[str, Path] | None = None) -> dict:
    """Compute E1-H1/H2/H3 from the worksheet and (optional) judge logs."""
    partition = load_partition(worksheet_csv)
    meta = load_partition_meta(worksheet_csv)
    classes = list(partition.values())
    pairs = [(partition[i], meta.get(i)) for i in partition]
    out = {
        "n_classified": len(partition),
        "class_counts": dict(Counter(classes)),
        "E1_H1": shifting_proportion_ci(classes),
        "E1_H2": h2_association(pairs),
        "E1_H3": {},
    }
    for model, log in (judge_logs or {}).items():
        if Path(log).exists():
            out["E1_H3"][model] = judge_agreement(partition, load_judge_verdicts(log))
    return out
