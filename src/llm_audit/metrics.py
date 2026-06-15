"""Metrics: soft-label assembly from logs, divergences, F1, hierarchical check.

Assembles model soft labels from the §6.2 call log (Format C aggregates K draws
into proportions; A/B use the single distribution), then scores them against the
EXIST gold soft labels (master spec §7.1): Jensen-Shannon divergence, KL,
cross-entropy, hard macro/per-class F1, and the §7.1 hierarchical-consistency
check. Only ``parsed_ok`` cells feed the soft metrics; refusal / disclaimer /
format_violation are reported as rates, never coerced (§6.2). ICM-Soft / ICM-Soft
Norm (the EXIST official soft metric) live in ``llm_audit.icm_soft``, which wraps
the organisers' PyEvALL in an isolated environment (master spec §7.1).
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import parsing
from .data import T11_CLASSES, T12_CLASSES, T13_CATEGORIES

_EPS = 1e-12
_CANON = {"1.1": T11_CLASSES, "1.2": T12_CLASSES, "1.3": T13_CATEGORIES}
_MONO = {"1.1", "1.2"}  # proper distributions (sum to 1); 1.3 is multi-label


# ── Pure metric primitives (operate on aligned vectors) ───────────────────────
def _kl(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(p, _EPS, 1.0)
    q = np.clip(q, _EPS, 1.0)
    return float(np.sum(p * np.log2(p / q)))


def jsd(p, q) -> float:
    """Jensen-Shannon divergence (bits) between two distributions."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    m = 0.5 * (p + q)
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def kl_divergence(p, q) -> float:
    """KL(p || q) in bits, with epsilon smoothing."""
    return _kl(np.asarray(p, float), np.asarray(q, float))


def cross_entropy(gold, model) -> float:
    """Cross-entropy H(gold, model) in bits."""
    g = np.asarray(gold, float)
    m = np.clip(np.asarray(model, float), _EPS, 1.0)
    return float(-np.sum(g * np.log2(m)))


def multilabel_jsd(p, q) -> float:
    """Mean per-category Bernoulli JSD for the multi-label 1.3 vectors."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    vals = [jsd([pi, 1 - pi], [qi, 1 - qi]) for pi, qi in zip(p, q)]
    return float(np.mean(vals)) if vals else 0.0


def f1_scores(y_true: list, y_pred: list, labels: list) -> dict:
    """Macro + per-class F1 for hard single-label predictions."""
    per = {}
    f1s = []
    for lab in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if p == lab and t == lab)
        fp = sum(1 for t, p in zip(y_true, y_pred) if p == lab and t != lab)
        fn = sum(1 for t, p in zip(y_true, y_pred) if p != lab and t == lab)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per[lab] = {"precision": prec, "recall": rec, "f1": f1, "support": tp + fn}
        f1s.append(f1)
    return {"macro_f1": float(np.mean(f1s)) if f1s else 0.0, "per_class": per}


def multilabel_f1(y_true: list[set], y_pred: list[set], labels: list) -> dict:
    """Macro + per-class F1 for multi-label (1.3) hard predictions."""
    per = {}
    f1s = []
    for lab in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if lab in p and lab in t)
        fp = sum(1 for t, p in zip(y_true, y_pred) if lab in p and lab not in t)
        fn = sum(1 for t, p in zip(y_true, y_pred) if lab not in p and lab in t)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per[lab] = {"precision": prec, "recall": rec, "f1": f1, "support": tp + fn}
        f1s.append(f1)
    return {"macro_f1": float(np.mean(f1s)) if f1s else 0.0, "per_class": per}


# ── Format-C aggregation (K draws -> soft label) ──────────────────────────────
def aggregate_format_c_single(draws: list[str], subtask: str) -> dict[str, float]:
    """1.1/1.2: proportions over classes from K single-label categorical draws."""
    classes = _CANON[subtask]
    n = len(draws)
    counts = {c: 0 for c in classes}
    for d in draws:
        if d in counts:
            counts[d] += 1
    return {c: (counts[c] / n if n else 0.0) for c in classes}


def aggregate_format_c_multi(draws: list[list[str]]) -> dict[str, float]:
    """1.3: per-category proportion of draws that named the category."""
    n = len(draws)
    counts = {c: 0 for c in T13_CATEGORIES}
    for cats in draws:
        for c in set(cats):
            if c in counts:
                counts[c] += 1
    return {c: (counts[c] / n if n else 0.0) for c in T13_CATEGORIES}


# ── Log loading + soft-label assembly ─────────────────────────────────────────
def iter_records(log_path: Path):
    """Yield the latest record per call key from a §6.2 JSONL log."""
    latest: dict[tuple, dict] = {}
    order: dict[tuple, int] = {}
    with open(log_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (rec["experiment"], rec["condition"], rec["subtask"], rec["language"],
                   rec["format"], rec["item_id"], rec["k_index"])
            latest[key] = rec
            order[key] = i
    yield from (latest[k] for k in sorted(latest, key=lambda k: order[k]))


def assemble_model_soft(records, k_cap: int | None = None) -> tuple[dict, dict]:
    """Build model soft labels and status-rate tallies from records.

    Returns ``(soft, status_counts)`` where ``soft`` is keyed by
    ``(condition, subtask, format, item_id)`` -> {"soft": dict, "n_valid": int} and
    only ``parsed_ok`` draws contribute. ``status_counts`` is keyed by
    ``(condition, subtask, format)`` -> {status: count}.

    ``k_cap`` limits Format-C aggregation to the first ``k_cap`` draws (by k_index).
    Used for the cross-lane (open-weight K=6 vs proprietary K=3) comparison so JSD is
    matched on *both* format and K — JSD is sensitive to elicitation format and to the
    soft-label granularity that K sets, so an unmatched comparison is confounded.
    """
    c_draws_single: dict[tuple, list] = defaultdict(list)
    c_draws_multi: dict[tuple, list] = defaultdict(list)
    ab_soft: dict[tuple, dict] = {}
    status_counts: dict[tuple, dict] = defaultdict(lambda: defaultdict(int))

    for rec in records:
        cond, sub, fmt, item = rec["condition"], rec["subtask"], rec["format"], rec["item_id"]
        status_counts[(cond, sub, fmt)][rec["parse_status"]] += 1
        if rec["parse_status"] != parsing.PARSED_OK:
            continue
        if k_cap is not None and fmt == "C" and rec.get("k_index", 0) >= k_cap:
            continue  # K-match: keep only the first k_cap categorical draws
        val = rec["parsed_value"]
        gkey = (cond, sub, fmt, item)
        if fmt in ("A", "B"):
            ab_soft[gkey] = val
        elif sub == "1.3":
            c_draws_multi[gkey].append(val or [])
        else:
            c_draws_single[gkey].append(val)

    soft: dict[tuple, dict] = {}
    for gkey, dist in ab_soft.items():
        soft[gkey] = {"soft": dist, "n_valid": 1}
    for gkey, draws in c_draws_single.items():
        soft[gkey] = {"soft": aggregate_format_c_single(draws, gkey[1]), "n_valid": len(draws)}
    for gkey, draws in c_draws_multi.items():
        soft[gkey] = {"soft": aggregate_format_c_multi(draws), "n_valid": len(draws)}
    return soft, {k: dict(v) for k, v in status_counts.items()}


# ── Hierarchical consistency (§7.1) ───────────────────────────────────────────
def hierarchical_consistency(p_not_sexist_11: float, p_not_sexist_12: float,
                             category_mass_13: float, p_sexist_11: float) -> dict:
    """Per-item consistency signals across the independent 1.1/1.2/1.3 elicitations."""
    return {
        "not_sexist_gap_11_12": abs(p_not_sexist_11 - p_not_sexist_12),
        # Category mass in 1.3 should not exceed P(sexist) from 1.1 (soft bound).
        "category_excess_over_sexist": max(0.0, category_mass_13 - p_sexist_11),
    }


# ── End-to-end scoring against gold ───────────────────────────────────────────
def _vec(dist: dict, subtask: str) -> np.ndarray:
    return np.array([dist.get(c, 0.0) for c in _CANON[subtask]], float)


def score_against_gold(soft: dict, gold_by_item: dict) -> dict:
    """Aggregate divergence/F1 metrics per (condition, subtask, format).

    Args:
        soft: output of ``assemble_model_soft`` (the ``soft`` dict).
        gold_by_item: {item_id: {"1.1": dist, "1.2": dist, "1.3": dist}}.
    """
    buckets: dict[tuple, list] = defaultdict(list)
    for (cond, sub, fmt, item), entry in soft.items():
        if item not in gold_by_item:
            continue
        buckets[(cond, sub, fmt)].append((item, entry["soft"]))

    results = {}
    for (cond, sub, fmt), rows in buckets.items():
        jsds, kls, ces = [], [], []
        yt, yp = [], []           # mono-label hard labels
        yt_ml, yp_ml = [], []     # multi-label hard sets
        for item, msoft in rows:
            g = gold_by_item[item][sub]
            gv, mv = _vec(g, sub), _vec(msoft, sub)
            if sub in _MONO:
                jsds.append(jsd(gv, mv))
                kls.append(kl_divergence(gv, mv))
                ces.append(cross_entropy(gv, mv))
                classes = list(_CANON[sub])
                yt.append(classes[int(np.argmax(gv))])
                yp.append(classes[int(np.argmax(mv))])
            else:
                jsds.append(multilabel_jsd(gv, mv))
                yt_ml.append({c for c, v in zip(_CANON[sub], gv) if v >= 0.5})
                yp_ml.append({c for c, v in zip(_CANON[sub], mv) if v >= 0.5})
        res = {"n_items": len(rows), "mean_jsd": float(np.mean(jsds)) if jsds else None}
        if sub in _MONO:
            res["mean_kl"] = float(np.mean(kls))
            res["mean_cross_entropy"] = float(np.mean(ces))
            res["f1"] = f1_scores(yt, yp, list(_CANON[sub]))
        else:
            res["mean_kl"] = None
            res["f1"] = multilabel_f1(yt_ml, yp_ml, list(_CANON[sub]))
        results[(cond, sub, fmt)] = res
    return results


def gold_lookup(df) -> dict:
    """Build {item_id: {"1.1": gold_1_1, "1.2": gold_1_2, "1.3": gold_1_3}}."""
    return {
        str(r["item_id"]): {"1.1": r["gold_1_1"], "1.2": r["gold_1_2"], "1.3": r["gold_1_3"]}
        for _, r in df.iterrows()
    }


def cross_lane_compare(logs: dict, gold_by_item: dict, k: int = 3, fmt: str = "C",
                       spec: str = "defined") -> dict:
    """Format- and K-matched JSD-to-gold across lanes (open-weight vs proprietary).

    Because JSD depends on elicitation format and on K (the soft-label granularity),
    a clean open-weight-vs-proprietary comparison must hold both constant and use the
    **shared item set**. This restricts every lane to ``fmt`` (the primary format) and
    caps Format-C to ``k`` draws (the proprietary lane's K), then scores each lane on
    the items present in *all* lanes.

    Args:
        logs: {lane_name: log_path}.
        k: common K (proprietary K=3); open-weight K=6 is capped to it.
    Returns per-subtask: shared-item count and per-lane mean JSD on those items.
    """
    # lane -> {(subtask, item): soft}
    lane_soft: dict[str, dict] = {}
    for name, path in logs.items():
        soft, _ = assemble_model_soft(iter_records(path), k_cap=k)
        lane_soft[name] = {
            (sub, item): entry["soft"]
            for (cond, sub, f, item), entry in soft.items()
            if f == fmt and cond == spec
        }
    out = {}
    for subtask in ("1.1", "1.2", "1.3"):
        # items present for this subtask in every lane and in gold
        per_lane_items = [
            {item for (s, item) in d if s == subtask and item in gold_by_item}
            for d in lane_soft.values()
        ]
        shared = set.intersection(*per_lane_items) if per_lane_items else set()
        if not shared:
            continue
        lanes = {}
        for name, d in lane_soft.items():
            divs = []
            for item in shared:
                g, m = _vec(gold_by_item[item][subtask], subtask), _vec(d[(subtask, item)], subtask)
                divs.append(multilabel_jsd(g, m) if subtask == "1.3" else jsd(g, m))
            lanes[name] = float(np.mean(divs))
        out[subtask] = {"n_shared": len(shared), "k": k, "format": fmt, "mean_jsd": lanes}
    return out
