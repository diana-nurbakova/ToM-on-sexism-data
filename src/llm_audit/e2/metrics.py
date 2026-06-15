"""E2 flip metrics: pair original vs swapped soft labels (master spec §5.3, §7.1).

From the E2 log, pairs each item's ``orig`` and ``swap`` soft labels (per spec ×
subtask × format) and computes flip-rate (hard-label change), soft divergence (JSD),
and the swap asymmetry |flip(F→M) − flip(M→F)|, with per-category and per-invariance
breakdowns. Reuses the §7.1 primitives from ``metrics``.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .. import metrics as M
from ..data import T13_CATEGORIES


def _hard_label(soft: dict, subtask: str):
    if subtask == "1.3":  # multi-label hard set at 0.5
        return frozenset(c for c in T13_CATEGORIES if soft.get(c, 0.0) >= 0.5)
    classes = M._CANON[subtask]
    return classes[int(np.argmax([soft.get(c, 0.0) for c in classes]))]


def _divergence(orig: dict, swap: dict, subtask: str) -> float:
    o = M._vec(orig, subtask)
    s = M._vec(swap, subtask)
    return M.multilabel_jsd(o, s) if subtask == "1.3" else M.jsd(o, s)


def pair_orig_swap(soft: dict) -> dict:
    """From assemble_model_soft output keyed (condition, subtask, fmt, item) where
    condition is '{spec}|{variant}', return {(spec, subtask, fmt, item): {orig, swap}}."""
    paired: dict[tuple, dict] = defaultdict(dict)
    for (condition, subtask, fmt, item), entry in soft.items():
        if "|" not in condition:
            continue
        spec, variant = condition.split("|", 1)
        paired[(spec, subtask, fmt, item)][variant] = entry["soft"]
    return {k: v for k, v in paired.items() if "orig" in v and "swap" in v}


def flip_metrics(paired: dict, item_meta: dict) -> dict:
    """Aggregate flip-rate / JSD / asymmetry per (spec, subtask, format).

    Args:
        paired: output of ``pair_orig_swap``.
        item_meta: {item_id: {"category": strat_category, "direction": F2M|M2F|...,
                    "invariance": class}} for the breakdowns.
    """
    buckets: dict[tuple, list] = defaultdict(list)
    for (spec, subtask, fmt, item), pair in paired.items():
        buckets[(spec, subtask, fmt)].append((item, pair["orig"], pair["swap"]))

    out = {}
    for key, rows in buckets.items():
        _, subtask, _ = key
        flips, divs = [], []
        by_dir = defaultdict(list)
        by_cat = defaultdict(list)
        by_inv = defaultdict(list)
        shift_dir = defaultdict(list)             # signed ΔP(sexist) by direction (1.1/1.2)
        shift_cat_dir = defaultdict(lambda: defaultdict(list))  # 1.3: per-category × direction
        for item, o, s in rows:
            flipped = int(_hard_label(o, subtask) != _hard_label(s, subtask))
            d = _divergence(o, s, subtask)
            flips.append(flipped); divs.append(d)
            meta = item_meta.get(item, {})
            direction = meta.get("direction")
            by_dir[direction].append(flipped)
            by_cat[meta.get("category")].append(flipped)
            by_inv[meta.get("invariance")].append(flipped)
            # Signed directional shift — the HMAIN crux (woman-targeting tracks sexism):
            # woman→man should lower, man→woman should raise the sexist mass.
            if subtask == "1.3":
                for cat in T13_CATEGORIES:
                    shift_cat_dir[cat][direction].append(s.get(cat, 0.0) - o.get(cat, 0.0))
            else:
                shift_dir[direction].append(_positive_mass(s, subtask) - _positive_mass(o, subtask))
        f2m = float(np.mean(by_dir["F2M"])) if by_dir.get("F2M") else None
        m2f = float(np.mean(by_dir["M2F"])) if by_dir.get("M2F") else None
        rec = {
            "n": len(rows),
            "flip_rate": float(np.mean(flips)),
            "mean_jsd": float(np.mean(divs)),
            "flip_F2M": f2m, "flip_M2F": m2f,
            "asymmetry": (abs(f2m - m2f) if f2m is not None and m2f is not None else None),
            "flip_by_category": {c: float(np.mean(v)) for c, v in by_cat.items() if c},
            "flip_by_invariance": {c: float(np.mean(v)) for c, v in by_inv.items() if c},
        }
        if subtask == "1.3":
            rec["shift_by_direction"] = {
                cat: {d: float(np.mean(v)) for d, v in dirs.items() if d}
                for cat, dirs in shift_cat_dir.items()}
        else:
            rec["shift_by_direction"] = {d: float(np.mean(v)) for d, v in shift_dir.items() if d}
        out["|".join(key)] = rec
    return out


def _positive_mass(soft: dict, subtask: str) -> float:
    """The 'sexist-ish' mass whose directional shift carries the HMAIN signal."""
    if subtask == "1.1":
        return soft.get("sexist", 0.0)
    return 1.0 - soft.get("not_sexist", 0.0)  # 1.2: any sexist intention


def compute_e2(log_path, item_meta: dict) -> dict:
    """End-to-end: read the E2 log, assemble soft labels, pair, and score."""
    soft, _status = M.assemble_model_soft(M.iter_records(log_path))
    return flip_metrics(pair_orig_swap(soft), item_meta)
