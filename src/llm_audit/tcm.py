"""Transport-based Confusion Matrix (TCM) — the unified evaluation backbone (§7.1).

Erbani, Portier, Egyed-Zsigmond & Nurbakova (2024), *Confusion Matrices: A Unified
Theory*, IEEE Access. TCM generalises the ordinary confusion matrix to multi-label
and soft-label settings under one construction, which fits the EXIST task family
(1.1/1.2 mono-label soft, 1.3 multi-label soft). For each instance it "transports"
the normalised label distribution onto the normalised prediction distribution:
matched mass ``min(y, ŷ)`` stays on the diagonal, and the residual (error) label
mass is spread across the residual prediction mass as the maximum-entropy
(independent) coupling of the two residual marginals. Summed over a dataset it
gives a ``C×C`` matrix with **rows = true/label classes, columns = predicted
classes**, reducing to the classic confusion matrix in the single-label case.

Per-instance contribution (normalised ``y₁``, ``ŷ₁``; ``m = min(y₁, ŷ₁)``)::

    diag      = diag(m)
    off_diag  = outer(y₁ − m, ŷ₁ − m) / Σ(y₁ − m)        # 0 if y₁ == ŷ₁
    contrib   = weight · (diag + off_diag)

Three weightings (per the reference): ``TCMone`` (every instance ×1), ``TCMlab``
(× the label row's pre-normalisation mass), ``TCMpred`` (× the prediction row's
pre-normalisation mass). This is a dependency-light numpy port of the reference
torch implementation (github.com/johan140391/TCM); the two agree by construction.
"""

from __future__ import annotations

import numpy as np

from .data import T11_CLASSES, T12_CLASSES, T13_CATEGORIES

WEIGHTINGS = ("TCMone", "TCMlab", "TCMpred")

# Class orderings per subtask. 1.3 prepends the not-sexist class so multi-label
# rows are never all-zero (its mass comes from the 1.1 detection distribution,
# matching the ICM-Soft convention in ``icm_soft``).
TCM_CLASSES = {
    "1.1": list(T11_CLASSES),
    "1.2": list(T12_CLASSES),
    "1.3": ["not_sexist", *T13_CATEGORIES],
}


def _contribution(y: np.ndarray, yhat: np.ndarray, weighting: str) -> np.ndarray:
    """Single-instance TCM contribution (``C×C``). See module docstring."""
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    sy, syh = y.sum(), yhat.sum()
    if sy <= 0 or syh <= 0:
        raise ValueError("TCM needs non-zero label and prediction mass per row.")
    y1, yhat1 = y / sy, yhat / syh
    common = np.minimum(y1, yhat1)
    contrib = np.diag(common)
    if not np.allclose(y1, yhat1):
        resid_lab = y1 - common
        resid_pred = yhat1 - common
        contrib = contrib + np.outer(resid_lab, resid_pred) / resid_lab.sum()
    weight = 1.0 if weighting == "TCMone" else (sy if weighting == "TCMlab" else syh)
    return weight * contrib


class TCM:
    """Accumulating Transport-based Confusion Matrix.

    Call :meth:`update` once per batch (or once for the whole dataset) with
    aligned ``labels`` and ``predictions`` arrays of shape ``(n, C)`` — soft
    label and prediction vectors, one row per instance — then :meth:`get` the
    summed ``C×C`` matrix. Mirrors the reference API (``update``/``get``).
    """

    def __init__(self, n_classes: int, weighting: str = "TCMlab"):
        if weighting not in WEIGHTINGS:
            raise ValueError(f"weighting must be one of {WEIGHTINGS}, got {weighting!r}")
        self.C = n_classes
        self.weighting = weighting
        self.M = np.zeros((n_classes, n_classes))

    def update(self, labels, predictions) -> "TCM":
        labels = np.asarray(labels, float)
        predictions = np.asarray(predictions, float)
        if labels.shape != predictions.shape:
            raise ValueError(f"shape mismatch: {labels.shape} vs {predictions.shape}")
        if labels.ndim != 2 or labels.shape[1] != self.C:
            raise ValueError(f"expected (n, {self.C}) arrays, got {labels.shape}")
        for y, yhat in zip(labels, predictions):
            self.M += _contribution(y, yhat, self.weighting)
        return self

    def get(self) -> np.ndarray:
        return self.M


def tcm_matrix(labels, predictions, weighting: str = "TCMlab") -> np.ndarray:
    """Convenience one-shot TCM over ``(n, C)`` label/prediction arrays."""
    n_classes = np.asarray(labels, float).shape[1]
    return TCM(n_classes, weighting).update(labels, predictions).get()


def row_normalise(matrix: np.ndarray) -> np.ndarray:
    """Row-normalise a TCM (each true-class row sums to 1) for plotting/§7.3.

    Zero rows are left as zeros rather than producing NaNs.
    """
    matrix = np.asarray(matrix, float)
    sums = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, sums, out=np.zeros_like(matrix), where=sums > 0)


# ── EXIST builder: TCM from assembled soft labels ─────────────────────────────
def _vectors(soft: dict, classes: list, no_mass: float | None) -> np.ndarray:
    vec = np.array([float(soft.get(c, 0.0)) for c in classes])
    if classes and classes[0] == "not_sexist" and no_mass is not None:
        vec[0] = float(no_mass)  # 1.3 not-sexist mass (rows[0] is not_sexist)
    return vec


def tcm_for_exist(soft: dict, gold_by_item: dict, weighting: str = "TCMlab") -> dict:
    """Build a TCM per ``(condition, subtask, format)`` from assembled soft labels.

    Args mirror ``icm_soft.score_icm_soft``: ``soft`` is the
    ``metrics.assemble_model_soft`` output (keyed ``(condition, subtask, format,
    item)``) and ``gold_by_item`` is ``{item: {"1.1": dist, "1.2": dist, "1.3":
    dist}}``. Gold rows are the label, model rows the prediction. The 1.3
    not-sexist mass is threaded from each side's 1.1 distribution (the ICM-Soft
    convention); items whose gold or model row has no mass are dropped and counted.

    Returns ``{"condition|subtask|format": {"classes": [...], "weighting": str,
    "n_items": int, "n_dropped": int, "matrix": list[list[float]]}}``.
    """
    from collections import defaultdict

    det = {
        (cond, fmt, item): entry["soft"]
        for (cond, sub, fmt, item), entry in soft.items() if sub == "1.1"
    }
    buckets: dict[tuple, list] = defaultdict(list)
    for (cond, sub, fmt, item), entry in soft.items():
        if item in gold_by_item:
            buckets[(cond, sub, fmt)].append((item, entry["soft"]))

    out: dict[str, dict] = {}
    for (cond, sub, fmt), rows in buckets.items():
        classes = TCM_CLASSES[sub]
        labels, preds, dropped = [], [], 0
        for item, model_soft in rows:
            model_no = gold_no = None
            if sub == "1.3":
                model_no = (det.get((cond, fmt, item)) or {}).get("not_sexist")
                gold_no = gold_by_item[item]["1.1"].get("not_sexist")
            gold_vec = _vectors(gold_by_item[item][sub], classes, gold_no)
            pred_vec = _vectors(model_soft, classes, model_no)
            if gold_vec.sum() <= 0 or pred_vec.sum() <= 0:
                dropped += 1
                continue
            labels.append(gold_vec)
            preds.append(pred_vec)
        if not labels:
            continue
        matrix = tcm_matrix(np.array(labels), np.array(preds), weighting)
        out["|".join((cond, sub, fmt))] = {
            "classes": classes,
            "weighting": weighting,
            "n_items": len(labels),
            "n_dropped": dropped,
            "matrix": matrix.tolist(),
        }
    return out
