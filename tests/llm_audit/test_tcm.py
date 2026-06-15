"""Tests for the Transport-based Confusion Matrix (master spec §7.1).

Cover the per-instance contribution against hand-computed examples, the
single-label reduction to an ordinary confusion matrix, the three weightings, the
accumulation API, and the EXIST builder. A self-contained reference port of the
published algorithm cross-checks the implementation on random data.
"""

import numpy as np
import pytest

from llm_audit import tcm


# ── Reference algorithm (faithful to github.com/johan140391/TCM) ──────────────
def _ref_contribution(y, yhat, weighting):
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    y1, yhat1 = y / y.sum(), yhat / yhat.sum()
    m = np.minimum(y1, yhat1)
    diag = np.diag(m)
    off = (np.zeros_like(diag) if np.allclose(y1, yhat1)
           else np.outer(y1 - m, yhat1 - m) / (y1 - m).sum())
    w = {"TCMone": 1.0, "TCMlab": y.sum(), "TCMpred": yhat.sum()}[weighting]
    return w * (diag + off)


def test_matches_reference_on_random_data():
    rng = np.random.default_rng(0)
    n, c = 40, 5
    labels = rng.random((n, c))
    preds = rng.random((n, c))
    # plant a few perfect matches (the off-diagonal == 0 branch)
    preds[[3, 7, 19]] = labels[[3, 7, 19]]
    for w in tcm.WEIGHTINGS:
        ref = sum(_ref_contribution(labels[i], preds[i], w) for i in range(n))
        got = tcm.tcm_matrix(labels, preds, w)
        assert np.allclose(got, ref)


# ── Hand examples ─────────────────────────────────────────────────────────────
def test_perfect_match_is_diagonal_only():
    y = np.array([0.5, 0.5])
    M = tcm._contribution(y, y.copy(), "TCMone")
    assert np.allclose(M, np.diag([0.5, 0.5]))
    assert M[0, 1] == 0.0 and M[1, 0] == 0.0


def test_full_misclassification_transports_off_diagonal():
    # true class 0, predicted class 1 -> all mass at [0, 1] (rows=true, cols=pred).
    M = tcm._contribution([1.0, 0.0], [0.0, 1.0], "TCMone")
    assert np.allclose(M, [[0.0, 1.0], [0.0, 0.0]])


def test_single_label_reduces_to_confusion_matrix():
    # one-hot rows + TCMone == the ordinary count confusion matrix.
    labels = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0]], float)
    preds = np.array([[1, 0, 0], [0, 1, 0], [0, 1, 0]], float)
    M = tcm.tcm_matrix(labels, preds, "TCMone")
    assert np.allclose(M, [[1, 1, 0], [0, 1, 0], [0, 0, 0]])


def test_off_diagonal_column_sums_match_residual_prediction():
    y = np.array([0.7, 0.2, 0.1])
    yhat = np.array([0.1, 0.6, 0.3])
    M = tcm._contribution(y, yhat, "TCMone")
    common = np.minimum(y, yhat)
    resid_pred = yhat - common
    off = M - np.diag(np.diag(M))
    assert np.allclose(off.sum(axis=0), resid_pred)  # cols carry predicted residual


# ── Weighting ─────────────────────────────────────────────────────────────────
def test_weightings_scale_contribution():
    y = np.array([2.0, 0.0])       # pre-norm label mass 2
    yhat = np.array([0.0, 3.0])    # pre-norm pred mass 3
    one = tcm._contribution(y, yhat, "TCMone")
    lab = tcm._contribution(y, yhat, "TCMlab")
    pred = tcm._contribution(y, yhat, "TCMpred")
    assert np.allclose(lab, 2.0 * one)
    assert np.allclose(pred, 3.0 * one)


def test_zero_mass_row_raises():
    with pytest.raises(ValueError):
        tcm._contribution([0.0, 0.0], [1.0, 0.0], "TCMone")


def test_bad_weighting_rejected():
    with pytest.raises(ValueError):
        tcm.TCM(2, "nope")


# ── Accumulation API ──────────────────────────────────────────────────────────
def test_update_accumulates_across_batches():
    m = tcm.TCM(2, "TCMone")
    m.update([[1.0, 0.0]], [[0.0, 1.0]])
    m.update([[1.0, 0.0]], [[0.0, 1.0]])
    assert np.allclose(m.get(), [[0.0, 2.0], [0.0, 0.0]])


def test_update_shape_check():
    with pytest.raises(ValueError):
        tcm.TCM(3, "TCMone").update([[1.0, 0.0]], [[1.0, 0.0]])


def test_row_normalise_handles_zero_rows():
    M = np.array([[2.0, 2.0], [0.0, 0.0]])
    out = tcm.row_normalise(M)
    assert np.allclose(out, [[0.5, 0.5], [0.0, 0.0]])


# ── EXIST builder ─────────────────────────────────────────────────────────────
def test_tcm_for_exist_shapes_and_1_3_no_threading():
    soft = {
        ("bare", "1.1", "C", "I1"): {"soft": {"not_sexist": 0.25, "sexist": 0.75}},
        ("bare", "1.3", "C", "I1"): {"soft": {
            "ideological_inequality": 0.0, "stereotyping_dominance": 0.0,
            "objectification": 0.5, "sexual_violence": 0.0,
            "misogyny_non_sexual_violence": 0.0}},
    }
    gold = {"I1": {
        "1.1": {"not_sexist": 0.5, "sexist": 0.5},
        "1.2": {"not_sexist": 0.5, "direct": 0.5, "reported": 0.0, "judgemental": 0.0},
        "1.3": {"ideological_inequality": 0.0, "stereotyping_dominance": 0.0,
                "objectification": 0.5, "sexual_violence": 0.0,
                "misogyny_non_sexual_violence": 0.0},
    }}
    out = tcm.tcm_for_exist(soft, gold, "TCMlab")
    assert set(out) == {"bare|1.1|C", "bare|1.3|C"}
    m13 = out["bare|1.3|C"]
    assert m13["classes"][0] == "not_sexist"           # NO prepended for 1.3
    assert len(m13["matrix"]) == 6 and len(m13["matrix"][0]) == 6
    assert m13["n_items"] == 1 and m13["n_dropped"] == 0
    # 1.1 is a flat 2x2.
    assert len(out["bare|1.1|C"]["matrix"]) == 2
