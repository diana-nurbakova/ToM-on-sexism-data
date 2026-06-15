"""Tests for E1-H1/H2/H3 analysis."""

import math

from llm_audit.e1 import analysis as A


def test_category_grouping():
    assert A.category_grouping("objectification") == "affective"
    assert A.category_grouping("ideological_inequality") == "cognitive"
    assert A.category_grouping("not_a_category") is None


def test_shifting_proportion_all_shifting():
    res = A.shifting_proportion_ci(["SHIFTING"] * 30, B=200, seed=1)
    assert res["p_shift"] == 1.0 and res["clears_10pct"]


def test_shifting_proportion_empty():
    res = A.shifting_proportion_ci([], B=10)
    assert res["n"] == 0 and res["p_shift"] is None


def test_shifting_proportion_includes_ambiguous_in_denominator():
    # 5 shifting, 5 invariant, 10 ambiguous -> p = 5/20 = 0.25
    classes = ["SHIFTING"] * 5 + ["INVARIANT"] * 5 + ["AMBIGUOUS"] * 10
    res = A.shifting_proportion_ci(classes, B=500, seed=2)
    assert math.isclose(res["p_shift"], 0.25)


def test_h2_association_affective_enriched():
    # affective heavily SHIFTING, cognitive heavily INVARIANT -> OR > 1, p_one small
    pairs = ([("SHIFTING", "objectification")] * 10
             + [("INVARIANT", "objectification")] * 2
             + [("INVARIANT", "ideological_inequality")] * 10
             + [("SHIFTING", "ideological_inequality")] * 2)
    res = A.h2_association(pairs)
    assert res["odds_ratio"] > 1
    assert res["p_one_sided_affective_enriched"] < 0.05
    assert not math.isnan(res["cramers_v"])


def test_cohen_kappa_perfect_and_partial():
    assert A.cohen_kappa(["A", "B", "C"], ["A", "B", "C"]) == 1.0
    k = A.cohen_kappa(["A", "A", "B", "B"], ["A", "B", "A", "B"])
    assert -1.0 <= k <= 1.0


def test_judge_agreement_confusion_and_kappa():
    partition = {"1": "INVARIANT", "2": "SHIFTING", "3": "AMBIGUOUS", "4": "SHIFTING"}
    verdicts = {"1": "INVARIANT", "2": "SHIFTING", "3": "INVARIANT"}  # item 4 missing
    res = A.judge_agreement(partition, verdicts)
    assert res["n"] == 3  # available-case
    assert res["confusion"]["SHIFTING"]["SHIFTING"] == 1
    assert res["confusion"]["AMBIGUOUS"]["INVARIANT"] == 1
    assert math.isclose(res["percent_agreement"], 2 / 3)
