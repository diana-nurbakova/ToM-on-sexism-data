"""Tests for gold soft-label derivation and stratification helpers."""

import math

from llm_audit.data import (
    soft_label_1_1, soft_label_1_2, soft_label_1_3,
    agreement_bin, stratification_category,
)


def test_soft_label_1_1_proportion():
    t11 = ["YES", "YES", "NO", "YES", "YES", "YES"]  # 5/6 yes
    soft = soft_label_1_1(t11)
    assert math.isclose(soft["sexist"], 5 / 6)
    assert math.isclose(soft["not_sexist"], 1 / 6)


def test_soft_label_1_2_no_annotator_maps_to_not_sexist():
    t11 = ["YES", "YES", "NO", "YES", "YES", "YES"]
    t12 = ["REPORTED", "JUDGEMENTAL", "-", "REPORTED", "JUDGEMENTAL", "REPORTED"]
    soft = soft_label_1_2(t11, t12)
    # 3 reported, 2 judgemental, 1 not_sexist (the NO annotator's '-').
    assert math.isclose(soft["reported"], 3 / 6)
    assert math.isclose(soft["judgemental"], 2 / 6)
    assert math.isclose(soft["not_sexist"], 1 / 6)
    assert math.isclose(soft["direct"], 0.0)
    assert math.isclose(sum(soft.values()), 1.0)


def test_soft_label_1_3_multilabel_proportions_over_six():
    t11 = ["YES", "YES", "NO", "YES", "YES", "YES"]
    t13 = [
        ["OBJECTIFICATION"],
        ["OBJECTIFICATION", "SEXUAL-VIOLENCE"],
        ["-"],
        ["STEREOTYPING-DOMINANCE"],
        ["SEXUAL-VIOLENCE"],
        ["IDEOLOGICAL-INEQUALITY", "MISOGYNY-NON-SEXUAL-VIOLENCE"],
    ]
    soft = soft_label_1_3(t11, t13)
    assert math.isclose(soft["objectification"], 2 / 6)
    assert math.isclose(soft["sexual_violence"], 2 / 6)
    assert math.isclose(soft["stereotyping_dominance"], 1 / 6)
    assert math.isclose(soft["ideological_inequality"], 1 / 6)
    assert math.isclose(soft["misogyny_non_sexual_violence"], 1 / 6)


def test_agreement_bins():
    assert agreement_bin(["YES"] * 6) == "high"            # 6-0
    assert agreement_bin(["YES"] * 5 + ["NO"]) == "high"   # 5-1
    assert agreement_bin(["YES"] * 4 + ["NO"] * 2) == "mixed"  # 4-2
    assert agreement_bin(["YES"] * 3 + ["NO"] * 3) == "low"    # 3-3


def test_stratification_category_tiebreak_and_none():
    t11 = ["YES", "YES", "NO", "YES", "YES", "YES"]
    t13 = [
        ["OBJECTIFICATION"], ["OBJECTIFICATION", "SEXUAL-VIOLENCE"], ["-"],
        ["STEREOTYPING-DOMINANCE"], ["SEXUAL-VIOLENCE"],
        ["IDEOLOGICAL-INEQUALITY", "MISOGYNY-NON-SEXUAL-VIOLENCE"],
    ]
    # objectification and sexual_violence tie at 2/6; tie-break order puts
    # objectification ahead of sexual_violence.
    assert stratification_category(t11, t13) == "objectification"
    # not-sexist item -> no category.
    assert stratification_category(["NO"] * 6, [["-"]] * 6) is None
