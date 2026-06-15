"""Tests for the regression-set builder and noise-band estimation."""

import math

import pytest

from llm_audit import regression
from llm_audit.data import load_pool


@pytest.fixture(scope="module")
def en_pool():
    return load_pool("en")


def test_regression_set_has_expected_slots(en_pool):
    items = regression.build_regression_set(en_pool, "en", seed=42)
    slots = [it["slot_type"] for it in items]
    assert sum(s == "clear_sexist" for s in slots) >= 1
    assert sum(s == "clear_not" for s in slots) == 3
    assert sum(s == "hard" for s in slots) == 3
    # All five category slots present.
    for cat in ("ideological_inequality", "stereotyping_dominance", "objectification",
                "sexual_violence", "misogyny_non_sexual_violence"):
        assert f"category_{cat}" in slots
    # Definition-sensitive items are flagged for author review.
    defs = [it for it in items if it["slot_type"] == "definition_sensitive"]
    assert defs and all(it.get("needs_review") for it in defs)


def test_regression_items_are_disjoint(en_pool):
    items = regression.build_regression_set(en_pool, "en", seed=42)
    ids = [it["exist_item_id"] for it in items]
    assert len(ids) == len(set(ids))  # no item reused across slots


def test_regression_items_carry_gold(en_pool):
    items = regression.build_regression_set(en_pool, "en", seed=42)
    it = items[0]
    assert set(it["gold_detect_dist"]) == {"sexist", "not_sexist"}
    assert math.isclose(sum(it["gold_detect_dist"].values()), 1.0)


def test_soft_labels_per_repeat_chunks_draws():
    draws = ["sexist"] * 6 + ["not_sexist"] * 6  # 2 repeats of K=6
    reps = regression.soft_labels_per_repeat(draws, "1.1", k=6)
    assert len(reps) == 2
    assert reps[0]["sexist"] == 1.0 and reps[1]["not_sexist"] == 1.0


def test_estimate_noise_band_zero_for_identical_repeats():
    repeats = {"100": [{"sexist": 0.5, "not_sexist": 0.5}] * 4}
    band = regression.estimate_noise_band(repeats, "1.1", B=100, seed=1)
    assert band["set_mean"] == 0.0
    assert band["upper_band"] == 0.0
