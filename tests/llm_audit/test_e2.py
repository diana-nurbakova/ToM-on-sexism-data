"""Tests for E2 sampling, construct-rule tagging, driver cells, and flip metrics."""

import math

import pandas as pd
import pytest

from llm_audit.data import load_pool
from llm_audit.e1 import swap
from llm_audit.e2 import sampling, tagging, metrics as E2M
from llm_audit.e2.driver import build_e2_cells


# ── swap direction flags ──────────────────────────────────────────────────────
def test_swap_direction_flags():
    assert swap.swap_text("she is a woman").female_source
    assert not swap.swap_text("she is a woman").male_source
    assert swap.swap_text("he is a man").male_source
    assert swap.swap_text("the weather").female_source is False


# ── sampling ──────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def en_pool():
    return load_pool("en")


def test_build_e2_sample_equal_n_and_swappable(en_pool):
    s = sampling.build_e2_sample(en_pool, "en", target=120, seed=42)
    assert s["swappable"].all()                      # only swappable items
    counts = s.groupby("strat_category").size()
    assert (counts <= 120).all()                     # never exceed target
    assert set(s["swap_direction"]).issubset({"F2M", "M2F", "MIXED", "NONE"})


# ── construct-rule first-pass tagging ─────────────────────────────────────────
def test_construct_rule_tag_cases():
    assert tagging.construct_rule_tag(0.2, True, False, False)[0] == "INVARIANT"   # not sexist
    assert tagging.construct_rule_tag(0.9, True, False, False)[0] == "SHIFTING"    # gendered swap
    assert tagging.construct_rule_tag(0.9, False, True, False)[0] == "AMBIGUOUS"   # name only
    assert tagging.construct_rule_tag(0.9, False, False, True)[0] == "UNSWAPPABLE"
    # SEXIST tags are flagged for author review; not-sexist/unswappable are not.
    assert tagging.construct_rule_tag(0.9, True, False, False)[1] is True
    assert tagging.construct_rule_tag(0.2, True, False, False)[1] is False


# ── driver cell expansion ─────────────────────────────────────────────────────
def test_build_e2_cells_orig_and_swap_variants():
    df = pd.DataFrame([{"item_id": "1", "text": "she is a woman", "swapped": "he is a man"}])
    cells = build_e2_cells(df, "en", subtasks=("1.1",), specs=("defined",), formats=("C",), k=2)
    # 1 item × 2 variants × 1 subtask × 1 spec × 2 draws = 4 cells.
    assert len(cells) == 4
    conds = {c.condition for c in cells}
    assert conds == {"defined|orig", "defined|swap"}
    orig = [c for c in cells if c.condition == "defined|orig"][0]
    swp = [c for c in cells if c.condition == "defined|swap"][0]
    assert orig.text == "she is a woman" and swp.text == "he is a man"


# ── flip metrics ──────────────────────────────────────────────────────────────
def test_pair_and_flip_metrics():
    soft = {
        ("defined|orig", "1.1", "C", "1"): {"soft": {"sexist": 1.0, "not_sexist": 0.0}, "n_valid": 6},
        ("defined|swap", "1.1", "C", "1"): {"soft": {"sexist": 0.0, "not_sexist": 1.0}, "n_valid": 6},
        ("defined|orig", "1.1", "C", "2"): {"soft": {"sexist": 1.0, "not_sexist": 0.0}, "n_valid": 6},
        ("defined|swap", "1.1", "C", "2"): {"soft": {"sexist": 1.0, "not_sexist": 0.0}, "n_valid": 6},
    }
    paired = E2M.pair_orig_swap(soft)
    assert len(paired) == 2
    meta = {"1": {"category": "objectification", "direction": "F2M", "invariance": "SHIFTING"},
            "2": {"category": "objectification", "direction": "M2F", "invariance": "INVARIANT"}}
    res = E2M.flip_metrics(paired, meta)["defined|1.1|C"]
    assert math.isclose(res["flip_rate"], 0.5)        # item 1 flips, item 2 doesn't
    assert res["flip_F2M"] == 1.0 and res["flip_M2F"] == 0.0
    assert math.isclose(res["asymmetry"], 1.0)
    # signed directional shift: item 1 (F2M) sexist 1->0 = -1; item 2 (M2F) 1->1 = 0
    assert math.isclose(res["shift_by_direction"]["F2M"], -1.0)
    assert math.isclose(res["shift_by_direction"]["M2F"], 0.0)
