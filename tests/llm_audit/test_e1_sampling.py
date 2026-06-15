"""Tests for E1 bin-first stratified sampling."""

import pytest

from llm_audit.data import load_pool
from llm_audit.e1 import sampling


@pytest.fixture(scope="module")
def en_pool():
    return load_pool("en")


def test_sample_size_and_grid(en_pool):
    s = sampling.build_sample(en_pool, "en", seed=42)
    assert 0 < len(s) <= sampling.TARGET_TOTAL
    # Every sampled item has a category and a bin in the grid.
    assert s["strat_category"].notna().all()
    assert set(s["agreement_bin"]).issubset(set(sampling.BINS))
    # Per-cell counts never exceed the target.
    counts = sampling.cell_report(s)
    assert (counts["n"] <= s.attrs["per_cell_target"]).all()


def test_sampling_is_deterministic(en_pool):
    a = sampling.build_sample(en_pool, "en", seed=42)
    b = sampling.build_sample(en_pool, "en", seed=42)
    assert list(a["item_id"]) == list(b["item_id"])


def test_disjoint_from_excluded_ids(en_pool):
    s_full = sampling.build_sample(en_pool, "en", seed=42)
    hold = set(s_full["item_id"].iloc[:10])
    s = sampling.build_sample(en_pool, "en", exclude_ids=hold, seed=42)
    assert not (set(s["item_id"]) & hold)


def test_shortfalls_recorded(en_pool):
    s = sampling.build_sample(en_pool, "en", seed=42)
    assert isinstance(s.attrs["shortfalls"], list)  # may be empty or populated
