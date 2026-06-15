"""Tests for the E3 / E4 / E5 / H1c sub-sample samplers (master spec §5.4-5.6).

E4 and H1c reuse the E3 500-item set; E5 is the 200-item category × invariance grid
overlapping with E1. Shared structural checks (grid membership, determinism,
disjointness, no-cell-overflow, shortfall recording) mirror ``test_e1_sampling``.
"""

import pytest

from llm_audit.data import load_pool
from llm_audit.e1.rule import INVARIANCE_CLASSES
from llm_audit.e3 import sampling as e3s
from llm_audit.e4 import sampling as e4s
from llm_audit.e5 import sampling as e5s
from llm_audit.h1c import sampling as h1cs


@pytest.fixture(scope="module")
def en_pool():
    return load_pool("en")


# ── E3: bin-first 500 ─────────────────────────────────────────────────────────
def test_e3_grid_and_no_overflow(en_pool):
    s = e3s.build_e3_sample(en_pool, "en", seed=42)
    assert 0 < len(s) <= e3s.TARGET_TOTAL
    assert s["strat_category"].notna().all()
    assert set(s["agreement_bin"]).issubset(set(e3s.BINS))
    counts = e3s.cell_report(s)
    assert (counts["n"] <= s.attrs["per_cell_target"]).all()
    assert isinstance(s.attrs["shortfalls"], list)


def test_e3_deterministic(en_pool):
    a = e3s.build_e3_sample(en_pool, "en", seed=42)
    b = e3s.build_e3_sample(en_pool, "en", seed=42)
    assert list(a["item_id"]) == list(b["item_id"])


def test_e3_disjoint_from_excluded(en_pool):
    full = e3s.build_e3_sample(en_pool, "en", seed=42)
    hold = set(full["item_id"].iloc[:20])
    s = e3s.build_e3_sample(en_pool, "en", exclude_ids=hold, seed=42)
    assert not (set(s["item_id"]) & hold)


def test_e3_no_duplicate_items(en_pool):
    s = e3s.build_e3_sample(en_pool, "en", seed=42)
    assert s["item_id"].is_unique


# ── E4 / H1c: full overlap with E3 ────────────────────────────────────────────
def test_e4_is_e3_set(en_pool):
    e3 = e3s.build_e3_sample(en_pool, "en", seed=42)
    e4 = e4s.build_e4_sample(en_pool, "en", seed=42)
    assert list(e4["item_id"]) == list(e3["item_id"])


def test_h1c_is_e3_set(en_pool):
    e3 = e3s.build_e3_sample(en_pool, "en", seed=42)
    h1c = h1cs.build_h1c_sample(en_pool, "en", seed=42)
    assert list(h1c["item_id"]) == list(e3["item_id"])


# ── E5: 200, category × invariance, E1 overlap ────────────────────────────────
def test_e5_grid_and_no_overflow(en_pool):
    s = e5s.build_e5_sample(en_pool, "en", seed=42)
    assert 0 < len(s) <= e5s.TARGET_TOTAL
    assert s["strat_category"].notna().all()
    assert s["swappable"].all()                              # UNSWAPPABLE excluded
    assert set(s["invariance_class"]).issubset(set(INVARIANCE_CLASSES))
    counts = e5s.cell_report(s)
    assert (counts["n"] <= s.attrs["per_cell_target"]).all()


def test_e5_deterministic(en_pool):
    a = e5s.build_e5_sample(en_pool, "en", seed=42)
    b = e5s.build_e5_sample(en_pool, "en", seed=42)
    assert list(a["item_id"]) == list(b["item_id"])


def test_e5_prefers_e1_overlap(en_pool):
    # Pick a set of real swappable, categorised E1-like ids to prefer.
    base = e5s.build_e5_sample(en_pool, "en", seed=42)
    prefer = set(base["item_id"].iloc[:30])
    s = e5s.build_e5_sample(en_pool, "en", e1_ids=prefer, seed=42)
    # Every preferred item that survives screening + lands in its cell is flagged.
    assert s.loc[s["item_id"].isin(prefer), "in_e1"].all()
    # Overlap preference can only increase (or hold) the count of preferred items kept.
    assert len(set(s["item_id"]) & prefer) >= 1


def test_e5_partition_override(en_pool):
    s = e5s.build_e5_sample(en_pool, "en", seed=42)
    # Force one sampled item to a chosen final class; it must be honoured.
    iid = str(s["item_id"].iloc[0])
    forced = e5s.build_e5_sample(en_pool, "en", e1_partition={iid: "AMBIGUOUS"}, seed=42)
    row = forced[forced["item_id"] == iid]
    if not row.empty:                                        # still selected
        assert row["invariance_class"].iloc[0] == "AMBIGUOUS"
