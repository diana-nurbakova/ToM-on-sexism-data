"""Tests for E5 Cialdini condition assembly, exemplar pairing, cells, and shift."""

import json

import pytest

from llm_audit import prompts
from llm_audit.e5 import exemplars
from llm_audit.e5.conditions import (
    CONDITIONS, AUTHORITY, SOCIALPROOF, COMMITMENT, build_e5_prompt,
)
from llm_audit.e5.driver import build_e5_cells, condition_shift, make_e5_prompt_fn
from llm_audit.runner import Cell


# ── condition inventory ───────────────────────────────────────────────────────
def test_thirteen_conditions():
    assert len(CONDITIONS) == 13
    assert CONDITIONS[0] == "neutral"
    assert len(AUTHORITY) == 4 and len(SOCIALPROOF) == 6 and len(COMMITMENT) == 2


# ── neutral is the E0 bare prompt exactly (harmonised base, §5.6 + PE2) ───────
def test_neutral_matches_e0_bare():
    sys_e5, user_e5 = build_e5_prompt("neutral", "1.1", "en", "C", "hi")
    sys_e0, user_e0 = prompts.build_prompt("1.1", "bare", "en", "C", "hi")
    assert user_e5 == user_e0 and sys_e5 == sys_e0


# ── authority: third-party verdict prepended to the bare body ─────────────────
def test_authority_inline_framing():
    _, user = build_e5_prompt("authority_pro_sexist", "1.1", "en", "C", "TWEET")
    assert user.startswith(
        "An expert researcher in gender studies has classified the following tweet "
        "as sexist. Classify the following tweet as SEXIST or NOT SEXIST."
    )
    assert "TWEET" in user and "{tweet}" not in user and "{suffix}" not in user
    # anti-detection stance + not-sexist verdict
    _, anti = build_e5_prompt("authority_anti_not", "1.1", "en", "C", "x")
    assert anti.startswith(
        "A legal scholar specialising in freedom of expression has classified the "
        "following tweet as not sexist."
    )


# ── social proof: intensity n and direction surface correctly ─────────────────
def test_socialproof_intensity_and_direction():
    _, u5 = build_e5_prompt("socialproof_sexist_5", "1.1", "en", "C", "x")
    assert u5.startswith("Of 6 annotators who reviewed this tweet, 5 classified it as sexist.")
    _, u4 = build_e5_prompt("socialproof_not_4", "1.1", "es", "C", "x")
    assert u4.startswith("De 6 personas anotadoras que revisaron este tuit, 4 lo "
                         "clasificaron como no sexista.")


# ── commitment: requires an exemplar; assembles the labelled-example block ────
def test_commitment_requires_exemplar():
    with pytest.raises(ValueError):
        build_e5_prompt("commitment_sexist", "1.1", "en", "C", "x")


def test_commitment_block_then_bare_body():
    _, user = build_e5_prompt("commitment_sexist", "1.1", "en", "C", "TWEET",
                              exemplar="a clearly sexist example")
    assert user.startswith('Consider this example:\n\nExample: "a clearly sexist example"\n'
                           'Classification: sexist\n\n')
    # the bare body follows the block
    assert "Classify the following tweet as SEXIST or NOT SEXIST." in user
    _, notu = build_e5_prompt("commitment_not", "1.1", "es", "C", "x",
                              exemplar="ejemplo no sexista")
    assert notu.startswith('Considere este ejemplo:\n\nEjemplo: "ejemplo no sexista"\n'
                           'Clasificación: no sexista\n\n')


# ── every condition assembles in both languages with no leftover placeholders ─
def test_all_conditions_assemble_both_langs():
    for cond in CONDITIONS:
        for lang in ("en", "es"):
            ex = "x" if cond in COMMITMENT else None
            system, user = build_e5_prompt(cond, "1.1", lang, "C", "TWEET", exemplar=ex)
            assert system and "TWEET" in user
            assert "{tweet}" not in user and "{suffix}" not in user


def test_invalid_condition_raises():
    with pytest.raises(ValueError):
        build_e5_prompt("bogus", "1.1", "en", "C", "x")


# ── exemplar pools + pairing ──────────────────────────────────────────────────
def test_build_pairing_rotation():
    pools = {
        "sexist": [{"item_id": "s1", "text": "S1"}, {"item_id": "s2", "text": "S2"}],
        "not_sexist": [{"item_id": "n1", "text": "N1"}],
    }
    pairing = exemplars.build_pairing(["a", "b", "c"], pools)
    assert pairing["a"]["sexist"] == "S1" and pairing["b"]["sexist"] == "S2"
    assert pairing["c"]["sexist"] == "S1"            # rotation wraps
    assert pairing["a"]["not_sexist"] == "N1"        # single-item pool reused


def test_select_pools_unanimous_only():
    import pandas as pd
    df = pd.DataFrame([
        {"item_id": "1", "lang": "en", "text": "all-yes", "gold_1_1": {"sexist": 1.0, "not_sexist": 0.0}},
        {"item_id": "2", "lang": "en", "text": "all-no", "gold_1_1": {"sexist": 0.0, "not_sexist": 1.0}},
        {"item_id": "3", "lang": "en", "text": "split", "gold_1_1": {"sexist": 0.5, "not_sexist": 0.5}},
    ])
    pools = exemplars.select_pools(df, "en")
    assert [e["item_id"] for e in pools["sexist"]] == ["1"]
    assert [e["item_id"] for e in pools["not_sexist"]] == ["2"]


# ── driver: cells + prompt hook ───────────────────────────────────────────────
def test_build_e5_cells_grid():
    cells = build_e5_cells([("1", "t")], "en", conditions=CONDITIONS,
                           subtasks=("1.1",), formats=("C",), k=6)
    # 1 item × 13 conditions × 1 subtask × 6 draws = 78 cells.
    assert len(cells) == 13 * 6
    assert all(c.experiment == "E5" for c in cells)
    assert {c.condition for c in cells} == set(CONDITIONS)


def test_prompt_fn_injects_exemplar():
    pairing = {"1": {"sexist": "EX-S", "not_sexist": "EX-N"}}
    fn = make_e5_prompt_fn(pairing)
    cell = Cell("E5", "commitment_sexist", "1.1", "en", "C", "1", "tweet", 0)
    _, user = fn(cell)
    assert "EX-S" in user


# ── analysis: per-condition shift vs neutral ──────────────────────────────────
def _write_log(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _rec(cond, item, p_sexist):
    return {
        "experiment": "E5", "condition": cond, "subtask": "1.1", "language": "en",
        "format": "A", "item_id": item, "k_index": 0, "parse_status": "parsed_ok",
        "parsed_value": {"not_sexist": 1 - p_sexist, "sexist": p_sexist},
    }


def test_condition_shift_paired(tmp_path):
    log = tmp_path / "m.jsonl"
    rows = []
    for item in ("1", "2"):
        rows.append(_rec("neutral", item, 0.2))
        rows.append(_rec("authority_pro_sexist", item, 0.8))
    _write_log(log, rows)
    out = condition_shift(log)["1.1|A"]
    assert out["neutral"]["mean_shift_vs_neutral"] == 0.0
    assert abs(out["authority_pro_sexist"]["mean_shift_vs_neutral"] - 0.6) < 1e-9
    assert out["authority_pro_sexist"]["n_paired"] == 2


def test_condition_shift_majority_split(tmp_path):
    log = tmp_path / "m.jsonl"
    # item 1 gold majority = sexist; condition raises P(sexist) -> toward majority.
    rows = [_rec("neutral", "1", 0.3), _rec("authority_pro_sexist", "1", 0.9)]
    _write_log(log, rows)
    gold = {"1": {"1.1": {"sexist": 1.0, "not_sexist": 0.0}}}
    out = condition_shift(log, gold)["1.1|A"]["authority_pro_sexist"]
    assert out["n_toward"] == 1 and out["n_against"] == 0
    assert abs(out["toward_majority_mean_abs_shift"] - 0.6) < 1e-9
