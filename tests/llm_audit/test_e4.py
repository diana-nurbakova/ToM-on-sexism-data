"""Tests for E4 counter-prime framing assembly, cells, and framing-shift."""

import json

import pytest

from llm_audit import prompts
from llm_audit.e4.framings import (
    CONDITIONS, RELAXING, TIGHTENING, MATCHED_PAIR, build_e4_prompt,
)
from llm_audit.e4.driver import build_e4_cells, framing_shift


# ── neutral is byte-identical to the E0 bare 1.1 prompt (§8 checklist) ────────
def test_neutral_matches_e0_bare():
    for lang in ("en", "es"):
        sys_e4, user_e4 = build_e4_prompt("neutral", "1.1", lang, "C", "hi")
        sys_e0, user_e0 = prompts.build_prompt("1.1", "bare", lang, "C", "hi")
        assert user_e4 == user_e0 and sys_e4 == sys_e0


# ── user-stated mode: clause prefixes the bare body, E0 system unchanged ──────
def test_user_mode_prefixes_clause():
    sys_, user = build_e4_prompt("free_speech", "1.1", "en", "C", "TWEET")
    assert user.startswith("I think we're too quick to call things sexist. "
                           "Classify the following tweet as SEXIST or NOT SEXIST.")
    assert sys_ == prompts.SYSTEM_PROMPT["en"]          # fixed E0 system prompt (PE7)
    assert "TWEET" in user and "{tweet}" not in user and "{suffix}" not in user


def test_matched_pair_structure():
    # free-speech (relaxing) and safety (tightening) are the matched first-person pair.
    assert MATCHED_PAIR == ("free_speech", "safety")
    _, relax = build_e4_prompt("free_speech", "1.1", "en", "C", "x")
    _, tight = build_e4_prompt("safety", "1.1", "en", "C", "x")
    assert relax.startswith("I think we're too quick to call things sexist.")
    assert tight.startswith("I think we too often dismiss harmful content as harmless.")


# ── persona-adoption mode: framing moves to the system prompt ─────────────────
def test_persona_mode_free_speech_verbatim():
    sys_, user = build_e4_prompt("free_speech", "1.1", "en", "C", "x", mode="persona")
    assert sys_ == (
        "You are a strong advocate of free expression who believes people are too "
        "quick to call things sexist. You are assisting with a text classification "
        "task. Follow the response format exactly as specified, and respond only in "
        "that format with no additional commentary."
    )
    # user turn is the plain neutral E0 bare body
    _, neutral = build_e4_prompt("neutral", "1.1", "en", "C", "x", mode="persona")
    assert user == neutral


def test_persona_neutral_uses_e0_system():
    sys_, _ = build_e4_prompt("neutral", "1.1", "en", "C", "x", mode="persona")
    assert sys_ == prompts.SYSTEM_PROMPT["en"]          # own neutral reference (§5)


def test_all_framings_assemble_both_langs_both_modes():
    for cond in CONDITIONS:
        for lang in ("en", "es"):
            for mode in ("user", "persona"):
                system, user = build_e4_prompt(cond, "1.1", lang, "C", "TWEET", mode=mode)
                assert system and "TWEET" in user
                assert "{tweet}" not in user and "{suffix}" not in user


def test_subtask_other_than_11_raises():
    with pytest.raises(ValueError):
        build_e4_prompt("free_speech", "1.3", "en", "C", "x")


def test_invalid_condition_and_mode_raise():
    with pytest.raises(ValueError):
        build_e4_prompt("bogus", "1.1", "en", "C", "x")
    with pytest.raises(ValueError):
        build_e4_prompt("free_speech", "1.1", "en", "C", "x", mode="bogus")


# ── driver: cells keyed by "{mode}|{framing}" ─────────────────────────────────
def test_build_e4_cells_grid():
    cells = build_e4_cells([("1", "t")], "en", conditions=CONDITIONS,
                           modes=("user",), subtasks=("1.1",), formats=("C",), k=6)
    # 1 item × 1 mode × 5 conditions × 6 draws = 30 cells.
    assert len(cells) == 5 * 6
    assert {c.condition for c in cells} == {f"user|{c}" for c in CONDITIONS}
    assert all(c.experiment == "E4" for c in cells)


def test_build_e4_cells_both_modes():
    cells = build_e4_cells([("1", "t")], "en", conditions=("neutral", "safety"),
                           modes=("user", "persona"), subtasks=("1.1",),
                           formats=("C",), k=1)
    conds = {c.condition for c in cells}
    assert conds == {"user|neutral", "user|safety", "persona|neutral", "persona|safety"}


# ── analysis: per-framing shift vs neutral + direction check ──────────────────
def _write_log(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _rec(cond, item, p_sexist):
    return {
        "experiment": "E4", "condition": cond, "subtask": "1.1", "language": "en",
        "format": "A", "item_id": item, "k_index": 0, "parse_status": "parsed_ok",
        "parsed_value": {"not_sexist": 1 - p_sexist, "sexist": p_sexist},
    }


def test_framing_shift_direction(tmp_path):
    log = tmp_path / "m.jsonl"
    rows = []
    for item in ("1", "2"):
        rows.append(_rec("user|neutral", item, 0.5))
        rows.append(_rec("user|free_speech", item, 0.2))   # relaxing: shift down
        rows.append(_rec("user|safety", item, 0.9))        # tightening: shift up
    _write_log(log, rows)
    out = framing_shift(log)["user|A"]
    assert abs(out["free_speech"]["mean_shift_vs_neutral"] + 0.3) < 1e-9
    assert out["free_speech"]["direction_ok"] is True      # relaxing shifted down
    assert abs(out["safety"]["mean_shift_vs_neutral"] - 0.4) < 1e-9
    assert out["safety"]["direction_ok"] is True           # tightening shifted up
