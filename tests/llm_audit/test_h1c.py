"""Tests for H1c persona system-prompt assembly, cells, and gender-shift."""

import json

import pytest

from llm_audit import prompts
from llm_audit.h1c.persona import (
    PERSONAS, build_persona_prompt, persona_system_prompt, _FORMAT_SENTENCE,
)
from llm_audit.h1c.driver import build_h1c_cells, gender_shift


def test_six_personas():
    assert len(PERSONAS) == 6
    assert set(PERSONAS) == {"F-young", "F-mid", "F-older", "M-young", "M-mid", "M-older"}


# ── persona lives in the system prompt; user turn is the unchanged E0 base ────
def test_user_turn_is_e0_bare_for_11():
    system, user = build_persona_prompt("F-mid", "1.1", "en", "C", "hi")
    _, e0_user = prompts.build_prompt("1.1", "bare", "en", "C", "hi")
    assert user == e0_user                              # user turn unchanged (§3.4)
    assert system.startswith("You are a woman aged 23-45")


def test_user_turn_is_e0_defined_for_13():
    _, user = build_persona_prompt("M-older", "1.3", "en", "C", "hi")
    _, e0_user = prompts.build_prompt("1.3", "defined", "en", "C", "hi")
    assert user == e0_user                              # 1.3 uses the defined base


def test_fmid_mmid_worked_examples():
    assert persona_system_prompt("F-mid", "en") == (
        "You are a woman aged 23-45, annotating tweets for a research study. Label "
        "each tweet as you would. Follow the response format exactly as specified, "
        "and respond only in that format with no additional commentary."
    )
    assert persona_system_prompt("M-mid", "es") == (
        "Usted es un hombre de 23 a 45 años, y está anotando tuits para un estudio de "
        "investigación. Etiquete cada tuit como usted lo haría. Siga exactamente el "
        "formato de respuesta indicado y responda únicamente en ese formato, sin "
        "comentarios adicionales."
    )


# ── the format-instruction sentence is byte-identical to E0 (§5 checklist) ────
def test_format_sentence_byte_identical_to_e0():
    for lang in ("en", "es"):
        assert prompts.SYSTEM_PROMPT[lang].endswith(_FORMAT_SENTENCE[lang])
        for persona in PERSONAS:
            assert persona_system_prompt(persona, lang).endswith(_FORMAT_SENTENCE[lang])


def test_all_personas_assemble_both_langs():
    for persona in PERSONAS:
        for lang in ("en", "es"):
            for subtask in ("1.1", "1.3"):
                system, user = build_persona_prompt(persona, subtask, lang, "C", "TWEET")
                assert "TWEET" in user and "{tweet}" not in user and "{suffix}" not in user
                # only the identity sentence varies; gender word present
                assert ("woman" in system or "man" in system
                        or "mujer" in system or "hombre" in system)


def test_invalid_persona_and_subtask_raise():
    with pytest.raises(ValueError):
        persona_system_prompt("X-mid", "en")
    with pytest.raises(ValueError):
        build_persona_prompt("F-mid", "1.2", "en", "C", "x")


# ── driver: cells ─────────────────────────────────────────────────────────────
def test_build_h1c_cells_grid():
    cells = build_h1c_cells([("1", "t")], "en", personas=PERSONAS,
                            subtasks=("1.1", "1.3"), formats=("C",), k=6)
    # 1 item × 6 personas × 2 subtasks × 6 draws = 72 cells.
    assert len(cells) == 6 * 2 * 6
    assert all(c.experiment == "H1c" for c in cells)
    assert {c.condition for c in cells} == set(PERSONAS)


# ── analysis: female-minus-male shift per band ────────────────────────────────
def _write_log(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _rec_11(persona, item, p_sexist):
    return {
        "experiment": "H1c", "condition": persona, "subtask": "1.1", "language": "en",
        "format": "A", "item_id": item, "k_index": 0, "parse_status": "parsed_ok",
        "parsed_value": {"not_sexist": 1 - p_sexist, "sexist": p_sexist},
    }


def _rec_13(persona, item, cats):
    return {
        "experiment": "H1c", "condition": persona, "subtask": "1.3", "language": "en",
        "format": "A", "item_id": item, "k_index": 0, "parse_status": "parsed_ok",
        "parsed_value": cats,
    }


def test_gender_shift_11(tmp_path):
    log = tmp_path / "m.jsonl"
    rows = [_rec_11("F-mid", "1", 0.7), _rec_11("M-mid", "1", 0.4),
            _rec_11("F-mid", "2", 0.6), _rec_11("M-mid", "2", 0.5)]
    _write_log(log, rows)
    out = gender_shift(log)["1.1|A"]["mid"]
    # mean female-minus-male = ((0.7-0.4)+(0.6-0.5))/2 = 0.2
    assert abs(out["shift_p_sexist"] - 0.2) < 1e-9
    assert out["n_paired"] == 2


def test_gender_shift_13_per_category(tmp_path):
    log = tmp_path / "m.jsonl"
    rows = [
        _rec_13("F-mid", "1", {"misogyny_non_sexual_violence": 0.8, "ideological_inequality": 0.3}),
        _rec_13("M-mid", "1", {"misogyny_non_sexual_violence": 0.5, "ideological_inequality": 0.3}),
    ]
    _write_log(log, rows)
    cats = gender_shift(log)["1.3|A"]["mid"]["shift_per_category"]
    assert abs(cats["misogyny_non_sexual_violence"] - 0.3) < 1e-9   # affective shift
    assert abs(cats["ideological_inequality"] - 0.0) < 1e-9         # cognitive flat
