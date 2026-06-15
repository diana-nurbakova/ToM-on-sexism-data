"""Tests for E3 scaffold prompt assembly and driver cells."""

import pytest

from llm_audit.e3 import scaffolds
from llm_audit.e3.driver import build_e3_cells
from llm_audit.e3.scaffolds import build_scaffold_prompt, CONDITIONS


def test_neutral_11_matches_e0_bare():
    from llm_audit import prompts
    _, e3 = build_scaffold_prompt("neutral", "1.1", "en", "C", "hi")
    _, e0 = prompts.build_prompt("1.1", "bare", "en", "C", "hi")
    assert e3 == e0  # neutral 1.1 is exactly the E0 bare prompt


def test_all_11_conditions_assemble_without_placeholders():
    for cond in CONDITIONS:
        for lang in ("en", "es"):
            system, user = build_scaffold_prompt(cond, "1.1", lang, "C", "TWEET")
            assert "TWEET" in user and "{tweet}" not in user and "{suffix}" not in user
            assert system  # fixed E0 system prompt


def test_cognitive_and_affective_clause_content():
    _, cog = build_scaffold_prompt("cognitive", "1.1", "en", "C", "x")
    assert "author's intent" in cog
    _, aff = build_scaffold_prompt("affective", "1.1", "en", "C", "x")
    # affective clause is target-neutral (PE10): must NOT name "woman".
    assert "person or group" in aff and "woman" not in aff.lower()


def test_13_uses_defined_rubric_with_clause_prepended():
    _, user = build_scaffold_prompt("affective", "1.3", "en", "C", "x")
    assert user.startswith("Consider the experience")          # clause prepended
    assert "IDEOLOGICAL AND INEQUALITY" in user                # verbatim E0 defined rubric
    # neutral 1.3 == E0 defined 1.3 (no clause)
    from llm_audit import prompts
    _, neu = build_scaffold_prompt("neutral", "1.3", "en", "C", "x")
    _, e0 = prompts.build_prompt("1.3", "defined", "en", "C", "x")
    assert neu == e0


def test_build_e3_cells_grid():
    cells = build_e3_cells([("1", "t")], "en", subtasks=("1.1",),
                           conditions=CONDITIONS, formats=("C",), k=6)
    # 1 item × 5 conditions × 1 subtask × 6 draws = 30 cells.
    assert len(cells) == 30
    assert {c.condition for c in cells} == set(CONDITIONS)
    assert all(c.experiment == "E3" for c in cells)


def test_invalid_condition_raises():
    with pytest.raises(ValueError):
        build_scaffold_prompt("bogus", "1.1", "en", "C", "x")
