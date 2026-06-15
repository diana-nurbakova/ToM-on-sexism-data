"""Tests for the Strachan battery administration + scoring scaffolding."""

import pytest

from llm_audit.e3 import strachan as S


def test_load_battery_has_core_tasks():
    b = S.load_battery()
    assert "Hinting" in b and "Faux Pas" in b and "Strange Stories" in b
    assert len(b["Hinting"]) == 16  # matches the extracted count


def test_build_cells_count_and_fields():
    b = S.load_battery()
    cells = S.build_strachan_cells(b, sessions=3, tasks=["Hinting"])
    assert len(cells) == 16 * 3
    c = cells[0]
    assert c.experiment == "strachan" and c.condition == "Hinting"
    assert c.text  # the item prompt


def test_free_text_parser():
    assert S._parse_free_text("", None)[1] == "empty"
    assert S._parse_free_text("He means he is tired.", None) == ("He means he is tired.", "parsed_ok")
    raw, status = S._parse_free_text("I'm sorry, I can't answer that.", None)
    assert status == "disclaimer_prefixed"


def test_judge_prompt_embeds_rules():
    fp = S.build_judge_prompt("Faux Pas", "STORY", "ANS")
    assert "did NOT know" in fp and "STORY" in fp and "ANS" in fp
    hint = S.build_judge_prompt("Hinting", "STORY", "ANS", target="means tired; wants a drink")
    assert "meaning" in hint.lower() and "action" in hint.lower() and "means tired" in hint
    ss = S.build_judge_prompt("Strange Stories", "STORY", "ANS",
                              criterion={"2pt": "X", "1pt": "Y", "0pt": "Z"})
    assert "2 points: X" in ss


def test_judge_prompt_rejects_rule_based_task():
    with pytest.raises(ValueError):
        S.build_judge_prompt("Irony - A", "s", "a")


def test_score_false_belief():
    key = {"correct_location": "basket", "distractor_location": "box"}
    assert S.score_false_belief("It is in the basket.", key) == 1
    assert S.score_false_belief("It is in the box now.", key) == 0   # distractor named
    assert S.score_false_belief("", key) is None


def test_score_irony():
    assert S.score_irony("Yes, she means it.", {"correct": "yes"}) == 1
    assert S.score_irony("No.", {"correct": "yes"}) == 0
    assert S.score_irony("Maybe, hard to say.", {"correct": "yes"}) is None  # ambiguous
