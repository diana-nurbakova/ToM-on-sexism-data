"""Tests for E1 judge prompt assembly and invariance parsing."""

from llm_audit.e1 import judge
from llm_audit.e1.judge import build_judge_user, parse_invariance


def test_judge_prompt_includes_both_texts_and_no_placeholders():
    user = build_judge_user("orig tweet", "swapped tweet")
    assert "orig tweet" in user and "swapped tweet" in user
    assert "{original}" not in user and "{swapped}" not in user


def test_parse_clean_json_verdict():
    val, status = parse_invariance('{"verdict": "SHIFTING", "reason": "woman-targeted"}')
    assert (val, status) == ("SHIFTING", "parsed_ok")


def test_parse_fenced_json():
    val, status = parse_invariance('```json\n{"verdict": "INVARIANT", "reason": "x"}\n```')
    assert val == "INVARIANT" and status == "parsed_ok"


def test_parse_bare_word_fallback():
    val, status = parse_invariance("AMBIGUOUS")
    assert val == "AMBIGUOUS" and status == "parsed_ok"


def test_parse_refusal():
    val, status = parse_invariance("I'm sorry, I can't make that judgement.")
    assert val is None and status == "refusal"


def test_parse_ambiguous_multiple_words_is_violation():
    # Two class words and no JSON -> cannot disambiguate.
    val, status = parse_invariance("It could be INVARIANT or SHIFTING, hard to say.")
    assert val is None and status == "format_violation"


def test_build_judge_cells_skips_unswappable():
    rows = [
        {"item_id": "1", "original": "a woman", "swapped": "a man", "unswappable": False},
        {"item_id": "2", "original": "the weather", "swapped": "the weather", "unswappable": True},
    ]
    cells = judge.build_judge_cells(rows)
    assert len(cells) == 1 and cells[0].item_id == "1"
    assert cells[0].experiment == "E1-judge"
