"""Tests for the parse-status taxonomy, JSON recovery, and label matcher."""

from llm_audit import parsing
from llm_audit.parsing import (
    PARSED_OK, REFUSAL, DISCLAIMER_PREFIXED, FORMAT_VIOLATION, EMPTY,
    match_label, normalize, parse_response,
)


# ── normalize / match_label ───────────────────────────────────────────────────
def test_normalize_strips_accents_and_punctuation():
    assert normalize("Cosificación!") == "cosificacion"
    assert normalize("NO_SEXISTA") == "no sexista"


def test_match_label_variants_resolve():
    assert match_label("cosificación", "1.3") == "objectification"
    assert match_label("crítico", "1.2") == "judgemental"
    assert match_label("enjuiciador", "1.2") == "judgemental"
    assert match_label("sexista", "1.1") == "sexist"
    assert match_label("no sexista", "1.1") == "not_sexist"


def test_match_label_fuzzy_catches_misspelling():
    # misojinia vs misoginia — one edit, should resolve to misogyny class.
    assert match_label("misojinia", "1.3") == "misogyny_non_sexual_violence"


def test_match_label_does_not_merge_distinct_classes():
    assert match_label("totally unrelated phrase", "1.3") is None


# ── Format A (JSON) ───────────────────────────────────────────────────────────
def test_format_a_clean_json():
    val, status = parse_response('{"sexist": 0.8, "not_sexist": 0.2}', "1.1", "A")
    assert status == PARSED_OK
    assert abs(val["sexist"] - 0.8) < 1e-9


def test_format_a_fenced_json_recovered():
    raw = '```json\n{"sexist": 1, "not_sexist": 0}\n```'
    val, status = parse_response(raw, "1.1", "A")
    assert status == PARSED_OK
    assert val["sexist"] == 1.0


def test_format_a_prose_wrapped_json_recovered():
    raw = 'Sure, here is the result: {"sexist": 0.5, "not_sexist": 0.5} done.'
    val, status = parse_response(raw, "1.1", "A")
    assert status == PARSED_OK
    assert val["sexist"] == 0.5


def test_format_a_1_3_multilabel_not_renormalised():
    raw = '{"objectification": 0.9, "sexual_violence": 0.7}'
    val, status = parse_response(raw, "1.3", "A")
    assert status == PARSED_OK
    assert val["objectification"] == 0.9 and val["sexual_violence"] == 0.7
    assert val["misogyny_non_sexual_violence"] == 0.0


# ── Format B (plain) ──────────────────────────────────────────────────────────
def test_format_b_pairs():
    val, status = parse_response("sexist: 0.6 | not_sexist: 0.4", "1.1", "B")
    assert status == PARSED_OK
    assert abs(val["sexist"] - 0.6) < 1e-9


# ── Format C (categorical) ────────────────────────────────────────────────────
def test_format_c_single_word():
    assert parse_response("sexist", "1.1", "C") == ("sexist", PARSED_OK)
    assert parse_response("judgemental", "1.2", "C")[0] == "judgemental"


def test_format_c_multilabel_list():
    val, status = parse_response("objectification, sexual_violence", "1.3", "C")
    assert status == PARSED_OK
    assert set(val) == {"objectification", "sexual_violence"}


def test_format_c_multilabel_none():
    assert parse_response("none", "1.3", "C") == ([], PARSED_OK)


# ── Refusal / disclaimer / empty / violation ──────────────────────────────────
def test_refusal_detected():
    val, status = parse_response("I'm sorry, I can't help with classifying that.", "1.1", "C")
    assert status == REFUSAL and val is None


def test_disclaimer_prefixed_still_parses():
    raw = 'Please note this is sensitive. {"sexist": 0.7, "not_sexist": 0.3}'
    val, status = parse_response(raw, "1.1", "A")
    assert status == DISCLAIMER_PREFIXED
    assert val["sexist"] == 0.7


def test_empty_response():
    assert parse_response("   ", "1.1", "A")[1] == EMPTY
    assert parse_response(None, "1.1", "A")[1] == EMPTY


def test_format_violation_on_garbage():
    val, status = parse_response("purple monkey dishwasher", "1.1", "A")
    assert status == FORMAT_VIOLATION and val is None
