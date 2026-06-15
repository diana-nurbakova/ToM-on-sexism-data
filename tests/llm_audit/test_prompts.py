"""Tests that assembled E0 prompts match the spec and have no stray placeholders."""

import pytest

from llm_audit import prompts
from llm_audit.prompts import build_prompt, SUBTASKS, SPECS, LANGS, FORMATS


def test_no_unfilled_placeholders_across_full_grid():
    for sub in SUBTASKS:
        for spec in SPECS:
            for lang in LANGS:
                for fmt in FORMATS:
                    system, user = build_prompt(sub, spec, lang, fmt, "TWEET_TEXT")
                    assert "TWEET_TEXT" in user
                    assert "{tweet}" not in user and "{suffix}" not in user
                    assert system == prompts.SYSTEM_PROMPT[lang]


def test_bare_1_1_en_verbatim():
    _, user = build_prompt("1.1", "bare", "en", "C", "hello world")
    assert user.startswith("Classify the following tweet as SEXIST or NOT SEXIST.")
    assert "Tweet: hello world" in user
    assert user.rstrip().endswith("Respond with exactly one word: sexist or not_sexist.")


def test_defined_1_1_en_includes_official_definition():
    _, user = build_prompt("1.1", "defined", "en", "A", "x")
    assert "A tweet is sexist if it expresses sexist ideas" in user
    assert "describes a sexist situation" in user


def test_defined_1_3_en_has_all_five_categories():
    _, user = build_prompt("1.3", "defined", "en", "A", "x")
    for cat in ("IDEOLOGICAL AND INEQUALITY", "STEREOTYPING AND DOMINANCE",
                "OBJECTIFICATION", "SEXUAL VIOLENCE", "MISOGYNY AND NON-SEXUAL VIOLENCE"):
        assert cat in user


def test_spanish_uses_formal_register_and_tokens():
    system, user = build_prompt("1.2", "bare", "es", "C", "x")
    assert system.startswith("Usted es un asistente")
    assert "crítico" in user  # chosen ES token for JUDGEMENTAL


def test_invalid_args_raise():
    with pytest.raises(ValueError):
        build_prompt("9.9", "bare", "en", "A", "x")
    with pytest.raises(ValueError):
        build_prompt("1.1", "bogus", "en", "A", "x")
