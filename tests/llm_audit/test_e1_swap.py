"""Tests for the E1 frozen woman↔man token swap."""

from llm_audit.e1.swap import swap_text


def test_basic_noun_and_pronoun_swap():
    r = swap_text("She is a woman and women matter.")
    assert r.swapped == "He is a man and men matter."
    assert not r.unswappable


def test_case_preserved():
    assert swap_text("Woman").swapped == "Man"
    assert swap_text("WOMEN").swapped == "MEN"
    assert swap_text("she").swapped == "he"


def test_bidirectional_male_to_female():
    assert swap_text("He is a man.").swapped == "She is a woman."


def test_her_possessive_vs_object():
    # possessive determiner before a noun -> his
    assert "his car" in swap_text("That is her car.").swapped.lower()
    # object pronoun at clause end -> him
    assert swap_text("I saw her.").swapped.lower().startswith("i saw him")


def test_his_reverse_possessive_and_pronoun():
    assert "her car" in swap_text("That is his car.").swapped.lower()
    assert swap_text("It is his.").swapped.lower().rstrip(".").endswith("hers")


def test_her_triggers_author_review():
    r = swap_text("her car")
    assert r.author_review and any("ambiguous" in x for x in r.review_reasons)


def test_honorific_maps_to_mr_and_flags_review():
    r = swap_text("Mrs Smith arrived.")
    assert r.swapped.startswith("Mr Smith")
    assert r.author_review


def test_name_swap_flagged():
    r = swap_text("Mary went home.")
    assert r.swapped.startswith("John")
    assert r.name_swapped


def test_possessive_noun_swaps_and_keeps_suffix():
    assert swap_text("Women's studies").swapped == "Men's studies"
    assert swap_text("the girls' team").swapped == "the boys' team"


def test_unswappable_when_no_gendered_token():
    r = swap_text("The weather is nice today.")
    assert r.unswappable
    assert r.swapped == "The weather is nice today."


def test_mention_handle_never_altered_and_not_counted():
    # A curated name inside a handle must stay intact and not make the item swappable.
    r = swap_text("@Mary hello world")
    assert r.swapped == "@Mary hello world"
    assert r.unswappable


def test_mention_ignored_but_real_token_still_swaps():
    r = swap_text("@SomeUser she is a woman")
    assert r.swapped == "@SomeUser he is a man"
    assert not r.unswappable


def test_plain_name_outside_handle_still_swaps():
    # Same name without the @ is a normal token and swaps.
    assert swap_text("John went home").swapped == "Mary went home"


def test_email_like_at_is_not_a_mention():
    # 'a@b' lookbehind: the swap of a real noun next to it is unaffected.
    r = swap_text("email me at sam@host about the woman")
    assert "man" in r.swapped and not r.unswappable


def test_swapped_tokens_recorded():
    r = swap_text("the mother and her daughter")
    srcs = [s for s, _ in r.swapped_tokens]
    assert "mother" in srcs and "daughter" in srcs and "her" in srcs
