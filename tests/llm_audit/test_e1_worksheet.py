"""Tests for the E1 adjudication worksheet workflow (build -> adjudicate -> finalise)."""

import csv
import json

import pandas as pd
import pytest

from llm_audit.e1 import rule, worksheet


def _sample():
    """Three synthetic items exercising the three rule outcomes."""
    return pd.DataFrame([
        # majority not-sexist -> confident INVARIANT, no adjudication
        {"item_id": "1", "strat_category": "ideological_inequality", "agreement_bin": "high",
         "gold_1_1": {"not_sexist": 0.8, "sexist": 0.2}, "gold_1_3": {}, "text": "she is a woman"},
        # sexist + gendered swap -> SHIFTING first-pass, needs adjudication
        {"item_id": "2", "strat_category": "misogyny_non_sexual_violence", "agreement_bin": "high",
         "gold_1_1": {"not_sexist": 0.1, "sexist": 0.9}, "gold_1_3": {}, "text": "women belong in the kitchen"},
        # no gendered token -> UNSWAPPABLE, excluded
        {"item_id": "3", "strat_category": "stereotyping_dominance", "agreement_bin": "low",
         "gold_1_1": {"not_sexist": 0.5, "sexist": 0.5}, "gold_1_3": {}, "text": "the weather is nice"},
    ])


def test_build_worksheet_rule_first_pass():
    df = worksheet.build_worksheet(_sample())
    by_id = {r["item_id"]: r for _, r in df.iterrows()}
    assert by_id["1"]["rule_class"] == "INVARIANT" and not by_id["1"]["needs_adjudication"]
    assert by_id["2"]["rule_class"] == "SHIFTING" and by_id["2"]["needs_adjudication"]
    assert by_id["3"]["rule_class"] == "UNSWAPPABLE" and not by_id["3"]["needs_adjudication"]
    # author cells start blank; swapped text was produced for the swappable rows.
    assert by_id["1"]["invariance_class"] == "" and by_id["1"]["justification"] == ""
    assert by_id["2"]["swapped"] == "men belong in the kitchen"


def test_validate_and_finalize_block_on_unresolved(tmp_path):
    path = tmp_path / "ws.csv"
    worksheet.write_worksheet(worksheet.build_worksheet(_sample()), path, merge_prior=False)
    unresolved = worksheet.validate_worksheet(path)
    assert [u["item_id"] for u in unresolved] == ["2"]          # only the flagged row
    with pytest.raises(ValueError, match="need adjudication"):
        worksheet.finalize_partition(path, lang="en")


def test_finalize_exports_partitions(tmp_path):
    path = tmp_path / "ws.csv"
    worksheet.write_worksheet(worksheet.build_worksheet(_sample()), path, merge_prior=False)
    # author adjudicates item 2 as SHIFTING (agreeing with the rule).
    _fill(path, {"2": ("SHIFTING", "woman-target stereotype emptied under swap")})

    summary = worksheet.finalize_partition(path, out_dir=tmp_path, lang="en")
    assert summary["n_items"] == 3
    assert summary["n_unswappable_excluded"] == 1
    assert summary["post_adjudication_counts"]["INVARIANT"] == 1
    assert summary["post_adjudication_counts"]["SHIFTING"] == 1
    assert summary["n_changed_from_rule"] == 0

    partition = json.loads((tmp_path / "e1_partition_en.json").read_text())
    assert partition["partition_post_adjudication"]["2"] == "SHIFTING"
    assert partition["partition_pre_adjudication"]["2"] == "SHIFTING"
    # the resolved worksheet carries the final class for the confident row too.
    resolved = {r["item_id"]: r["invariance_class"]
                for r in csv.DictReader((tmp_path / "e1_sample_en_resolved.csv").open(encoding="utf-8"))}
    assert resolved == {"1": "INVARIANT", "2": "SHIFTING", "3": "UNSWAPPABLE"}


def test_finalize_logs_confident_override(tmp_path):
    path = tmp_path / "ws.csv"
    worksheet.write_worksheet(worksheet.build_worksheet(_sample()), path, merge_prior=False)
    # author overrides the confident INVARIANT row 1, adjudicates row 2 in line with the rule.
    _fill(path, {"1": ("SHIFTING", "override"), "2": ("SHIFTING", "agrees with rule")})
    summary = worksheet.finalize_partition(path, out_dir=tmp_path, lang="en")
    assert summary["confident_row_overrides"] == ["1"]   # only the non-flagged override is a deviation
    assert summary["n_changed_from_rule"] == 1           # row 2 matched the rule, so no change


def test_write_worksheet_merges_prior_author_cells(tmp_path):
    path = tmp_path / "ws.csv"
    worksheet.write_worksheet(worksheet.build_worksheet(_sample()), path, merge_prior=False)
    _fill(path, {"2": ("SHIFTING", "keep me")})
    # rebuild with merge: the author cell for item 2 must survive.
    worksheet.write_worksheet(worksheet.build_worksheet(_sample()), path, merge_prior=True)
    rows = {r["item_id"]: r for r in csv.DictReader(path.open(encoding="utf-8"))}
    assert rows["2"]["invariance_class"] == "SHIFTING" and rows["2"]["justification"] == "keep me"


def _fill(path, decisions):
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    fields = rows[0].keys()
    for r in rows:
        if r["item_id"] in decisions:
            r["invariance_class"], r["justification"] = decisions[r["item_id"]]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fields))
        w.writeheader()
        w.writerows(rows)
