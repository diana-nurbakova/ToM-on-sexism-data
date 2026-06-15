"""Tests for the Strachan scoring pipeline, judge runner, and calibration."""

import json
import uuid

import pytest

from llm_audit import parsing
from llm_audit.clients import FakeClient
from llm_audit.config import get_model
from llm_audit.e3 import strachan as S
from llm_audit.e3 import strachan_scoring as SC


def _rec(task, item, k, answer, **over):
    """A runner-style admin record for one response."""
    r = {
        "call_id": uuid.uuid4().hex,
        "experiment": "strachan", "condition": task, "subtask": task,
        "language": "en", "format": "free", "item_id": f"{task}#{item}",
        "k_index": k, "model": "gpt-4-strachan",
        "user_prompt": f"STORY for {task} {item}",
        "raw_response": answer, "parsed_value": answer,
        "parse_status": parsing.PARSED_OK if answer else parsing.EMPTY,
    }
    r.update(over)
    return r


def _write_log(path, recs):
    path.write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")


# ── rule-based routing ────────────────────────────────────────────────────────
def test_score_rule_irony_and_false_belief():
    keys = SC.S.load_keys()["keys"]
    # Irony A#1 is ironic -> correct "no".
    assert SC.score_rule("Irony - A", keys["Irony - A"]["1"], "No.") == 1
    assert SC.score_rule("Irony - A", keys["Irony - A"]["1"], "Yes.") == 0
    # False Belief A#3 is a false-belief location item.
    k3 = keys["False Belief - A"]["3"]
    assert SC.score_rule("False Belief - A", k3, f"He looks by the {k3['correct_location']}.") == 1


def test_score_rule_catch_item_is_yes_no():
    keys = SC.S.load_keys()["keys"]
    catch = keys["False Belief - A"]["7"]
    assert catch.get("question_type") == "surprise_yesno"
    assert SC.score_rule("False Belief - A", catch, "Yes, he will be surprised.") == 1
    assert SC.score_rule("False Belief - A", catch, "No.") == 0


# ── judge verdict parsing ─────────────────────────────────────────────────────
def test_parse_judge_verdict():
    assert SC.parse_judge_verdict('{"score": 1, "criterion": "x"}')[:2] == (1.0, "x")
    assert SC.parse_judge_verdict('noise {"score": 2} tail', max_score=2)[0] == 2.0
    assert SC.parse_judge_verdict("", 1)[2] == parsing.EMPTY
    assert SC.parse_judge_verdict("no json here", 1)[2] == parsing.FORMAT_VIOLATION
    # Out-of-range score is rejected, not clamped.
    assert SC.parse_judge_verdict('{"score": 2}', max_score=1)[2] == parsing.FORMAT_VIOLATION


def test_strange_criteria_resolves_as_previous():
    crit = SC.load_strange_criteria()
    assert len(crit) == 12
    # Item 6 ("[as previous]") inherits item 5's white-lie criteria.
    assert crit[5]["2pt"] == crit[4]["2pt"]
    assert "[as previous]" not in str(crit)


# ── judge runner (FakeClient) + resume ────────────────────────────────────────
def _fake_judge(monkeypatch):
    def responses(system, user, params):
        if "Strange Stories" in user:
            return '{"score": 2, "criterion": "2pt"}'
        return '{"score": 1, "criterion": "ok"}'
    spec = get_model("claude-4-sonnet")
    monkeypatch.setattr(SC, "make_client",
                        lambda *a, **k: FakeClient(spec, responses))


def test_run_judge_and_resume(tmp_path, monkeypatch):
    _fake_judge(monkeypatch)
    admin = tmp_path / "gpt-4-strachan.jsonl"
    recs = [
        _rec("Hinting", 1, 0, "He wants a drink."),
        _rec("Faux Pas", 5, 0, "No, Richard had forgotten."),
        _rec("Strange Stories", 1, 0, "Jim thinks Simon is lying."),
        _rec("Hinting", 2, 0, ""),                 # empty -> not judged
        _rec("Irony - A", 1, 0, "No."),            # rule-based -> not judged
    ]
    _write_log(admin, recs)
    out = tmp_path / "judge.jsonl"

    stats = SC.run_judge(admin, "claude-4-sonnet", out)
    assert stats["judged"] == 3 and stats["circuit_tripped"] is None
    verdicts = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    assert {v["task"] for v in verdicts} == {"Hinting", "Faux Pas", "Strange Stories"}
    assert next(v for v in verdicts if v["task"] == "Strange Stories")["score"] == 2.0

    # Resume: everything already settled -> nothing re-judged.
    stats2 = SC.run_judge(admin, "claude-4-sonnet", out)
    assert stats2["judged"] == 0


# ── combine + aggregate ───────────────────────────────────────────────────────
def test_score_responses_combines_rule_and_judge(tmp_path, monkeypatch):
    _fake_judge(monkeypatch)
    keys = SC.S.load_keys()["keys"]
    iro = keys["Irony - A"]["1"]["correct"]  # "no"
    admin = tmp_path / "gpt-4-strachan.jsonl"
    recs = [
        _rec("Irony - A", 1, 0, iro.capitalize() + "."),     # rule -> 1
        _rec("Irony - A", 1, 1, "Yes."),                     # rule -> 0
        _rec("Strange Stories", 1, 0, "Jim knows Simon lies."),  # judge 2 -> 1.0
        _rec("Hinting", 1, 0, "wants a drink"),              # judge 1 -> 1.0
    ]
    _write_log(admin, recs)
    out = tmp_path / "judge.jsonl"
    SC.run_judge(admin, "claude-4-sonnet", out)

    rows = SC.score_responses(admin, out, keys=keys)
    by = {(r["task"], r["item"], r["k_index"]): r for r in rows}
    assert by[("Irony - A", 1, 0)]["score"] == 1
    assert by[("Irony - A", 1, 1)]["score"] == 0
    # Strange Stories rescaled 0/1/2 -> 0/0.5/1.
    ss = by[("Strange Stories", 1, 0)]
    assert ss["raw_score"] == 2.0 and ss["score"] == 1.0

    agg = SC.aggregate(rows)
    assert agg["n_scored"] == 4


# ── calibration mechanics ─────────────────────────────────────────────────────
def test_load_published_shape():
    pub = SC.load_published("gpt-4-strachan")
    assert set(pub.columns) == {"csv_task", "item", "pub_mean"}
    assert (pub["pub_mean"].dropna().between(0, 1)).all()
    assert "Hinting" in set(pub["csv_task"])


def test_calibration_pass_and_fail():
    import pandas as pd
    rev = {v: k for k, v in SC.TASK_TO_CSV.items()}
    pub = SC.load_published("gpt-4-strachan").dropna(subset=["pub_mean"])
    # Build "our" rows that reproduce the published per-item means exactly -> pass.
    rows = [{"model": "gpt-4-strachan", "task": rev[r.csv_task], "item": int(r.item),
             "k_index": 0, "score": r.pub_mean, "raw_score": r.pub_mean,
             "parse_status": "parsed_ok"} for r in pub.itertuples()]
    good = SC.calibration_table(rows, "gpt-4-strachan")
    assert good["passed"] and good["overall_abs_diff"] < 1e-9

    # Shift every score by 0.5 -> well outside the band -> fail.
    bad_rows = [{**x, "score": min(1.0, x["score"] + 0.5)} for x in rows]
    bad = SC.calibration_table(bad_rows, "gpt-4-strachan")
    assert not bad["passed"] and bad["worst_abs_diff"] > 0.12
