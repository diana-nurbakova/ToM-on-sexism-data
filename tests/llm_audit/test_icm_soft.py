"""Tests for the ICM-Soft / PyEvALL wrapper (master spec §7.1).

These cover the load-bearing, in-process parts: the internal -> EXIST token
conversion, the 1.3 ``NO``-mass threading from 1.1, the hierarchy choice, and the
subprocess handoff (with PyEvALL faked, since it lives in a separate env). The
ICM-Soft *numbers* themselves come from PyEvALL and are validated in that env.
"""

import json

import pytest

from llm_audit import config, icm_soft


# ── Token conversion ──────────────────────────────────────────────────────────
def test_to_exist_value_1_1():
    v = icm_soft.to_exist_value({"not_sexist": 0.2, "sexist": 0.8}, "1.1")
    assert v == {"NO": 0.2, "YES": 0.8}


def test_to_exist_value_1_2_maps_all_classes():
    soft = {"not_sexist": 0.5, "direct": 0.0, "reported": 0.3, "judgemental": 0.2}
    v = icm_soft.to_exist_value(soft, "1.2")
    assert v == {"NO": 0.5, "DIRECT": 0.0, "REPORTED": 0.3, "JUDGEMENTAL": 0.2}


def test_to_exist_value_1_3_uses_exist_tokens_and_no_mass():
    soft = {
        "ideological_inequality": 0.1, "stereotyping_dominance": 0.0,
        "objectification": 0.5, "sexual_violence": 0.3,
        "misogyny_non_sexual_violence": 0.2,
    }
    v = icm_soft.to_exist_value(soft, "1.3", no_mass=0.4)
    assert v["OBJECTIFICATION"] == 0.5
    assert v["MISOGYNY-NON-SEXUAL-VIOLENCE"] == 0.2
    assert v["NO"] == 0.4  # not-sexist mass threaded in
    assert "ideological_inequality" not in v  # internal keys not leaked


def test_to_exist_value_1_3_without_no_mass_omits_no():
    v = icm_soft.to_exist_value({"objectification": 1.0}, "1.3")
    assert "NO" not in v


def test_build_records_shape():
    recs = icm_soft.build_records({"100": {"not_sexist": 0.5, "sexist": 0.5}}, "1.1")
    assert recs == [{"test_case": icm_soft.TEST_CASE, "id": "100",
                     "value": {"NO": 0.5, "YES": 0.5}}]


# ── Hierarchy choice ──────────────────────────────────────────────────────────
def test_hierarchy_flat_for_1_1_hierarchical_for_others():
    assert icm_soft.HIERARCHY["1.1"] is None
    assert "YES" in icm_soft.HIERARCHY["1.2"]
    assert set(icm_soft.HIERARCHY["1.3"]["YES"]) == {
        "IDEOLOGICAL-INEQUALITY", "STEREOTYPING-DOMINANCE", "OBJECTIFICATION",
        "SEXUAL-VIOLENCE", "MISOGYNY-NON-SEXUAL-VIOLENCE",
    }


# ── Subprocess handoff (PyEvALL faked) ────────────────────────────────────────
class _FakeProc:
    returncode = 0
    stdout = ""
    stderr = ""


def _fake_run_factory(capture, scores):
    """Build a fake subprocess.run that records the command and writes scores."""
    def fake_run(cmd, capture_output, text):
        flags = {}
        for flag in ("--pred", "--gold", "--out", "--metrics", "--hierarchy"):
            if flag in cmd:
                flags[flag] = cmd[cmd.index(flag) + 1]
        capture["cmd"] = cmd
        capture["flags"] = flags
        # The pred/gold files must have been written before the call.
        capture["pred"] = json.loads(open(flags["--pred"], encoding="utf-8").read())
        capture["gold"] = json.loads(open(flags["--gold"], encoding="utf-8").read())
        open(flags["--out"], "w", encoding="utf-8").write(json.dumps(scores))
        return _FakeProc()
    return fake_run


def test_run_pyevall_writes_files_and_passes_hierarchy(monkeypatch):
    cap = {}
    monkeypatch.setattr(icm_soft.subprocess, "run",
                        _fake_run_factory(cap, {"ICMSoft": 0.5, "ICMSoftNorm": 0.75}))
    pred = icm_soft.build_records({"1": {"objectification": 1.0}}, "1.3", {"1": 0.0})
    gold = icm_soft.build_records({"1": {"objectification": 1.0}}, "1.3", {"1": 0.0})
    res = icm_soft.run_pyevall(pred, gold, "1.3", "py", workdir=None)
    assert res == {"ICMSoft": 0.5, "ICMSoftNorm": 0.75}
    assert "--hierarchy" in cap["flags"]          # 1.3 is hierarchical
    assert cap["pred"][0]["value"]["OBJECTIFICATION"] == 1.0


def test_run_pyevall_no_hierarchy_for_1_1(monkeypatch):
    cap = {}
    monkeypatch.setattr(icm_soft.subprocess, "run",
                        _fake_run_factory(cap, {"ICMSoft": 1.0, "ICMSoftNorm": 1.0}))
    recs = icm_soft.build_records({"1": {"not_sexist": 0.0, "sexist": 1.0}}, "1.1")
    icm_soft.run_pyevall(recs, recs, "1.1", "py")
    assert "--hierarchy" not in cap["flags"]


def test_run_pyevall_raises_on_failure(monkeypatch):
    def boom(cmd, capture_output, text):
        p = _FakeProc()
        p.returncode = 1
        p.stderr = "ModuleNotFoundError: pyevall"
        return p
    monkeypatch.setattr(icm_soft.subprocess, "run", boom)
    recs = icm_soft.build_records({"1": {"not_sexist": 1.0, "sexist": 0.0}}, "1.1")
    with pytest.raises(RuntimeError, match="PyEvALL runner failed"):
        icm_soft.run_pyevall(recs, recs, "1.1", "py")


# ── End-to-end bucketing + 1.3 NO threading (run_pyevall faked) ────────────────
def test_score_icm_soft_buckets_and_threads_1_3_no(monkeypatch):
    calls = []

    def fake_run_pyevall(pred, gold, subtask, python_exe, metrics=icm_soft.METRICS,
                         workdir=None):
        calls.append({"subtask": subtask, "pred": pred, "gold": gold})
        return {"ICMSoft": 0.42, "ICMSoftNorm": 0.6}

    monkeypatch.setattr(icm_soft, "run_pyevall", fake_run_pyevall)

    soft = {
        ("bare", "1.1", "C", "I1"): {"soft": {"not_sexist": 0.25, "sexist": 0.75}},
        ("bare", "1.3", "C", "I1"): {"soft": {
            "ideological_inequality": 0.0, "stereotyping_dominance": 0.0,
            "objectification": 0.5, "sexual_violence": 0.0,
            "misogyny_non_sexual_violence": 0.0}},
    }
    gold_by_item = {"I1": {
        "1.1": {"not_sexist": 0.5, "sexist": 0.5},
        "1.2": {"not_sexist": 0.5, "direct": 0.5, "reported": 0.0, "judgemental": 0.0},
        "1.3": {"ideological_inequality": 0.0, "stereotyping_dominance": 0.0,
                "objectification": 0.5, "sexual_violence": 0.0,
                "misogyny_non_sexual_violence": 0.0},
    }}

    out = icm_soft.score_icm_soft(soft, gold_by_item, python_exe="py")
    assert set(out) == {"bare|1.1|C", "bare|1.3|C"}
    assert out["bare|1.1|C"]["ICMSoft"] == 0.42

    call_13 = next(c for c in calls if c["subtask"] == "1.3")
    # model NO from the model's own 1.1 prediction; gold NO from gold 1.1.
    assert call_13["pred"][0]["value"]["NO"] == 0.25
    assert call_13["gold"][0]["value"]["NO"] == 0.5


def test_score_icm_soft_requires_interpreter(monkeypatch):
    monkeypatch.setattr(config, "pyevall_python", lambda: None)
    with pytest.raises(RuntimeError, match="No PyEvALL interpreter"):
        icm_soft.score_icm_soft({}, {}, python_exe=None)


# ── config resolver ───────────────────────────────────────────────────────────
def test_pyevall_python_from_env(monkeypatch):
    monkeypatch.setenv(config.PYEVALL_PYTHON_ENV, "/opt/pyevall/bin/python")
    monkeypatch.setattr(config, "load_env", lambda: {})
    assert config.pyevall_python() == "/opt/pyevall/bin/python"


def test_pyevall_python_none_when_unset(monkeypatch):
    monkeypatch.delenv(config.PYEVALL_PYTHON_ENV, raising=False)
    monkeypatch.setattr(config, "load_env", lambda: {})
    assert config.pyevall_python() is None
