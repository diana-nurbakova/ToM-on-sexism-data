"""ICM-Soft / ICM-Soft Norm via the official PyEvALL library (master spec §7.1).

EXIST's official *soft* metric is ICM-Soft and its normalised form ICM-Soft Norm
— a hierarchical, soft-label extension of the Information Contrast Measure
(Plaza et al. 2025 overview §4: ``ICM(s,g) = 2·IC(s) + 2·IC(g) − 3·IC(s∪g)`` over
a sexist/not-sexist hierarchy with annotator-proportion soft labels). The metric
is intricate and the organisers ship a reference implementation (PyEvALL,
github.com/UNEDLENAR/PyEvALL), so — per the project decision — we **wrap** it
rather than reimplement, guaranteeing leaderboard-comparable numbers.

PyEvALL pins heavy dependencies, so it lives in a **separate environment** and is
called out-of-process: this module converts our internal soft-label dicts
(``data.py`` keys) into PyEvALL's JSON input format + hierarchy, shells out to
``scripts/pyevall_runner.py`` under that env's interpreter (``config.pyevall_python()``),
and parses the scores back. The format conversion is the load-bearing part and
lives here, in one place, so every experiment formats EXIST input identically.

PyEvALL input format (per record): ``{"test_case", "id", "value"}`` where a soft
value is ``{class: proportion}``. EXIST class tokens and the hierarchy are below;
tasks 1.2/1.3 are hierarchical (parent ``YES`` over the sub-classes, plus the
sibling leaf ``NO``), 1.1 is flat binary.

**1.3 not-sexist mass.** The multilabel 1.3 soft value needs a ``NO`` entry so
not-sexist instances are representable. We take it from the *detection* (1.1)
distribution of the same source — gold ``NO`` from the gold 1.1 label, model
``NO`` from the model's own 1.1 prediction for the same (condition, format, item)
— mirroring EXIST's shared sexist/not-sexist first level. This convention is the
one place to revisit against an official EXIST gold file once one is in hand.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

from . import config

# ── EXIST class tokens (internal key -> EXIST label string) ───────────────────
_TOKENS = {
    "1.1": {"not_sexist": "NO", "sexist": "YES"},
    "1.2": {
        "not_sexist": "NO",
        "direct": "DIRECT",
        "reported": "REPORTED",
        "judgemental": "JUDGEMENTAL",
    },
    "1.3": {
        "ideological_inequality": "IDEOLOGICAL-INEQUALITY",
        "stereotyping_dominance": "STEREOTYPING-DOMINANCE",
        "objectification": "OBJECTIFICATION",
        "sexual_violence": "SEXUAL-VIOLENCE",
        "misogyny_non_sexual_violence": "MISOGYNY-NON-SEXUAL-VIOLENCE",
    },
}

# Sexist/not-sexist hierarchy passed to PyEvALL (PARAM_HIERARCHY) for the
# hierarchical tasks; 1.1 is flat. Parent YES over the sub-classes; NO is a leaf.
HIERARCHY = {
    "1.1": None,
    "1.2": {"YES": ["DIRECT", "REPORTED", "JUDGEMENTAL"], "NO": []},
    "1.3": {
        "YES": [
            "IDEOLOGICAL-INEQUALITY",
            "STEREOTYPING-DOMINANCE",
            "OBJECTIFICATION",
            "SEXUAL-VIOLENCE",
            "MISOGYNY-NON-SEXUAL-VIOLENCE",
        ],
        "NO": [],
    },
}

TEST_CASE = "EXIST2025"          # single test case per evaluation bucket
METRICS = ("ICMSoft", "ICMSoftNorm")


# ── Internal soft-label dict -> EXIST PyEvALL value ───────────────────────────
def to_exist_value(soft: dict, subtask: str, no_mass: float | None = None) -> dict:
    """Map one internal soft-label dict to an EXIST-token ``{class: prob}`` value.

    ``no_mass`` supplies the not-sexist proportion for 1.3 (taken from the 1.1
    distribution; see module docstring). It is ignored for 1.1/1.2, whose ``NO``
    is already represented as a class.
    """
    tok = _TOKENS[subtask]
    value = {tok[k]: float(soft.get(k, 0.0)) for k in tok}
    if subtask == "1.3" and no_mass is not None:
        value["NO"] = float(no_mass)
    return value


def build_records(
    soft_by_item: dict[str, dict], subtask: str,
    no_by_item: dict[str, float] | None = None,
) -> list[dict]:
    """Build a PyEvALL-format record list from per-item soft labels."""
    no_by_item = no_by_item or {}
    return [
        {
            "test_case": TEST_CASE,
            "id": str(item),
            "value": to_exist_value(soft, subtask, no_by_item.get(item)),
        }
        for item, soft in soft_by_item.items()
    ]


# ── Out-of-process PyEvALL call ───────────────────────────────────────────────
_RUNNER = Path(__file__).resolve().parents[2] / "scripts" / "pyevall_runner.py"


def run_pyevall(
    pred: list[dict], gold: list[dict], subtask: str,
    python_exe: str, metrics=METRICS, workdir: str | Path | None = None,
) -> dict:
    """Write PyEvALL input files, invoke the isolated runner, return its scores.

    Returns ``{metric_name: average_or_None}``. Raises ``RuntimeError`` with the
    runner's stderr if the subprocess fails (e.g. PyEvALL not installed in that
    env, or a format error).
    """
    tmp = Path(tempfile.mkdtemp(prefix="icm_soft_", dir=workdir))
    try:
        pred_p, gold_p, out_p = tmp / "pred.json", tmp / "gold.json", tmp / "out.json"
        pred_p.write_text(json.dumps(pred, ensure_ascii=False), encoding="utf-8")
        gold_p.write_text(json.dumps(gold, ensure_ascii=False), encoding="utf-8")
        cmd = [
            python_exe, str(_RUNNER),
            "--pred", str(pred_p), "--gold", str(gold_p),
            "--metrics", ",".join(metrics), "--out", str(out_p),
        ]
        hier = HIERARCHY.get(subtask)
        if hier:
            hier_p = tmp / "hierarchy.json"
            hier_p.write_text(json.dumps(hier), encoding="utf-8")
            cmd += ["--hierarchy", str(hier_p)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"PyEvALL runner failed (exit {proc.returncode}) for subtask "
                f"{subtask}.\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
            )
        return json.loads(out_p.read_text(encoding="utf-8"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ── End-to-end scoring against gold ───────────────────────────────────────────
def score_icm_soft(
    soft: dict, gold_by_item: dict, *,
    python_exe: str | None = None, metrics=METRICS,
    workdir: str | Path | None = None,
) -> dict:
    """Compute ICM-Soft / ICM-Soft Norm per (condition, subtask, format).

    Buckets the assembled model soft labels (``metrics.assemble_model_soft``
    output, keyed ``(condition, subtask, format, item)``) by
    ``(condition, subtask, format)``, restricts gold to the items the model
    covered, formats both for PyEvALL, and runs the isolated evaluator once per
    bucket. The 1.3 ``NO`` mass is threaded from each side's 1.1 distribution.

    Args:
        soft: ``assemble_model_soft`` ``soft`` dict.
        gold_by_item: ``{item_id: {"1.1": dist, "1.2": dist, "1.3": dist}}``.
        python_exe: isolated-env interpreter; defaults to ``config.pyevall_python()``.

    Returns ``{"condition|subtask|format": {metric: score}}``.
    """
    python_exe = python_exe or config.pyevall_python()
    if not python_exe:
        raise RuntimeError(
            "No PyEvALL interpreter configured. Set PYEVALL_PYTHON to the python "
            "of the isolated environment where PyEvALL is installed (master spec "
            "§7.1), or pass python_exe=..."
        )

    # Model detection (1.1) soft by (condition, format, item) for the 1.3 NO mass.
    det = {
        (cond, fmt, item): entry["soft"]
        for (cond, sub, fmt, item), entry in soft.items() if sub == "1.1"
    }

    buckets: dict[tuple, dict] = defaultdict(dict)
    for (cond, sub, fmt, item), entry in soft.items():
        if item in gold_by_item:
            buckets[(cond, sub, fmt)][item] = entry["soft"]

    out: dict[str, dict] = {}
    for (cond, sub, fmt), model_soft in buckets.items():
        items = list(model_soft)
        model_no = gold_no = None
        if sub == "1.3":
            model_no = {
                i: (det.get((cond, fmt, i)) or {}).get("not_sexist") for i in items
            }
            gold_no = {i: gold_by_item[i]["1.1"].get("not_sexist") for i in items}
        pred = build_records(model_soft, sub, model_no)
        gold = build_records({i: gold_by_item[i][sub] for i in items}, sub, gold_no)
        out["|".join((cond, sub, fmt))] = run_pyevall(
            pred, gold, sub, python_exe, metrics=metrics, workdir=workdir
        )
    return out
