"""Strachan battery scoring pipeline, LLM-judge runner, and calibration check.

Reads an administration log written by :func:`strachan.run_strachan` and produces
per-model / per-subtest scores under the **Strachan-simplified key**:

  * **Rule-based** (closed-form) subtests — False Belief (location naming; the catch
    item is a surprise Yes/No) and Irony (Yes/No discrimination) — are scored
    directly from ``strachan_keys.json`` by :mod:`strachan`'s rule scorers.
  * **Free-text** subtests — Hinting (meaning+action), Faux Pas (final
    speaker-awareness question), Strange Stories (released 0/1/2 rubric) — are scored
    by an **LLM-judge** (:func:`run_judge`), which embeds the per-item rubric/target
    and returns a structured 0/1(/2) verdict. Strange Stories is rescaled 0/1/2 ->
    0/0.5/1 to match the published scale.

The calibration models (GPT-4, GPT-3.5, LLaMA2-70B) are scored the same way and
compared to Strachan's published per-item scores (``scores_gpt.csv`` /
``scores_llama.csv``) via :func:`calibration_table` — agreement within the ~5 %
overall / ~12 % worst-subtest band means the auto-scorer is "on the same ruler"
(spec §5). Missing/unparseable responses are dropped (available-case), never imputed.
"""

from __future__ import annotations

import json
from pathlib import Path

from .. import parsing
from ..clients import CircuitBreakerTrip, make_client
from ..config import REPO_ROOT, get_model
from ..metrics import iter_records
from . import strachan as S

DATA_DIR = REPO_ROOT / "data" / "strachan"

# Battery task -> published-CSV task label (the CSVs drop the " - " before the list).
TASK_TO_CSV = {
    "False Belief - A": "False Belief A",
    "False Belief - B": "False Belief B",
    "Irony - A": "Irony A",
    "Irony - B": "Irony B",
    "Hinting": "Hinting",
    "Faux Pas": "Faux Pas",
    "Strange Stories": "Strange Stories",
}

# Reproduction model -> (published CSV, row filter, per-session column template).
# scores_gpt.csv is long (filter the `model` column); scores_llama.csv is wide
# (one column block per size, score_70B-1..15).
CALIB_SOURCES = {
    "gpt-4-strachan": ("scores_gpt.csv", {"model": "GPT-4"}, "score{s}"),
    "gpt-3.5-turbo-strachan": ("scores_gpt.csv", {"model": "GPT-3.5"}, "score{s}"),
    "llama-2-70b-strachan": ("scores_llama.csv", {}, "score_70B-{s}"),
}

# Deterministic, terse judging.
JUDGE_SAMPLING = {"temperature": 0.0, "max_tokens": 200}
_MAX_SCORE = {"Hinting": 1, "Faux Pas": 1, "Strange Stories": 2}


# ── helpers ───────────────────────────────────────────────────────────────────
def _item_no(item_id: str) -> int:
    """'Irony - A#3' -> 3."""
    return int(str(item_id).rsplit("#", 1)[-1])


def _answer_of(rec: dict) -> str | None:
    """The model's free-text answer, or None if the cell produced nothing usable."""
    ans = rec.get("parsed_value")
    if ans is None:
        ans = rec.get("raw_response")
    ans = None if ans is None else str(ans).strip()
    return ans or None


def load_strange_criteria(path: Path = S.BATTERY_PATH) -> list[dict]:
    """Released Strange Stories 2pt/1pt/0pt criteria, in item order.

    Resolves the OSF's ``"[as previous]"`` placeholders (item 6 reuses item 5's
    white-lie criteria) by carrying the prior item forward.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8")).get("strange_stories_coding", [])
    out: list[dict] = []
    for c in raw:
        resolved = {}
        for k in ("2pt", "1pt", "0pt"):
            v = c.get(k, "")
            if isinstance(v, str) and v.strip().lower() == "[as previous]" and out:
                v = out[-1][k]
            resolved[k] = v
        out.append(resolved)
    return out


def parse_judge_verdict(raw: str | None, max_score: int = 1):
    """(score, criterion, status) from a judge reply. ``score`` is float in [0, max]."""
    if raw is None or not str(raw).strip():
        return None, None, parsing.EMPTY
    obj = parsing._extract_json(str(raw))
    if not obj or "score" not in obj:
        return None, None, parsing.FORMAT_VIOLATION
    try:
        score = float(obj["score"])
    except (TypeError, ValueError):
        return None, None, parsing.FORMAT_VIOLATION
    if not (0 <= score <= max_score):
        return None, obj.get("criterion"), parsing.FORMAT_VIOLATION
    return score, obj.get("criterion"), parsing.PARSED_OK


# ── rule-based scoring ────────────────────────────────────────────────────────
def score_rule(task: str, item_key: dict, answer: str):
    """Score a rule-based response. 1/0, or None if unscorable/ambiguous."""
    if task.startswith("Irony"):
        return S.score_irony(answer, item_key)
    # False Belief: the catch story is a surprise Yes/No, not a location question.
    if item_key.get("question_type") == "surprise_yesno":
        return S.score_irony(answer, item_key)  # reuses the Yes/No matcher (key has 'correct')
    return S.score_false_belief(answer, item_key)


# ── LLM-judge runner (free-text subtests) ─────────────────────────────────────
def _iter_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _done_judge_keys(out_path: Path) -> set[str]:
    """admin_call_ids already settled in the judge log (latest record wins)."""
    if not Path(out_path).exists():
        return set()
    latest: dict[str, str] = {}
    for v in _iter_jsonl(Path(out_path)):
        latest[v.get("admin_call_id")] = v.get("parse_status")
    settled = {parsing.PARSED_OK, parsing.FORMAT_VIOLATION, parsing.EMPTY}
    return {cid for cid, st in latest.items() if st in settled}


def _append_jsonl(path: Path, rec: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()


def build_judge_prompt_for(rec: dict, keys: dict, strange: list[dict]):
    """(prompt, max_score) for a judge-based admin record, or (None, None) to skip."""
    task = rec["condition"]
    if task not in S.JUDGE_BASED:
        return None, None
    answer = _answer_of(rec)
    if answer is None:
        return None, None
    item = _item_no(rec["item_id"])
    story = rec.get("user_prompt", "")
    item_key = keys.get(task, {}).get(str(item))
    if task == "Hinting":
        prompt = S.build_judge_prompt("Hinting", story, answer,
                                      target=(item_key or {}).get("target"))
    elif task == "Faux Pas":
        prompt = S.build_judge_prompt("Faux Pas", story, answer, key=item_key)
    else:  # Strange Stories
        crit = strange[item - 1] if 1 <= item <= len(strange) else None
        prompt = S.build_judge_prompt("Strange Stories", story, answer, criterion=crit)
    return prompt, _MAX_SCORE[task]


def run_judge(admin_log, judge_model: str, out_path, env=None, keys=None,
              strange=None, limit: int | None = None) -> dict:
    """Judge every free-text response in ``admin_log`` with ``judge_model``.

    Append-only and resumable (keyed by admin ``call_id``); halts cleanly on a
    circuit-breaker trip, leaving the rest for a later resume.
    """
    keys = keys if keys is not None else S.load_keys().get("keys", {})
    strange = strange if strange is not None else load_strange_criteria()
    spec = get_model(judge_model)
    client = make_client(spec, env)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _done_judge_keys(out_path)
    stats = {"judged": 0, "skipped": 0, "circuit_tripped": None, "by_status": {}}

    for rec in iter_records(Path(admin_log)):
        if rec["condition"] not in S.JUDGE_BASED:
            continue
        if rec["call_id"] in done:
            stats["skipped"] += 1
            continue
        prompt, max_score = build_judge_prompt_for(rec, keys, strange)
        if prompt is None:  # missing/empty answer -> available-case, not judged
            stats["skipped"] += 1
            continue
        try:
            raw, meta = client.complete(S.JUDGE_SYSTEM, prompt, JUDGE_SAMPLING)
        except CircuitBreakerTrip as trip:
            stats["circuit_tripped"] = trip.provider
            break
        except Exception as exc:  # noqa: BLE001 — logged; left for resume
            _append_jsonl(out_path, _verdict_rec(rec, judge_model, None, None, None,
                                                 parsing.ERROR, repr(exc)))
            stats["by_status"][parsing.ERROR] = stats["by_status"].get(parsing.ERROR, 0) + 1
            continue
        score, criterion, status = parse_judge_verdict(raw, max_score)
        _append_jsonl(out_path, _verdict_rec(rec, judge_model, raw, score, criterion, status))
        stats["by_status"][status] = stats["by_status"].get(status, 0) + 1
        stats["judged"] += 1
        if limit and stats["judged"] >= limit:
            break
    return stats


def _verdict_rec(rec, judge_model, raw, score, criterion, status, error=None):
    out = {
        "admin_call_id": rec["call_id"],
        "judge_model": judge_model,
        "model": rec.get("model"),
        "task": rec["condition"],
        "item": _item_no(rec["item_id"]),
        "k_index": rec["k_index"],
        "judge_raw": raw,
        "score": score,
        "criterion": criterion,
        "parse_status": status,
    }
    if error:
        out["error"] = error
    return out


# ── combine rule + judge into per-response scores ─────────────────────────────
def score_responses(admin_log, judge_log=None, keys=None) -> list[dict]:
    """One row per administered response: rule-scored or judge-scored.

    ``score`` is on the 0..1 scale (Strange Stories rescaled from 0/1/2);
    ``raw_score`` keeps the native scale. ``score`` is None for missing/unscorable.
    """
    keys = keys if keys is not None else S.load_keys().get("keys", {})
    verdicts: dict[str, dict] = {}
    if judge_log and Path(judge_log).exists():
        for v in _iter_jsonl(Path(judge_log)):
            verdicts[v.get("admin_call_id")] = v  # latest wins

    rows = []
    for rec in iter_records(Path(admin_log)):
        task = rec["condition"]
        if task not in TASK_TO_CSV:
            continue
        item = _item_no(rec["item_id"])
        answer = _answer_of(rec)
        score = raw_score = None
        if task in S.RULE_BASED:
            item_key = (keys.get(task, {}) or {}).get(str(item))
            if item_key is not None and answer is not None:
                raw_score = score_rule(task, item_key, answer)
                score = raw_score
        else:  # judge-based
            v = verdicts.get(rec["call_id"])
            if v and v.get("score") is not None:
                raw_score = v["score"]
                score = raw_score / 2.0 if task == "Strange Stories" else raw_score
        rows.append({
            "model": rec.get("model"), "task": task, "item": item,
            "k_index": rec["k_index"], "score": score, "raw_score": raw_score,
            "parse_status": rec.get("parse_status"),
        })
    return rows


# ── aggregation + calibration ─────────────────────────────────────────────────
def aggregate(rows):
    """Per-(model,task,item), per-(model,task), and per-model means (available-case)."""
    import pandas as pd

    df = pd.DataFrame(rows)
    scored = df.dropna(subset=["score"])
    return {
        "per_item": scored.groupby(["model", "task", "item"], as_index=False).score.mean(),
        "per_task": scored.groupby(["model", "task"], as_index=False).score.mean(),
        "per_model": scored.groupby("model", as_index=False).score.mean(),
        "n": len(df), "n_scored": len(scored),
    }


def load_published(model_name: str):
    """Published per-item mean pass-rate (available-case over the 15 sessions)."""
    import pandas as pd

    fname, filt, col_tmpl = CALIB_SOURCES[model_name]
    df = pd.read_csv(DATA_DIR / fname)
    for col, val in filt.items():
        df = df[df[col] == val]
    cols = [col_tmpl.format(s=s) for s in range(1, 16)]
    recs = []
    for _, r in df.iterrows():
        mean = pd.to_numeric(pd.Series([r[c] for c in cols]), errors="coerce").dropna().mean()
        recs.append({"csv_task": r["task"], "item": int(r["item"]), "pub_mean": mean})
    return pd.DataFrame(recs)


def calibration_table(rows, model_name: str, tol_overall: float = 0.05,
                      tol_subtest: float = 0.12):
    """Compare our auto-scores to Strachan's published per-item scores.

    Returns ``{table, overall_abs_diff, worst_abs_diff, passed_overall,
    passed_subtest, passed}``. ``table`` is per published subtest with our mean, the
    published mean, and the absolute difference; pass = overall ≤ ``tol_overall`` and
    every subtest ≤ ``tol_subtest`` (spec §5 tolerance band).
    """
    import pandas as pd

    ours = aggregate(rows)["per_item"].copy()
    if "model" in ours and ours["model"].notna().any():
        ours = ours[ours["model"] == model_name] if (ours["model"] == model_name).any() else ours
    ours["csv_task"] = ours["task"].map(TASK_TO_CSV)
    pub = load_published(model_name)
    merged = ours.merge(pub, on=["csv_task", "item"], how="inner")

    grp = merged.groupby("csv_task")
    table = pd.DataFrame({
        "ours": grp["score"].mean(),
        "published": grp["pub_mean"].mean(),
        "n_items": grp.size(),
    }).reset_index()
    table["abs_diff"] = (table["ours"] - table["published"]).abs()

    overall = abs(merged["score"].mean() - merged["pub_mean"].mean())
    worst = float(table["abs_diff"].max()) if len(table) else float("nan")
    return {
        "table": table,
        "overall_abs_diff": float(overall),
        "worst_abs_diff": worst,
        "passed_overall": overall <= tol_overall,
        "passed_subtest": worst <= tol_subtest,
        "passed": (overall <= tol_overall) and (worst <= tol_subtest),
    }
