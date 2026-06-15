"""Administer, judge-score, and calibrate the Strachan ToM battery (spec sec. 4-5).

Per model: administer the released battery (15 sessions/item, resumable), run the
LLM-judge over the free-text subtests, combine with the rule-based scores, and write
per-item / per-subtest scores under the Strachan-simplified key. For the three
reproduction models it also writes the calibration table vs Strachan's published
scores and prints pass/fail against the ~5% overall / ~12% worst-subtest band.

Lane etiquette: open-weight administration shares the DeepInfra lane with the running
E0 jobs, so keep --workers <= 4 (the battery is small, ~6-11k calls/model).

Usage (examples):
    python scripts/run_strachan_scoring.py --calibration            # 3 reproduction models
    python scripts/run_strachan_scoring.py --models gpt-4o claude-4-sonnet
    python scripts/run_strachan_scoring.py --calibration --score-only   # skip admin/judge
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_audit.config import get_model
from llm_audit.e3 import strachan as S
from llm_audit.e3 import strachan_scoring as SC

OUT = Path("results/strachan")
CALIBRATION_MODELS = ["gpt-4-strachan", "gpt-3.5-turbo-strachan", "llama-2-70b-strachan"]
DEFAULT_JUDGE = "claude-4-sonnet"


def process(model: str, judge_model: str, workers: int, score_only: bool,
            calibrate: bool) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    admin_log = OUT / f"{model}.jsonl"
    judge_log = OUT / f"{model}.judge.jsonl"

    if not score_only:
        # Proprietary backends must stay serial (rate-limit/ban safety); the runner
        # enforces this, so only request parallelism on open-weight lanes.
        w = 1 if get_model(model).is_proprietary else max(1, workers)
        stats = S.run_strachan(model, OUT, workers=w)
        print(f"[strachan] {model}: admin ran={stats['ran']} skipped={stats['skipped']} "
              f"circuit={stats['circuit_tripped']}")
        jstats = SC.run_judge(admin_log, judge_model, judge_log)
        print(f"[strachan] {model}: judge judged={jstats['judged']} "
              f"by_status={jstats['by_status']} circuit={jstats['circuit_tripped']}")

    if not admin_log.exists():
        print(f"[strachan] {model}: no admin log at {admin_log}; skipping scoring")
        return

    rows = SC.score_responses(admin_log, judge_log if judge_log.exists() else None)
    agg = SC.aggregate(rows)
    (OUT / f"{model}.scores.json").write_text(json.dumps({
        "model": model, "n": agg["n"], "n_scored": agg["n_scored"],
        "per_task": agg["per_task"].to_dict(orient="records"),
        "per_model": agg["per_model"].to_dict(orient="records"),
        "per_item": agg["per_item"].to_dict(orient="records"),
    }, indent=2), encoding="utf-8")
    print(f"[strachan] {model}: scored {agg['n_scored']}/{agg['n']} -> {model}.scores.json")

    if calibrate and model in SC.CALIB_SOURCES:
        cal = SC.calibration_table(rows, model)
        (OUT / f"{model}.calibration.json").write_text(json.dumps({
            "model": model,
            "overall_abs_diff": cal["overall_abs_diff"],
            "worst_abs_diff": cal["worst_abs_diff"],
            "passed": cal["passed"],
            "passed_overall": cal["passed_overall"],
            "passed_subtest": cal["passed_subtest"],
            "table": cal["table"].to_dict(orient="records"),
        }, indent=2), encoding="utf-8")
        verdict = "PASS" if cal["passed"] else "FAIL"
        print(f"[strachan] {model}: calibration {verdict} "
              f"(overall diff={cal['overall_abs_diff']:.3f}, "
              f"worst diff={cal['worst_abs_diff']:.3f})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="*", default=None,
                    help="models to run (default: the 3 calibration models)")
    ap.add_argument("--calibration", action="store_true",
                    help="run the 3 reproduction models and the calibration check")
    ap.add_argument("--judge-model", default=DEFAULT_JUDGE)
    ap.add_argument("--workers", type=int, default=4, help="open-weight lane only; <=4")
    ap.add_argument("--score-only", action="store_true",
                    help="skip administration + judging; rescore existing logs")
    args = ap.parse_args()

    models = args.models or (CALIBRATION_MODELS if args.calibration else CALIBRATION_MODELS)
    for model in models:
        process(model, args.judge_model, args.workers, args.score_only,
                calibrate=args.calibration or model in SC.CALIB_SOURCES)


if __name__ == "__main__":
    main()
