"""E1 adjudication worksheet workflow CLI (OSF pre-reg §6).

Two subcommands frame the author's construct-derivation loop:

    # 1. build the worksheet from the bin-first sample (rule first-pass pre-filled)
    python scripts/run_e1_worksheet.py build  --lang en

    # ... author fills `invariance_class` + `justification` for needs_adjudication rows ...

    # 2. check what is still unresolved at any time
    python scripts/run_e1_worksheet.py status --lang en

    # 3. finalise: validate + export the frozen partition E2/E5 consume
    python scripts/run_e1_worksheet.py finalize --lang en

``build`` preserves any author cells already entered (rebuild-safe). ``finalize``
refuses to export while any adjudication row is blank.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_audit.data import load_pool
from llm_audit.e1 import sampling, worksheet
from llm_audit.regression import build_regression_set  # disjointness (sample excludes these)

RESULTS = Path(__file__).resolve().parents[1] / "results" / "e1"


def _worksheet_path(lang: str) -> Path:
    return RESULTS / f"e1_sample_{lang}_worksheet.csv"


def _regression_ids(pool, lang: str) -> set:
    """Item_ids in the prompt-regression set, held out for disjointness (pre-reg §4)."""
    return {str(it["exist_item_id"]) for it in build_regression_set(pool, lang)}


def cmd_build(args) -> None:
    pool = load_pool(args.lang)
    exclude = _regression_ids(pool, args.lang)
    sample = sampling.build_sample(pool, lang=args.lang, exclude_ids=exclude, seed=args.seed)
    df = worksheet.build_worksheet(sample)
    path = worksheet.write_worksheet(df, _worksheet_path(args.lang), merge_prior=not args.no_merge)
    flagged = int(df["needs_adjudication"].sum())
    print(f"wrote {path} ({len(df)} items, {flagged} need adjudication)")
    shortfalls = sample.attrs.get("shortfalls", [])
    if shortfalls:
        print(f"sparse-cell shortfalls (take-all, recorded): {shortfalls}")


def cmd_status(args) -> None:
    path = _worksheet_path(args.lang)
    unresolved = worksheet.validate_worksheet(path)
    if not unresolved:
        print(f"{path.name}: ready to finalise (no unresolved adjudication rows)")
        return
    print(f"{path.name}: {len(unresolved)} item(s) still need `invariance_class`:")
    for u in unresolved[:30]:
        print(f"  {u['item_id']:>10}  [{u['rule_class']}]  {u['rule_reason']}")
    if len(unresolved) > 30:
        print(f"  … and {len(unresolved) - 30} more")


def cmd_finalize(args) -> None:
    summary = worksheet.finalize_partition(_worksheet_path(args.lang), lang=args.lang)
    print(json.dumps(summary, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description="E1 adjudication worksheet workflow")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="build/refresh the worksheet from the sample")
    b.add_argument("--lang", default="en")
    b.add_argument("--seed", type=int, default=sampling.SEED)
    b.add_argument("--no-merge", action="store_true",
                   help="do not preserve author cells from an existing worksheet")
    b.set_defaults(func=cmd_build)

    s = sub.add_parser("status", help="list rows still needing adjudication")
    s.add_argument("--lang", default="en")
    s.set_defaults(func=cmd_status)

    f = sub.add_parser("finalize", help="validate + export the frozen partition")
    f.add_argument("--lang", default="en")
    f.set_defaults(func=cmd_finalize)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
