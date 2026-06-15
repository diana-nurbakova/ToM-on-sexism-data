"""E0 CLI orchestrator: select items, run cells, optionally score.

Examples (from repo root, inside the venv):
    python -m llm_audit.run_e0 --itemset pilot --lang en --model claude-4-sonnet
    python -m llm_audit.run_e0 --itemset smoke --lang en --model claude-4-sonnet --metrics
    python -m llm_audit.run_e0 --itemset noise-band --lang en --model claude-4-sonnet --repeats 5

Item sets:
    pilot       20 stratified items, Format A, both specs, all subtasks (parseability).
    regression  the regression set (Format C, K=6).
    noise-band  the regression set, Format C, K = 6*repeats (split for the noise band).
    smoke       ~300 stratified items, all formats/specs/subtasks (full pipeline).
    full        the whole language pool.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import config, metrics as M, regression
from .clients import make_client
from .data import load_pool, stratified_sample
from .runner import Runner, build_e0_cells

DEFAULTS = {
    "pilot": {"n": 20, "formats": ("A",), "specs": ("bare", "defined"),
              "subtasks": ("1.1", "1.2", "1.3")},
    "regression": {"formats": ("C",), "specs": ("bare", "defined"),
                   "subtasks": ("1.1", "1.2", "1.3")},
    "noise-band": {"formats": ("C",), "specs": ("bare",), "subtasks": ("1.1", "1.2", "1.3")},
    "smoke": {"n": 300, "formats": ("A", "B", "C"), "specs": ("bare", "defined"),
              "subtasks": ("1.1", "1.2", "1.3")},
    "full": {"formats": ("A", "B", "C"), "specs": ("bare", "defined"),
             "subtasks": ("1.1", "1.2", "1.3")},
}


def select_items(df, itemset: str, lang: str, n: int | None, seed: int):
    """Return a list of (item_id, text) for the chosen item set."""
    if itemset in ("regression", "noise-band"):
        reg = regression.build_regression_set(df, lang, seed)
        return [(r["exist_item_id"], r["text"]) for r in reg]
    if itemset == "full":
        return list(zip(df["item_id"], df["text"]))
    n = n or DEFAULTS[itemset]["n"]
    sample = stratified_sample(df, n, by=["agreement_bin"], seed=seed)
    return list(zip(sample["item_id"], sample["text"]))


def default_log_path(itemset: str, lang: str, model: str) -> Path:
    name = "calls.jsonl" if itemset not in ("regression", "noise-band") else f"{itemset}.jsonl"
    return config.RESULTS_ROOT / "e0" / lang / model / name


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="E0 run harness")
    p.add_argument("--itemset", required=True, choices=list(DEFAULTS))
    p.add_argument("--lang", default="en", choices=("en", "es"))
    p.add_argument("--model", default="claude-4-sonnet")
    p.add_argument("--subtasks", nargs="+", default=None)
    p.add_argument("--formats", nargs="+", default=None)
    p.add_argument("--specs", nargs="+", default=None)
    p.add_argument("--n", type=int, default=None, help="item count for pilot/smoke")
    p.add_argument("--k", type=int, default=config.K_SAMPLES, help="K draws for Format C")
    p.add_argument("--repeats", type=int, default=5, help="repeats for noise-band")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--workers", type=int, default=1,
                   help="concurrent calls; >1 only for DeepInfra/local, never proprietary")
    p.add_argument("--out", default=None, help="override log path")
    p.add_argument("--metrics", action="store_true", help="score against gold after the run")
    p.add_argument("--icm", action="store_true",
                   help="also compute EXIST ICM-Soft/Norm via the isolated PyEvALL env "
                        "(needs PYEVALL_PYTHON; implies --metrics)")
    p.add_argument("--dry-run", action="store_true", help="build cells, print plan, do not call")
    args = p.parse_args(argv)

    d = DEFAULTS[args.itemset]
    subtasks = tuple(args.subtasks or d["subtasks"])
    formats = tuple(args.formats or d["formats"])
    specs = tuple(args.specs or d["specs"])
    k = args.k * args.repeats if args.itemset == "noise-band" else args.k

    df = load_pool(args.lang)
    items = select_items(df, args.itemset, args.lang, args.n, args.seed)
    cells = build_e0_cells(items, args.lang, subtasks, specs, formats, k_samples=k)
    log_path = Path(args.out) if args.out else default_log_path(args.itemset, args.lang, args.model)

    print(f"[E0] itemset={args.itemset} lang={args.lang} model={args.model} "
          f"items={len(items)} cells={len(cells)} -> {log_path}")
    if args.dry_run:
        return 0

    spec = config.get_model(args.model)
    client = make_client(spec)
    if args.workers > 1 and spec.is_proprietary:
        p.error(f"--workers>1 is not allowed for proprietary backend '{spec.backend}' "
                "(rate-limit/ban safety); use serial.")
    runner = Runner(client, spec, log_path, seed=args.seed,
                    sampling={"max_tokens": args.max_tokens}, max_workers=args.workers)
    stats = runner.run(cells)
    print(f"[E0] ran={stats['ran']} skipped={stats['skipped']} "
          f"by_status={stats['by_status']} circuit={stats['circuit_tripped']}")

    if args.metrics or args.icm:
        records = list(M.iter_records(log_path))
        soft, status_counts = M.assemble_model_soft(records)
        gold_by_item = M.gold_lookup(df)
        scored = M.score_against_gold(soft, gold_by_item)
        out = {
            "model": args.model, "lang": args.lang, "itemset": args.itemset,
            "status_counts": {"|".join(k): v for k, v in status_counts.items()},
            "metrics": {"|".join(k): v for k, v in scored.items()},
        }
        from . import tcm as TCM_mod
        out["tcm"] = TCM_mod.tcm_for_exist(soft, gold_by_item)
        if args.icm:
            from . import icm_soft
            try:
                out["icm_soft"] = icm_soft.score_icm_soft(soft, gold_by_item)
            except RuntimeError as exc:  # no interpreter / runner failure
                out["icm_soft_error"] = str(exc)
                print(f"[E0] ICM-Soft skipped: {exc}")
        mpath = log_path.with_name("metrics.json")
        mpath.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[E0] metrics -> {mpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
