"""Re-score completed E0 logs with TCM + ICM-Soft, then render the headline grid.

This does NOT run any model calls. It reads the existing per-model E0 logs
(``full.jsonl`` for the open-weight lane, ``subsample.jsonl`` for the proprietary
lane), assembles soft labels, scores them against the EXIST gold distribution,
and writes a metrics file carrying the JSD/KL/CE/F1 block *plus* the TCM matrices
(``tcm``) and ICM-Soft / ICM-Soft Norm (``icm_soft``, via the isolated PyEvALL
env named by ``PYEVALL_PYTHON``). It then builds the multi-panel TCM grid figure
(master spec §7.3) across whichever models were scored.

Metrics are written next to each log as ``<logstem>_metrics.json`` so the existing
``metrics.json`` (smoke / pre-ICM scoring) is never clobbered.

Usage (from repo root):
    python scripts/rescore_icm_tcm.py                      # default panel, EN
    python scripts/rescore_icm_tcm.py --models llama-3.3-70b-instruct claude-4-sonnet
    python scripts/rescore_icm_tcm.py --no-icm             # TCM only, skip PyEvALL
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # make llm_audit importable without install

from llm_audit import metrics as M  # noqa: E402
from llm_audit import tcm as TCM_mod  # noqa: E402
from llm_audit import icm_soft  # noqa: E402
from llm_audit import figures  # noqa: E402
from llm_audit.data import load_pool  # noqa: E402

# Default panel: open-weight full lane + proprietary subsample lane (EN).
OPEN_WEIGHT_FULL = [
    "deepseek-v3", "llama-3.1-8b-instruct", "llama-3.3-70b-instruct",
    "mistral-7b-instruct", "qwen-2.5-72b-instruct", "gemma-4-31b",
]
PROPRIETARY_SUB = ["claude-4-sonnet", "gpt-4.1", "gpt-4o"]
DEFAULT_PANEL = OPEN_WEIGHT_FULL + PROPRIETARY_SUB


def find_log(model: str, lang: str) -> tuple[Path, str] | None:
    """Locate a model's E0 log, preferring the full lane over the subsample lane."""
    base = ROOT / "results" / "e0" / lang / model
    for stem in ("full", "subsample"):
        p = base / f"{stem}.jsonl"
        if p.exists():
            return p, stem
    return None


def score_model(model: str, lang: str, gold_by_item: dict, *, run_icm: bool) -> dict | None:
    """Score one model's log, write ``<logstem>_metrics.json``, return its ``tcm`` block."""
    found = find_log(model, lang)
    if found is None:
        print(f"[rescore] {model}: no full.jsonl/subsample.jsonl under results/e0/{lang} — skip")
        return None
    log_path, stem = found
    records = list(M.iter_records(log_path))
    soft, status_counts = M.assemble_model_soft(records)
    scored = M.score_against_gold(soft, gold_by_item)
    out = {
        "model": model, "lang": lang, "itemset": stem, "log": str(log_path.name),
        "status_counts": {"|".join(k): v for k, v in status_counts.items()},
        "metrics": {"|".join(k): v for k, v in scored.items()},
    }
    out["tcm"] = TCM_mod.tcm_for_exist(soft, gold_by_item)
    if run_icm:
        try:
            out["icm_soft"] = icm_soft.score_icm_soft(soft, gold_by_item)
        except RuntimeError as exc:  # no interpreter / PyEvALL failure
            out["icm_soft_error"] = str(exc)
            print(f"[rescore] {model}: ICM-Soft skipped — {exc}")
    mpath = log_path.with_name(f"{stem}_metrics.json")
    mpath.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    n_icm = "ok" if "icm_soft" in out else ("err" if "icm_soft_error" in out else "off")
    print(f"[rescore] {model}: scored {len(records)} records from {stem}.jsonl "
          f"(tcm cells={len(out['tcm'])}, icm={n_icm}) -> {mpath.name}")
    return out["tcm"]


def build_figure(tcm_by_row: dict[str, dict], lang: str) -> None:
    """Render the multi-panel TCM grid over the Format-C cells common to all models."""
    if not tcm_by_row:
        print("[rescore] no scored models — figure skipped")
        return
    # Columns = Format-C selectors present in every scored model (intersection),
    # so the panel is consistent across the open-weight and proprietary lanes.
    common = None
    for block in tcm_by_row.values():
        keys = {k for k in block if k.endswith("|C")}
        common = keys if common is None else (common & keys)
    common = sorted(common or [])
    if not common:
        print("[rescore] no shared Format-C TCM cells across models — figure skipped")
        return
    columns = {key: key for key in common}  # label == selector for the first-draft grid
    panel = figures.panel_from_metrics(tcm_by_row, columns)
    out_dir = ROOT / "outputs" / "figures"
    path = figures.tcm_grid(
        panel, out_dir, name=f"fig_tcm_grid_{lang}",
        title=f"TCM grid (EXIST {lang.upper()}, Format C) — true→predicted mass",
    )
    print(f"[rescore] TCM grid -> {path} ({len(panel)} rows x {len(common)} cols)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Re-score E0 logs with TCM + ICM-Soft.")
    ap.add_argument("--models", nargs="+", default=DEFAULT_PANEL)
    ap.add_argument("--lang", default="en", choices=("en", "es"))
    ap.add_argument("--no-icm", action="store_true", help="skip ICM-Soft (TCM only)")
    ap.add_argument("--no-figure", action="store_true", help="skip the TCM grid figure")
    args = ap.parse_args(argv)

    df = load_pool(args.lang)
    gold_by_item = M.gold_lookup(df)

    tcm_by_row: dict[str, dict] = {}
    for model in args.models:
        block = score_model(model, args.lang, gold_by_item, run_icm=not args.no_icm)
        if block is not None:
            tcm_by_row[model] = block

    if not args.no_figure:
        build_figure(tcm_by_row, args.lang)
    print(f"[rescore] done: {len(tcm_by_row)}/{len(args.models)} models scored")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
