"""Proprietary lane (master spec §5.1 cost-control policy): Claude 4 Sonnet, GPT-4.1, GPT-4o.

E0 on a SHARED 1,500-item stratified subsample (5 categories × 3 agreement bins,
~100/cell, fixed seed, identical items across the three models), Format C only
(the primary format), K=3, both bare/defined, all three subtasks. Then E2 on the
600-item swap sample at K=3 Format C. All serial (proprietary backends; the runner
forbids concurrency there). Idempotent/resumable. EN only (language-outer).

The reduced K/format is recorded with the results; cross-lane comparison to the
open-weight K=6 lane uses metrics.cross_lane_compare (format + K matched).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from llm_audit import config
from llm_audit.clients import make_client
from llm_audit.data import T13_CATEGORIES, load_pool
from llm_audit.runner import Runner, build_e0_cells
from llm_audit.e2 import driver, sampling, tagging
from llm_audit.e2 import metrics as E2M

PROPRIETARY = ["claude-4-sonnet", "gpt-4.1", "gpt-4o"]
E0_SUBSAMPLE_PER_CELL = 100
SEED = 42
E0_OUT = Path("results/e0/en")
E2_OUT = Path("results/e2/en")


def build_e0_subsample(df: pd.DataFrame, per_cell=E0_SUBSAMPLE_PER_CELL, seed=SEED):
    """Shared 1,500-item subsample: 5 categories × 3 agreement bins, ~per_cell each."""
    pool = df[df["strat_category"].notna()]
    parts, shortfalls = [], []
    for cat in T13_CATEGORIES:
        for b in ("high", "mixed", "low"):
            cell = pool[(pool["strat_category"] == cat) & (pool["agreement_bin"] == b)]
            take = min(per_cell, len(cell))
            if len(cell) < per_cell:
                shortfalls.append({"category": cat, "bin": b, "available": len(cell)})
            if take:
                parts.append(cell.sample(n=take, random_state=seed))
    s = pd.concat(parts).reset_index(drop=True)
    s.attrs["shortfalls"] = shortfalls
    return s


def main() -> None:
    df = load_pool("en")
    sub = build_e0_subsample(df)
    print(f"[PROP] E0 subsample n={len(sub)} shortfalls={sub.attrs['shortfalls']}")
    e0_items = list(zip(sub["item_id"], sub["text"]))

    # E2 sample (600) + per-item metadata for the flip breakdowns.
    e2_sample = tagging.tag_e2_sample(sampling.build_e2_sample(df, "en", target=120, seed=SEED))
    e2_meta = {str(r.item_id): {"category": r.strat_category, "direction": r.swap_direction,
                                "invariance": r.invariance_class} for r in e2_sample.itertuples()}

    for model in PROPRIETARY:
        spec = config.get_model(model)
        client = make_client(spec)
        # ── E0: 1,500 subsample, Format C, K=3, both specs, all subtasks ──
        cells = build_e0_cells(e0_items, "en", ("1.1", "1.2", "1.3"),
                               ("bare", "defined"), ("C",), k_samples=3)
        r = Runner(client, spec, E0_OUT / model / "subsample.jsonl",
                   sampling={"max_tokens": 256})  # serial (proprietary)
        st = r.run(cells)
        print(f"[PROP] {model} E0: ran={st['ran']} skipped={st['skipped']} "
              f"by_status={st['by_status']} circuit={st['circuit_tripped']}")
        if st["circuit_tripped"]:
            print(f"[PROP] circuit tripped on {model}; halting proprietary lane.")
            break
        # ── E2: 600 swap sample, Format C, K=3, both specs, 1.1+1.3 ──
        e2st = driver.run_e2(model, e2_sample, E2_OUT, lang="en", workers=1,
                             max_tokens=256, subtasks=("1.1", "1.3"),
                             specs=("bare", "defined"), formats=("C",), k=3)
        print(f"[PROP] {model} E2: ran={e2st['ran']} by_status={e2st['by_status']} "
              f"circuit={e2st['circuit_tripped']}")
        if not e2st["circuit_tripped"]:
            res = E2M.compute_e2(E2_OUT / f"{model}.jsonl", e2_meta)
            (E2_OUT / f"{model}_flip.json").write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
