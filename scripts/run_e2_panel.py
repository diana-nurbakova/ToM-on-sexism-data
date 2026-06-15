"""Run E2 (counterfactual swap) across the open-weight panel on DeepInfra/OpenRouter.

600-item equal-N sample (120/category), original+swapped × {bare,defined} × {1.1,1.3}
× Format C K=6. Per-model: run via the E2 driver (idempotent/resumable), then compute
flip-rate / JSD / asymmetry with the per-category, per-direction, per-invariance
breakdowns and write a *_flip.json. EN only (the swap engine is English).
"""

from __future__ import annotations

import json
from pathlib import Path

from llm_audit.data import load_pool
from llm_audit.e2 import driver, sampling, tagging
from llm_audit.e2 import metrics as E2M

OUT = Path("results/e2/en")
# Open-weight panel; Gemma routes via OpenRouter (fewer workers), rest via DeepInfra.
MODELS = ["llama-3.3-70b-instruct", "qwen-2.5-72b-instruct", "deepseek-v3",
          "mistral-7b-instruct", "llama-3.1-8b-instruct", "gemma-4-31b"]
_OPENROUTER = {"gemma-4-31b", "gemma-3-27b-it"}  # fewer workers on OpenRouter


def main() -> None:
    df = load_pool("en")
    sample = tagging.tag_e2_sample(sampling.build_e2_sample(df, "en", target=120, seed=42))
    meta = {str(r.item_id): {"category": r.strat_category, "direction": r.swap_direction,
                             "invariance": r.invariance_class}
            for r in sample.itertuples()}
    print(f"[E2] sample={len(sample)} items; panel={MODELS}")
    for model in MODELS:
        workers = 4 if model in _OPENROUTER else 8
        stats = driver.run_e2(model, sample, OUT, lang="en", workers=workers, max_tokens=256)
        print(f"[E2] {model}: ran={stats['ran']} skipped={stats['skipped']} "
              f"by_status={stats['by_status']} circuit={stats['circuit_tripped']}")
        res = E2M.compute_e2(OUT / f"{model}.jsonl", meta)
        (OUT / f"{model}_flip.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(f"[E2] {model}: flip metrics -> {model}_flip.json")


if __name__ == "__main__":
    main()
