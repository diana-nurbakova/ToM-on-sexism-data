"""E2 run driver: build original+swapped cells and run them through the E0 runner.

Each E2 item yields two variants — ``orig`` (original text) and ``swap`` (frozen
woman↔man swapped text) — crossed with the bare/defined prompt factor (§5.1b) and
subtasks/formats. The runner's prompt hook decodes ``condition = "{spec}|{variant}"``
and builds the E0 prompt on the variant's text. Open-weight models run concurrently
on DeepInfra; proprietary stay serial (the runner guards this).
"""

from __future__ import annotations

from pathlib import Path

from .. import config, prompts
from ..clients import make_client
from ..runner import Cell, Runner

E2_SUBTASKS = ("1.1", "1.3")   # detection flip + per-category flip
E2_SPECS = ("bare", "defined")
E2_FORMATS = ("C",)            # behaviourally-grounded soft label; A/B addable


def _e2_prompt(cell: Cell):
    spec, _variant = cell.condition.split("|", 1)
    return prompts.build_prompt(cell.subtask, spec, cell.language, cell.format, cell.text)


def build_e2_cells(sample, lang: str = "en", subtasks=E2_SUBTASKS,
                   specs=E2_SPECS, formats=E2_FORMATS, k=config.K_SAMPLES):
    """Expand the E2 sample into original/swapped cells."""
    cells: list[Cell] = []
    for _, r in sample.iterrows():
        variants = (("orig", r["text"]), ("swap", r["swapped"]))
        for variant, text in variants:
            for subtask in subtasks:
                for spec in specs:
                    for fmt in formats:
                        draws = range(k) if fmt == "C" else range(1)
                        for ki in draws:
                            cells.append(Cell("E2", f"{spec}|{variant}", subtask, lang,
                                              fmt, str(r["item_id"]), text, ki))
    return cells


def run_e2(model: str, sample, out_dir: Path, env=None, lang: str = "en",
           workers: int = 1, max_tokens: int = 256, **cell_kw) -> dict:
    """Run E2 for one model. ``workers>1`` only permitted on non-proprietary backends."""
    spec = config.get_model(model)
    if workers > 1 and spec.is_proprietary:
        raise ValueError(f"workers>1 not allowed for proprietary backend {spec.backend}")
    client = make_client(spec, env)
    log_path = Path(out_dir) / f"{model}.jsonl"
    runner = Runner(client, spec, log_path, sampling={"max_tokens": max_tokens},
                    prompt_fn=_e2_prompt, max_workers=workers)
    return runner.run(build_e2_cells(sample, lang, **cell_kw))
