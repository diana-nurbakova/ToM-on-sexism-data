"""E3 run driver + shift-vs-neutral analysis.

Builds scaffold cells (condition = scaffold name) and runs them through the E0
runner via its prompt hook. The within-E3 measurement is each scaffold's shift in
alignment-with-gold (JSD) relative to the neutral baseline, per subtask and — for
1.3 — per category (the scaffold × category interaction, H7c). The cross-model H7b
step (residual correlation with the Strachan covariate) is separate and needs the
external benchmark scores.
"""

from __future__ import annotations

from pathlib import Path

from .. import config, metrics as M
from ..clients import make_client
from ..runner import Cell, Runner
from .scaffolds import CONDITIONS, build_scaffold_prompt

E3_SUBTASKS = ("1.1", "1.3")
E3_FORMATS = ("C",)


def _e3_prompt(cell: Cell):
    return build_scaffold_prompt(cell.condition, cell.subtask, cell.language,
                                 cell.format, cell.text)


def build_e3_cells(items, lang="en", subtasks=E3_SUBTASKS, conditions=CONDITIONS,
                   formats=E3_FORMATS, k=config.K_SAMPLES):
    """``items`` = iterable of (item_id, text). condition = scaffold name."""
    cells = []
    for item_id, text in items:
        for cond in conditions:
            for subtask in subtasks:
                for fmt in formats:
                    draws = range(k) if fmt == "C" else range(1)
                    for ki in draws:
                        cells.append(Cell("E3", cond, subtask, lang, fmt,
                                          str(item_id), text, ki))
    return cells


def run_e3(model: str, items, out_dir: Path, env=None, lang="en",
           workers: int = 1, max_tokens: int = 256, **cell_kw) -> dict:
    spec = config.get_model(model)
    if workers > 1 and spec.is_proprietary:
        raise ValueError(f"workers>1 not allowed for proprietary backend {spec.backend}")
    client = make_client(spec, env)
    runner = Runner(client, spec, Path(out_dir) / f"{model}.jsonl",
                    sampling={"max_tokens": max_tokens}, prompt_fn=_e3_prompt,
                    max_workers=workers)
    return runner.run(build_e3_cells(items, lang, **cell_kw))


def shift_vs_neutral(log_path, gold_by_item: dict) -> dict:
    """Per (subtask, format): each scaffold's mean JSD-to-gold and its shift vs neutral.

    A negative shift = the scaffold moved the model *closer* to the human distribution
    (helped); positive = hurt. The per-category 1.3 breakdown is the H7c interaction.
    """
    soft, _ = M.assemble_model_soft(M.iter_records(log_path))
    scored = M.score_against_gold(soft, gold_by_item)  # keyed (cond, subtask, fmt)
    out = {}
    # group by (subtask, fmt); compute shift relative to the neutral condition
    by_sf: dict[tuple, dict] = {}
    for (cond, subtask, fmt), res in scored.items():
        by_sf.setdefault((subtask, fmt), {})[cond] = res
    for (subtask, fmt), conds in by_sf.items():
        neutral = conds.get("neutral", {}).get("mean_jsd")
        out["|".join((subtask, fmt))] = {
            cond: {
                "mean_jsd": res["mean_jsd"],
                "shift_vs_neutral": (res["mean_jsd"] - neutral) if neutral is not None else None,
                "macro_f1": res["f1"]["macro_f1"],
            }
            for cond, res in conds.items()
        }
    return out
