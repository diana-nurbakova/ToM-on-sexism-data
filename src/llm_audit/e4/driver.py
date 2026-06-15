"""E4 run driver (base-vs-instruct counter-priming) + framing-shift analysis.

Builds E4 cells (condition = ``"{mode}|{framing}"``) and runs them through the E0
``Runner`` via its prompt hook. The counter-prime conditions run on subtask 1.1,
originals only (H6b); the H6a base-instruct safety-prior gap reuses the E0 neutral
prompts on originals + E2 pronoun-swap and the E0 defined 1.3, so no E4-specific
prompt is needed there. E4 is open-weight-only (base/instruct pairs), so the
high-pressure framings carry no provider-suspension risk and may run concurrently.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .. import config, metrics as M
from ..clients import make_client
from ..runner import Cell, Runner
from .framings import CONDITIONS, RELAXING, TIGHTENING, build_e4_prompt

E4_SUBTASKS = ("1.1",)
E4_FORMATS = ("C",)               # Format C primary (base-model parseability, §1/§7.4)
E4_MODES = ("user",)              # user-stated primary; ("user","persona") for the slice


def _e4_prompt(cell: Cell):
    mode, framing = cell.condition.split("|", 1)
    return build_e4_prompt(framing, cell.subtask, cell.language, cell.format,
                           cell.text, mode=mode)


def build_e4_cells(items, lang="en", subtasks=E4_SUBTASKS, conditions=CONDITIONS,
                   modes=E4_MODES, formats=E4_FORMATS, k=config.K_SAMPLES):
    """``items`` = iterable of (item_id, text). condition = ``"{mode}|{framing}"``."""
    cells: list[Cell] = []
    for item_id, text in items:
        for mode in modes:
            for cond in conditions:
                for subtask in subtasks:
                    for fmt in formats:
                        draws = range(k) if fmt == "C" else range(1)
                        for ki in draws:
                            cells.append(Cell("E4", f"{mode}|{cond}", subtask, lang,
                                              fmt, str(item_id), text, ki))
    return cells


def run_e4(model: str, items, out_dir: Path, env=None, lang="en",
           workers: int = 1, max_tokens: int = 256, **cell_kw) -> dict:
    spec = config.get_model(model)
    if workers > 1 and spec.is_proprietary:
        raise ValueError(f"workers>1 not allowed for proprietary backend {spec.backend}")
    client = make_client(spec, env)
    runner = Runner(client, spec, Path(out_dir) / f"{model}.jsonl",
                    sampling={"max_tokens": max_tokens}, prompt_fn=_e4_prompt,
                    max_workers=workers)
    return runner.run(build_e4_cells(items, lang, **cell_kw))


# ── Analysis: per-framing shift in P(sexist) vs the same-mode neutral (H6b) ───
def framing_shift(log_path) -> dict:
    """Per (mode, format): each framing's mean P(sexist) and paired shift vs neutral.

    Sign convention: relaxing framings should shift P(sexist) *down* (negative),
    the tightening framing *up* (positive); ``direction_ok`` flags whether the
    measured sign matches the predicted pole (the §8 direction check). Paired by item.
    """
    soft, _ = M.assemble_model_soft(M.iter_records(log_path))
    by_mf: dict[tuple, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for (cond, sub, fmt, item), entry in soft.items():
        if sub != "1.1":
            continue
        mode, framing = cond.split("|", 1)
        by_mf[(mode, fmt)][framing][item] = entry["soft"].get("sexist", 0.0)

    out: dict[str, dict] = {}
    for (mode, fmt), framings in by_mf.items():
        neutral = framings.get("neutral", {})
        block: dict[str, dict] = {}
        for framing, per_item in framings.items():
            shifts = [p - neutral[item] for item, p in per_item.items() if item in neutral]
            mean_shift = _mean(shifts) if framing != "neutral" else 0.0
            entry = {
                "mean_p_sexist": _mean(list(per_item.values())),
                "mean_shift_vs_neutral": mean_shift,
                "n_paired": len(shifts),
            }
            if framing in RELAXING:
                entry["direction_ok"] = mean_shift is not None and mean_shift <= 0
            elif framing in TIGHTENING:
                entry["direction_ok"] = mean_shift is not None and mean_shift >= 0
            block[framing] = entry
        out["|".join((mode, fmt))] = block
    return out


def _mean(xs):
    return float(sum(xs) / len(xs)) if xs else None
