"""E5 run driver (Cialdini persuasion) + per-condition shift analysis.

Builds E5 cells (condition = Cialdini condition name) and runs them through the E0
``Runner`` via its prompt hook. The commitment conditions need a per-item exemplar,
supplied through a fixed pairing schedule captured in the prompt closure. The
within-E5 measurement is each condition's shift in P(sexist) relative to the neutral
baseline (H9a effect sizes); the verdict-vs-majority direction split (H9b) is layered
on top with the gold majority per item.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .. import config, metrics as M
from ..clients import make_client
from ..runner import Cell, Runner
from .conditions import CONDITIONS, COMMITMENT_SET, build_e5_prompt

E5_SUBTASKS = ("1.1",)              # 1.1 primary; 1.3 secondary if requested
E5_FORMATS = ("A", "B", "C")       # elicitation format is an E5 methods sub-question (§9)


def _commitment_direction(condition: str) -> str:
    return "sexist" if condition == "commitment_sexist" else "not_sexist"


def make_e5_prompt_fn(pairing: dict[str, dict[str, str]]):
    """Build the runner prompt hook, closing over the exemplar pairing schedule."""
    def _fn(cell: Cell):
        exemplar = None
        if cell.condition in COMMITMENT_SET:
            exemplar = pairing[cell.item_id][_commitment_direction(cell.condition)]
        return build_e5_prompt(cell.condition, cell.subtask, cell.language,
                               cell.format, cell.text, exemplar)
    return _fn


def build_e5_cells(items, lang="en", subtasks=E5_SUBTASKS, conditions=CONDITIONS,
                   formats=E5_FORMATS, k=config.K_SAMPLES):
    """``items`` = iterable of (item_id, text). condition = Cialdini condition name."""
    cells: list[Cell] = []
    for item_id, text in items:
        for cond in conditions:
            for subtask in subtasks:
                for fmt in formats:
                    draws = range(k) if fmt == "C" else range(1)
                    for ki in draws:
                        cells.append(Cell("E5", cond, subtask, lang, fmt,
                                          str(item_id), text, ki))
    return cells


def run_e5(model: str, items, pairing, out_dir: Path, env=None, lang="en",
           workers: int = 1, max_tokens: int = 256, **cell_kw) -> dict:
    """Run E5 for one model. ``pairing`` is the exemplar schedule (``exemplars`` module).

    Open-weight-first (master spec §5.6): the high-risk proprietary cells stay serial
    and are gated by the runner's circuit-breaker; ``workers>1`` is rejected on
    proprietary backends.
    """
    spec = config.get_model(model)
    if workers > 1 and spec.is_proprietary:
        raise ValueError(f"workers>1 not allowed for proprietary backend {spec.backend}")
    client = make_client(spec, env)
    runner = Runner(client, spec, Path(out_dir) / f"{model}.jsonl",
                    sampling={"max_tokens": max_tokens},
                    prompt_fn=make_e5_prompt_fn(pairing), max_workers=workers)
    return runner.run(build_e5_cells(items, lang, **cell_kw))


# ── Analysis: per-condition shift in P(sexist) vs neutral (H9a) ───────────────
def condition_shift(log_path, gold_by_item: dict | None = None) -> dict:
    """Per (subtask, format): each condition's mean P(sexist) and paired shift vs neutral.

    The shift is paired by item (condition minus neutral on the same item), so only
    items with both a neutral and a condition soft label contribute. When
    ``gold_by_item`` is supplied, the shift is additionally split by whether the
    condition's framed verdict matched or contradicted the human majority (H9b): for
    each condition the ``toward_majority``/``against_majority`` means report whether
    the persuasion pushed P(sexist) toward or away from the gold majority label.
    """
    soft, _ = M.assemble_model_soft(M.iter_records(log_path))
    # group P(sexist) by (subtask, fmt) -> {condition: {item: p_sexist}}
    by_sf: dict[tuple, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for (cond, sub, fmt, item), entry in soft.items():
        if sub != "1.1":
            continue
        by_sf[(sub, fmt)][cond][item] = entry["soft"].get("sexist", 0.0)

    out: dict[str, dict] = {}
    for (sub, fmt), conds in by_sf.items():
        neutral = conds.get("neutral", {})
        block: dict[str, dict] = {}
        for cond, per_item in conds.items():
            shifts = [p - neutral[item] for item, p in per_item.items() if item in neutral]
            entry = {
                "mean_p_sexist": _mean(list(per_item.values())),
                "n_items": len(per_item),
                "mean_shift_vs_neutral": _mean(shifts) if cond != "neutral" else 0.0,
                "n_paired": len(shifts),
            }
            if gold_by_item is not None and cond != "neutral":
                entry.update(_majority_split(cond, per_item, neutral, gold_by_item))
            block[cond] = entry
        out["|".join((sub, fmt))] = block
    return out


def _majority_split(condition: str, per_item: dict, neutral: dict, gold_by_item: dict) -> dict:
    """Split the paired shift by whether the framed verdict matched the gold majority (H9b)."""
    toward, against = [], []
    for item, p in per_item.items():
        if item not in neutral or item not in gold_by_item:
            continue
        gold_sexist = gold_by_item[item]["1.1"]["sexist"] >= 0.5
        shift = p - neutral[item]
        # A shift "toward majority" raises P(sexist) when the majority is sexist,
        # or lowers it when the majority is not sexist.
        signed = shift if gold_sexist else -shift
        (toward if signed >= 0 else against).append(abs(shift))
    return {
        "toward_majority_mean_abs_shift": _mean(toward), "n_toward": len(toward),
        "against_majority_mean_abs_shift": _mean(against), "n_against": len(against),
    }


def _mean(xs):
    return float(sum(xs) / len(xs)) if xs else None
