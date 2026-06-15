"""H1c run driver (persona prompting) + gender-shift analysis.

Builds persona cells (condition = persona cell name, e.g. ``F-mid``) and runs them
through the E0 ``Runner`` via its prompt hook. The within-H1c measurement is the
female-minus-male soft-label shift within each age band, per subtask and — for 1.3 —
per category (the selectivity test: affective > cognitive, §4.2). The human reference
(female-minus-male annotator shift from EXIST) and the directional-match comparison
are layered on top in the analysis notebook. The no-persona reference is the
bare-prompt E0 result, computed separately on the same sub-sample (§2.2, §5).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .. import config, metrics as M
from ..clients import make_client
from ..data import T13_CATEGORIES
from ..runner import Cell, Runner
from .persona import PERSONAS, H1C_SUBTASKS, build_persona_prompt

H1C_FORMATS = ("C",)
_BANDS = ("young", "mid", "older")


def _h1c_prompt(cell: Cell):
    return build_persona_prompt(cell.condition, cell.subtask, cell.language,
                                cell.format, cell.text)


def build_h1c_cells(items, lang="en", subtasks=H1C_SUBTASKS, personas=PERSONAS,
                    formats=H1C_FORMATS, k=config.K_SAMPLES):
    """``items`` = iterable of (item_id, text). condition = persona cell name."""
    cells: list[Cell] = []
    for item_id, text in items:
        for persona in personas:
            for subtask in subtasks:
                for fmt in formats:
                    draws = range(k) if fmt == "C" else range(1)
                    for ki in draws:
                        cells.append(Cell("H1c", persona, subtask, lang, fmt,
                                          str(item_id), text, ki))
    return cells


def run_h1c(model: str, items, out_dir: Path, env=None, lang="en",
            workers: int = 1, max_tokens: int = 256, **cell_kw) -> dict:
    spec = config.get_model(model)
    if workers > 1 and spec.is_proprietary:
        raise ValueError(f"workers>1 not allowed for proprietary backend {spec.backend}")
    client = make_client(spec, env)
    runner = Runner(client, spec, Path(out_dir) / f"{model}.jsonl",
                    sampling={"max_tokens": max_tokens}, prompt_fn=_h1c_prompt,
                    max_workers=workers)
    return runner.run(build_h1c_cells(items, lang, **cell_kw))


# ── Analysis: female-minus-male shift within age band (H1c selectivity) ───────
def gender_shift(log_path) -> dict:
    """Per (subtask, format, age band): the female-minus-male soft-label shift.

    For 1.1 the shifted quantity is P(sexist); for 1.3 it is each category's
    annotator-proportion (the per-category selectivity test — the shift should
    concentrate on the affective categories). Paired by item: only items annotated
    under both the female and the male persona of the band contribute.
    """
    soft, _ = M.assemble_model_soft(M.iter_records(log_path))
    # (subtask, fmt) -> persona -> item -> soft dict
    by_sf: dict[tuple, dict[str, dict[str, dict]]] = defaultdict(lambda: defaultdict(dict))
    for (persona, sub, fmt, item), entry in soft.items():
        by_sf[(sub, fmt)][persona][item] = entry["soft"]

    out: dict[str, dict] = {}
    for (sub, fmt), per_persona in by_sf.items():
        band_block: dict[str, dict] = {}
        for band in _BANDS:
            f_items = per_persona.get(f"F-{band}", {})
            m_items = per_persona.get(f"M-{band}", {})
            shared = [i for i in f_items if i in m_items]
            if not shared:
                continue
            if sub == "1.1":
                shifts = [f_items[i].get("sexist", 0.0) - m_items[i].get("sexist", 0.0)
                          for i in shared]
                band_block[band] = {"shift_p_sexist": _mean(shifts), "n_paired": len(shared)}
            else:  # 1.3: per-category female-minus-male shift
                cats = {}
                for cat in T13_CATEGORIES:
                    diffs = [f_items[i].get(cat, 0.0) - m_items[i].get(cat, 0.0)
                             for i in shared]
                    cats[cat] = _mean(diffs)
                band_block[band] = {"shift_per_category": cats, "n_paired": len(shared)}
        out["|".join((sub, fmt))] = band_block
    return out


def _mean(xs):
    return float(sum(xs) / len(xs)) if xs else None
