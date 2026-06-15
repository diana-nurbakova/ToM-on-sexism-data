"""E1 LLM-judge cross-check (OSF pre-reg §6, E1-H3).

GPT-4.1 and Claude 4 Sonnet judge whether the correct EXIST sexism label should
stay the same (INVARIANT), change (SHIFTING), or is undeterminable (AMBIGUOUS)
under the woman↔man swap. This is a **convergent check only** — the verdicts are
compared to the author's construct-derived partition (E1-H3) and never used to
define it. The circularity (LLMs judging a partition other LLMs are tested
against) is disclosed in the pre-registration.

The judge reuses the validated ``runner.Runner`` via its prompt/parse hooks, so it
inherits idempotent resume, the circuit-breaker, and §6.2 logging. Prompt wording
here is a scaffold (the pre-reg specifies the task, not verbatim text); freeze it
before the confirmatory run.
"""

from __future__ import annotations

from pathlib import Path

from .. import config, parsing
from ..clients import make_client
from ..runner import Cell, Runner

INVARIANCE_CLASSES = ("INVARIANT", "SHIFTING", "AMBIGUOUS")

JUDGE_SYSTEM = (
    "You are an expert annotator applying the EXIST 2025 sexism construct. EXIST "
    "defines sexism as attitudes toward women: the construct is woman-focused and "
    "therefore asymmetric. You will compare an original tweet with a version in "
    "which the gender has been swapped (woman ↔ man) and judge whether the "
    "correct sexism label should stay the same or change. Respond only in the "
    "requested format."
)

_JUDGE_USER = (
    "Under EXIST, a tweet is SEXIST if it expresses sexist ideas — because it "
    "is sexist itself, describes a sexist situation, or criticises sexist "
    "behaviour. The construct is woman-focused (sexism = attitudes toward women).\n\n"
    "Original tweet:\n{original}\n\n"
    "Gender-swapped tweet (woman ↔ man):\n{swapped}\n\n"
    "Under the EXIST woman-focused construct, should the correct sexism label be:\n"
    "- INVARIANT: the same for both (the sexism does not hinge on the target being "
    "a woman, or neither version is sexist)\n"
    "- SHIFTING: different (the original engages the woman-focused construct but the "
    "swapped version no longer does, or engages it differently)\n"
    "- AMBIGUOUS: cannot be determined confidently from the construct and the text\n\n"
    'Respond with a JSON object: {{"verdict": "INVARIANT" | "SHIFTING" | '
    '"AMBIGUOUS", "reason": "<one short sentence>"}}. No other text.'
)


def build_judge_user(original: str, swapped: str) -> str:
    return _JUDGE_USER.format(original=original, swapped=swapped)


def parse_invariance(raw: str | None) -> tuple[str | None, str]:
    """Parse a judge response to an invariance verdict + parse status."""
    if raw is None or not str(raw).strip():
        return None, parsing.EMPTY
    obj = parsing._extract_json(raw)
    verdict = None
    if obj is not None:
        verdict = _canon(obj.get("verdict"))
    if verdict is None:  # fall back to scanning the raw text for a class word
        up = str(raw).upper()
        hits = [c for c in INVARIANCE_CLASSES if c in up]
        verdict = hits[0] if len(hits) == 1 else None
    if verdict is not None:
        if parsing._has_cue(raw, parsing._REFUSAL_CUES) or parsing._has_cue(raw, parsing._DISCLAIMER_CUES):
            return verdict, parsing.DISCLAIMER_PREFIXED
        return verdict, parsing.PARSED_OK
    if parsing._has_cue(raw, parsing._REFUSAL_CUES):
        return None, parsing.REFUSAL
    return None, parsing.FORMAT_VIOLATION


def _canon(v) -> str | None:
    if not isinstance(v, str):
        return None
    u = v.strip().upper()
    return u if u in INVARIANCE_CLASSES else None


def build_judge_cells(rows, lang: str = "en"):
    """Build judge cells from worksheet rows.

    ``rows`` is an iterable of dicts with ``item_id``, ``original``, ``swapped``,
    and ``unswappable``. Unswappable items are skipped (excluded from the partition).
    The precomputed judge prompt is stored in ``Cell.text`` so the runner's
    prompt hook is a trivial passthrough.
    """
    cells = []
    for r in rows:
        if r.get("unswappable"):
            continue
        user = build_judge_user(r["original"], r["swapped"])
        cells.append(Cell("E1-judge", "invariance", "invariance", lang, "J",
                          str(r["item_id"]), user, 0))
    return cells


def run_judge(model: str, rows, out_dir: Path, env=None, lang: str = "en",
              max_tokens: int = 256) -> dict:
    """Run one LLM judge over the swappable items; returns runner stats."""
    spec = config.get_model(model)
    client = make_client(spec, env)
    log_path = Path(out_dir) / f"judge_{model}.jsonl"
    runner = Runner(
        client, spec, log_path, sampling={"max_tokens": max_tokens},
        prompt_fn=lambda c: (JUDGE_SYSTEM, c.text),
        parse_fn=lambda raw, c: parse_invariance(raw),
    )
    return runner.run(build_judge_cells(rows, lang))
