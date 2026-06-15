"""Strachan ToM battery — administration + scoring (strachan-scoring-spec.md).

Administers the released Strachan items (data/strachan/strachan_items.json) to our
panel at 15 independent sessions/item via the E0 runner, then scores per the
**Strachan-simplified key**: rule-based for the near-closed-form subtests (False
Belief location-naming, Irony Yes/No discrimination) and an LLM-judge for the
free-text subtests (Faux Pas final-question, Hinting meaning+action, Strange
Stories released rubric). The calibration run (GPT-4/GPT-3.5/LLaMA2-70B) is scored
the same way and compared to Strachan's published per-item scores to validate the
auto-scorer ("same ruler", ~5–12% tolerance) before the panel is scored.

Per-item answer keys live in ``data/strachan/strachan_keys.json`` (a companion the
scorer loads). Strange Stories criteria come from the items JSON. The Hinting
meaning+action targets (Gil et al. 2012 Appendix A) are the one external input not
in the OSF release — flagged where missing; the judge degrades to rule-only.
"""

from __future__ import annotations

import json
from pathlib import Path

from .. import config, parsing
from ..clients import make_client
from ..config import REPO_ROOT
from ..runner import Cell, Runner

BATTERY_PATH = REPO_ROOT / "data" / "strachan" / "strachan_items.json"
KEYS_PATH = REPO_ROOT / "data" / "strachan" / "strachan_keys.json"
SESSIONS = 15  # Strachan's repetition design for LLMs

# Subtest -> scoring path.
RULE_BASED = {"False Belief - A", "False Belief - B", "Irony - A", "Irony - B"}
JUDGE_BASED = {"Hinting", "Faux Pas", "Strange Stories"}

# Neutral, content-free system prompt: the items are self-contained ToM vignettes.
STRACHAN_SYSTEM = (
    "You are answering questions about short stories. Read each story carefully and "
    "answer the question that follows as directly as you can."
)


# ── Administration ────────────────────────────────────────────────────────────
def load_battery(path: Path = BATTERY_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))["battery"]


def _strachan_prompt(cell: Cell):
    return STRACHAN_SYSTEM, cell.text


def _parse_free_text(raw, _cell):
    if raw is None or not str(raw).strip():
        return None, parsing.EMPTY
    if parsing._has_cue(raw, parsing._REFUSAL_CUES):
        return raw, parsing.DISCLAIMER_PREFIXED  # keep text; refusal-flavoured
    return raw.strip(), parsing.PARSED_OK


def build_strachan_cells(battery: dict, lang="en", sessions=SESSIONS, tasks=None):
    """One cell per (task, item, session). condition=task, subtask=task, text=prompt."""
    cells = []
    for task, items in battery.items():
        if tasks and task not in tasks:
            continue
        for it in items:
            for s in range(sessions):
                cells.append(Cell("strachan", task, task, lang, "free",
                                  f"{task}#{it['item']}", it["prompt"], s))
    return cells


def run_strachan(model: str, out_dir: Path, env=None, sessions=SESSIONS,
                 workers: int = 1, max_tokens: int = 512, tasks=None) -> dict:
    """Administer the battery to one model (panel or calibration). Resumable."""
    spec = config.get_model(model)
    if workers > 1 and spec.is_proprietary:
        raise ValueError(f"workers>1 not allowed for proprietary backend {spec.backend}")
    client = make_client(spec, env)
    runner = Runner(client, spec, Path(out_dir) / f"{model}.jsonl",
                    sampling={"max_tokens": max_tokens}, prompt_fn=_strachan_prompt,
                    parse_fn=_parse_free_text, max_workers=workers)
    return runner.run(build_strachan_cells(load_battery(), sessions=sessions, tasks=tasks))


# ── Scoring: judge prompts (Strachan-simplified key embedded) ─────────────────
JUDGE_SYSTEM = (
    "You are a careful grader scoring a model's answer to a Theory-of-Mind test item "
    "against a fixed rubric. Apply the rubric exactly. Respond only in the requested "
    "JSON format."
)

_FAUX_PAS_RUBRIC = (
    "Faux Pas — score ONLY whether the answer correctly recognises that the speaker "
    "did NOT know / did not remember the relevant fact (the final speaker-awareness "
    "question). Correct (1) = answer indicates the speaker was unaware / did not know "
    "/ did not remember. Incorrect (0) = says the speaker knew, or refuses with 'not "
    "enough information'. Score the speaker-awareness judgement only, not other details."
)
_HINTING_RUBRIC = (
    "Hinting — score 1 only if the answer identifies BOTH (a) the intended meaning of "
    "the speaker's remark AND (b) the concrete action the speaker is trying to get the "
    "other person to do. Identifying the meaning but not naming the requested action "
    "scores 0. {target}"
)
_STRANGE_RUBRIC = (
    "Strange Stories — apply this released coding criterion. 2 points: {two}. "
    "1 point: {one}. 0 points: {zero}. Return the integer score 0, 1, or 2."
)


def build_judge_prompt(task: str, story_prompt: str, answer: str,
                       criterion: dict | None = None, target: str | None = None,
                       key: dict | None = None) -> str:
    """Judge prompt for a free-text subtest. ``key`` is the per-item entry from
    strachan_keys.json; for Faux Pas it pins the exact question to score (the story
    asks four) and applies the lenient coding for the story-competition item."""
    if task == "Faux Pas":
        rubric, scale = _FAUX_PAS_RUBRIC, '"score": 0 or 1'
        if key:
            fq = key.get("final_question", "")
            if key.get("lenient"):
                rubric += (
                    f' The ONLY question to score is: "{fq}". This item is scored '
                    "LENIENTLY: either 'Yes' or 'No' (i.e. did or did not know/realise) "
                    "counts as correct (1); score 0 only for an irrelevant answer or a "
                    "refusal such as 'not enough information'.")
            else:
                rubric += (
                    f' The ONLY question to score is: "{fq}". Correct (1) = the answer to '
                    "THIS question indicates the speaker did not know / did not remember "
                    "(typically 'No'); 0 = says they knew, or refuses with 'not enough "
                    "information'.")
    elif task == "Hinting":
        rubric = _HINTING_RUBRIC.format(
            target=f"Intended meaning + action for this item: {target}" if target
            else "(No per-item target supplied — infer the intended meaning+action "
                 "from the story; flag lower confidence.)")
        scale = '"score": 0 or 1'
    elif task == "Strange Stories":
        c = criterion or {}
        rubric = _STRANGE_RUBRIC.format(two=c.get("2pt"), one=c.get("1pt"), zero=c.get("0pt"))
        scale = '"score": 0, 1, or 2'
    else:
        raise ValueError(f"{task} is rule-based, not judge-scored")
    return (f"{rubric}\n\nITEM:\n{story_prompt}\n\nMODEL ANSWER:\n{answer}\n\n"
            f'Respond with a JSON object: {{{scale}, "criterion": "<which rubric clause matched>"}}.'
            " No other text.")


# ── Scoring: rule-based (closed-form) ─────────────────────────────────────────
# Fold smart quotes/dashes to ASCII (and lowercase) before any string comparison.
# The administered items carry curly apostrophes/quotes and en/em dashes while the
# keys use straight ASCII (spec §5b); normalise BOTH sides or exact-match scoring
# throws spurious failures that are pure encoding artefacts. Kept punctuation-light
# on purpose so meaningful characters (e.g. the "/shared_folder/tmp" path) survive.
_SMART_MAP = str.maketrans({
    "‘": "'", "’": "'", "‛": "'",   # ‘ ’ ‛ -> '
    "“": '"', "”": '"',                   # “ ” -> "
    "–": "-", "—": "-", "−": "-",     # – — − -> -
    " ": " ",                                   # nbsp -> space
})


def _norm(s: str) -> str:
    """Lowercase and fold smart quotes/dashes to ASCII, for robust matching."""
    return str(s).translate(_SMART_MAP).lower()


def score_false_belief(answer: str, key: dict) -> int | None:
    """1 iff the answer names the keyed correct location (and not the distractor)."""
    if not answer:
        return None
    low = _norm(answer)
    correct = _norm(key["correct_location"])
    distractor = _norm(key.get("distractor_location", ""))
    has_correct = correct in low
    has_distractor = bool(distractor) and distractor in low
    return int(has_correct and not has_distractor)


def score_irony(answer: str, key: dict) -> int | None:
    """1 iff the Yes/No answer matches the keyed polarity for this (ironic/sincere) item."""
    if not answer:
        return None
    low = _norm(answer).strip()
    said_yes = low.startswith("yes") or " yes" in low[:12]
    said_no = low.startswith("no") or " no" in low[:12]
    if said_yes == said_no:  # ambiguous / neither
        return None
    return int(("yes" if said_yes else "no") == _norm(key["correct"]))


def load_keys(path: Path = KEYS_PATH) -> dict:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
