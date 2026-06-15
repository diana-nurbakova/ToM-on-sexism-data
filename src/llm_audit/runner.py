"""Run loop: cell expansion, idempotent resume, append-only §6.2 logging.

A *cell* is one LLM call, uniquely keyed by
``(experiment, condition, subtask, language, format, item_id, K_index)``. The
runner reads any existing log, skips cells already done, and appends one flushed
JSON record per call so a crash or circuit-breaker halt only ever costs the
in-flight call. Format A/B draw once (K_index 0); Format C draws ``K`` times.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

from . import config, parsing, prompts
from .clients import BaseClient, CircuitBreakerTrip
from .config import ModelSpec

# Statuses that count a cell as done on resume. ``error`` is retryable; everything
# else (incl. refusal/format_violation/empty) is a settled outcome we keep.
_DONE_STATUSES = frozenset(
    {parsing.PARSED_OK, parsing.REFUSAL, parsing.DISCLAIMER_PREFIXED,
     parsing.FORMAT_VIOLATION, parsing.EMPTY}
)
_KEY_FIELDS = ("experiment", "condition", "subtask", "language", "format", "item_id", "k_index")


@dataclass(frozen=True)
class Cell:
    experiment: str
    condition: str       # E0: the bare/defined spec
    subtask: str
    language: str
    format: str
    item_id: str
    text: str
    k_index: int

    def key(self) -> tuple:
        return (self.experiment, self.condition, self.subtask,
                self.language, self.format, self.item_id, self.k_index)


def build_e0_cells(items, lang, subtasks, specs, formats, k_samples=config.K_SAMPLES):
    """Expand the E0 grid into cells. ``items`` is an iterable of (item_id, text)."""
    cells: list[Cell] = []
    for item_id, text in items:
        for subtask in subtasks:
            for spec in specs:
                for fmt in formats:
                    draws = range(k_samples) if fmt == "C" else range(1)
                    for k in draws:
                        cells.append(Cell("E0", spec, subtask, lang, fmt,
                                          str(item_id), text, k))
    return cells


def _record_key(rec: dict) -> tuple:
    return tuple(rec[f] for f in _KEY_FIELDS)


def load_done_keys(log_path: Path) -> set[tuple]:
    """Return keys of settled cells in an existing log (latest record per key)."""
    if not Path(log_path).exists():
        return set()
    latest: dict[tuple, str] = {}
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # tolerate a torn final line (append-only crash)
            latest[_record_key(rec)] = rec.get("parse_status")
    return {k for k, status in latest.items() if status in _DONE_STATUSES}


class Runner:
    """Drives a list of cells through one model client into one JSONL log."""

    def __init__(self, client: BaseClient, spec: ModelSpec, log_path: Path,
                 seed: int = 42, sampling: dict | None = None,
                 prompt_fn=None, parse_fn=None, max_workers: int = 1):
        self.client = client
        self.spec = spec
        self.log_path = Path(log_path)
        self.seed = seed
        self.sampling = {"temperature": config.TEMPERATURE, "max_tokens": 512, **(sampling or {})}
        # Concurrency: >1 issues calls in parallel (DeepInfra throughput lane).
        # Default 1 keeps the serial path for proprietary lanes (rate-limit/ban safety).
        self.max_workers = max(1, int(max_workers))
        self._write_lock = threading.Lock()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        # Pluggable prompt/parse hooks; default to the E0 behaviour. The judge and
        # other experiments inject their own here and reuse all the resume /
        # circuit-breaker / logging machinery below.
        self.prompt_fn = prompt_fn or (lambda c: prompts.build_prompt(
            c.subtask, c.condition, c.language, c.format, c.text))
        self.parse_fn = parse_fn or (lambda raw, c: parsing.parse_response(
            raw, c.subtask, c.format))

    def _write(self, rec: dict, _attempts: int = 8) -> None:
        line = json.dumps(rec, ensure_ascii=False) + "\n"
        with self._write_lock:  # thread-safe append for the concurrent path
            # OneDrive/AV can briefly lock a synced file (Errno 13); retry transiently
            # so a concurrent append is never lost to a momentary lock.
            for attempt in range(_attempts):
                try:
                    with open(self.log_path, "a", encoding="utf-8") as f:
                        f.write(line)
                        f.flush()
                        os.fsync(f.fileno())
                    return
                except (PermissionError, OSError):
                    if attempt == _attempts - 1:
                        raise
                    time.sleep(0.25 * (attempt + 1))

    def _base_record(self, cell: Cell) -> dict:
        return {
            "call_id": uuid.uuid4().hex,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": self.spec.name,
            "model_version": self.spec.model_id,
            "backend": self.spec.backend,
            "quantisation": self.spec.quantisation,
            **{f: getattr(cell, f) for f in _KEY_FIELDS},
            "seed": self.seed,
            "sampling_params": self.sampling,
        }

    def _process(self, cell: "Cell") -> tuple[dict, tuple[str, str]]:
        """Run one cell -> (record, ('circuit'|'status', value)). Never raises."""
        system, user = self.prompt_fn(cell)
        rec = self._base_record(cell)
        rec["system_prompt"] = system
        rec["user_prompt"] = user
        try:
            raw, meta = self.client.complete(system, user, self.sampling)
        except CircuitBreakerTrip as trip:
            rec.update(raw_response=None, parsed_value=None,
                       parse_status=parsing.ERROR, meta={"circuit_breaker": str(trip)})
            return rec, ("circuit", trip.provider)
        except Exception as exc:  # noqa: BLE001 — logged as a retryable error
            rec.update(raw_response=None, parsed_value=None,
                       parse_status=parsing.ERROR, meta={"error": repr(exc)})
            return rec, ("status", parsing.ERROR)
        parsed, status = self.parse_fn(raw, cell)
        rec.update(raw_response=raw, parsed_value=parsed, parse_status=status, meta=meta)
        return rec, ("status", status)

    def run(self, cells, progress=True) -> dict:
        """Run all not-yet-done cells. Returns a stats dict. Stops on a breaker trip.

        Serial when ``max_workers == 1`` (default, proprietary-safe); otherwise issues
        up to ``max_workers`` concurrent calls (DeepInfra throughput lane).
        """
        done = load_done_keys(self.log_path)
        todo = [c for c in cells if c.key() not in done]
        stats = {"total": len(cells), "skipped": len(cells) - len(todo),
                 "ran": 0, "by_status": {}, "circuit_tripped": None}

        def _tally(kind: tuple[str, str]) -> None:
            if kind[0] == "circuit":
                stats["circuit_tripped"] = kind[1]
            else:
                stats["by_status"][kind[1]] = stats["by_status"].get(kind[1], 0) + 1
                stats["ran"] += 1

        def _bar(it):
            if not progress:
                return it
            try:
                from tqdm import tqdm
                return tqdm(it, total=len(todo), desc=f"{self.spec.name}")
            except ImportError:
                return it

        if self.max_workers == 1:
            for cell in _bar(todo):
                rec, kind = self._process(cell)
                self._write(rec)
                _tally(kind)
                if kind[0] == "circuit":
                    break  # halt the provider lane immediately
            return stats

        # Concurrent path: in-flight calls finish; once a breaker trips, queued
        # cells short-circuit (left for idempotent resume), and the provider is recorded.
        stop = threading.Event()

        def worker(cell):
            if stop.is_set():
                return None
            rec, kind = self._process(cell)
            self._write(rec)
            if kind[0] == "circuit":
                stop.set()
            return kind

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = [ex.submit(worker, c) for c in todo]
            for fut in _bar(as_completed(futures)):
                kind = fut.result()
                if kind is not None:
                    _tally(kind)
        return stats
