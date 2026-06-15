"""Tests for cell expansion, append-only logging, and idempotent resume."""

import json

from llm_audit.clients import FakeClient
from llm_audit.config import ModelSpec
from llm_audit.runner import Runner, build_e0_cells, load_done_keys

SPEC = ModelSpec("fake", "anthropic", "fake-model")
ITEMS = [("100", "tweet one"), ("101", "tweet two")]


def _runner(log_path, responses):
    client = FakeClient(SPEC, responses, sleep=lambda s: None)
    return Runner(client, SPEC, log_path), client


def test_build_e0_cells_format_c_expands_to_k():
    cells = build_e0_cells(ITEMS, "en", ("1.1",), ("bare",), ("A", "C"), k_samples=6)
    # 2 items × (1 A draw + 6 C draws) = 14 cells.
    assert len(cells) == 14
    a = [c for c in cells if c.format == "A"]
    c = [c for c in cells if c.format == "C"]
    assert len(a) == 2 and len(c) == 12


def test_run_writes_one_record_per_cell_and_flushes(tmp_path):
    log = tmp_path / "calls.jsonl"
    cells = build_e0_cells(ITEMS, "en", ("1.1",), ("bare",), ("C",), k_samples=3)
    runner, client = _runner(log, ["sexist", "not_sexist", "sexist"])
    stats = runner.run(cells, progress=False)
    assert stats["ran"] == len(cells) == 6
    lines = log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 6
    rec = json.loads(lines[0])
    for field in ("call_id", "timestamp", "model", "backend", "system_prompt",
                  "user_prompt", "raw_response", "parsed_value", "parse_status"):
        assert field in rec


def test_idempotent_resume_runs_zero_duplicates(tmp_path):
    log = tmp_path / "calls.jsonl"
    cells = build_e0_cells(ITEMS, "en", ("1.1",), ("bare",), ("C",), k_samples=3)
    runner1, client1 = _runner(log, ["sexist"])
    runner1.run(cells, progress=False)
    n_first = len(client1.calls)
    assert n_first == 6

    # Fresh runner + fresh client over the same log = a simulated restart.
    runner2, client2 = _runner(log, ["sexist"])
    stats = runner2.run(cells, progress=False)
    assert stats["skipped"] == 6 and stats["ran"] == 0
    assert len(client2.calls) == 0  # no duplicate API calls


def test_partial_resume_only_runs_missing(tmp_path):
    log = tmp_path / "calls.jsonl"
    cells = build_e0_cells(ITEMS, "en", ("1.1",), ("bare",), ("C",), k_samples=3)
    runner1, _ = _runner(log, ["sexist"])
    runner1.run(cells[:4], progress=False)        # only 4 of 6 done
    runner2, client2 = _runner(log, ["not_sexist"])
    stats = runner2.run(cells, progress=False)
    assert stats["skipped"] == 4 and stats["ran"] == 2
    assert len(client2.calls) == 2


def test_concurrent_run_completes_all_with_no_duplicates(tmp_path):
    log = tmp_path / "calls.jsonl"
    items = [(str(i), f"tweet {i}") for i in range(10)]
    cells = build_e0_cells(items, "en", ("1.1",), ("bare",), ("C",), k_samples=2)
    client = FakeClient(SPEC, ["sexist", "not_sexist"], sleep=lambda s: None)
    runner = Runner(client, SPEC, log, max_workers=4)
    stats = runner.run(cells, progress=False)
    assert stats["ran"] == len(cells) == 20
    lines = log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 20
    # Resume under concurrency: no duplicate calls.
    runner2 = Runner(FakeClient(SPEC, ["sexist"], sleep=lambda s: None), SPEC, log, max_workers=4)
    stats2 = runner2.run(cells, progress=False)
    assert stats2["skipped"] == 20 and stats2["ran"] == 0


def test_done_keys_excludes_error_records(tmp_path):
    log = tmp_path / "calls.jsonl"
    base = {"experiment": "E0", "condition": "bare", "subtask": "1.1",
            "language": "en", "format": "C", "item_id": "100", "k_index": 0}
    with open(log, "w", encoding="utf-8") as f:
        f.write(json.dumps({**base, "parse_status": "error"}) + "\n")
        f.write(json.dumps({**base, "item_id": "101", "parse_status": "parsed_ok"}) + "\n")
    done = load_done_keys(log)
    assert ("E0", "bare", "1.1", "en", "C", "101", 0) in done
    assert ("E0", "bare", "1.1", "en", "C", "100", 0) not in done  # error is retryable
