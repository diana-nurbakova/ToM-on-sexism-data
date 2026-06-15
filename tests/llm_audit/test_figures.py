"""Tests for the TCM grid headline figure (master spec §7.3).

Rendering is checked as a smoke test (files written, no exceptions) under the Agg
backend; the panel-assembly selector logic is checked directly.
"""

import matplotlib

matplotlib.use("Agg")

from llm_audit import figures, tcm


def _entry(subtask="1.3"):
    classes = tcm.TCM_CLASSES[subtask]
    n = len(classes)
    # identity-ish matrix so it row-normalises cleanly
    matrix = [[1.0 if i == j else 0.1 for j in range(n)] for i in range(n)]
    return {"classes": classes, "matrix": matrix, "n_items": 5, "n_dropped": 0}


def test_tcm_grid_writes_pdf_and_png(tmp_path):
    panel = {
        "llama-3.3-70b": {"bare": _entry(), "defined": _entry()},
        "claude-4-sonnet": {"bare": _entry(), "defined": _entry()},
    }
    pdf = figures.tcm_grid(panel, tmp_path, name="grid_test", title="TCM grid")
    assert pdf.exists() and pdf.suffix == ".pdf"
    assert (tmp_path / "grid_test.png").exists()


def test_tcm_grid_handles_missing_cells_and_mixed_subtasks(tmp_path):
    panel = {
        "m1": {"1.1": _entry("1.1"), "1.3": _entry("1.3")},
        "m2": {"1.1": _entry("1.1")},  # missing 1.3 -> blank cell
    }
    pdf = figures.tcm_grid(panel, tmp_path, name="mixed", normalise=False)
    assert pdf.exists()


def test_tcm_grid_empty_panel_raises(tmp_path):
    import pytest
    with pytest.raises(ValueError):
        figures.tcm_grid({}, tmp_path)


def test_panel_from_metrics_selects_and_omits():
    tcm_by_row = {
        "modelA": {"bare|1.3|C": _entry(), "defined|1.3|C": _entry()},
        "modelB": {"bare|1.3|C": _entry()},  # no defined
    }
    panel = figures.panel_from_metrics(
        tcm_by_row, {"Bare": "bare|1.3|C", "Defined": "defined|1.3|C"})
    assert set(panel["modelA"]) == {"Bare", "Defined"}
    assert set(panel["modelB"]) == {"Bare"}  # missing selector omitted


def test_abbrev_known_and_unknown():
    assert figures._abbrev(["sexual_violence", "weird_class"]) == ["SXV", "WEI"]
