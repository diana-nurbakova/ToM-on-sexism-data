"""Headline figure: the multi-panel TCM grid (master spec §7.3).

One row per model, one column per condition / sub-study; each cell is a
Transport-based Confusion Matrix (``tcm.tcm_for_exist`` output) drawn as a
heatmap. Row-normalised by default, so the colour scale reads as *where does this
model send each true class's mass* — a strong diagonal is a faithful model, mass
off the diagonal shows where confusions concentrate. A single shared colour bar
spans the grid. British spelling; serif / whitegrid to match the project style.

A cell ``entry`` is a ``tcm_for_exist`` value: ``{"classes": [...], "matrix":
[[...]], ...}`` (rows = true classes, columns = predicted). The ``panel`` is a
nested dict ``{row_label: {column_label: entry}}``; missing cells are left blank.
When every cell shares the same classes (e.g. a single subtask across conditions)
the axis ticks are drawn once on the grid edges; otherwise each column gets its
own predicted-class ticks on the bottom row.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import tcm as _tcm

# Compact tick labels for the EXIST classes (keeps the grid readable).
_ABBREV = {
    "not_sexist": "NO",
    "sexist": "SEX",
    "direct": "DIR",
    "reported": "REP",
    "judgemental": "JUD",
    "ideological_inequality": "IDE",
    "stereotyping_dominance": "STE",
    "objectification": "OBJ",
    "sexual_violence": "SXV",
    "misogyny_non_sexual_violence": "MIS",
}


def _abbrev(classes: list[str]) -> list[str]:
    return [_ABBREV.get(c, c[:3].upper()) for c in classes]


def setup_style() -> None:
    """Serif / whitegrid paper style, matching ``nlpercep.figures``."""
    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "serif",
    })


def _save(fig, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{name}.pdf"
    fig.savefig(pdf_path)
    fig.savefig(out_dir / f"{name}.png")
    plt.close(fig)
    return pdf_path


def _matrix(entry: dict, normalise: bool) -> np.ndarray:
    m = np.asarray(entry["matrix"], float)
    return _tcm.row_normalise(m) if normalise else m


def _draw_cell(ax, entry, *, normalise, annotate, cmap, vmin, vmax,
               show_xticks, show_yticks):
    m = _matrix(entry, normalise)
    im = ax.imshow(m, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    classes = _abbrev(entry["classes"])
    n = len(classes)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(classes if show_xticks else [], fontsize=6, rotation=90)
    ax.set_yticklabels(classes if show_yticks else [], fontsize=6)
    ax.tick_params(length=0)
    if annotate:
        thresh = (vmax + vmin) / 2 if vmax is not None else m.max() / 2
        for i in range(n):
            for j in range(n):
                val = m[i, j]
                if val <= 1e-3:
                    continue
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=5,
                        color="white" if val < thresh else "black")
    return im


def tcm_grid(
    panel: dict[str, dict[str, dict]],
    out_dir: Path,
    *,
    name: str = "fig_tcm_grid",
    title: str | None = None,
    normalise: bool = True,
    cmap: str = "magma",
    annotate: bool | None = None,
    row_order: list[str] | None = None,
    col_order: list[str] | None = None,
) -> Path:
    """Render the multi-panel TCM grid (rows = models, columns = conditions).

    Args:
        panel: ``{row_label: {column_label: tcm_entry}}``.
        normalise: row-normalise each matrix (default; colour scale 0–1).
        annotate: write cell values; defaults to on when the largest matrix is
            ≤ 6×6 (legible), off otherwise.
        row_order / col_order: explicit ordering; defaults to insertion order
            (rows) and first-seen order across rows (columns).

    Returns the path to the saved PDF (a PNG is written alongside).
    """
    setup_style()
    rows = row_order or list(panel)
    if col_order is None:
        col_order = []
        for r in rows:
            for c in panel.get(r, {}):
                if c not in col_order:
                    col_order.append(c)
    cols = col_order
    if not rows or not cols:
        raise ValueError("panel has no rows or columns to plot")

    entries = [e for r in rows for e in panel.get(r, {}).values()]
    max_dim = max((len(e["classes"]) for e in entries), default=0)
    if annotate is None:
        annotate = max_dim <= 6
    uniform = len({tuple(e["classes"]) for e in entries}) == 1

    if normalise:
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = 0.0, max((np.asarray(e["matrix"], float).max() for e in entries),
                              default=1.0)

    nrows, ncols = len(rows), len(cols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(2.4 * ncols + 1.5, 2.4 * nrows + 1.0),
        squeeze=False,
    )

    im = None
    for ri, r in enumerate(rows):
        for ci, c in enumerate(cols):
            ax = axes[ri][ci]
            entry = panel.get(r, {}).get(c)
            if entry is None:
                ax.axis("off")
                continue
            is_bottom = ri == nrows - 1
            is_left = ci == 0
            # With uniform classes show ticks only on the edges; otherwise show
            # predicted-class ticks per column (bottom row) and true-class ticks
            # on the left column of each row.
            show_x = is_bottom
            show_y = is_left
            im = _draw_cell(
                ax, entry, normalise=normalise, annotate=annotate, cmap=cmap,
                vmin=vmin, vmax=vmax, show_xticks=show_x, show_yticks=show_y,
            )
            if ri == 0:
                ax.set_title(c, fontsize=9)
            if is_left:
                ax.set_ylabel(r, fontsize=9, rotation=90, labelpad=8)

    if uniform:
        fig.text(0.5, 0.04, "Predicted class", ha="center", fontsize=9)
        fig.text(0.02, 0.5, "True class", va="center", rotation=90, fontsize=9)

    if title:
        fig.suptitle(title, fontsize=12)

    fig.tight_layout(rect=(0.03, 0.05, 0.92, 0.96 if title else 0.98))
    if im is not None:
        cbar_ax = fig.add_axes((0.94, 0.15, 0.015, 0.7))
        label = "Row-normalised mass" if normalise else "Transported mass"
        fig.colorbar(im, cax=cbar_ax, label=label)

    return _save(fig, out_dir, name)


def panel_from_metrics(
    tcm_by_row: dict[str, dict],
    columns: dict[str, str],
) -> dict[str, dict[str, dict]]:
    """Assemble a ``tcm_grid`` panel from per-model ``metrics.json`` ``tcm`` blocks.

    Args:
        tcm_by_row: ``{row_label: tcm_block}`` where ``tcm_block`` is the ``"tcm"``
            field written by ``run_e0`` (keys ``"condition|subtask|format"``).
        columns: ordered ``{column_label: "condition|subtask|format"}`` selectors.

    Cells whose selector is absent for a row are simply omitted (left blank).
    """
    panel: dict[str, dict[str, dict]] = {}
    for row, block in tcm_by_row.items():
        cells = {col: block[key] for col, key in columns.items() if key in block}
        if cells:
            panel[row] = cells
    return panel
