"""E1 adjudication worksheet workflow (OSF pre-reg §6).

Turns the bin-first sample into the construct-derivation worksheet the author works
from, and turns the author-completed worksheet back into the frozen partition that
E2/E5 condition on. Three steps, matching the pre-registration:

  1. ``build_worksheet`` — apply the frozen swap (``swap.swap_text``) and the frozen
     construct rule (``rule.construct_rule_tag``) to every sampled item, emitting a
     worksheet with a *rule first-pass* (``rule_class`` / ``rule_reason`` /
     ``needs_adjudication``) plus blank author columns (``invariance_class`` /
     ``justification``). Confident rows (UNSWAPPABLE, majority-not-sexist INVARIANT)
     come pre-resolved; every SEXIST-derived row is flagged for adjudication.
  2. The author fills ``invariance_class`` (+ a one-line ``justification``) for each
     ``needs_adjudication`` row, and may override a confident row (logged as a
     deviation). ``--merge`` on rebuild preserves any author cells already entered.
  3. ``finalize_partition`` — validate that no adjudication row is unresolved, then
     emit (a) the *pre-adjudication* partition (rule only), (b) the *post-adjudication*
     partition (author-resolved; the one E2/E5 use), (c) a per-item decision log, and
     (d) a fully-populated *resolved* worksheet that ``analysis.run_analysis`` reads.

Both the pre- and post-adjudication partitions are retained, per pre-reg §6.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from .rule import EXCLUSION, INVARIANCE_CLASSES, construct_rule_tag
from .swap import swap_text

# Worksheet column order (frozen; the author edits only the last two).
COLUMNS = [
    "item_id", "strat_category", "agreement_bin", "gold_1_1", "gold_1_3",
    "original", "swapped",
    "name_swapped", "gendered_swapped", "female_source", "male_source",
    "unswappable", "author_review", "review_reasons", "n_swapped_tokens",
    "rule_class", "rule_reason", "needs_adjudication",
    "invariance_class", "justification",
]
AUTHOR_COLUMNS = ("invariance_class", "justification")
RESOLVED_CLASSES = (*INVARIANCE_CLASSES, EXCLUSION)


# ── Step 1: build the worksheet ───────────────────────────────────────────────
def build_worksheet(sample: pd.DataFrame) -> pd.DataFrame:
    """Apply the frozen swap + construct rule to the E1 sample.

    ``sample`` is the bin-first sample (``sampling.build_sample``) carrying
    ``item_id``, ``strat_category``, ``agreement_bin``, ``gold_1_1``, ``gold_1_3``
    and the tweet ``text``. Returns a worksheet DataFrame with ``COLUMNS`` and blank
    author cells.
    """
    text_col = "text" if "text" in sample.columns else "original"
    rows = []
    for _, r in sample.iterrows():
        sw = swap_text(r[text_col])
        rule_class, needs_adj, rule_reason = construct_rule_tag(
            float(r["gold_1_1"]["sexist"]), sw.gendered_swapped,
            sw.name_swapped, sw.unswappable)
        rows.append({
            "item_id": str(r["item_id"]),
            "strat_category": r["strat_category"],
            "agreement_bin": r["agreement_bin"],
            "gold_1_1": json.dumps(r["gold_1_1"]),
            "gold_1_3": json.dumps(r["gold_1_3"]),
            "original": sw.original,
            "swapped": sw.swapped,
            "name_swapped": sw.name_swapped,
            "gendered_swapped": sw.gendered_swapped,
            "female_source": sw.female_source,
            "male_source": sw.male_source,
            "unswappable": sw.unswappable,
            "author_review": sw.author_review,
            "review_reasons": "; ".join(sw.review_reasons),
            "n_swapped_tokens": len(sw.swapped_tokens),
            "rule_class": rule_class,
            "rule_reason": rule_reason,
            "needs_adjudication": needs_adj,
            "invariance_class": "",
            "justification": "",
        })
    return pd.DataFrame(rows, columns=COLUMNS)


def write_worksheet(df: pd.DataFrame, path: Path, merge_prior: bool = True) -> Path:
    """Write the worksheet CSV, optionally preserving prior author cells.

    With ``merge_prior`` (default), any ``invariance_class`` / ``justification``
    already entered in an existing worksheet at ``path`` is carried over by
    ``item_id`` so a rebuild never discards adjudication work.
    """
    path = Path(path)
    df = df.copy()
    if merge_prior and path.exists():
        prior = {r["item_id"]: r for r in _read_rows(path)}
        for i, item_id in df["item_id"].items():
            pr = prior.get(str(item_id))
            if not pr:
                continue
            for col in AUTHOR_COLUMNS:
                val = (pr.get(col) or "").strip()
                if val:
                    df.at[i, col] = val
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


# ── Step 3: finalise the partition ────────────────────────────────────────────
def _read_rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _truthy(v) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes")


def _author_class(row: dict) -> str | None:
    cls = (row.get("invariance_class") or "").strip().upper()
    return cls if cls in RESOLVED_CLASSES else None


def validate_worksheet(worksheet_csv: Path) -> list[dict]:
    """Return the rows still blocking finalisation.

    A row is unresolved when the rule flagged it ``needs_adjudication`` but the
    author has not entered a valid ``invariance_class``. An empty list means the
    worksheet is ready to finalise.
    """
    unresolved = []
    for row in _read_rows(Path(worksheet_csv)):
        if _truthy(row.get("needs_adjudication")) and _author_class(row) is None:
            unresolved.append({
                "item_id": row["item_id"],
                "rule_class": row.get("rule_class"),
                "rule_reason": row.get("rule_reason"),
            })
    return unresolved


def finalize_partition(worksheet_csv: Path, out_dir: Path | None = None,
                       lang: str = "en") -> dict:
    """Validate the completed worksheet and export the frozen partition.

    Resolution rule per row:
      * ``needs_adjudication`` -> the author's ``invariance_class`` (must be present;
        otherwise the worksheet is not ready — see :func:`validate_worksheet`).
      * confident row (not flagged) -> the author's override if they entered one
        (recorded as a deviation), else the ``rule_class``.

    Writes, into ``out_dir`` (default: the worksheet's directory):
      * ``e1_partition_<lang>.json`` — the post-adjudication partition + metadata,
        the export E2/E5 condition on.
      * ``e1_decisions_<lang>.csv`` — per-item decision log (rule vs final, changes).
      * ``e1_sample_<lang>_resolved.csv`` — worksheet with ``invariance_class`` fully
        populated to the final class (what ``analysis.run_analysis`` reads).

    Returns a summary dict. Raises ``ValueError`` if any adjudication row is unresolved.
    """
    worksheet_csv = Path(worksheet_csv)
    out_dir = Path(out_dir) if out_dir else worksheet_csv.parent

    unresolved = validate_worksheet(worksheet_csv)
    if unresolved:
        raise ValueError(
            f"{len(unresolved)} item(s) still need adjudication; fill "
            f"`invariance_class` for: {', '.join(u['item_id'] for u in unresolved[:10])}"
            + (" …" if len(unresolved) > 10 else ""))

    rows = _read_rows(worksheet_csv)
    pre, post, decisions, resolved_rows = {}, {}, [], []
    for row in rows:
        item_id = row["item_id"]
        rule_class = (row.get("rule_class") or "").strip().upper()
        needs_adj = _truthy(row.get("needs_adjudication"))
        author = _author_class(row)

        final = author if (needs_adj or author) else rule_class
        changed = author is not None and author != rule_class

        pre[item_id] = rule_class
        post[item_id] = final
        decisions.append({
            "item_id": item_id,
            "strat_category": row.get("strat_category"),
            "rule_class": rule_class,
            "needs_adjudication": needs_adj,
            "author_filled": author is not None,
            "final_class": final,
            "changed_from_rule": changed,
            "justification": (row.get("justification") or "").strip(),
        })
        out_row = dict(row)
        out_row["invariance_class"] = final
        resolved_rows.append(out_row)

    summary = _summarise(pre, post, decisions, lang)

    out_dir.mkdir(parents=True, exist_ok=True)
    partition_path = out_dir / f"e1_partition_{lang}.json"
    decisions_path = out_dir / f"e1_decisions_{lang}.csv"
    resolved_path = out_dir / f"e1_sample_{lang}_resolved.csv"

    with open(partition_path, "w", encoding="utf-8") as f:
        json.dump({"lang": lang, "summary": summary,
                   "partition_post_adjudication": post,
                   "partition_pre_adjudication": pre}, f, indent=2)
    _write_csv(decisions_path, decisions,
               ["item_id", "strat_category", "rule_class", "needs_adjudication",
                "author_filled", "final_class", "changed_from_rule", "justification"])
    _write_csv(resolved_path, resolved_rows, list(rows[0].keys()) if rows else COLUMNS)

    summary["outputs"] = {
        "partition": str(partition_path),
        "decisions": str(decisions_path),
        "resolved_worksheet": str(resolved_path),
    }
    return summary


def _summarise(pre: dict, post: dict, decisions: list[dict], lang: str) -> dict:
    """Counts that go into the pre-reg's reporting + the deviation tally."""
    n_changed = sum(1 for d in decisions if d["changed_from_rule"])
    overrides = [d["item_id"] for d in decisions
                 if d["changed_from_rule"] and not d["needs_adjudication"]]
    n_excluded = sum(1 for c in post.values() if c == EXCLUSION)
    return {
        "lang": lang,
        "n_items": len(post),
        "n_unswappable_excluded": n_excluded,
        "n_classified": len(post) - n_excluded,
        "pre_adjudication_counts": dict(Counter(pre.values())),
        "post_adjudication_counts": dict(Counter(post.values())),
        "n_adjudicated": sum(1 for d in decisions if d["needs_adjudication"]),
        "n_changed_from_rule": n_changed,
        "confident_row_overrides": overrides,  # deviations: author overrode a non-flagged row
    }


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
