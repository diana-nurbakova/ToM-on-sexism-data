"""Isolated-environment entry point for ICM-Soft via PyEvALL (master spec §7.1).

Run this script **under the interpreter of the environment where PyEvALL is
installed** (point ``PYEVALL_PYTHON`` at it). It reads PyEvALL-format prediction
and gold JSON files, runs the requested metrics (default ICM-Soft + ICM-Soft
Norm), and writes the dataset-level average per metric to ``--out`` as JSON.

It is invoked out-of-process by ``llm_audit.icm_soft`` so PyEvALL's heavy, pinned
dependencies never enter the main analysis environment. Nothing here imports the
``llm_audit`` package — the only contract is the JSON in/out and the CLI flags
below — so the isolated env needs only PyEvALL, not the project.

Standalone usage (inside the isolated env)::

    python scripts/pyevall_runner.py --pred pred.json --gold gold.json \\
        --metrics ICMSoft,ICMSoftNorm --hierarchy hier.json --out out.json

``--pred`` / ``--gold`` follow PyEvALL's input format: a JSON list of records
``{"test_case", "id", "value"}`` where a soft value is ``{class: proportion}``.
``--hierarchy`` (optional, for the hierarchical EXIST tasks 1.2/1.3) is a JSON
object mapping each parent class to its list of child classes.
"""

import argparse
import json

# Metric identifiers PyEvALL exposes via MetricFactory; the report keys results by
# the metric class name, which is what ``.value`` returns.
_METRIC_ALIASES = ("ICM", "ICMNorm", "ICMSoft", "ICMSoftNorm",
                   "Accuracy", "Precision", "Recall", "FMeasure", "Kappa")


def _resolve_metrics(requested, factory):
    """Map requested metric names to the MetricFactory values PyEvALL expects."""
    out = []
    for name in requested:
        member = getattr(factory, name, None)
        out.append(member.value if member is not None else name)
    return out


def _extract_average(report, metric_values):
    """Pull the dataset-level average for each metric out of ``report.report``.

    Report shape (PyEvALL ``PyEvALLReport``)::

        report.report["metrics"][<metric_class>]["results"]["average_per_test_case"]
    """
    rep = getattr(report, "report", None) or {}
    metrics = rep.get("metrics", {}) or {}
    results = {}
    # Match requested -> report key tolerantly (exact, then case-insensitive).
    lower = {str(k).lower(): k for k in metrics}
    for mv in metric_values:
        key = mv if mv in metrics else lower.get(str(mv).lower())
        if key is None:
            results[mv] = None
            continue
        res = (metrics[key] or {}).get("results", {}) or {}
        results[mv] = res.get("average_per_test_case")
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run PyEvALL ICM-Soft out-of-process.")
    ap.add_argument("--pred", required=True, help="PyEvALL-format predictions JSON")
    ap.add_argument("--gold", required=True, help="PyEvALL-format gold JSON")
    ap.add_argument("--metrics", default="ICMSoft,ICMSoftNorm",
                    help="comma-separated metric names (default ICMSoft,ICMSoftNorm)")
    ap.add_argument("--hierarchy", help="optional hierarchy JSON (tasks 1.2/1.3)")
    ap.add_argument("--out", required=True, help="output JSON path")
    args = ap.parse_args(argv)

    from pyevall.evaluation import PyEvALLEvaluation
    from pyevall.metrics.metricfactory import MetricFactory
    from pyevall.utils.utils import PyEvALLUtils

    requested = [m.strip() for m in args.metrics.split(",") if m.strip()]
    metric_values = _resolve_metrics(requested, MetricFactory)

    params = {PyEvALLUtils.PARAM_REPORT: PyEvALLUtils.PARAM_OPTION_REPORT_EMBEDDED}
    if args.hierarchy:
        with open(args.hierarchy, encoding="utf-8") as fh:
            params[PyEvALLUtils.PARAM_HIERARCHY] = json.load(fh)

    report = PyEvALLEvaluation().evaluate(args.pred, args.gold, metric_values, **params)
    results = _extract_average(report, metric_values)

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
