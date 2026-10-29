"""Run benchmark experiments and export CSV, Markdown, and plot reports.

Run from the ``code`` directory with:
    python -m sfmpe.benchmark.report --config configs/benchmark.yaml
"""

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import yaml

from sfmpe.benchmark.runner import run_benchmark


_PROJECT_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "benchmark.yaml"


def _json_value(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Benchmark result file does not exist: {path}")
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(
                    f"Expected a JSON object at {path}:{line_number}."
                )
            rows.append(row)
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot export an empty benchmark table to {path}.")
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        _json_value(value)
                        if isinstance(value, (dict, list, tuple))
                        else value
                    )
                    for key, value in row.items()
                }
            )


def _flatten_target_row(row: Mapping[str, Any]) -> dict[str, Any]:
    flattened = {key: value for key, value in row.items() if key != "metrics"}
    metrics = row.get("metrics", {})
    if not isinstance(metrics, Mapping):
        raise TypeError("Each target row's 'metrics' field must be an object.")
    flattened.update({f"metric_{key}": value for key, value in metrics.items()})
    return flattened


def _numeric_metric_columns(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    excluded = {
        "mean_setup_simulator_draws",
        "mean_ppc_simulator_draws",
        "mean_sensitivity_simulator_draws",
        "mean_sensitivity_perturbation_scale",
    }
    columns = set()
    for row in rows:
        for key, value in row.items():
            if (
                key.startswith("mean_")
                and key not in excluded
                and isinstance(value, (int, float))
                and math.isfinite(value)
            ):
                columns.add(key)
    return sorted(columns)


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _format_cell(row: Mapping[str, Any], mean_column: str) -> str:
    mean = row.get(mean_column)
    if not isinstance(mean, (int, float)) or not math.isfinite(mean):
        return "n/a"
    metric = mean_column.removeprefix("mean_")
    std = row.get(f"std_{metric}")
    if isinstance(std, (int, float)) and math.isfinite(std):
        return f"{mean:.4g} ± {std:.3g}"
    return f"{mean:.4g}"


def _write_markdown_report(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    metrics: Sequence[str],
) -> None:
    by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row.get("task", "unknown"))].append(row)
    # Rounds axis: only sequential models emit integer rounds; single-shot
    # methods keep `round: null` (rendered as "-").
    has_rounds = any(row.get("round") is not None for row in rows)

    lines = [
        "# Sequential SBI benchmark report",
        "",
        "Each cell reports mean ± standard deviation across target replicates "
        "when a standard deviation is available. Inference simulation counts "
        "exclude PPC and sensitivity diagnostic simulations.",
        "",
        "SBC/TARP and predictive/sensitivity diagnostics describe different "
        "properties; lower values are not universally better for every metric. "
        "Interpret each metric according to its definition.",
        "",
    ]
    if has_rounds:
        lines.extend(
            [
                "Rows are (budget, round) pairs: `round` is the number of "
                "completed sequential training rounds (the model is scored "
                "after each round). Methods without sequential rounds "
                "(SMC-ABC) have `round = -`.",
                "",
            ]
        )
    for task in sorted(by_task):
        task_rows = sorted(
            by_task[task],
            key=lambda row: (
                float(row.get("budget", 0)),
                -1 if row.get("round") is None else int(row["round"]),
                str(row.get("method", "")),
            ),
        )
        lines.extend([f"## {task}", ""])
        for mean_column in metrics:
            metric_name = mean_column.removeprefix("mean_")
            available = [
                row for row in task_rows
                if isinstance(row.get(mean_column), (int, float))
            ]
            if not available:
                continue
            methods = sorted({str(row.get("method", "unknown")) for row in available})
            round_prefix = "Round | " if has_rounds else ""
            lines.extend(
                [
                    f"### {metric_name}",
                    "",
                    "| Budget | " + round_prefix + " | ".join(methods) + " |",
                    "|---:|" + ("---:|" if has_rounds else "")
                    + "|".join("---:" for _ in methods) + "|",
                ]
            )
            budget_round_pairs = sorted(
                {
                    (
                        float(budget),
                        -1 if row.get("round") is None else int(row["round"]),
                    )
                    for row in available
                    if isinstance((budget := row.get("budget")), (int, float))
                }
            )
            for budget, round_value in budget_round_pairs:
                cells = []
                for method in methods:
                    row = next(
                        (
                            candidate
                            for candidate in available
                            if (
                                isinstance(candidate.get("budget"), (int, float))
                                and float(candidate["budget"]) == budget
                            )
                            and candidate.get("method") == method
                            and (
                                -1
                                if candidate.get("round") is None
                                else int(candidate["round"])
                            )
                            == round_value
                        ),
                        {},
                    )
                    cells.append(_format_cell(row, mean_column))
                round_cell = (
                    f"{round_value if round_value >= 0 else '-'} | "
                    if has_rounds
                    else ""
                )
                lines.append(
                    f"| {budget} | " + round_cell + " | ".join(cells) + " |"
                )
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _final_round_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Keep only the last-round summary per (task, method, budget) for
    budget-axis plots; single-shot rows (round=None) are kept as-is."""
    best: dict[tuple, tuple[int, Mapping[str, Any]]] = {}
    for row in rows:
        key = (row.get("task"), row.get("method"), row.get("budget"))
        round_value = row.get("round")
        rank = -1 if round_value is None else int(round_value)
        if key not in best or rank > best[key][0]:
            best[key] = (rank, row)
    return [row for _rank, row in best.values()]


def _plot_metric(
    path: Path,
    task: str,
    metric_column: str,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_method: dict[str, list[tuple[float, float, Any]]] = defaultdict(list)
    for row in _final_round_rows(rows):
        budget = row.get("budget")
        metric_value = row.get(metric_column)
        if (
            row.get("task") == task
            and isinstance(budget, (int, float))
            and budget > 0
            and isinstance(metric_value, (int, float))
            and math.isfinite(metric_value)
        ):
            metric_name = metric_column.removeprefix("mean_")
            by_method[str(row.get("method", "unknown"))].append(
                (
                    float(budget),
                    float(metric_value),
                    row.get(f"std_{metric_name}"),
                )
            )
    if not by_method:
        return

    figure, axis = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    for method, method_rows in sorted(by_method.items()):
        method_rows.sort(key=lambda row: row[0])
        budgets = [row[0] for row in method_rows]
        means = [row[1] for row in method_rows]
        standard_deviations = [row[2] for row in method_rows]
        if all(
            isinstance(value, (int, float)) and math.isfinite(value)
            for value in standard_deviations
        ):
            axis.errorbar(
                budgets,
                means,
                yerr=[float(value) for value in standard_deviations],
                marker="o",
                capsize=3,
                label=method,
            )
        else:
            axis.plot(budgets, means, marker="o", label=method)
    axis.set_xscale("log")
    axis.set_xlabel("Inference simulation budget")
    axis.set_ylabel(metric_column.removeprefix("mean_").replace("_", " "))
    axis.set_title(f"{task}: {metric_column.removeprefix('mean_')}")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_rounds_metric(
    path: Path,
    task: str,
    metric_column: str,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    """Plot a metric against the number of sequential rounds at the largest
    budget of this task (one line per sequential method)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eligible = [
        row
        for row in rows
        if row.get("task") == task
        and row.get("round") is not None
        and isinstance(row.get(metric_column), (int, float))
        and math.isfinite(cast(float, row.get(metric_column)))
    ]
    if not eligible:
        return
    max_budget = max(
        float(cast(float, row["budget"]))
        for row in eligible
        if isinstance(row.get("budget"), (int, float))
    )
    by_method: dict[str, list[tuple[int, float, Any]]] = defaultdict(list)
    for row in eligible:
        if float(cast(float, row["budget"])) != max_budget:
            continue
        metric_name = metric_column.removeprefix("mean_")
        by_method[str(row.get("method", "unknown"))].append(
            (
                int(cast(int, row["round"])),
                float(cast(float, row[metric_column])),
                row.get(f"std_{metric_name}"),
            )
        )
    if not by_method:
        return

    figure, axis = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    for method, method_rows in sorted(by_method.items()):
        method_rows.sort(key=lambda row: row[0])
        rounds = [row[0] for row in method_rows]
        means = [row[1] for row in method_rows]
        standard_deviations = [row[2] for row in method_rows]
        if all(
            isinstance(value, (int, float)) and math.isfinite(value)
            for value in standard_deviations
        ):
            axis.errorbar(
                rounds,
                means,
                yerr=[float(value) for value in standard_deviations],
                marker="o",
                capsize=3,
                label=method,
            )
        else:
            axis.plot(rounds, means, marker="o", label=method)
    axis.set_xlabel("Sequential training rounds")
    axis.set_ylabel(metric_column.removeprefix("mean_").replace("_", " "))
    axis.set_title(
        f"{task}: {metric_column.removeprefix('mean_')} vs rounds "
        f"(budget {max_budget:g})"
    )
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")


def merge_benchmark_results(
    input_dirs: Sequence[str | Path],
    output_dir: str | Path,
) -> Path:
    """Merge per-shard result directories into one and re-export reports.

    Shards are launched as separate Kaggle kernels; this concatenates their
    ``target_metrics.jsonl``/``benchmark_summary.jsonl`` (dedupe by the
    runner's row keys, later inputs win) and regenerates CSV/Markdown/plots.
    """
    from sfmpe.benchmark.runner import _summary_key, _target_key

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    targets: dict[tuple, dict[str, Any]] = {}
    summaries: dict[tuple, dict[str, Any]] = {}
    for directory in input_dirs:
        directory_path = Path(directory)
        for row in _read_jsonl(directory_path / "target_metrics.jsonl"):
            targets[_target_key(row)] = row
        for row in _read_jsonl(directory_path / "benchmark_summary.jsonl"):
            summaries[_summary_key(row)] = row
    if not targets or not summaries:
        raise FileNotFoundError(
            "No shard results to merge: each input directory needs "
            "target_metrics.jsonl and benchmark_summary.jsonl."
        )
    _write_jsonl(output / "target_metrics.jsonl", list(targets.values()))
    _write_jsonl(output / "benchmark_summary.jsonl", list(summaries.values()))
    export_benchmark_results(output)
    return output


def export_benchmark_results(
    result_dir: str | Path,
    report_dir: str | Path | None = None,
) -> Path:
    """Export runner JSONL results to flat CSVs, plots, and a Markdown report."""
    result_path = Path(result_dir).resolve()
    destination = (
        Path(report_dir).resolve()
        if report_dir is not None
        else result_path
    )
    destination.mkdir(parents=True, exist_ok=True)
    summary_rows = _read_jsonl(result_path / "benchmark_summary.jsonl")
    target_rows = _read_jsonl(result_path / "target_metrics.jsonl")
    _write_csv(destination / "benchmark_summary.csv", summary_rows)
    _write_csv(
        destination / "target_metrics.csv",
        [_flatten_target_row(row) for row in target_rows],
    )

    metric_columns = _numeric_metric_columns(summary_rows)
    _write_markdown_report(
        destination / "benchmark_report.md",
        summary_rows,
        metric_columns,
    )
    figures_dir = destination / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    task_names = sorted({str(row.get("task", "unknown")) for row in summary_rows})
    for task in task_names:
        for metric_column in metric_columns:
            _plot_metric(
                figures_dir
                / f"{_safe_filename(task)}_{_safe_filename(metric_column)}.png",
                task,
                metric_column,
                summary_rows,
            )
            _plot_rounds_metric(
                figures_dir
                / f"{_safe_filename(task)}_rounds_"
                f"{_safe_filename(metric_column)}.png",
                task,
                metric_column,
                summary_rows,
            )
    return destination


def _configured_result_dir(config_path: Path) -> Path:
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, Mapping):
        raise TypeError("Benchmark configuration must be a mapping.")
    output = config.get("output", "./results/benchmark")
    return (config_path.parent / output).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the configured SBI benchmark and export CSV tables, "
            "a Markdown report, and metric-vs-budget plots."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_PROJECT_CONFIG,
        help="Benchmark YAML configuration (default: code/configs/benchmark.yaml).",
    )
    parser.add_argument(
        "--process-only",
        action="store_true",
        help="Skip inference and process existing JSONL results from this config.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="Report destination (default: configured benchmark output directory).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Override the configured output directory for both raw benchmark "
            "results and generated reports."
        ),
    )
    parser.add_argument(
        "--merge-inputs",
        type=Path,
        nargs="+",
        default=None,
        help=(
            "Merge these per-shard result directories (each containing "
            "target_metrics.jsonl and benchmark_summary.jsonl) instead of "
            "running inference."
        ),
    )
    parser.add_argument(
        "--merge-output",
        type=Path,
        default=None,
        help="Destination for merged results (default: configured output dir).",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    result_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else _configured_result_dir(config_path)
    )
    if args.merge_inputs:
        merged_dir = merge_benchmark_results(
            args.merge_inputs,
            args.merge_output.resolve()
            if args.merge_output is not None
            else result_dir,
        )
        print(f"Merged shard results and reports saved to: {merged_dir}")
        return
    if not args.process_only:
        summaries = run_benchmark(config_path, output_dir=result_dir)
        print(f"Benchmark completed: {len(summaries)} task/method/budget groups.")
    report_dir = export_benchmark_results(
        result_dir,
        args.report_dir if args.report_dir is not None else result_dir,
    )
    print(f"CSV, tables, and plots saved to: {report_dir}")


if __name__ == "__main__":
    main()
