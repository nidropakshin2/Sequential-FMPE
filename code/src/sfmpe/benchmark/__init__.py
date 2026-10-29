"""Target-specific simulation-based inference benchmarking."""

from pathlib import Path

from sfmpe.benchmark.metrics import (
    exact_posterior_metrics,
    posterior_predictive_metrics,
    sensitivity_metrics,
)
from sfmpe.benchmark.runner import BenchmarkRunner, run_benchmark


def export_benchmark_results(
    result_dir: str | Path,
    report_dir: str | Path | None = None,
) -> Path:
    from sfmpe.benchmark.report import export_benchmark_results as export

    return export(result_dir, report_dir)


def merge_benchmark_results(
    input_dirs: list[str | Path] | tuple[str | Path, ...],
    output_dir: str | Path,
) -> Path:
    from sfmpe.benchmark.report import merge_benchmark_results as merge

    return merge(input_dirs, output_dir)

__all__ = [
    "BenchmarkRunner",
    "exact_posterior_metrics",
    "export_benchmark_results",
    "merge_benchmark_results",
    "posterior_predictive_metrics",
    "run_benchmark",
    "sensitivity_metrics",
]
