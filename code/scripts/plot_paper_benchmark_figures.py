"""Generate the StochVol posterior comparison used by the paper."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import gaussian_kde


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "code" / "results"
RESULTS_NEW = ROOT / "code" / "results_new"
FIGURES = ROOT / "paper" / "src" / "images"
PLOT_BUDGETS = (100, 1000, 10000)
METHODS = {
    "SFMPE": "SFMPE",
    "SNPE_C": "SNPE-C",
    "SMC_ABC": "SMC-ABC",
}
COLORS = {
    "SFMPE": "#0072B2",
    "SNPE-C": "#D55E00",
    "SMC-ABC": "#009E73",
}
POSTERIOR_SAMPLES = {
    "SFMPE": (
        RESULTS
        / "all-tasks-sfmpe"
        / "benchmark"
        / "posterior_samples"
        / "StochVol__SFMPE__10000__1__0.pt",
        "#0072B2",
    ),
    "SNPE-C": (
        RESULTS
        / "stochvol-snpe-c"
        / "output"
        / "posterior_samples"
        / "StochVol__SNPE_C__10000__1__0.pt",
        "#D55E00",
    ),
}
PARAMETERS = (r"$\mu$", r"$\phi$", r"$\sigma$", r"$m$")


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _summary_sources() -> list[tuple[str, Path]]:
    return [
        ("SFMPE", RESULTS / "all-tasks-sfmpe" / "benchmark"),
        ("SNPE-C", RESULTS / "gaussian_snpe-c" / "output"),
        ("SNPE-C", RESULTS / "sir-snpe-c" / "output"),
        ("SNPE-C", RESULTS_NEW / "stochvol_snpe-c" / "output"),
        ("SMC-ABC", RESULTS / "gaussian_smc_abc" / "output"),
        ("SMC-ABC", RESULTS / "sir-smc-abc" / "output"),
        ("SMC-ABC", RESULTS / "stochvol_smc_abc" / "output"),
    ]


def plot_metric_curves() -> None:
    rows_by_key: dict[tuple[str, str, int], dict] = {}
    for expected_method, directory in _summary_sources():
        for row in _read_jsonl(directory / "benchmark_summary.jsonl"):
            method = METHODS.get(row.get("method"))
            if method != expected_method:
                continue
            budget = int(row["budget"])
            if budget not in PLOT_BUDGETS:
                continue
            rounds = row.get("rounds")
            if method != "SMC-ABC" and rounds != 1:
                continue
            if method == "SMC-ABC" and rounds is not None:
                continue
            key = (str(row["task"]), method, budget)
            if key in rows_by_key:
                raise ValueError(f"Duplicate benchmark summary group: {key}")
            rows_by_key[key] = row

    figure, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), constrained_layout=True)
    for axis, task in zip(axes, ("Gaussian", "SIR", "StochVol")):
        metric = "mean_c2st" if task == "Gaussian" else "tarp_area_to_curve"
        for method in METHODS.values():
            points = sorted(
                (
                    (budget, row[metric], row.get("std_c2st"))
                    for (row_task, row_method, budget), row in rows_by_key.items()
                    if row_task == task and row_method == method
                ),
                key=lambda point: point[0],
            )
            if not points:
                raise ValueError(f"Missing {method} results for {task}")
            budgets, values, deviations = zip(*points)
            if task == "Gaussian":
                axis.errorbar(
                    budgets,
                    values,
                    yerr=deviations,
                    marker="o",
                    linewidth=1.8,
                    capsize=3,
                    color=COLORS[method],
                    label=method,
                )
            else:
                axis.plot(
                    budgets,
                    values,
                    marker="o",
                    linewidth=1.8,
                    color=COLORS[method],
                    label=method,
                )

        axis.set_xscale("log")
        axis.set_xticks(PLOT_BUDGETS, labels=("100", "1 000", "10 000"))
        axis.set_xlabel("Число симуляций")
        axis.set_title(task)
        axis.grid(True, which="both", alpha=0.25)
        axis.axhline(
            0.5 if task == "Gaussian" else 0,
            color="#555555",
            linestyle="--",
            linewidth=1,
        )
        axis.set_ylabel("C2ST" if task == "Gaussian" else "ATC")

    axes[0].set_ylim(0.46, 0.70)
    axes[1].set_ylim(-0.08, 0.15)
    axes[2].set_ylim(-0.08, 0.18)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=3,
        frameon=False,
    )
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES / "benchmark_metrics.png", dpi=300)
    plt.close(figure)


def _load_samples(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Posterior samples not found: {path}")
    samples = torch.load(path, map_location="cpu", weights_only=True).numpy()
    if samples.ndim != 2 or samples.shape[1] != len(PARAMETERS):
        raise ValueError(
            f"Expected posterior samples with shape (n, 4), got {samples.shape} "
            f"from {path}"
        )
    if samples.shape[0] < 2 or not np.isfinite(samples).all():
        raise ValueError(f"Invalid posterior samples in {path}")
    return samples


def plot_stochvol_marginals() -> None:
    samples = {
        method: _load_samples(path)
        for method, (path, _) in POSTERIOR_SAMPLES.items()
    }
    figure, axes = plt.subplots(1, 4, figsize=(11, 3.2))

    for parameter_index, (axis, parameter) in enumerate(
        zip(axes, PARAMETERS)
    ):
        combined = np.concatenate(
            [draws[:, parameter_index] for draws in samples.values()]
        )
        lower, upper = np.quantile(combined, (0.005, 0.995))
        padding = max((upper - lower) * 0.08, 1e-6)
        x_values = np.linspace(lower - padding, upper + padding, 400)

        for method, draws in samples.items():
            density = gaussian_kde(draws[:, parameter_index])(x_values)
            color = POSTERIOR_SAMPLES[method][1]
            axis.fill_between(x_values, density, color=color, alpha=0.2)
            axis.plot(x_values, density, color=color, linewidth=1.8, label=method)

        axis.set_title(parameter)
        axis.set_xlabel(parameter)
        axis.grid(alpha=0.2)

    axes[0].set_ylabel("Плотность")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    figure.tight_layout(rect=(0, 0, 1, 0.88))
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES / "stochvol_posterior.png", dpi=300)
    plt.close(figure)


if __name__ == "__main__":
    plot_metric_curves()
    plot_stochvol_marginals()
