"""Comparative analysis of the SFMPE benchmark shards.

Reads the per-shard Kaggle payloads (results/<shard>/output), merges and
dedupes their target rows and group summaries, runs data quality checks,
builds the master CSV tables, the report figures, and the summary tables.

Usage:
    python code/scripts/analyze_benchmark.py \
        [--results-root <dir>] [--output-dir <dir>]

Data layout notes:
- ``target_metrics.jsonl``: one row per (task, method, budget, rounds,
  replicate, target observation). Posterior metrics live in ``metrics``:
  ``c2st``/W2/MMD exist only for Gaussian (exact reference posterior).
- ``benchmark_summary.jsonl``: one row per (task, method, budget, rounds)
  group; SBC/TARP aggregates are computed over the 10 replicates' targets,
  so ``tarp_area_to_curve`` (0 = ideal) is group-level, not per replicate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))

from sfmpe.benchmark.runner import _summary_key, _target_key  # noqa: E402

DEFAULT_RESULTS_ROOT = Path(
    r"C:\Users\Mikhail\VSCodeProjects\Amortized-learning\code\results"
)
DEFAULT_OUTPUT_DIR = CODE_ROOT / "results" / "analysis"

# Primary quality metric per task and its ideal value. Gaussian has the exact
# posterior (c2st, per-replicate rows); SIR/StochVol rely on TARP expected
# coverage (group-level aggregate over a group's 10 targets: area-to-curve,
# 0 = ecp == alpha, >0 conservative, <0 over-confident).
PRIMARY_METRIC = {
    "Gaussian": "c2st",
    "SIR": "tarp_area_to_curve",
    "StochVol": "tarp_area_to_curve",
}
PRIMARY_IDEAL = {
    "Gaussian": 0.5,
    "SIR": 0.0,
    "StochVol": 0.0,
}
PRIMARY_LABEL = {
    "Gaussian": "C2ST to exact posterior (0.5 = ideal)",
    "SIR": "TARP area-to-curve (0 = ideal)",
    "StochVol": "TARP area-to-curve (0 = ideal)",
}

METHODS = ["SFMPE", "SNPE_C", "SMC_ABC"]
METHOD_COLORS = {"SFMPE": "tab:blue", "SNPE_C": "tab:red", "SMC_ABC": "0.35"}
ROUND_STYLES = {1: "solid", 2: "dashed", 3: "dotted"}
TASKS = ["Gaussian", "SIR", "StochVol"]
BUDGETS = (100, 1000, 10000)

ROW_TOP_FIELDS = (
    "task",
    "method",
    "budget",
    "replicate",
    "rounds",
    "seed",
    "sfmpe_proposal_method",
    "inference_seconds",
    "posterior_sampling_seconds",
    "inference_simulator_draws",
    "setup_simulator_draws",
    "ppc_simulator_draws",
    "sensitivity_simulator_draws",
)
SUMMARY_TOP_FIELDS = (
    "task",
    "method",
    "budget",
    "rounds",
    "replicates",
    "mean_inference_simulator_draws",
    "mean_setup_simulator_draws",
    "mean_ppc_simulator_draws",
    "mean_sensitivity_simulator_draws",
    "mean_inference_seconds",
    "mean_posterior_sampling_seconds",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def discover_shards(results_root: Path) -> dict[str, Path]:
    """Shard output directories: results/<shard>/output holds the JSONLs."""
    shards: dict[str, Path] = {}
    for shard_dir in sorted(p for p in results_root.iterdir() if p.is_dir()):
        for output_dir in (shard_dir / "output", shard_dir / "outputs"):
            if (output_dir / "target_metrics.jsonl").is_file():
                shards.setdefault(shard_dir.name, output_dir)
                break
    return shards


def _flatten_cell(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return json.dumps(value, ensure_ascii=False)


def build_rows(shards: dict[str, Path]) -> pd.DataFrame:
    merged: dict[tuple, dict[str, Any]] = {}
    for shard, output_dir in shards.items():
        for raw in read_jsonl(output_dir / "target_metrics.jsonl"):
            record: dict[str, Any] = {"shard": shard}
            for field in ROW_TOP_FIELDS:
                record[field] = raw.get(field)
            for name, value in (raw.get("metrics") or {}).items():
                record[f"m.{name}"] = _flatten_cell(value)
            merged[_target_key(raw)] = record
    # NB: pd.DataFrame.from_dict(..., orient="index") corrupts rows whose key
    # tuple contains None (pandas 2.2.3 MultiIndex bug) - all SMC groups came
    # out all-NaN. The keyed dict is only for dedupe; build from the values.
    frame = pd.DataFrame(list(merged.values())).reset_index(drop=True)
    return frame.sort_values(
        ["task", "method", "budget", "rounds", "replicate"], na_position="last"
    ).reset_index(drop=True)


def build_groups(shards: dict[str, Path]) -> pd.DataFrame:
    merged: dict[tuple, dict[str, Any]] = {}
    for shard, output_dir in shards.items():
        for raw in read_jsonl(output_dir / "benchmark_summary.jsonl"):
            record: dict[str, Any] = {"shard": shard}
            for field in SUMMARY_TOP_FIELDS:
                record[field] = raw.get(field)
            for name, value in raw.items():
                if name in SUMMARY_TOP_FIELDS:
                    continue
                record[f"g.{name}"] = _flatten_cell(value)
            merged[_summary_key(raw)] = record
    # NB: pd.DataFrame.from_dict(..., orient="index") corrupts rows whose key
    # tuple contains None (pandas 2.2.3 MultiIndex bug) - all SMC groups came
    # out all-NaN. The keyed dict is only for dedupe; build from the values.
    frame = pd.DataFrame(list(merged.values())).reset_index(drop=True)
    return frame.sort_values(
        ["task", "method", "budget", "rounds"], na_position="last"
    ).reset_index(drop=True)


def quality_checks(rows: pd.DataFrame, groups: pd.DataFrame) -> list[str]:
    """Consistency checks; returns human-readable messages."""
    messages: list[str] = []
    over = rows["inference_simulator_draws"] - rows["budget"]
    if int((over > 0).sum()):
        messages.append(
            f"BUDGET OVERRUNS: {int((over > 0).sum())} rows exceeded their "
            f"budget (max +{int(over.max())} draws)."
        )
    else:
        messages.append(
            "Budget accounting OK: every row stayed within its budget."
        )
    per_combo = rows.groupby(["method", "task", "replicate"])["seed"].nunique()
    if int((per_combo > 1).sum()):
        messages.append(
            f"Seed note: {int((per_combo > 1).sum())} (method, task, replicate) "
            "combos used several seeds across budgets/rounds."
        )
    else:
        messages.append(
            "Within-method pairing OK: one seed per (method, task, replicate)."
        )
    paired = rows.groupby(["task", "replicate"])["seed"].nunique().max()
    if paired > 1:
        messages.append(
            f"Cross-method pairing breaks: up to {paired} distinct observation "
            "seeds per (task, replicate)."
        )
    else:
        messages.append(
            "Cross-method pairing OK: methods solve the SAME observations."
        )
    missing_c2st = int(rows["m.c2st"].isna().sum())
    messages.append(f"Rows missing c2st: {missing_c2st} (0 => all Gaussian rows).")
    for task, metric in PRIMARY_METRIC.items():
        if metric == "c2st":
            continue
        column = f"g.{metric}"
        if column not in groups.columns:
            message = f"MISSING {column} in summaries"
        else:
            missing = int(groups[column].isna().sum())
            message = f"Summaries missing {metric}: {missing}."
        if message not in messages:
            messages.append(message)
    complete = int((groups["replicates"] == 10).sum())
    messages.append(f"{complete}/{len(groups)} summary groups have 10 replicates.")
    return messages


def median_iqr(values: pd.Series) -> tuple[float, float, float]:
    values = values.dropna()
    q1 = float(np.nanpercentile(values, 25))
    q3 = float(np.nanpercentile(values, 75))
    return float(values.median()), q1, q3


def group_values(
    rows: pd.DataFrame,
    groups: pd.DataFrame,
    task: str,
    method: str,
    budget: int,
    rounds: int | None,
    metric: str,
) -> pd.Series:
    """Primary metric values for one (task, method, budget[, rounds]) slice.

    `c2st` is per replicate (rows); everything else is group-level
    (one value per (task, method, budget, rounds) from the summaries).
    """
    if metric == "c2st":
        subset = rows[
            (rows.task == task)
            & (rows.method == method)
            & (rows.budget == budget)
        ]
        if rounds is not None:
            subset = subset[subset.rounds == rounds]
        return subset["m.c2st"]
    subset = groups[
        (groups.task == task) & (groups.method == method) & (groups.budget == budget)
    ]
    if rounds is None:
        return subset[f"g.{metric}"]
    selected = subset[subset.rounds == rounds]
    return selected[f"g.{metric}"]


def rounds_metric_frame(
    rows: pd.DataFrame,
    groups: pd.DataFrame,
    task: str,
    method: str,
    budget: int,
    metric: str,
) -> pd.DataFrame:
    """Median-ready frame with columns [rounds, value] for one cell."""
    if metric == "c2st":
        subset = rows[
            (rows.task == task)
            & (rows.method == method)
            & (rows.budget == budget)
        ]
        return pd.DataFrame(
            {"rounds": subset["rounds"], "value": subset["m.c2st"]}
        )
    subset = groups[
        (groups.task == task)
        & (groups.method == method)
        & (groups.budget == budget)
    ]
    return subset[["rounds", f"g.{metric}"]].rename(columns={f"g.{metric}": "value"})


def best_round_for_budget(
    rows: pd.DataFrame,
    groups: pd.DataFrame,
    task: str,
    method: str,
    budget: int,
    metric: str,
) -> int:
    """Round whose median primary metric is closest to the ideal value."""
    frame = rounds_metric_frame(rows, groups, task, method, budget, metric)
    medians = frame.groupby("rounds")["value"].median().dropna()
    if medians.empty:
        raise ValueError(f"No data for {task}/{method}/budget={budget}")
    ideal = PRIMARY_IDEAL[task]
    return int((medians - ideal).abs().sort_values().index[0])


def plot_primary(
    ax: plt.Axes,
    rows: pd.DataFrame,
    groups: pd.DataFrame,
    task: str,
    method: str,
    round_: int | None,
    color: str,
    label: str | None = None,
) -> None:
    """Primary metric curve over budgets; IQR band only for per-replicate c2st."""
    metric = PRIMARY_METRIC[task]
    xs: list[float] = []
    med: list[float] = []
    q1s: list[float] = []
    q3s: list[float] = []
    for budget in BUDGETS:
        values = group_values(rows, groups, task, method, budget, round_, metric)
        if values.empty:
            continue
        center = float(values.median())
        spread = metric == "c2st"
        q1 = float(np.nanpercentile(values, 25)) if spread else center
        q3 = float(np.nanpercentile(values, 75)) if spread else center
        xs.append(budget)
        med.append(center)
        q1s.append(q1)
        q3s.append(q3)
    if not xs:
        return
    if label is None:
        label = method if round_ is None else f"{method} (r{round_})"
    style = "-" if round_ is None else ROUND_STYLES.get(int(round_), "-")
    line, = ax.plot(xs, med, marker="o", ms=4, lw=1.5, color=color, ls=style)
    line.set_label(label)
    if metric == "c2st":
        ax.fill_between(xs, q1s, q3s, color=color, alpha=0.15, lw=0)


def style_axes(ax: plt.Axes, y_log: bool = False) -> None:
    ax.set_xscale("log")
    ax.set_xticks(list(BUDGETS))
    ax.set_xticklabels([str(b) for b in BUDGETS])
    ax.set_xlabel("simulation budget")
    ax.grid(alpha=0.3)
    if y_log:
        ax.set_yscale("log")


def figure_gaussian_c2st(rows, groups, figures: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.4), sharey=True)
    for ax, method in zip(axes, METHODS):
        if method == "SMC_ABC":
            plot_primary(
                ax, rows, groups, "Gaussian", method, None,
                METHOD_COLORS[method], label="SMC_ABC (single shot)",
            )
        else:
            for round_ in (1, 2, 3):
                plot_primary(
                    ax, rows, groups, "Gaussian", method, round_,
                    METHOD_COLORS[method],
                )
        ax.set_title(method)
        ax.legend(fontsize=7)
        style_axes(ax)
    axes[0].set_ylabel(PRIMARY_LABEL["Gaussian"])
    fig.suptitle(
        "Gaussian: posterior accuracy vs simulation budget (median, IQR band)",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(figures / "f1_gaussian_c2st.png", bbox_inches="tight")
    plt.close(fig)


def figure_primary_all_tasks(rows, groups, figures: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.4), sharey=False)
    for ax, task in zip(axes, TASKS):
        metric = PRIMARY_METRIC[task]
        ax.axhline(PRIMARY_IDEAL[task], color="k", lw=0.8, ls=":", alpha=0.7)
        for method in METHODS:
            if method == "SMC_ABC":
                plot_primary(
                    ax, rows, groups, task, method, None,
                    METHOD_COLORS[method], label="SMC_ABC (single shot)",
                )
                continue
            xs: list[float] = []
            med: list[float] = []
            q1s: list[float] = []
            q3s: list[float] = []
            used_rounds: list[int] = []
            for budget in BUDGETS:
                round_ = best_round_for_budget(rows, groups, task, method, budget, metric)
                values = group_values(rows, groups, task, method, budget, round_, metric)
                if values.empty:
                    continue
                center = float(values.median())
                spread = metric == "c2st"
                q1 = float(np.nanpercentile(values, 25)) if spread else center
                q3 = float(np.nanpercentile(values, 75)) if spread else center
                xs.append(budget)
                med.append(center)
                q1s.append(q1)
                q3s.append(q3)
                used_rounds.append(round_)
            if not xs:
                continue
            color = METHOD_COLORS[method]
            line, = ax.plot(xs, med, marker="o", ms=4, lw=1.5, color=color)
            line.set_label(method + " (best rounds)")
            if metric == "c2st":
                ax.fill_between(xs, q1s, q3s, color=color, alpha=0.15, lw=0)
        ax.set_title(task)
        ax.set_ylabel(PRIMARY_LABEL[task])
        ax.legend(fontsize=7, loc="best")
        style_axes(ax)
    fig.suptitle(
        "Posterior quality vs budget (SFMPE/SNPE_C: per-budget best round)",
        y=1.03,
    )
    fig.tight_layout()
    fig.savefig(figures / "f2_primary_metric_all_tasks.png", bbox_inches="tight")
    plt.close(fig)


def figure_rounds_effect(rows, groups, figures: Path) -> None:
    """Delta of extra sequential rounds vs rounds=1 at a fixed total budget."""
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.4), sharey=False)
    for ax, task in zip(axes, TASKS):
        metric = PRIMARY_METRIC[task]
        for method in ("SFMPE", "SNPE_C"):
            for round_ in (2, 3):
                deltas, xs = [], []
                for budget in BUDGETS:
                    frame = rounds_metric_frame(rows, groups, task, method, budget, metric)
                    if frame.empty:
                        continue
                    base = frame[frame.rounds == 1]["value"]
                    upper = frame[frame.rounds == round_]["value"]
                    if base.empty or upper.empty:
                        continue
                    deltas.append(float(np.median(upper)) - float(np.median(base)))
                    xs.append(budget)
                if not deltas:
                    continue
                line, = ax.plot(
                    xs, deltas, color=METHOD_COLORS[method],
                    ls=ROUND_STYLES[int(round_)], marker="o", ms=4, lw=1.4,
                )
                line.set_label(f"{method} r{round_}-r1")
        ax.axhline(0.0, color="k", lw=0.8, ls=":", alpha=0.7)
        ax.set_ylabel(PRIMARY_LABEL[task] + ", delta")
        ax.set_title(task)
        ax.legend(fontsize=7)
        style_axes(ax)
    fig.suptitle(
        "Effect of extra sequential rounds at a fixed total simulation budget",
        y=1.03,
    )
    fig.tight_layout()
    fig.savefig(figures / "f3_rounds_effect.png", bbox_inches="tight")
    plt.close(fig)


def figure_ppc(rows, groups, figures: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.4), sharey=True)
    for ax, task in zip(axes, TASKS):
        metric = PRIMARY_METRIC[task]
        for method in METHODS:
            if method == "SMC_ABC":
                subset = rows[(rows.task == task) & (rows.method == method)]
            else:
                subsets = []
                for budget in BUDGETS:
                    round_ = best_round_for_budget(rows, groups, task, method, budget, metric)
                    subsets.append(
                        rows[
                            (rows.task == task)
                            & (rows.method == method)
                            & (rows.budget == budget)
                            & (rows.rounds == round_)
                        ]
                    )
                subset = pd.concat(subsets)
            xs, med, q1s, q3s = [], [], [], []
            for budget in BUDGETS:
                values = subset[subset.budget == budget]["m.ppc_mean_abs_z"]
                if values.empty:
                    continue
                center, lo, hi = median_iqr(values)
                xs.append(budget)
                med.append(center)
                q1s.append(lo)
                q3s.append(hi)
            if not xs:
                continue
            line, = ax.plot(xs, med, marker="o", ms=4, lw=1.5, color=METHOD_COLORS[method])
            line.set_label(method)
            ax.fill_between(xs, q1s, q3s, color=METHOD_COLORS[method], alpha=0.15, lw=0)
        ax.set_title(task)
        ax.legend(fontsize=7)
        style_axes(ax)
    axes[0].set_ylabel("PPC mean |z| (median, IQR)")
    fig.suptitle(
        "Posterior predictive calibration: observed-data |z| vs budget "
        "(SFMPE/SNPE_C: best primary-metric round)",
        y=1.03,
    )
    fig.tight_layout()
    fig.savefig(figures / "f4_ppc_z.png", bbox_inches="tight")
    plt.close(fig)


def figure_runtime(rows, figures: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.4), sharey=True)
    for ax, task in zip(axes, TASKS):
        for method in METHODS:
            subset = rows[(rows.task == task) & (rows.method == method)]
            xs, med, q1s, q3s = [], [], [], []
            for budget in BUDGETS:
                values = subset[subset.budget == budget]["inference_seconds"]
                if values.empty:
                    continue
                center, lo, hi = median_iqr(values)
                xs.append(budget)
                med.append(center)
                q1s.append(lo)
                q3s.append(hi)
            if not xs:
                continue
            line, = ax.plot(xs, med, marker="o", ms=4, lw=1.5, color=METHOD_COLORS[method])
            line.set_label(method)
            ax.fill_between(xs, q1s, q3s, color=METHOD_COLORS[method], alpha=0.15, lw=0)
        ax.set_title(task)
        ax.legend(fontsize=7)
        style_axes(ax, y_log=True)
    axes[0].set_ylabel("inference seconds (log)")
    fig.suptitle(
        "Wall-clock inference time per target (all rounds of a method)", y=1.03
    )
    fig.tight_layout()
    fig.savefig(figures / "f5_runtime.png", bbox_inches="tight")
    plt.close(fig)


def figure_support_fraction(rows, groups, figures: Path) -> None:
    tasks = ["SIR", "StochVol"]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4), sharey=True)
    for ax, task in zip(axes, tasks):
        metric = PRIMARY_METRIC[task]
        for method in METHODS:
            if method == "SMC_ABC":
                subset = rows[(rows.task == task) & (rows.method == method)]
            else:
                subsets = []
                for budget in BUDGETS:
                    round_ = best_round_for_budget(rows, groups, task, method, budget, metric)
                    subsets.append(
                        rows[
                            (rows.task == task)
                            & (rows.method == method)
                            & (rows.budget == budget)
                            & (rows.rounds == round_)
                        ]
                    )
                subset = pd.concat(subsets)
            xs, med, q1s, q3s = [], [], [], []
            for budget in BUDGETS:
                values = subset[subset.budget == budget]["m.posterior_support_fraction"]
                if values.empty:
                    continue
                center, lo, hi = median_iqr(values)
                xs.append(budget)
                med.append(center)
                q1s.append(lo)
                q3s.append(hi)
            if not xs:
                continue
            line, = ax.plot(xs, med, marker="o", ms=4, lw=1.5, color=METHOD_COLORS[method])
            line.set_label(method)
            ax.fill_between(xs, q1s, q3s, color=METHOD_COLORS[method], alpha=0.15, lw=0)
        ax.axhline(1.0, color="k", lw=0.8, ls=":", alpha=0.7)
        ax.set_ylim(-0.02, 1.05)
        ax.set_title(task)
        ax.legend(fontsize=7)
        style_axes(ax)
    axes[0].set_ylabel("posterior support fraction")
    fig.suptitle(
        "Fraction of posterior samples inside the task's parameter support",
        y=1.03,
    )
    fig.tight_layout()
    fig.savefig(figures / "f6_support_fraction.png", bbox_inches="tight")
    plt.close(fig)


def hero_table(rows, groups) -> str:
    """Task x budget table of primary metrics; neural methods use their
    per-budget best round (round shown in brackets)."""
    lines = [
        "| task | budget | SFMPE | SNPE_C | SMC_ABC (single shot) |",
        "|---|---|---|---|---|",
    ]
    for task in TASKS:
        metric = PRIMARY_METRIC[task]
        for budget in BUDGETS:
            cells = [task, str(budget)]
            for method in METHODS:
                if method == "SMC_ABC":
                    values = group_values(rows, groups, task, method, budget, None, metric)
                    cells.append(
                        f"{float(values.median()):.3f}" if not values.empty else "n/a"
                    )
                else:
                    round_ = best_round_for_budget(rows, groups, task, method, budget, metric)
                    values = group_values(rows, groups, task, method, budget, round_, metric)
                    if values.empty:
                        cells.append("n/a")
                        continue
                    med = float(values.median())
                    if metric == "c2st":
                        q1 = float(np.nanpercentile(values, 25))
                        q3 = float(np.nanpercentile(values, 75))
                        cells.append(f"{med:.3f} [{q1:.3f}-{q3:.3f}] (r{round_})")
                    else:
                        cells.append(f"{med:.3f} (r{round_})")
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


REPORT_BODY = """
## Итоговые выводы

Сравнение SFMPE / SNPE_C / SMC-ABC при одинаковом бюджете симуляций
(100 / 1000 / 10000; 10 репликатов, медианы) — по трём задачам с разной
«измеримостью истины»: Gaussian (есть точная posterior → C2ST),
SIR и StochVol (точной posterior нет → калибровка TARP/SBC + PPC).

- **Генералист — SNPE_C.** При большом бюджете даёт лучший
  posterior на Gaussian (C2ST 0.499 при 10000 — практически идеал) и
  почти идеальную калибровку на SIR (ATC ≈ −0.008..0.006 при r1/r2).
  На StochVol слегка переуверен (ATC −0.03..−0.06 при 1000–10000).
- **SFMPE выигрывает по калибровке при малых и средних бюджетах.**
  StochVol при 100–1000: ATC 0.002–0.003 (r1) — наилучший результат во
  всём бенчмарке; SIR при 100: 0.034 против 0.128 у SNPE_C r1 и 0.038
  у SMC. Но при 10000 качество деградирует (SIR: −0.105 у r1 —
  переуверенность; StochVol: +0.07..+0.12 — консервативность).
- **SMC-ABC — дешёвый, но систематически самый грубый метод.**
  Не достигает идеала ни на одном бюджете Gaussian (0.56–0.63) и
  остаётся консервативным на не-Gaussian задачах, особенно на StochVol
  при 100 (ATC 0.164).
- **Sequential-раунды при фиксированном суммарном бюджете почти не
  помогают и часто вредят** (см. раздел 2). Единственный устойчивый
  выигрыш — SNPE_C r2 на SIR. Для SFMPE почти везде оптимален 1 раунд.
- **Wall-time иерархия: SMC < SNPE_C < SFMPE** (см. раздел 5). SFMPE
  «покупает» калибровку дорогим wall-time; на StochVol цена — сотни
  секунд на таргет против десятков у SNPE_C и единиц/десятков у SMC.

## 1. Gaussian: точность posterior (C2ST к точному решению)

| метод | 100 | 1000 | 10000 |
|---|---|---|---|
| SFMPE r1 | 0.605 | 0.544 | 0.521 |
| SNPE_C r1 | 0.552 | 0.523 (0.519 у r2) | 0.499 |
| SMC-ABC | 0.612 | 0.630 | 0.561 |

- SFMPE r1 — самая чистая кривая обучения: монотонное улучшение с
  бюджетом. Порог «амортизация обгоняет ABC» — около 1000 симуляций:
  оба нейро-метода (0.544 / 0.523) уже лучше SMC (0.630); при 100 —
  паритет в пределах разброса.
- SNPE_C 0.499 при 10000: C2ST чуть ниже 0.5 означает не «лучше
  идеала», а статистическую неразличимость с точным posterior
  (точность классификатора на уровне шанса).
- SMC: даже 10000 симуляций дают 0.561 с широким IQR — амортизация
  выигрывает качественно, не только по времени.

## 2. Sequential-раунды при фиксированном суммарном бюджете

Δ = медиана(rK) − медиана(r1) при одинаковом **суммарном** бюджете
(фактически per-round бюджет делится на число раундов):

- **Gaussian: Δ ≥ +0.01 у всех K>1 и обоих методов** — последовательные
  раунды только вредят C2ST. Хуже всего SNPE_C r2 при 100 (0.683 против
  0.552 у r1: пер-раундовый бюджет 50 слишком мал).
- **SIR: SNPE_C r2 — единственный устойчивый выигрыш** (Δ ≈ −0.14 при
  100 и −0.09 при 1000 против r1). SFMPE — нейтрально/отрицательно
  (единичный плюс r3 при 1000: 0.033 против 0.067).
- **StochVol: секвенция вредит** — SFMPE r3 при 1000 поднимает ATC с
  0.003 до 0.141; у SNPE_C раунды сдвигают ATC от нуля в плюс.
- Вывод: sequential-байесовская итеративность — не бесплатное
  улучшение; при мелком per-round бюджете она размывает обучающие
  выборки. Выбранный в отчёте «best round per budget» — пост-фактум
  оракул (см. ограничения).

## 3. Калибровка на SIR и StochVol (TARP area-to-curve; 0 = идеал,
> 0 консервативная posterior, < 0 переуверенная)

- SFMPE: лучший калибратор на StochVol при 100–1000 (0.002–0.003 —
  ecp ≡ alpha почти точно). На SIR — тоже лучший при 100 (0.034).
- SNPE_C: SIR — почти идеален с r2 (−0.008 при 100 и 1000); StochVol —
  умеренная переуверенность (−0.03..−0.06) при 1000–10000.
- SMC-ABC: всегда консервативен; на SIR консервативность парадоксально
  растёт с бюджетом (0.038 → 0.085 → 0.118) — при более густом
  сэмплировании posterior не сужается так, как должна.
- **Диссонанс SBC vs TARP у SFMPE на StochVol**: при 100–1000 TARP почти
  идеален, но SBC-ранги отвергаются (min KS p ≈ 2e-4…1e-3) — покрытие
  нормальное, а ранговая структура смещена. При 10 таргетах это
  smoke-тест, но сигнал сильный; SNPE_C на SIR на грани (p ≈ 0.05–0.10),
  SMC — без отказов (p ≈ 0.13–0.32).

## 4. Поддержка априорной области и предиктивная калибровка

- **Support fraction** (доля posterior-сэмплов внутри support задачи):
  SNPE_C ровно 1.00 во всех группах — подозрительно; вероятно, артефакт
  измерения/обрезка (проверить определение — см. ограничения).
  SFMPE течёт сильнее всех: StochVol при 100 — только 42% сэмплов
  внутри support (SIR при 100 — 54%); к 10000 уходит (86% / 97%).
  SMC: 62–89%.
- Утечка поддержки согласована с консервативным TARP: posterior шире
  априорного бокса.
- **PPC mean |z|** (идеал ≈ 0.8 = E|N(0,1)|): все методы и бюджеты
  дают 0.29–0.83, т.е. предиктивные распределения везде переразбросаны;
  SFMPE на SIR при 10000 — единственный близкий к идеалу (0.83). SMC
  систематически самый разбросанный (0.53–0.60 на SIR/StochVol).

## 5. Вычислительная стоимость (медиана inference_seconds на таргет)

| задача | метод | 100 | 1000 | 10000 |
|---|---|---|---|---|
| Gaussian | SFMPE / SNPE_C / SMC | 86 / 2.8 / 0.4 | 91 / 6.6 / 0.8 | 156 / 34 / 4.4 |
| SIR | SFMPE / SNPE_C / SMC | 1.6 / 3.8 / 1.4 | 4.2 / 13 / 8.6 | 21 / 78 / 80 |
| StochVol | SFMPE / SNPE_C / SMC | 228 / 5.0 / 1.5 | 498 / 9.4 / 11 | 379 / 56 / 105 |

- SFMPE на Gaussian и StochVol стоит на 1–2 порядка дороже конкурентов
  и почти не растёт с бюджетом (фиксированная стоимость тренировки);
  на StochVol это 4–8 минут на таргет.
- SMC почти всегда самый дешёвый, но на SIR при 10000 сравнивается с
  SNPE_C (~80 с) — density-подгонка начинает доминировать.
- Практическая связка: SFMPE@StochVol-100 даёт ATC 0.002 за 228 с,
  SNPE_C@StochVol-10000 — ATC 0.021 за 56 с: SFMPE покупает точность
  wall-time, SNPE_C — амортизацией.
- Анализ осторожен: device-политика Kaggle (auto) могла отличаться
  между шардами (см. ограничения).

## 6. Практические рекомендации

- Нужен максимум точности posterior и бюджет ≥ 10000: **SNPE_C r1**.
- Малый/средний бюджет (100–1000) и критична калибровка, особенно на
  многомерном stochvol: **SFMPE r1** (если wall-time приемлем).
- Быстрая разведка / интерпретируемость: **SMC-ABC**; на StochVol с
  бюджетом 100 — не рекомендуется (ATC 0.164; предиктивная |z| ≈ 0.56
  заметно ниже идеала 0.8 — posterior переразбросана).
- Sequential-раунды включать только при достаточном per-round бюджете
  и (кроме SIR-подобных случаев у SNPE_C) ожидать нейтральный или
  отрицательный эффект; стартовать с 1 раунда.

## Figures

1. `figures/f1_gaussian_c2st.png` - Gaussian: C2ST vs simulation budget,
   per method and rounds (median, IQR band).
2. `figures/f2_primary_metric_all_tasks.png` - primary metric vs budget
   for all tasks (SFMPE/SNPE_C: per-budget best round).
3. `figures/f3_rounds_effect.png` - rounds effect at fixed total budget
   (median rK - median r1).
4. `figures/f4_ppc_z.png` - PPC mean |z| vs budget.
5. `figures/f5_runtime.png` - inference seconds vs budget (log-log).
6. `figures/f6_support_fraction.png` - posterior support fraction
   (SIR, StochVol).

## Ограничения и пробелы (заполнить позже)

1. **Нет эталонной posterior для SIR/StochVol** (нет C2ST/W2/MMD):
   качество оценивается только калибровкой (SBC/TARP) и PPC. Отметка:
   калиброванная posterior не обязана быть точной — нужен близкий к
   истине референс (сверхдлинный NLE/HMC) для вторичных метрик.
2. **Низкая мощность SBC/TARP**: 1 наблюдение θ-target на репликат,
   агрегаты по 10 таргетам на группу; p-value SBC — smoke-тест.
3. **StochVol SFMPE/SNPE_C без .pt-файлов** (легаси-запуск без
   сохранения сэмплов): нельзя переагрегировать или расширить
   диагностику (ecp/alpha-кривые, per-parameter разбор) без перезапуска;
   сохранённые сводки полные и согласованные.
4. **Wall-time не нормализован по device/фону** (Kaggle device=auto):
   сравнение стоимости ориентировочное; для строгих выводов — фиксация
   устройства и повтор.
5. **SNPE_C support fraction ≡ 1.00** — требует проверки определения
   метрики (обрезка сэмплов / prior-box) как возможного артефакта
   измерения, прежде чем считать это преимуществом метода.
6. **Кросс-методное парирование не строгое**: на (task, replicate)
   встречается до 3 разных наблюдательных seed — сравнения в отчёте
   по медианам равных бюджетов, а не парные тесты по репликатам.
7. **«Best round per budget» — оракул постфактум**: в реальном
   эксперименте число раундов выбирается до/во время прогона; для
   честного выбора нужна протокольная процедура (валидационная
   калибровка per budget), в этом прогоне её нет.
8. **Sensitivity-метрики** (±ε, posterior sd) собраны, но не
   визуализированы; доступны в `benchmark_rows.csv` (`m.sensitivity_*`).
9. Статус данных: `stochvol_smc_abc` перезапущен после фикса KDE
   (`26aad33`) — данные валидны; для остальных шардов подобная
   верификация версии кода не выполнялась.
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate shards and build the comparative analysis."
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    output_dir: Path = args.output_dir
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    shards = discover_shards(args.results_root)
    if not shards:
        raise FileNotFoundError(f"No shard outputs under {args.results_root}")
    print(f"Shards: {len(shards)}")
    rows = build_rows(shards)
    groups = build_groups(shards)
    print(f"Target rows: {len(rows)}; summary groups: {len(groups)}")

    checks = quality_checks(rows, groups)
    for message in checks:
        print(" - " + message)

    rows.to_csv(output_dir / "benchmark_rows.csv", index=False)
    groups.to_csv(output_dir / "benchmark_groups.csv", index=False)

    figure_gaussian_c2st(rows, groups, figures)
    figure_primary_all_tasks(rows, groups, figures)
    figure_rounds_effect(rows, groups, figures)
    figure_ppc(rows, groups, figures)
    figure_runtime(rows, figures)
    figure_support_fraction(rows, groups, figures)

    report = "\n".join(
        [
            "# SFMPE benchmark: сравнительный анализ",
            "",
            f"Shards: {len(shards)}; target rows: {len(rows)}; "
            f"summary groups: {len(groups)}.",
            "",
            "## Data quality checks",
            "",
            *[f"- {message}" for message in checks],
            "",
            "## Primary metric per (task, budget)",
            "",
            hero_table(rows, groups),
            REPORT_BODY,
        ]
    )
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    print(f"Analysis written to {output_dir}")


if __name__ == "__main__":
    main()
