"""Reproduce the StochVol target parameters used by the canonical benchmark.

Run from ``code/``:

    python scripts/reproduce_stochvol_true_theta.py

The script writes CSV to stdout. Use ``--output`` to save it to a file.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import torch
import yaml

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))

from sfmpe.tasks.StochVol import StochVolTask

DEFAULT_CONFIG = CODE_ROOT / "configs" / "benchmark.yaml"
PARAMETERS = ("mu", "phi", "sigma", "m")


def _read_mapping(path: Path, label: str) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"{label} must contain a YAML mapping: {path}")
    return value


def _resolve_device(config: dict[str, Any], override: str | None) -> str:
    benchmark_device = override or config.get("device", "auto")
    if benchmark_device == "auto":
        benchmark_device = "cuda" if torch.cuda.is_available() else "cpu"
    if benchmark_device not in {"cpu", "cuda"}:
        raise ValueError(
            f"device must be 'auto', 'cpu', or 'cuda', got {benchmark_device!r}"
        )
    if benchmark_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    devices = config.get("devices", {})
    if not isinstance(devices, dict):
        raise TypeError("The benchmark 'devices' setting must be a mapping.")
    task_device = devices.get("nn")
    if task_device in {None, "auto"}:
        return benchmark_device
    if task_device not in {"cpu", "cuda"}:
        raise ValueError(
            f"devices.nn must be 'auto', 'cpu', or 'cuda', got {task_device!r}"
        )
    if task_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("devices.nn requests CUDA, but CUDA is unavailable.")
    return task_device


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _target_config(
    config_path: Path, benchmark_config: dict[str, Any]
) -> tuple[int, int, Path]:
    tasks = benchmark_config.get("tasks")
    if not isinstance(tasks, dict):
        raise TypeError("The benchmark 'tasks' setting must be a mapping.")
    task_names = list(tasks)
    task_name = next(
        (name for name in task_names if name.casefold() == "stochvol"),
        None,
    )
    if task_name is None:
        raise ValueError("The benchmark config does not include a StochVol task.")

    task_config_path = (config_path.parent / tasks[task_name]).resolve()
    task_seed = int(benchmark_config.get("seed", 0)) + task_names.index(
        task_name
    ) * 100_000
    replicates = int(benchmark_config.get("replicates", 1))
    if replicates < 1:
        raise ValueError(f"replicates must be positive, got {replicates}")
    return task_seed, replicates, task_config_path


def _write_rows(rows: list[dict[str, Any]], output: TextIO) -> None:
    fields = ("task", "target_seed", "device", "replicate", *PARAMETERS)
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce true_theta values for the benchmark's StochVol targets."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="benchmark YAML (default: configs/benchmark.yaml)",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default=None,
        help="override the device policy in the benchmark config",
    )
    parser.add_argument(
        "--replicate",
        type=int,
        help="print only this zero-based replicate (default: all)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="write CSV to this path instead of stdout",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    benchmark_config = _read_mapping(config_path, "Benchmark config")
    seed, replicates, task_config_path = _target_config(
        config_path, benchmark_config
    )
    device = _resolve_device(benchmark_config, args.device)
    if args.replicate is not None and not 0 <= args.replicate < replicates:
        raise ValueError(
            f"replicate must be between 0 and {replicates - 1}, "
            f"got {args.replicate}"
        )

    _set_seed(seed)
    task_config = _read_mapping(task_config_path, "StochVol task config")
    task = StochVolTask(task_config, device=device)
    target_thetas = task.sample_prior((replicates,)).detach().cpu()

    replicate_indices = (
        [args.replicate]
        if args.replicate is not None
        else list(range(replicates))
    )
    rows = [
        {
            "task": "StochVol",
            "target_seed": seed,
            "device": device,
            "replicate": index,
            **dict(zip(PARAMETERS, target_thetas[index].tolist(), strict=True)),
        }
        for index in replicate_indices
    ]

    if args.output is None:
        _write_rows(rows, sys.stdout)
    else:
        output_path = args.output.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as stream:
            _write_rows(rows, stream)
        print(f"Wrote {len(rows)} target row(s) to {output_path}")


if __name__ == "__main__":
    main()
