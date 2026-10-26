"""Run a self-contained sequential SBI benchmark on Kaggle.

Submit from the repository's ``code`` directory with:

    kagglex run --dir . --file notebooks/benchmark_kaggle_standalone.py --include-data src pyproject.toml --gpu t4-2x

Bundling ``src`` and ``pyproject.toml`` makes the remote run install this exact
checkout, including the CPU-only SMC-ABC fix, rather than an older GitHub copy.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TextIO


def install_dependencies() -> str:
    import torch

    torch_version = torch.__version__
    constraint = Path(tempfile.gettempdir()) / "sfmpe-torch-constraint.txt"
    constraint.write_text(f"torch=={torch_version}\n", encoding="utf-8")
    project_root = Path.cwd()
    has_bundled_source = (
        (project_root / "pyproject.toml").is_file()
        and (project_root / "src" / "sfmpe").is_dir()
    )
    package_requirement = (
        ["-e", str(project_root)]
        if has_bundled_source
        else [
            "git+https://github.com/nidropakshin2/Sequential-FMPE.git#subdirectory=code"
        ]
    )
    if not has_bundled_source:
        raise FileNotFoundError(
            "The benchmark requires the updated local SFMPE source. Submit with "
            "`--include-data src pyproject.toml` so Kaggle installs this checkout."
        )
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "-c",
            str(constraint),
            *package_requirement,
            "sbi==0.21.0",
            "pandas",
            "matplotlib",
            "torchdiffeq",
            "PyYAML",
        ]
    )

    import torch

    if torch.__version__ != torch_version:
        raise RuntimeError(
            f"Dependency installation changed Kaggle PyTorch from "
            f"{torch_version} to {torch.__version__}."
        )
    print(f"Dependencies installed; preserving Kaggle PyTorch {torch_version}.")
    return torch_version


TASK_CONFIGS = {
    "Gaussian": {
        "prior": {
            "mean": [0.0, 0.0],
            "covariance": [[1.5, 0.3], [0.3, 1.0]],
        },
        "simulator": {
            "likelihood_shift": [0.0, 0.0],
            "likelihood_covariance": [[0.4, 0.1], [0.1, 0.6]],
        },
        "summary": {"type": "identity"},
        "logger": {
            "name": "Gaussian",
            "level": "WARNING",
            "log_to_file": False,
        },
    },
    "SIR": {
        "prior": {
            "gamma_range": [0.05, 0.5],
            "beta_range": [0.5, 5.0],
        },
        "simulator": {"S": [1000, 2000], "I": [100, 200], "R": [0, 50], "T": [60, 120]},
        "summary": {
            "type": "handmade",
            "input_dim": 3,
            "hidden_dim": 32,
            "num_layers": 2,
            "output_dim": 5,
            "dropout": 0.1,
        },
        "logger": {
            "name": "SIR",
            "level": "WARNING",
            "log_file_path": "sir.log",
        },
    },
    "StochVol": {
        "prior": {
            "mu_loc": 0.0,
            "mu_scale": 0.2,
            "phi_alpha": 20.0,
            "phi_beta": 1.5,
            "sigma_ig_shape": 2.5,
            "sigma_ig_scale": 0.025,
            "m_loc": -1.5,
            "m_scale": 1.0,
        },
        "simulator": {"T": [100, 250]},
        "summary": {"type": "handmade"},
        "logger": {
            "name": "StochVol",
            "level": "WARNING",
            "log_file_path": "stochvol.log",
        },
    },
}

SFMPE_CONFIG = {
    "flow": {
        "init_dist": {"type": "normal"},
        "velocity_model": {
            "type": "simple",
            "params": {"hidden_dim": 32, "num_layers": 2},
        },
        "path": {"type": "affine"},
    },
    "optimizer": {
        "type": "Adam",
        "params": {
            "task": {"summary": {"lr": 0.0001}},
            "flow_model": {"lr": 0.0001},
        },
    },
    "loss": "MSELoss",
    "proposal": {
        "method": "Truncated",
        "n_steps": 8,
        "params": {"quantile": 0.5},
    },
    "logger": {
        "name": "Benchmark",
        "level": "WARNING",
        "log_to_console": False,
        "log_to_file": False,
    },
    "run_sequential": {
        "num_rounds": 3,
        "sims_per_round": 100,
        "sims_per_sample": 1,
        "clean_sampling": False,
        "upd_x": False,
        "training": {
            "epochs": 8,
            "show_every": 2,
            "batch_size": 100,
            "shuffle": True,
            "num_workers": 0,
        },
    },
}

# Keep smoke-test inference budgets at or below 100 simulations.
TASKS = ["Gaussian"]#, "SIR", "StochVol"]
METHODS = ["SFMPE", "SNPE_C", "SMC_ABC"]
BUDGETS = [100, 100, 100]
# SBC/TARP are evaluated across independent synthetic targets; a single target
# cannot define the reference range required by TARP.
REPLICATES = 5
ROUNDS = 2
POSTERIOR_SAMPLES = 64
SEED = 2026

TRAINING = {
    "epochs": 8,
    "stop_after_epochs": 3,
    "batch_size": 32,
    "learning_rate": 0.0005,
    "validation_fraction": 0.2,
}
SMC_ABC = {
    "particles": 20,
    "initial_population": 20,
    "epsilon_decay": 0.5,
}
DIAGNOSTICS = {
    "ppc_samples": 16,
    "sensitivity_enabled": False,
    "sensitivity_scale": 0.1,
    "sensitivity_samples": 16,
    "sbc_c2st": False,
}


class _Tee:
    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def validate_settings() -> None:
    if not TASKS or not METHODS or not BUDGETS:
        raise ValueError("TASKS, METHODS, and BUDGETS must not be empty.")
    unknown_tasks = set(TASKS) - set(TASK_CONFIGS)
    if unknown_tasks:
        raise ValueError(f"Missing task configurations: {sorted(unknown_tasks)}")
    # if any(budget < 1 or budget > 100 for budget in BUDGETS):
    #     raise ValueError("Smoke-test budgets must be between 1 and 100.")
    if "SMC_ABC" in METHODS and any(budget < 20 for budget in BUDGETS):
        raise ValueError("SMC_ABC requires a simulation budget of at least 20.")
    # if REPLICATES < 10:
    #     raise ValueError(
    #         "TARP calibration needs at least 10 independent target replicates; "
    #         "increase REPLICATES."
    #     )


def save_report_image(report_path: Path, image_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report_lines = report_path.read_text(encoding="utf-8").splitlines()
    height = max(5, min(0.24 * len(report_lines) + 1, 80))
    figure = plt.figure(figsize=(16, height))
    figure.text(
        0.015,
        0.99,
        "\n".join(report_lines),
        va="top",
        ha="left",
        family="monospace",
        fontsize=9,
    )
    figure.savefig(image_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    import torch

    install_dependencies()
    validate_settings()

    from sfmpe.benchmark.report import export_benchmark_results
    from sfmpe.benchmark.runner import run_benchmark

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is unavailable; running on CPU.")

    working_dir = Path("/kaggle/working")
    if not working_dir.is_dir():
        working_dir = Path.cwd()
    output_dir = working_dir / "output"
    config_dir = output_dir / "configs"
    logs_dir = output_dir / "logs"
    config_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    original_stdout, original_stderr = sys.stdout, sys.stderr
    with (logs_dir / "benchmark.log").open("w", encoding="utf-8") as log_stream:
        sys.stdout = _Tee(original_stdout, log_stream)  # type: ignore[assignment]
        sys.stderr = _Tee(original_stderr, log_stream)  # type: ignore[assignment]
        try:
            run_benchmark_and_export(
                device=device,
                output_dir=output_dir,
                config_dir=config_dir,
                logs_dir=logs_dir,
                run_benchmark=run_benchmark,
                export_benchmark_results=export_benchmark_results,
            )
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


def run_benchmark_and_export(
    *,
    device: str,
    output_dir: Path,
    config_dir: Path,
    logs_dir: Path,
    run_benchmark,
    export_benchmark_results,
) -> None:
    import yaml

    task_paths: dict[str, str] = {}
    for task_name in TASKS:
        task_config = {
            **TASK_CONFIGS[task_name],
            "logger": {
                **TASK_CONFIGS[task_name].get("logger", {}),
                "log_file_path": str(logs_dir / f"{task_name.lower()}.log"),
            },
        }
        task_path = config_dir / f"{task_name.lower()}.yaml"
        task_path.write_text(
            yaml.safe_dump(task_config, sort_keys=False),
            encoding="utf-8",
        )
        task_paths[task_name] = task_path.name

    sfmpe_path = config_dir / "sfmpe.yaml"
    sfmpe_path.write_text(
        yaml.safe_dump(SFMPE_CONFIG, sort_keys=False),
        encoding="utf-8",
    )
    benchmark_config = {
        "seed": SEED,
        "device": device,
        "tasks": task_paths,
        "methods": METHODS,
        "budgets": BUDGETS,
        "replicates": REPLICATES,
        "rounds": ROUNDS,
        "posterior_samples": POSTERIOR_SAMPLES,
        "output": str(output_dir),
        "sfmpe_config": sfmpe_path.name,
        "sfmpe_proposal_methods": {"SIR": "Truncated", "StochVol": "Truncated", "Gaussian": "Truncated"},
        "training": TRAINING,
        "smc_abc": SMC_ABC,
        "diagnostics": DIAGNOSTICS,
    }
    benchmark_path = config_dir / "benchmark.yaml"
    benchmark_path.write_text(
        yaml.safe_dump(benchmark_config, sort_keys=False),
        encoding="utf-8",
    )

    print(
        f"Starting benchmark: tasks={TASKS}, methods={METHODS}, "
        f"budgets={BUDGETS}, replicates={REPLICATES}, device={device}"
    )
    summaries = run_benchmark(benchmark_path, output_dir=output_dir)
    export_benchmark_results(output_dir)
    save_report_image(
        output_dir / "benchmark_report.md",
        output_dir / "benchmark_report.png",
    )
    print(f"Completed {len(summaries)} task/method/budget groups.")
    print(f"CSV and Markdown tables: {output_dir}")
    print(f"Summary image: {output_dir / 'benchmark_report.png'}")
    print(f"Metric plot images: {output_dir / 'figures'}")


if __name__ == "__main__":
    main()