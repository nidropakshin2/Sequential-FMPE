"""Run an SFMPE benchmark on Kaggle from a bundled YAML config.

The script is config-driven: all task, method, training, and Kaggle settings
live in a benchmark YAML file (see ``configs/benchmark_kaggle_smoke.yaml``).
Everything is bundled by the Kaggle submit wrapper:

    kagglex run --dir . --file notebooks/benchmark_kaggle_standalone.py \
        --include-data src pyproject.toml configs --gpu t4-2x

or, after phase 2, simply ``sfmpe-kaggle --config configs/benchmark_kaggle_smoke.yaml``.

Import bootstrap: the packaged sources are resolved explicitly — the script
searches for the bundled project root relative to its own location, inserts
``<root>/src`` at the front of ``sys.path`` before the first ``sfmpe`` import,
and does not rely on a pip editable install. Missing third-party dependencies
are installed via pip only when absent (pinned sbi, torchdiffeq, ...), with
torch constrained to the preinstalled build. A fast first diagnostic on
Kaggle is:

    python benchmark_kaggle_standalone.py --check-imports

Config selection on Kaggle (kagglex replaces the remote command via
``--command``, so ``--config`` can be appended there; the bare name is
resolved against the bundled ``configs/``):

    kagglex run --dir . --file notebooks/benchmark_kaggle_standalone.py \
        --include-data src pyproject.toml configs --gpu t4-2x \
        --command "python benchmark_kaggle_standalone.py --config configs/benchmark.yaml"

or equivalently without changing the command:

    kagglex run --dir . --file notebooks/benchmark_kaggle_standalone.py \
        --include-data src pyproject.toml configs --gpu t4-2x \
        --env SFMPE_BENCHMARK_CONFIG=benchmark.yaml

Without either, the bundled smoke profile (``configs/benchmark_kaggle_smoke.yaml``)
runs.

Run schedules and sharding
==========================

The full benchmark is split across Kaggle sessions via
``configs/benchmark_schedule.yaml`` (one shard per task x method; a shard may
also pin ``budgets``/``replicates``). Every shard writes results incrementally
and skips completed targets on re-launch, so a run that hits the wall-clock
limit can simply be submitted again. ``time_limit_hours`` in the config (or
``--time-limit-hours``) stops cleanly between targets.

    python benchmark_kaggle_standalone.py --list-shards
    python benchmark_kaggle_standalone.py --shard sir-snpe-c
    kagglex run ... --env SFMPE_BENCHMARK_SHARD=sir-snpe-c

A shard without an explicit config runs the schedule's ``base_config``
(``benchmark.yaml``), not the smoke profile. When passing several envs to
``kagglex``, put them after ONE ``--env``: ``--env "K1=V1 K2=V2"`` (its
``--env`` uses ``nargs="*"`` and repeated flags overwrite each other).

Rounds axis: ``rounds: [1, 2, 3]`` runs one independent experiment per
value, each splitting the SAME total simulation budget evenly across that
many rounds (1000 -> [1000], [500, 500], [334, 333, 333]). Metrics carry a
``rounds`` field, so models trained with 1/2/3 rounds are compared at a
fixed simulation budget; single-shot SMC-ABC rows carry ``rounds: null``.

Merging shard outputs (run locally after downloading each session):

    python -m sfmpe.benchmark.report --config configs/benchmark.yaml \
        --merge-inputs <shard-output-dir> ... \
        --merge-output results/benchmark-merged

Locally the same script works from the ``code`` directory (a Kaggle profile
declares ``device: cuda``, so use ``--device auto/cpu`` when no GPU is present):

    python notebooks/benchmark_kaggle_standalone.py --config configs/benchmark_kaggle_smoke.yaml --device auto
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TextIO

BENCHMARK_CONFIG_ENV = "SFMPE_BENCHMARK_CONFIG"
BENCHMARK_SHARD_ENV = "SFMPE_BENCHMARK_SHARD"
DEFAULT_SCHEDULE_NAME = "benchmark_schedule.yaml"


def find_project_root() -> Path:
    """Locate the bundled project root (contains pyproject.toml and src/).

    Kaggle's bootstrap extracts the data payload (``src/``, ``pyproject.toml``,
    ``configs/``) directly into ``/kaggle/working``, where this script runs;
    some layouts instead place the payload under ``pkg_src`` or next to the
    script. Locally the script lives in ``code/notebooks``, so the package
    root sits two levels up. Search is performed relative to both the current
    working directory and the script location.
    """
    callers = [Path.cwd(), Path(__file__).resolve().parent]
    checked: list[Path] = []
    for caller in callers:
        for root in (caller, caller.parent, caller / "pkg_src"):
            if not root.is_dir():
                continue
            if root in checked:
                continue
            checked.append(root)
            if (root / "pyproject.toml").is_file() and (
                root / "src" / "sfmpe"
            ).is_dir():
                return root
    listing = "\n".join(f"  - {path}" for path in checked)
    raise FileNotFoundError(
        "Could not find the bundled SFMPE project root (pyproject.toml + "
        "src/sfmpe). Submit with `--include-data src pyproject.toml configs`.\n"
        f"Paths checked:\n{listing}"
    )


def ensure_source_import_path(project_root: Path) -> bool:
    """Put the bundled ``src`` at the front of sys.path for `import sfmpe`.

    Kaggle does not add the payload's ``src`` directory to ``sys.path``, and
    we deliberately do not rely on a pip editable install resolving the
    package remotely. Inserting the bundled sources explicitly before the
    first ``sfmpe`` import is the authoritative resolution path. Returns
    True when this call performed the insertion.
    """
    src_dir = (project_root / "src").resolve()
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
        print(f"Inserted bundled source into sys.path: {src_dir}")
        return True
    return False


def find_config_path(
    explicit: str | None,
    project_root: Path,
    env_value: str | None = None,
) -> Path:
    """Resolve the benchmark YAML to execute, and assert it exists.

    Priority: ``--config``, then the ``SFMPE_BENCHMARK_CONFIG`` environment
    variable (injectable remotely via
    ``kagglex run --env SFMPE_BENCHMARK_CONFIG=<value>``), then the bundled
    smoke config. A bare file name is resolved against ``configs/`` inside
    the bundle (absolute and path-with-directories values are used as given).
    """
    requested = explicit or env_value
    if requested is not None:
        path = Path(requested)
        if path.is_absolute():
            candidates = [path]
        else:
            # Relative values try the current directory first, then the
            # bundled project root; a bare file name is additionally tried
            # against configs/ inside the bundle.
            candidates = [path, project_root / path]
            if path.parent == Path("."):
                candidates.append(project_root / "configs" / path)
        if not path.suffix:
            candidates = [candidate.with_suffix(".yaml") for candidate in candidates]
        for candidate in candidates:
            resolved = candidate.expanduser().resolve()
            if resolved.is_file():
                return resolved
        listing = "\n".join(str(candidate.resolve()) for candidate in candidates)
        raise FileNotFoundError(
            f"Benchmark config not found for {requested!r}. Checked:\n{listing}"
        )
    config_candidates = [
        project_root / "configs" / "benchmark_kaggle_smoke.yaml",
        Path.cwd() / "configs" / "benchmark_kaggle_smoke.yaml",
    ]
    for candidate in config_candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "No benchmark config found; pass --config <path to benchmark YAML> "
        f"or set {BENCHMARK_CONFIG_ENV}."
    )


def find_schedule_path(
    explicit: str | None,
    project_root: Path,
) -> Path:
    """Resolve the run-schedule YAML (``--schedule`` or the bundled default)."""
    if explicit is not None:
        path = Path(explicit)
        if path.is_absolute():
            candidates = [path]
        else:
            candidates = [path, project_root / path]
            if path.parent == Path("."):
                candidates.append(project_root / "configs" / path)
        for candidate in candidates:
            resolved = candidate.expanduser().resolve()
            if resolved.is_file():
                return resolved
        listing = "\n".join(str(candidate.resolve()) for candidate in candidates)
        raise FileNotFoundError(
            f"Benchmark schedule not found for {explicit!r}. Checked:\n{listing}"
        )
    for candidate in (
        project_root / "configs" / DEFAULT_SCHEDULE_NAME,
        Path.cwd() / "configs" / DEFAULT_SCHEDULE_NAME,
    ):
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"No run schedule found; pass --schedule <path> or create "
        f"configs/{DEFAULT_SCHEDULE_NAME}."
    )


def load_schedule(path: Path) -> dict:
    """Load a schedule YAML with a top-level ``shards`` list."""
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("shards"), list):
        raise ValueError(
            f"Schedule {path} must define a top-level 'shards' list."
        )
    return data


def shard_overrides(schedule: dict, name: str) -> dict:
    """Config overlay for one schedule shard (everything except `name`)."""
    for entry in schedule["shards"]:
        if str(entry.get("name")) == name:
            return {key: value for key, value in entry.items() if key != "name"}
    available = ", ".join(str(entry.get("name")) for entry in schedule["shards"])
    raise KeyError(f"Unknown shard {name!r}; available shards: {available}")


def print_schedule(schedule: dict, schedule_path: Path) -> None:
    """Print shard names and ready-to-copy kagglex commands."""
    base_config = schedule.get("base_config", "benchmark.yaml")
    print(f"Schedule: {schedule_path}")
    print(f"Base config: configs/{base_config}")
    print("")
    for entry in schedule["shards"]:
        name = entry.get("name")
        summary = ", ".join(
            f"{key}={value}"
            for key, value in entry.items()
            if key != "name"
        )
        print(f"- {name} ({summary})")
        print(
            "    kagglex run --dir . --file notebooks/benchmark_kaggle_standalone.py "
            "--include-data src pyproject.toml configs --gpu t4-2x "
            f'--env "{BENCHMARK_CONFIG_ENV}={base_config} '
            f'{BENCHMARK_SHARD_ENV}={name}"'
        )
    print("")
    print(
        "Merge the downloaded shard outputs with:\n"
        "    python -m sfmpe.benchmark.report --config configs/"
        f"{base_config} --merge-inputs <shard-dir> ... --merge-output "
        "results/benchmark-merged"
    )


def resolve_benchmark_config(
    explicit: str | None,
    env_value: str | None,
    schedule: dict | None,
    project_root: Path,
) -> Path:
    """Pick the benchmark YAML: --config > env > schedule base_config > smoke.

    A shard selected without an explicit config runs the schedule's
    ``base_config`` (e.g. benchmark.yaml) instead of the smoke profile, so
    setting only ``SFMPE_BENCHMARK_SHARD`` does not silently shrink the run.
    """
    if explicit is None and env_value is None and schedule is not None:
        base = str(schedule.get("base_config", "benchmark.yaml"))
        print(f"Shard run uses the schedule base config: {base}")
        return find_config_path(base, project_root)
    return find_config_path(explicit, project_root, env_value=env_value)


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


_THIRD_PARTY_REQUIREMENTS: dict[str, str] = {
    # import-name -> pip requirement, checked with importlib before any pip call
    "yaml": "PyYAML",
    "torchdiffeq": "torchdiffeq",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "sbi": "sbi==0.21.0",
}


def missing_dependencies() -> list[str]:
    """Pip requirement specs for third-party deps absent from this environment.

    ``torch`` is deliberately absent here: the Kaggle image ships PyTorch and
    the benchmark must never let pip replace that CUDA build.
    """
    import importlib.util

    return [
        requirement
        for module, requirement in _THIRD_PARTY_REQUIREMENTS.items()
        if importlib.util.find_spec(module) is None
    ]


def install_dependencies(missing: list[str]) -> None:
    """Install only the missing third-party requirements.

    A constraint file pins ``torch`` to the already installed version so no
    dependency can pull a different (e.g. CPU-only) build.
    """
    if not missing:
        print("All third-party dependencies are present; skipping pip install.")
        return

    import torch

    torch_version = torch.__version__
    constraint = Path(tempfile.gettempdir()) / "sfmpe-torch-constraint.txt"
    constraint.write_text(f"torch=={torch_version}\n", encoding="utf-8")

    print(
        f"Installing missing dependencies ({', '.join(missing)}) "
        f"with torch=={torch_version} pinned..."
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
            *missing,
        ]
    )

    still_missing = missing_dependencies()
    if still_missing:
        raise RuntimeError(
            f"pip install finished but {still_missing} remained unimportable."
        )
    print(f"Dependencies installed; preserving PyTorch {torch_version}.")


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a bundled SFMPE benchmark from a YAML config."
    )
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Path to the benchmark YAML. Defaults to "
            "configs/benchmark_kaggle_smoke.yaml inside the bundle."
        ),
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["auto", "cpu", "cuda"],
        help=(
            "Optional device override for the top-level `device` entry in "
            "the config. Useful locally to run a Kaggle (cuda) profile in "
            "auto/cpu mode."
        ),
    )
    parser.add_argument(
        "--check-imports",
        action="store_true",
        help=(
            "Verify bundled-source resolution and dependency imports, print "
            "provenance paths, and exit without running the benchmark. Use "
            "it as a fast first diagnostic on Kaggle."
        ),
    )
    parser.add_argument(
        "--schedule",
        default=None,
        help=(
            "Path to a run schedule YAML (default: configs/"
            "benchmark_schedule.yaml inside the bundle)."
        ),
    )
    parser.add_argument(
        "--shard",
        default=None,
        help=(
            "Run only the named shard from the schedule (also settable via "
            f"the {BENCHMARK_SHARD_ENV} environment variable)."
        ),
    )
    parser.add_argument(
        "--list-shards",
        action="store_true",
        help="Print schedule shards and ready-to-copy kagglex commands, then exit.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Override the task list (e.g. --tasks Gaussian SIR).",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Override the method list (e.g. --methods SFMPE SNPE_C).",
    )
    parser.add_argument(
        "--budgets",
        nargs="+",
        type=int,
        default=None,
        help="Override the inference-simulation budgets.",
    )
    parser.add_argument(
        "--replicates",
        type=int,
        default=None,
        help="Override the number of target replicates.",
    )
    parser.add_argument(
        "--rounds",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Override the rounds axis, e.g. --rounds 1 2 3: one experiment "
            "per value, each splitting the total budget evenly."
        ),
    )
    parser.add_argument(
        "--time-limit-hours",
        type=float,
        default=None,
        help=(
            "Stop cleanly after this many hours; results are written "
            "incrementally and a later re-run resumes completed targets."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    project_root = find_project_root()
    # Insert the bundled sources BEFORE any sfmpe import: on Kaggle the
    # payload's src/ is not on sys.path, and we do not rely on a pip editable
    # install to resolve the package remotely.
    ensure_source_import_path(project_root)

    try:
        import torch
    except ImportError as error:  # torch ships with Kaggle images
        raise RuntimeError(
            "torch is unavailable in this environment; run the benchmark on "
            "a Kaggle image with PyTorch preinstalled or install it locally."
        ) from error

    missing = missing_dependencies()
    if missing:
        install_dependencies(missing)
    else:
        print("All third-party dependencies are present; skipping pip install.")

    try:
        import sfmpe
    except ImportError as error:
        src_dir = (project_root / "src").resolve()
        raise RuntimeError(
            f"import sfmpe failed even with bundled sources on sys.path "
            f"({src_dir}). Bundled project root: {project_root}"
        ) from error
    print(f"sfmpe: {Path(sfmpe.__file__).resolve()}")

    if args.check_imports:
        print(f"torch {torch.__version__}; cuda={torch.cuda.is_available()}")
        print("Import check passed.")
        return

    from sfmpe.benchmark.report import export_benchmark_results
    from sfmpe.benchmark.runner import run_benchmark

    shard_name = args.shard or os.environ.get(BENCHMARK_SHARD_ENV)
    schedule = None
    if args.list_shards or shard_name or args.schedule:
        schedule_path = find_schedule_path(args.schedule, project_root)
        schedule = load_schedule(schedule_path)
    if args.list_shards:
        print_schedule(schedule, schedule_path)
        return

    env_config = os.environ.get(BENCHMARK_CONFIG_ENV)
    if env_config and args.config is None:
        print(f"Using benchmark config from {BENCHMARK_CONFIG_ENV}: {env_config}")
    config_path = resolve_benchmark_config(
        args.config, env_config, schedule, project_root
    )

    overrides: dict = {}
    if shard_name:
        overrides.update(shard_overrides(schedule, shard_name))
        print(f"Using shard {shard_name!r} from {schedule_path}")
    if args.tasks:
        overrides["tasks"] = list(args.tasks)
    if args.methods:
        overrides["methods"] = list(args.methods)
    if args.budgets:
        overrides["budgets"] = list(args.budgets)
    if args.replicates is not None:
        overrides["replicates"] = args.replicates
    if args.rounds:
        overrides["rounds"] = list(args.rounds)
    if args.time_limit_hours is not None:
        overrides["time_limit_hours"] = args.time_limit_hours

    if Path("/kaggle/working").is_dir():
        # Only /kaggle/working persists after the session, so on Kaggle the
        # results directory is forced there instead of taken from the config.
        output_dir: Path | None = Path("/kaggle/working/output")
    else:
        # Locally `None` lets the benchmark YAML decide via its `output:`
        # entry, resolved against the config directory by the runner.
        output_dir = None

    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is unavailable; running on CPU.")

    results_dir = results_dir_for(config_path, output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    original_stdout, original_stderr = sys.stdout, sys.stderr
    log_path = results_dir / "benchmark.log"
    with log_path.open("w", encoding="utf-8") as log_stream:
        sys.stdout = _Tee(original_stdout, log_stream)  # type: ignore[assignment]
        sys.stderr = _Tee(original_stderr, log_stream)  # type: ignore[assignment]
        try:
            run_benchmark_and_export(
                config_path=config_path,
                output_dir=output_dir,
                device=args.device,
                run_benchmark=run_benchmark,
                export_benchmark_results=export_benchmark_results,
                overrides=overrides,
            )
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
    print(f"Log saved to {log_path}")


def results_dir_for(config_path: Path, output_dir: Path | None) -> Path:
    """Mirror the runner's output resolution so exports find the JSONL files."""
    if output_dir is not None:
        return Path(output_dir).resolve()
    import yaml

    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    return (
        Path(config_path).parent / config.get("output", "./results/benchmark")
    ).resolve()


def run_benchmark_and_export(
    *,
    config_path: Path,
    output_dir: Path | None,
    device: str | None,
    run_benchmark,
    export_benchmark_results,
    overrides: dict | None = None,
) -> None:
    print(f"Benchmark config: {config_path} (device override: {device})")
    if overrides:
        print(f"Config overrides: {overrides}")
    summaries = run_benchmark(
        config_path,
        output_dir=output_dir,
        device=device,
        overrides=overrides,
    )
    results_dir = results_dir_for(config_path, output_dir)
    export_benchmark_results(results_dir)
    report_md = results_dir / "benchmark_report.md"
    if report_md.is_file():
        save_report_image(report_md, results_dir / "benchmark_report.png")
        print(f"Summary image: {results_dir / 'benchmark_report.png'}")
    print(f"Completed {len(summaries)} task/method/budget groups.")
    print(f"CSV and Markdown tables: {results_dir}")


if __name__ == "__main__":
    main()
