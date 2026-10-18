import json
import ast
from pathlib import Path
from zipfile import ZipFile

import torch
import yaml

from sfmpe.benchmark.metrics import (
    exact_posterior_metrics,
    posterior_predictive_metrics,
)
from sfmpe.benchmark.report import export_benchmark_results
from sfmpe.benchmark.runner import (
    BenchmarkRunner,
    _SimulatorCallMeter,
    _balanced_round_budgets,
)
from sfmpe.tasks.Gaussian import GaussianTask
from scripts.package_kaggle_bundle import create_bundle


def test_round_budgets_sum_to_requested_total():
    assert _balanced_round_budgets(10, 3) == [4, 3, 3]
    assert _balanced_round_budgets(2, 4) == [1, 1]


def test_simulator_meter_counts_theta_vectors_not_batches():
    task = GaussianTask(
        {
            "prior": {"mean": [0.0, 0.0], "covariance": [[1.0, 0.0], [0.0, 1.0]]},
            "simulator": {
                "likelihood_covariance": [[1.0, 0.0], [0.0, 1.0]]
            },
        }
    )
    meter = _SimulatorCallMeter(task.simulator)
    meter.set_phase("inference")

    task.simulate(torch.zeros(2, 3, task.theta_dim))

    assert meter.counts["inference"] == 6


def test_exact_posterior_metrics_are_finite_for_matching_samples():
    torch.manual_seed(11)
    samples = torch.randn(64, 2)
    metrics = exact_posterior_metrics(samples, samples.clone())

    assert metrics["gaussian_wasserstein2"] == 0.0
    assert abs(metrics["mmd_rbf"]) < 1e-6
    assert metrics["c2st"] < 0.7


def test_posterior_predictive_metrics_only_simulate_supported_samples():
    class SupportedTask:
        def check_support(self, samples: torch.Tensor) -> torch.Tensor:
            return samples[:, 0] >= 0

        def summarize(self, data: torch.Tensor) -> torch.Tensor:
            return data

    samples = torch.tensor([[1.0], [-1.0], [2.0], [-2.0]])

    def simulate(theta: torch.Tensor) -> torch.Tensor:
        assert torch.all(theta >= 0)
        return theta

    metrics = posterior_predictive_metrics(
        SupportedTask(),
        torch.tensor([1.0]),
        samples,
        simulate,
    )

    assert metrics["ppc_valid_sample_fraction"] == 0.5


def test_posterior_predictive_metrics_report_when_all_samples_are_unsupported():
    class SupportedTask:
        def check_support(self, samples: torch.Tensor) -> torch.Tensor:
            return torch.zeros(samples.shape[0], dtype=torch.bool)

    def simulate(theta: torch.Tensor) -> torch.Tensor:
        raise AssertionError("No simulation should run for unsupported samples.")

    metrics = posterior_predictive_metrics(
        SupportedTask(),
        torch.tensor([0.0]),
        torch.ones(3, 1),
        simulate,
    )

    assert metrics["ppc_available"] == 0.0
    assert metrics["ppc_valid_sample_fraction"] == 0.0
    assert isinstance(metrics["ppc_unavailable_reason"], str)


def test_benchmark_report_exports_csv_markdown_and_figures(tmp_path: Path):
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    summaries = [
        {
            "task": "Gaussian",
            "method": "SFMPE",
            "budget": 40,
            "mean_gaussian_wasserstein2": 0.5,
            "std_gaussian_wasserstein2": 0.1,
            "mean_inference_seconds": 1.2,
            "mean_inference_simulator_draws": 40.0,
            "sbc_ecp": [0.2, 0.8],
        },
        {
            "task": "Gaussian",
            "method": "SNPE_C",
            "budget": 40,
            "mean_gaussian_wasserstein2": 0.7,
            "std_gaussian_wasserstein2": 0.2,
            "mean_inference_seconds": 1.5,
            "mean_inference_simulator_draws": 40.0,
            "sbc_ecp": [0.3, 0.7],
        },
    ]
    targets = [
        {
            "task": "Gaussian",
            "method": "SFMPE",
            "budget": 40,
            "replicate": 0,
            "metrics": {"mean_error": 0.25, "metric_curve": [0.1, 0.2]},
        }
    ]
    (result_dir / "benchmark_summary.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in summaries),
        encoding="utf-8",
    )
    (result_dir / "target_metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in targets),
        encoding="utf-8",
    )

    report_dir = export_benchmark_results(result_dir)

    assert (report_dir / "benchmark_summary.csv").is_file()
    assert (report_dir / "target_metrics.csv").is_file()
    report = (report_dir / "benchmark_report.md").read_text(encoding="utf-8")
    assert "gaussian_wasserstein2" in report
    assert "0.5 ± 0.1" in report
    figure_paths = list((report_dir / "figures").glob("*.png"))
    assert len(figure_paths) == 3


def test_kaggle_bundle_contains_source_and_smoke_config(tmp_path: Path):
    bundle_path = create_bundle(tmp_path / "sfmpe_kaggle_bundle.zip")

    with ZipFile(bundle_path) as bundle:
        names = set(bundle.namelist())
    assert "src/sfmpe/benchmark/runner.py" in names
    assert "configs/benchmark_kaggle_smoke.yaml" in names
    assert "requirements-kaggle.txt" in names
    assert not any("__pycache__" in name or ".egg-info" in name for name in names)

    code_dir = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (code_dir / "configs" / "benchmark_kaggle_smoke.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert max(config["budgets"]) <= 100


def test_benchmark_runner_allows_writable_output_override(tmp_path: Path):
    config_path = tmp_path / "benchmark.yaml"
    config_path.write_text(
        yaml.safe_dump({"device": "cpu", "output": "./readonly-default"}),
        encoding="utf-8",
    )
    output_dir = tmp_path / "kaggle-working"

    runner = BenchmarkRunner(config_path, output_dir=output_dir)

    assert runner.output_dir == output_dir.resolve()
    assert runner.output_dir.is_dir()


def test_smc_abc_uses_cpu_task_even_when_benchmark_device_is_cuda(
    tmp_path: Path,
    monkeypatch,
):
    code_dir = Path(__file__).resolve().parents[1]
    config_path = tmp_path / "benchmark.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "device": "cpu",
                "tasks": {"Gaussian": str(code_dir / "configs" / "gaussian.yaml")},
                "posterior_samples": 8,
                "diagnostics": {
                    "ppc_samples": 2,
                    "sensitivity_enabled": False,
                },
            }
        ),
        encoding="utf-8",
    )
    runner = BenchmarkRunner(config_path)
    runner.device = "cuda"
    observed_task_devices: list[torch.device] = []

    def fake_fit(task_name, method, task, observation, budget, meter):
        observed_task_devices.append(task.device)

        def sample_posterior(x, num_samples):
            return task.true_posterior(x).sample((num_samples,))

        return sample_posterior

    monkeypatch.setattr(runner, "_fit", fake_fit)
    observation = torch.tensor([0.2, -0.1])
    row, samples = runner._evaluate_target(
        task_name="Gaussian",
        method="SMC_ABC",
        budget=40,
        replicate=0,
        seed=1,
        observation=observation,
        observation_scale=torch.ones(2),
    )

    assert observed_task_devices == ["cpu"]
    assert samples.device.type == "cpu"
    assert row["inference_simulator_draws"] == 0


def test_kaggle_notebook_code_cells_are_valid_python():
    notebook_path = (
        Path(__file__).resolve().parents[1]
        / "notebooks"
        / "benchmark_kaggle.ipynb"
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))


def test_gaussian_benchmark_smoke_runs_all_methods_with_budget(tmp_path: Path):
    code_dir = Path(__file__).resolve().parents[1]
    config = {
        "seed": 12,
        "device": "cpu",
        "tasks": {"Gaussian": str(code_dir / "configs" / "gaussian.yaml")},
        "methods": ["SFMPE", "SNPE_C", "SMC_ABC"],
        "budgets": [40],
        "replicates": 4,
        "rounds": 2,
        "posterior_samples": 8,
        "output": str(tmp_path / "benchmark-results"),
        "sfmpe_config": str(code_dir / "configs" / "benchmark_sfmpe.yaml"),
        "training": {
            "epochs": 1,
            "stop_after_epochs": 1,
            "batch_size": 8,
            "validation_fraction": 0.2,
        },
        "smc_abc": {"particles": 20, "initial_population": 20},
        "diagnostics": {
            "ppc_samples": 2,
            "sensitivity_enabled": False,
            "sbc_c2st": False,
        },
    }
    config_path = tmp_path / "benchmark.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    summaries = BenchmarkRunner(config_path).run()

    assert {summary["method"] for summary in summaries} == {
        "SFMPE",
        "SNPE_C",
        "SMC_ABC",
    }
    assert all(summary["mean_inference_simulator_draws"] <= 40 for summary in summaries)
    assert (tmp_path / "benchmark-results" / "target_metrics.jsonl").is_file()
    assert (tmp_path / "benchmark-results" / "benchmark_summary.jsonl").is_file()
