import json
import ast
import importlib.util
import shutil
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

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


def test_benchmark_configs_wire_pertask_sfmpe_and_devices():
    code_dir = Path(__file__).resolve().parents[1]
    configs_dir = code_dir / "configs"

    for name in ("benchmark.yaml", "benchmark_kaggle_smoke.yaml"):
        config = yaml.safe_load(
            (configs_dir / name).read_text(encoding="utf-8")
        )

        task_names = set(config["tasks"])
        assert task_names == set(config["sfmpe_configs"]), (
            f"{name}: sfmpe_configs must cover exactly the configured tasks"
        )
        for task_name, relative_path in config["sfmpe_configs"].items():
            sfmpe_path = configs_dir / relative_path
            assert sfmpe_path.is_file(), f"{name}: missing {sfmpe_path}"
            sfmpe_config = yaml.safe_load(sfmpe_path.read_text(encoding="utf-8"))
            method = sfmpe_config["proposal"]["method"]
            # Truncated is the only canonical benchmark proposal method
            # (NPE-A/NPE-B are not valid benchmark methods).
            assert method == "Truncated", (
                f"{name}: {task_name} must use the Truncated proposal, "
                f"got {method!r}"
            )
            assert {
                "init_dist",
                "velocity_model",
                "path",
            } <= set(sfmpe_config["flow"])
            # The benchmark runner consumes run_sequential.clean_sampling
            # from the per-task SFMPE config; keep the key declared.
            assert "clean_sampling" in sfmpe_config["run_sequential"]

        devices = config["devices"]
        assert devices["smc_abc"] in {"cpu"}
        # sbi(>=0.21) requires the SNPE-C prior on the training device, so
        # 'auto' (inherit nn device) is the canonical value there.
        assert devices["sbi_prior"] in {"auto"}

        # Per-method training blocks must expose the complete option sets:
        # every key the runner forwards to fm_estimator.train / sbi SNPE-C
        # train() / sbi SMCABC.__call__ (minus the deliberately unexposed
        # budget accounting knobs and Python callables).
        assert set(config["sfmpe_training"]) == {
            "epochs",
            "stop_after_epochs",
            "batch_size",
            "shuffle",
            "num_workers",
            "show_every",
            "cfg_mode",
            "path",
            "validation_fraction",
            "min_delta",
            "validation_seed",
        }
        assert set(config["snpe_c_training"]) == {
            "max_num_epochs",
            "stop_after_epochs",
            "training_batch_size",
            "learning_rate",
            "validation_fraction",
            "num_atoms",
            "clip_max_norm",
            "resume_training",
            "force_first_round_loss",
            "discard_prior_samples",
            "use_combined_loss",
            "retrain_from_scratch",
            "show_train_summary",
            "exclude_invalid_x",
            "dataloader_kwargs",
            "calibration_kernel",
        }
        assert set(config["smc_abc"]) == {
            "particles",
            "initial_population",
            "num_workers",
            "epsilon_decay",
            "distance_based_decay",
            "ess_min",
            "kernel_variance_scale",
            "use_last_pop_samples",
            "kde",
            "kde_kwargs",
            "kde_sample_weights",
            "lra",
            "lra_with_weights",
            "sass",
            "sass_fraction",
            "sass_expansion_degree",
        }

    smoke = yaml.safe_load(
        (configs_dir / "benchmark_kaggle_smoke.yaml").read_text(encoding="utf-8")
    )
    assert max(smoke["budgets"]) <= 100
    assert smoke["replicates"] >= 2  # TARP needs theta variation across targets
    assert smoke["kaggle"]["max_hours"] <= 12
    kaggle_config = yaml.safe_load(
        (configs_dir / "benchmark.yaml").read_text(encoding="utf-8")
    )
    assert kaggle_config["kaggle"]["max_hours"] <= 12


def test_runner_device_policy(tmp_path: Path):
    def make_config(devices_block):
        config_path = tmp_path / "benchmark.yaml"
        config_path.write_text(
            yaml.safe_dump({"device": "auto", **devices_block}, sort_keys=False),
            encoding="utf-8",
        )
        return BenchmarkRunner(config_path)

    # Explicit nn override wins over the resolved top-level device.
    runner = make_config({"devices": {"nn": "cpu"}})
    assert runner._nn_device() == "cpu"
    assert runner._task_device("SFMPE") == "cpu"
    assert runner._task_device("SFMPE") != "cuda"

    # Defaults: nn and the sbi prior inherit the benchmark device; SMC-ABC
    # stays on CPU by policy.
    runner = make_config({})
    assert runner._nn_device() == runner.device
    assert runner._task_device("SMC_ABC") == "cpu"
    assert runner._task_device("SFMPE") == runner.device
    assert runner._sbi_prior_device() == runner.device
    assert runner._sbi_prior_device() == runner._nn_device()

    # sbi_prior may be forced to cpu for CPU-only runs; cuda requires an
    # available accelerator like any other device entry.
    runner = make_config({"devices": {"sbi_prior": "cpu"}})
    assert runner._sbi_prior_device() == "cpu"

    runner = make_config({"devices": {"nn": "cpu", "sbi_prior": "cuda"}})
    if torch.cuda.is_available():
        assert runner._sbi_prior_device() == "cuda"
    else:
        try:
            runner._sbi_prior_device()
        except RuntimeError as error:
            assert "CUDA" in str(error)
        else:
            raise AssertionError("sbi_prior=cuda must fail without CUDA")

    # CUDA requested at policy level is validated like a top-level device.
    runner = make_config({"devices": {"nn": "cuda"}})
    if torch.cuda.is_available():
        assert runner._nn_device() == "cuda"
    else:
        try:
            runner._nn_device()
        except RuntimeError as error:
            assert "CUDA" in str(error)
        else:
            raise AssertionError("nn=cuda must fail when CUDA is unavailable")

    # smc_abc must never be CUDA: sbi SMCABC computes kernel covariance via
    # numpy internally.
    runner = make_config({"devices": {"smc_abc": "cuda"}})
    try:
        runner._task_device("SMC_ABC")
    except ValueError as error:
        assert "smc_abc" in str(error)
    else:
        raise AssertionError("smc_abc=cuda must be rejected")


def test_runner_resolves_pertask_sfmpe_config_paths(tmp_path: Path):
    shared = tmp_path / "shared_sfmpe.yaml"
    shared.write_text("proposal:\n  method: Truncated\n", encoding="utf-8")
    stochvol = tmp_path / "stochvol_sfmpe.yaml"
    stochvol.write_text("proposal:\n  method: Truncated\n", encoding="utf-8")

    config_path = tmp_path / "benchmark.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "sfmpe_configs": {"StochVol": "stochvol_sfmpe.yaml"},
            }
        ),
        encoding="utf-8",
    )
    runner = BenchmarkRunner(config_path)

    assert runner._sfmpe_config_path("StochVol") == stochvol.resolve()
    # Tasks absent from the mapping fall back to the shared default.
    assert (
        runner._sfmpe_config_path("SIR") == (tmp_path / "benchmark_sfmpe.yaml").resolve()
    )

    no_mapping_path = tmp_path / "no_mapping.yaml"
    no_mapping_path.write_text(yaml.safe_dump({}), encoding="utf-8")
    runner = BenchmarkRunner(no_mapping_path)
    assert (
        runner._sfmpe_config_path("Gaussian")
        == (tmp_path / "benchmark_sfmpe.yaml").resolve()
    )


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

    def fake_fit(task_name, method, task, observation, budget, meter, on_round=None):
        observed_task_devices.append(task.device)

        def sample_posterior(x, num_samples):
            return task.true_posterior(x).sample((num_samples,))

        return sample_posterior

    monkeypatch.setattr(runner, "_fit", fake_fit)
    observation = torch.tensor([0.2, -0.1])
    scored = runner._evaluate_target(
        task_name="Gaussian",
        method="SMC_ABC",
        budget=40,
        replicate=0,
        seed=1,
        observation=observation,
        observation_scale=torch.ones(2),
    )
    row, samples = scored[-1]

    assert observed_task_devices == ["cpu"]
    assert samples.device.type == "cpu"
    assert row["inference_simulator_draws"] == 0
    assert row["round"] is None


def test_sfmpe_device_override_matches_benchmark_device(tmp_path: Path):
    from sfmpe.inference.SFMPE import SFMPE

    task = GaussianTask(
        {
            "prior": {"mean": [0.0, 0.0], "covariance": [[1.0, 0.0], [0.0, 1.0]]},
            "simulator": {"likelihood_covariance": [[1.0, 0.0], [0.0, 1.0]]},
        },
        device="cpu",
    )
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "device": "cuda",  # ignored: the override below wins
                "flow": {
                    "init_dist": {"type": "normal"},
                    "velocity_model": {
                        "type": "simple",
                        "params": {"hidden_dim": 8, "num_layers": 1},
                    },
                    "path": {"type": "affine"},
                },
                "proposal": {"method": "Truncated", "n_steps": 2},
                "run_sequential": {
                    "num_rounds": 1,
                    "sims_per_round": 2,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    pipeline = SFMPE(task, config_path, device="cpu")

    assert pipeline.device == "cpu"


def test_snpe_c_prior_on_nn_device_with_cpu_sim_outputs(monkeypatch):
    """sbi(>=0.21) asserts the prior lives on the training device
    (check_if_prior_on_device), so the runner must build the SNPE-C task on
    the nn device while the simulator wrapper still hands sbi CPU outputs
    (sbi's validate_theta_and_x moves them onto the training device).
    """
    observed_task_devices: list[str] = []
    observed_sim_devices: list[torch.device] = []
    observed_training_devices: list[str] = []

    class FakePosterior:
        def set_default_x(self, observation):
            return self

    class FakeInference:
        def __init__(self, prior, device, **kwargs):
            observed_training_devices.append(device)
            assert device == "cuda"

        def append_simulations(self, theta, x, proposal=None, **kwargs):
            # sbi itself moves data onto the training device; the runner is
            # only responsible for returning CPU sim outputs.
            assert x.device.type == "cpu"
            assert kwargs.get("exclude_invalid_x") is None
            return self

        def train(self, **kwargs):
            return object()

        def build_posterior(self, density_estimator, **kwargs):
            return FakePosterior()

    def simulate_for_sbi(simulator, proposal, **kwargs):
        theta = torch.zeros(2, 2)
        x = simulator(theta)
        assert x.device.type == "cpu"
        return theta, x

    inference_module = types.ModuleType("sbi.inference")
    inference_module.SNPE_C = FakeInference
    inference_module.simulate_for_sbi = simulate_for_sbi
    monkeypatch.setitem(sys.modules, "sbi.inference", inference_module)

    prior = torch.distributions.MultivariateNormal(
        torch.zeros(2),
        covariance_matrix=torch.eye(2),
    )

    def make_task(task_name, device):
        observed_task_devices.append(device)
        return SimpleNamespace(prior=prior)

    task = SimpleNamespace(device="cpu", theta_dim=2)
    # NB: this host's torch build has no CUDA runtime, so the simulator
    # wrapper's theta.to(task.device) stays on cpu here; on a CUDA host it
    # moves tensors to the task's (cuda) device. The policy assertion is
    # `observed_task_devices == ["cuda"]` above.
    runner = _FitRunner(
        config={"training": {}, "rounds": 1},
        device="cuda",
        _make_task=make_task,
        _make_simulator_callable=lambda _task: lambda theta: (
            observed_sim_devices.append(theta.device) or theta
        ),
        _nn_device=lambda: "cuda",
        _sbi_prior_device=lambda: "cuda",
    )
    meter = SimpleNamespace(set_phase=lambda _phase: None)

    BenchmarkRunner._fit_snpe_c(
        runner,
        "Gaussian",
        task,
        observation=torch.zeros(2),
        budget=4,
        meter=meter,
    )

    # The SNPE-C task (and its prior) must be built on the training device.
    assert observed_task_devices == ["cuda"]
    assert observed_training_devices == ["cuda"]
    # On this host (no CUDA runtime) the wrapper keeps theta on cpu.
    assert observed_sim_devices == [torch.device("cpu")]


class _FitRunner(SimpleNamespace):
    """Stand-in for unbound ``BenchmarkRunner._fit_*`` calls.

    `_training_block` is inherited as a class attribute so it binds to the
    instance like on a real runner; behavioral knobs stay instance attrs.
    """

    _training_block = BenchmarkRunner._training_block
    _sfmpe_sampler = staticmethod(BenchmarkRunner._sfmpe_sampler)
    _snpe_c_sampler = staticmethod(BenchmarkRunner._snpe_c_sampler)


def test_sfmpe_training_block_full_option_wiring(tmp_path: Path, monkeypatch):
    """Every sfmpe_training key reaches fm_estimator.train via
    manager.train_estimator (batch_size clamped per round; the checkpoint
    path kwarg is only passed when configured).
    """
    import sfmpe.benchmark.runner as runner_module

    observed_calls: list[dict] = []

    class FakeProposalParams:
        method = "Truncated"
        method_params = {"quantile": 0.5}

    class FakeManager:
        def run_round_experimental(self, round_id, samples, sims_per_sample, clean_sampling):
            assert clean_sampling is True
            return 0

        def train_estimator(self, rounds, **kwargs):
            observed_calls.append({"rounds": list(rounds), **kwargs})

        def build_posterior(self, update_x=True):
            return object()

        def update_proposal(self, posterior):
            fake_manager.proposal = SimpleNamespace(
                params=SimpleNamespace(x_0=None),
                sample=lambda shape, device=None: torch.zeros(2),
            )

    fake_manager = FakeManager()
    fake_pipeline = SimpleNamespace(
        config={"run_sequential": {"clean_sampling": True}},
        proposal_params=FakeProposalParams(),
        manager=fake_manager,
    )

    observed_pipeline_devices: list = []

    def fake_sfmpe(task, config_path, device=None):
        observed_pipeline_devices.append(device)
        return fake_pipeline

    monkeypatch.setattr(runner_module, "SFMPE", fake_sfmpe)

    checkpoint = str(tmp_path / "ckpt.pt")
    task = SimpleNamespace(device="cpu")
    runner = _FitRunner(
        config={
            "rounds": 3,
            "sfmpe_training": {
                "epochs": 7,
                "stop_after_epochs": 2,
                "batch_size": 8,
                "shuffle": False,
                "num_workers": 2,
                "show_every": 2,
                "cfg_mode": True,
                "path": checkpoint,
                "validation_fraction": 0.25,
                "min_delta": 0.01,
                "validation_seed": 5,
            },
        },
        _sfmpe_config_path=lambda _name: "pipeline.yaml",
        _nn_device=lambda: "cpu",
    )
    meter = SimpleNamespace(counts={"setup": 2}, set_phase=lambda _p: None)

    sample_posterior, method = BenchmarkRunner._fit_sfmpe(
        runner, "Gaussian", task, torch.zeros(2), budget=10, meter=meter
    )

    assert method == "Truncated"
    assert observed_pipeline_devices == ["cpu"]
    # remaining budget = 10 - 2 setup sims, split over 3 rounds: [3, 3, 2]
    # remaining budget = 10 - 2 setup sims, split over 3 rounds: [3, 3, 2];
    # every round trains cumulatively on all stored rounds so far
    assert [call["rounds"] for call in observed_calls] == [
        [0],
        [0, 1],
        [0, 1, 2],
    ]
    assert [call["batch_size"] for call in observed_calls] == [3, 3, 2]
    expected_common = {
        "epochs": 7,
        "shuffle": False,
        "num_workers": 2,
        "show_every": 2,
        "cfg_mode": True,
        "path": checkpoint,
        "validation_fraction": 0.25,
        "stop_after_epochs": 2,
        "min_delta": 0.01,
        "validation_seed": 5,
        "device": "cpu",
    }
    for call in observed_calls:
        call_without_batch = {k: v for k, v in call.items() if k not in {"batch_size", "rounds"}}
        assert call_without_batch == expected_common
    sample = sample_posterior(torch.zeros(2), 4)
    assert sample.shape == (2,)


def test_sfmpe_training_defaults_and_optional_checkpoint(monkeypatch):
    """Unconfigured keys fall back to the runner defaults; `path` stays
    absent from the forwarded kwargs unless configured."""
    import sfmpe.benchmark.runner as runner_module

    observed_calls: list[dict] = []

    class FakeManager:
        def run_round_experimental(self, round_id, samples, sims_per_sample, clean_sampling):
            return 0

        def train_estimator(self, rounds, **kwargs):
            observed_calls.append(kwargs)

        def build_posterior(self, update_x=True):
            return object()

        def update_proposal(self, posterior):
            return None

    fake_pipeline = SimpleNamespace(
        config={},  # clean_sampling defaults to True in _fit_sfmpe
        proposal_params=SimpleNamespace(
            method="Truncated",
            method_params={"quantile": 0.5},
            x_0=None,
        ),
        manager=FakeManager(),
    )

    def fake_sfmpe(task, config_path, device=None):
        return fake_pipeline

    monkeypatch.setattr(runner_module, "SFMPE", fake_sfmpe)
    task = SimpleNamespace(device="cpu")
    runner = _FitRunner(
        config={"rounds": 1, "sfmpe_training": {}},
        _sfmpe_config_path=lambda _name: "pipeline.yaml",
        _nn_device=lambda: "cpu",
    )
    meter = SimpleNamespace(counts={"setup": 0}, set_phase=lambda _p: None)

    BenchmarkRunner._fit_sfmpe(
        runner, "Gaussian", task, torch.zeros(2), budget=6, meter=meter
    )

    assert len(observed_calls) == 1
    assert observed_calls[0] == {
        "epochs": 20,
        "batch_size": 6,
        "shuffle": True,
        "num_workers": 0,
        "show_every": None,
        "cfg_mode": False,
        "validation_fraction": 0.1,
        "stop_after_epochs": 20,
        "min_delta": 0.0,
        "device": "cpu",
    }


def _install_fake_sbi(monkeypatch, fake_sbi_cls):
    inference_module = types.ModuleType("sbi.inference")
    inference_module.SNPE_C = fake_sbi_cls
    inference_module.simulate_for_sbi = lambda simulator, proposal, **kwargs: (
        torch.zeros(2, 2),
        simulator(torch.zeros(2, 2)),
    )
    monkeypatch.setitem(sys.modules, "sbi.inference", inference_module)
    return inference_module


def test_sfmpe_clean_truncated_without_quantile_fails(monkeypatch):
    """Clean-sampled Truncated without proposal.params.quantile must fail
    fast in the runner (same validation SFMPE.run performs)."""
    import sfmpe.benchmark.runner as runner_module

    fake_pipeline = SimpleNamespace(
        config={"run_sequential": {"clean_sampling": True}},
        proposal_params=SimpleNamespace(
            method="Truncated",
            method_params={},
            x_0=None,
        ),
        manager=object(),
    )

    def fake_sfmpe(task, config_path, device=None):
        return fake_pipeline

    monkeypatch.setattr(runner_module, "SFMPE", fake_sfmpe)
    runner = _FitRunner(
        config={"rounds": 1, "sfmpe_training": {}},
        _sfmpe_config_path=lambda _name: "pipeline.yaml",
        _nn_device=lambda: "cpu",
    )
    meter = SimpleNamespace(counts={"setup": 0}, set_phase=lambda _p: None)

    try:
        BenchmarkRunner._fit_sfmpe(
            runner,
            "Gaussian",
            SimpleNamespace(device="cpu"),
            torch.zeros(2),
            budget=6,
            meter=meter,
        )
    except ValueError as error:
        assert "quantile" in str(error)
    else:
        raise AssertionError("clean Truncated without quantile must raise")


def test_snpe_c_training_block_full_option_wiring(monkeypatch):
    """Every snpe_c_training key reaches sbi's append_simulations /
    train() calls (training_batch_size clamped to the round budget)."""
    observed_train: list[dict] = []
    observed_exclusions: list = []

    class FakePosterior:
        def set_default_x(self, observation):
            return self

    class FakeInference:
        def __init__(self, prior, device, **kwargs):
            self._kwargs = kwargs

        def append_simulations(self, theta, x, proposal=None, **kwargs):
            observed_exclusions.append(kwargs.get("exclude_invalid_x"))
            return self

        def train(self, **kwargs):
            observed_train.append(kwargs)
            return object()

        def build_posterior(self, density_estimator, **kwargs):
            return FakePosterior()

    _install_fake_sbi(monkeypatch, FakeInference)

    def make_task(_name, device):
        observed_task_devices.append(device)
        return SimpleNamespace(prior=None)

    observed_task_devices: list = []
    task = SimpleNamespace(device="cpu", theta_dim=2)
    runner = _FitRunner(
        config={
            "rounds": 1,
            "snpe_c_training": {
                "max_num_epochs": 11,
                "stop_after_epochs": 4,
                "training_batch_size": 64,
                "learning_rate": 1e-3,
                "validation_fraction": 0.2,
                "num_atoms": 3,
                "clip_max_norm": None,
                "resume_training": True,
                "force_first_round_loss": True,
                "discard_prior_samples": True,
                "use_combined_loss": True,
                "retrain_from_scratch": True,
                "show_train_summary": True,
                "exclude_invalid_x": False,
                "dataloader_kwargs": {"num_workers": 2},
                "calibration_kernel": None,
            },
        },
        device="cpu",
        _make_task=make_task,
        _make_simulator_callable=lambda _task: lambda theta: theta,
        _nn_device=lambda: "cpu",
        _sbi_prior_device=lambda: "cpu",
    )
    meter = SimpleNamespace(set_phase=lambda _p: None)

    sample_posterior = BenchmarkRunner._fit_snpe_c(
        runner, "Gaussian", task, torch.zeros(2), budget=8, meter=meter
    )

    assert callable(sample_posterior)
    assert observed_task_devices == ["cpu"]
    assert observed_exclusions == [False]
    assert len(observed_train) == 1
    assert observed_train[0] == {
        "num_atoms": 3,
        "training_batch_size": 8,  # min(64, max(2, 8))
        "learning_rate": 1e-3,
        "validation_fraction": 0.2,
        "stop_after_epochs": 4,
        "max_num_epochs": 11,
        "clip_max_norm": None,
        "calibration_kernel": None,
        "resume_training": True,
        "force_first_round_loss": True,
        "discard_prior_samples": True,
        "use_combined_loss": True,
        "retrain_from_scratch": True,
        "show_train_summary": True,
        "dataloader_kwargs": {"num_workers": 2},
    }


def test_snpe_c_legacy_training_block_mapping(monkeypatch):
    """The pre-split merged `training:` block still drives SBPE-C: legacy
    `epochs`/`batch_size` map to max_num_epochs/training_batch_size."""
    observed_train: list[dict] = []

    class FakePosterior:
        def set_default_x(self, observation):
            return self

    class FakeInference:
        def __init__(self, prior, device, **kwargs):
            pass

        def append_simulations(self, theta, x, proposal=None, **kwargs):
            return self

        def train(self, **kwargs):
            observed_train.append(kwargs)
            return object()

        def build_posterior(self, density_estimator, **kwargs):
            return FakePosterior()

    _install_fake_sbi(monkeypatch, FakeInference)

    task = SimpleNamespace(device="cpu", theta_dim=2)
    runner = _FitRunner(
        config={
            "rounds": 1,
            "training": {
                "epochs": 5,
                "batch_size": 16,
                "learning_rate": 2e-3,
                "validation_fraction": 0.3,
                "stop_after_epochs": 2,
            },
        },
        device="cpu",
        _make_task=lambda _name, device: SimpleNamespace(prior=None),
        _make_simulator_callable=lambda _task: lambda theta: theta,
        _nn_device=lambda: "cpu",
        _sbi_prior_device=lambda: "cpu",
    )
    meter = SimpleNamespace(set_phase=lambda _p: None)

    BenchmarkRunner._fit_snpe_c(
        runner, "Gaussian", task, torch.zeros(2), budget=8, meter=meter
    )

    assert len(observed_train) == 1
    assert observed_train[0] == {
        "num_atoms": 10,
        "training_batch_size": min(16, max(2, 8)),
        "learning_rate": 2e-3,
        "validation_fraction": 0.3,
        "stop_after_epochs": 2,
        "max_num_epochs": 5,
        "clip_max_norm": 5.0,
        "calibration_kernel": None,
        "resume_training": False,
        "force_first_round_loss": False,
        "discard_prior_samples": False,
        "use_combined_loss": False,
        "retrain_from_scratch": False,
        "show_train_summary": False,
        "dataloader_kwargs": None,
    }


def test_training_block_split_and_legacy_fallback(tmp_path: Path):
    """`_training_block`: method blocks win; the merged `training:` block is
    the fallback for both methods."""
    config_path = tmp_path / "benchmark.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "device": "cpu",
                "training": {"epochs": 1, "batch_size": 4},
                "snpe_c_training": {"max_num_epochs": 9},
            }
        ),
        encoding="utf-8",
    )
    runner = BenchmarkRunner(config_path)

    legacy = {"epochs": 1, "batch_size": 4}
    # SFMPE has no dedicated block -> falls back to legacy.
    assert runner._training_block("sfmpe_training") == legacy
    # SNPE-C has a dedicated block -> it wins wholesale.
    assert runner._training_block("snpe_c_training") == {"max_num_epochs": 9}

    config_path2 = tmp_path / "legacy.yaml"
    config_path2.write_text(
        yaml.safe_dump({"device": "cpu", "training": legacy}),
        encoding="utf-8",
    )
    legacy_only = BenchmarkRunner(config_path2)
    assert legacy_only._training_block("sfmpe_training") == legacy
    assert legacy_only._training_block("snpe_c_training") == legacy


def test_smc_abc_full_option_wiring(monkeypatch):
    """Every smc_abc key reaches sbi's SMCABC.__call__; particles and the
    initial population keep their budget clamps."""
    observed_calls: list[dict] = []
    observed_worker_counts: list = []

    class FakeSMCDistribution:
        def sample(self, shape):
            return torch.zeros(*shape, 2)

    class FakeSMCABC:
        def __init__(self, simulator, prior, num_workers=1, **kwargs):
            observed_worker_counts.append(num_workers)

        def __call__(self, x_o, num_particles, num_initial_pop, num_simulations, **kwargs):
            observed_calls.append(
                {
                    "num_particles": num_particles,
                    "num_initial_pop": num_initial_pop,
                    "num_simulations": num_simulations,
                    **kwargs,
                }
            )
            return FakeSMCDistribution()

    inference_module = types.ModuleType("sbi.inference")
    inference_module.SMCABC = FakeSMCABC
    monkeypatch.setitem(sys.modules, "sbi.inference", inference_module)

    task = SimpleNamespace(prior=torch.distributions.Normal(torch.zeros(2), torch.ones(2)))
    runner = _FitRunner(
        config={
            "smc_abc": {
                "particles": 30,
                "initial_population": 40,
                "num_workers": 3,
                "epsilon_decay": 0.7,
                "distance_based_decay": False,
                "ess_min": 0.5,
                "kernel_variance_scale": 2.0,
                "use_last_pop_samples": False,
                "kde": True,
                "kde_kwargs": {"bandwidth": None},
                "kde_sample_weights": True,
                "lra": False,
                "lra_with_weights": False,
                "sass": False,
                "sass_fraction": 0.25,
                "sass_expansion_degree": 1,
            },
        },
        _make_simulator_callable=lambda _task: lambda theta: theta,
    )
    meter = SimpleNamespace(set_phase=lambda _p: None)

    sample_posterior = BenchmarkRunner._fit_smc_abc(
        runner, task, torch.zeros(2), budget=1000, meter=meter
    )

    assert observed_worker_counts == [3]
    assert len(observed_calls) == 1
    # particles = min(max(20, min(30, 1000 // 5)), 1000) = 30
    # initial_population = min(1000, max(30, min(40, 500))) = 40
    assert observed_calls[0] == {
        "num_particles": 30,
        "num_initial_pop": 40,
        "num_simulations": 1000,
        "epsilon_decay": 0.7,
        "distance_based_decay": False,
        "ess_min": 0.5,
        "kernel_variance_scale": 2.0,
        "use_last_pop_samples": False,
        "kde": True,
        "kde_kwargs": {"bandwidth": None},
        "kde_sample_weights": True,
        "lra": False,
        "lra_with_weights": False,
        "sass": False,
        "sass_fraction": 0.25,
        "sass_expansion_degree": 1,
    }
    samples = sample_posterior(torch.zeros(2), 5)
    assert samples.shape == (5, 2)


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


def test_kaggle_standalone_imports_bundled_source_not_editable_install(tmp_path: Path):
    """Regression: on Kaggle the payload lands in the working dir and `import
    sfmpe` must resolve against the bundled sources (sys.path insert), never
    depend on a pip editable install.

    Mirrors the staged Kaggle layout: the standalone script, ``src/``,
    ``pyproject.toml`` and ``configs/`` all sit in one working directory, and
    the script is executed from there (`--check-imports` geo-only path).
    """
    code_dir = Path(__file__).resolve().parents[1]
    script = code_dir / "notebooks" / "benchmark_kaggle_standalone.py"
    assert script.is_file(), "standalone script must exist in notebooks/"

    shutil.copytree(code_dir / "src" / "sfmpe", tmp_path / "src" / "sfmpe")
    shutil.copyfile(code_dir / "pyproject.toml", tmp_path / "pyproject.toml")
    shutil.copytree(code_dir / "configs", tmp_path / "configs")
    shutil.copyfile(script, tmp_path / "benchmark_kaggle_standalone.py")

    # The assertion against the tmp copy proves the fix: the sys.path insert
    # routes `import sfmpe` to the bundled sources even though a pip editable
    # install of the repo also exists in this environment.
    result = subprocess.run(
        [sys.executable, str(tmp_path / "benchmark_kaggle_standalone.py"),
         "--check-imports"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 0, (
        f"check-imports failed:\n--- stdout ---\n{result.stdout}"
        f"\n--- stderr ---\n{result.stderr}"
    )
    bundled_init = (tmp_path / "src" / "sfmpe" / "__init__.py").resolve()
    assert f"sfmpe: {bundled_init}" in result.stdout, result.stdout
    assert "Inserted bundled source into sys.path" in result.stdout, result.stdout
    assert "skipping pip install" in result.stdout, result.stdout


def test_kaggle_standalone_reports_checked_paths_when_bundle_missing(tmp_path: Path):
    """Missing bundle -> a deliberate error listing every path that was checked."""
    code_dir = Path(__file__).resolve().parents[1]
    standalone = tmp_path / "benchmark_kaggle_standalone.py"
    shutil.copyfile(code_dir / "notebooks" / "benchmark_kaggle_standalone.py", standalone)

    result = subprocess.run(
        [sys.executable, str(standalone), "--check-imports"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "Paths checked:" in combined, combined
    assert f"  - {(tmp_path).resolve()}" in combined, combined


def _load_standalone_module():
    code_dir = Path(__file__).resolve().parents[1]
    script = code_dir / "notebooks" / "benchmark_kaggle_standalone.py"
    spec = importlib.util.spec_from_file_location(
        "benchmark_kaggle_standalone", script
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_config_env_var_resolution(tmp_path: Path):
    """`SFMPE_BENCHMARK_CONFIG` (kagglex --env) + bare names resolve against
    configs/ inside the bundled root; --config keeps precedence."""
    module = _load_standalone_module()
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    full = configs_dir / "benchmark_full.yaml"
    full.write_text("seed: 1\n", encoding="utf-8")

    # Bare name from the env resolves against <root>/configs/.
    assert (
        module.find_config_path(None, tmp_path, env_value="benchmark_full.yaml")
        == full.resolve()
    )
    # A bare stem also works (auto .yaml).
    assert (
        module.find_config_path(None, tmp_path, env_value="benchmark_full")
        == full.resolve()
    )
    # --config wins over the env value.
    assert (
        module.find_config_path(str(full), tmp_path, env_value="missing.yaml")
        == full.resolve()
    )
    # A path with directories is used as given (relative to cwd).
    assert (
        module.find_config_path(None, tmp_path, env_value="configs/benchmark_full.yaml")
        == full.resolve()
    )
    # Missing values raise with the list of checked paths.
    try:
        module.find_config_path(None, tmp_path, env_value="nope.yaml")
    except FileNotFoundError as error:
        assert "nope.yaml" in str(error) and "Checked:" in str(error)
    else:
        raise AssertionError("missing env config must raise")


def _fake_scored_row(method: str, round_number: int | None) -> dict:
    return {
        "task": "Gaussian",
        "method": method,
        "budget": 10,
        "replicate": 0,
        "round": round_number,
        "seed": 0,
        "inference_simulator_draws": 10,
        "setup_simulator_draws": 0,
        "sfmpe_proposal_method": None,
        "ppc_simulator_draws": 0,
        "sensitivity_simulator_draws": 0,
        "inference_seconds": 0.1,
        "posterior_sampling_seconds": 0.01,
        "metrics": {
            "mean_error": float(1.0 / (round_number or 1)),
            "mmd_rbf": float(1.0 / (round_number or 1)),
        },
    }


def test_rounds_axis_evaluation_emits_one_row_per_round(tmp_path: Path, monkeypatch):
    """evaluate_rounds: sequential methods report a row per round, SMC-ABC a
    single row with round=None."""
    code_dir = Path(__file__).resolve().parents[1]
    config_path = tmp_path / "benchmark.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "device": "cpu",
                "tasks": {"Gaussian": str(code_dir / "configs" / "gaussian.yaml")},
                "methods": ["SFMPE", "SMC_ABC"],
                "budgets": [10],
                "replicates": 2,
                "posterior_samples": 8,
                "evaluate_rounds": True,
                "diagnostics": {
                    "ppc_samples": 2,
                    "sensitivity_enabled": False,
                    "sbc_c2st": False,
                },
                "output": str(tmp_path / "results"),
            }
        ),
        encoding="utf-8",
    )
    runner = BenchmarkRunner(config_path)

    def fake_evaluate(task_name, method, budget, replicate, seed, observation, scale):
        rounds = [1, 2] if method == "SFMPE" else [None]
        scored = []
        for round_number in rounds:
            row = _fake_scored_row(method, round_number)
            row["replicate"] = replicate
            scored.append((row, torch.randn(8, 2)))
        return scored

    monkeypatch.setattr(runner, "_evaluate_target", fake_evaluate)
    summaries = runner.run()

    keys = {(summary["method"], summary["round"]) for summary in summaries}
    assert keys == {("SFMPE", 1), ("SFMPE", 2), ("SMC_ABC", None)}

    target_rows = [
        json.loads(line)
        for line in (tmp_path / "results" / "target_metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert {(row["method"], row["round"]) for row in target_rows} == {
        ("SFMPE", 1),
        ("SFMPE", 2),
        ("SMC_ABC", None),
    }
    assert len(target_rows) == 2 * 2 + 2 * 1  # 2 replicates per method/round

    # Resume: a re-run with the same output directory evaluates nothing.
    calls: list[tuple] = []

    def tracking_evaluate(*args, **kwargs):
        calls.append(args)
        return []

    resumed = BenchmarkRunner(config_path)
    monkeypatch.setattr(resumed, "_evaluate_target", tracking_evaluate)
    summaries_again = resumed.run()
    assert calls == []
    assert len(summaries_again) == 3


def test_overrides_select_shards_and_time_limit(tmp_path: Path, monkeypatch):
    """Shard/CLI overrides filter tasks/methods/budgets; a zero time limit
    stops before evaluating anything."""
    code_dir = Path(__file__).resolve().parents[1]
    config_path = tmp_path / "benchmark.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "device": "cpu",
                "tasks": {
                    "Gaussian": str(code_dir / "configs" / "gaussian.yaml"),
                    "SIR": str(code_dir / "configs" / "benchmark_sir.yaml"),
                },
                "methods": ["SFMPE", "SNPE_C", "SMC_ABC"],
                "budgets": [10, 20],
                "replicates": 2,
                "posterior_samples": 4,
                "diagnostics": {
                    "ppc_samples": 2,
                    "sensitivity_enabled": False,
                    "sbc_c2st": False,
                },
                "output": str(tmp_path / "results"),
            }
        ),
        encoding="utf-8",
    )
    runner = BenchmarkRunner(
        config_path,
        overrides={
            "tasks": ["Gaussian"],
            "methods": ["SNPE_C"],
            "budgets": [20],
        },
    )
    assert list(runner.config["tasks"]) == ["Gaussian"]
    assert runner.config["methods"] == ["SNPE_C"]
    assert runner.config["budgets"] == [20]

    evaluated: list[tuple] = []

    def fake_evaluate(task_name, method, budget, replicate, seed, observation, scale):
        evaluated.append((task_name, method, budget, replicate))
        row = _fake_scored_row(method, None)
        row["replicate"] = replicate
        return [(row, torch.randn(8, 2))]

    monkeypatch.setattr(runner, "_evaluate_target", fake_evaluate)
    summaries = runner.run()
    assert evaluated == [
        ("Gaussian", "SNPE_C", 20, 0),
        ("Gaussian", "SNPE_C", 20, 1),
    ]
    assert len(summaries) == 1
    assert summaries[0]["method"] == "SNPE_C"

    # Unknown task override fails loudly.
    try:
        BenchmarkRunner(config_path, overrides={"tasks": ["NoSuchTask"]})
    except ValueError as error:
        assert "NoSuchTask" in str(error)
    else:
        raise AssertionError("unknown task override must raise")

    # Time limit 0 h: stop before any evaluation, return no new summaries.
    limited = BenchmarkRunner(
        config_path,
        overrides={"time_limit_hours": 0, "output": str(tmp_path / "limited")},
    )
    limited_calls: list[tuple] = []
    monkeypatch.setattr(
        limited,
        "_evaluate_target",
        lambda *args, **kwargs: limited_calls.append(args) or [],
    )
    assert limited.run() == []
    assert limited_calls == []


def test_merge_shard_results_dedupes_and_exports(tmp_path: Path):
    from sfmpe.benchmark.report import merge_benchmark_results

    def write_shard(directory: Path, task: str, mean_error: float) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        target = {
            "task": task,
            "method": "SFMPE",
            "budget": 10,
            "replicate": 0,
            "round": 1,
            "metrics": {"mean_error": mean_error},
        }
        summary = {
            "task": task,
            "method": "SFMPE",
            "budget": 10,
            "round": 1,
            "replicates": 1,
            "mean_error": mean_error,
        }
        (directory / "target_metrics.jsonl").write_text(
            json.dumps(target) + "\n", encoding="utf-8"
        )
        (directory / "benchmark_summary.jsonl").write_text(
            json.dumps(summary) + "\n", encoding="utf-8"
        )

    shard_a = tmp_path / "shard-a"
    shard_b = tmp_path / "shard-b"
    write_shard(shard_a, "Gaussian", 1.0)
    write_shard(shard_b, "SIR", 2.0)

    merged = merge_benchmark_results([shard_a, shard_b], tmp_path / "merged")
    summaries = [
        json.loads(line)
        for line in (merged / "benchmark_summary.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert {row["task"] for row in summaries} == {"Gaussian", "SIR"}
    report = (merged / "benchmark_report.md").read_text(encoding="utf-8")
    assert "Gaussian" in report and "SIR" in report
    assert "Round" in report  # rounds axis present in tables
    assert (merged / "figures" / "Gaussian_rounds_mean_error.png").is_file()


def test_schedule_file_matches_benchmark_wiring(tmp_path: Path):
    """The canonical schedule covers every task x method pair and the
    resolved overlays are valid config overrides."""
    module = _load_standalone_module()
    code_dir = Path(__file__).resolve().parents[1]
    schedule_path = code_dir / "configs" / "benchmark_schedule.yaml"
    schedule = module.load_schedule(schedule_path)
    benchmark = yaml.safe_load(
        (code_dir / "configs" / "benchmark.yaml").read_text(encoding="utf-8")
    )

    assert schedule["base_config"] == "benchmark.yaml"
    expected = {
        (task, method)
        for task in benchmark["tasks"]
        for method in benchmark["methods"]
    }
    covered = {
        (entry["tasks"][0], entry["methods"][0])
        for entry in schedule["shards"]
    }
    assert covered == expected

    overlay = module.shard_overrides(schedule, "gaussian-sfmpe")
    assert overlay == {"tasks": ["Gaussian"], "methods": ["SFMPE"]}
    try:
        module.shard_overrides(schedule, "nope")
    except KeyError as error:
        assert "gaussian-sfmpe" in str(error)
    else:
        raise AssertionError("unknown shard must raise")

    # Schedule resolution works from the bundle root (configs/ fallback).
    assert module.find_schedule_path(None, code_dir) == schedule_path.resolve()


def test_rounds_axis_in_markdown_report(tmp_path: Path):
    """Markdown tables gain a Round column and rounds figures are produced."""
    from sfmpe.benchmark.report import export_benchmark_results

    results = tmp_path / "results"
    results.mkdir()
    summaries = []
    for round_number in (1, 2):
        summaries.append(
            {
                "task": "Gaussian",
                "method": "SFMPE",
                "budget": 10,
                "round": round_number,
                "replicates": 2,
                "mean_error": 1.0 / round_number,
                "std_mean_error": 0.1,
            }
        )
    summaries.append(
        {
            "task": "Gaussian",
            "method": "SMC_ABC",
            "budget": 10,
            "round": None,
            "replicates": 2,
            "mean_error": 0.5,
            "std_mean_error": 0.2,
        }
    )
    (results / "benchmark_summary.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in summaries),
        encoding="utf-8",
    )
    (results / "target_metrics.jsonl").write_text(
        json.dumps({"task": "Gaussian", "budget": 10, "round": 1}) + "\n",
        encoding="utf-8",
    )

    export_benchmark_results(results)
    report = (results / "benchmark_report.md").read_text(encoding="utf-8")
    assert "| Budget | Round |" in report
    assert "| 10.0 | 1 |" in report
    assert "| 10.0 | - |" in report  # SMC-ABC has no rounds
    assert (results / "figures" / "Gaussian_rounds_mean_error.png").is_file()
