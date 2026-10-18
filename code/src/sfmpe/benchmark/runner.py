"""Run target-specific sequential SBI benchmarks with matched simulation budgets."""

import argparse
import json
import math
import random
import time
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import yaml

from sfmpe.benchmark.metrics import (
    calibration_metrics,
    exact_posterior_metrics,
    posterior_predictive_metrics,
    sensitivity_metrics,
)
from sfmpe.inference.SFMPE import SFMPE
from sfmpe.tasks.Gaussian import GaussianTask
from sfmpe.tasks.SIR import SIRTask
from sfmpe.tasks.StochVol import StochVolTask


TASK_TYPES = {
    "gaussian": GaussianTask,
    "sir": SIRTask,
    "stochvol": StochVolTask,
    "stochasticvolatility": StochVolTask,
}


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'.")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return device


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _balanced_round_budgets(total: int, rounds: int) -> list[int]:
    if total < 1 or rounds < 1:
        raise ValueError("total and rounds must be positive.")
    active_rounds = min(total, rounds)
    base, remainder = divmod(total, active_rounds)
    return [
        base + (1 if round_index < remainder else 0)
        for round_index in range(active_rounds)
    ]


class _SimulatorCallMeter:
    """Count simulated theta vectors by phase without counting Python batches."""

    def __init__(self, simulator):
        self.simulator = simulator
        self.original_simulate = simulator.simulate
        self.phase = "setup"
        self.counts: dict[str, int] = defaultdict(int)

        def counted_simulate(theta, *args, **kwargs):
            self.counts[self.phase] += math.prod(theta.shape[:-1])
            return self.original_simulate(theta, *args, **kwargs)

        simulator.simulate = counted_simulate

    def set_phase(self, phase: str) -> None:
        self.phase = phase


class BenchmarkRunner:
    """Execute a configurable task × method × budget benchmark."""

    def __init__(
        self,
        config_path: str | Path,
        output_dir: str | Path | None = None,
    ):
        self.config_path = Path(config_path).resolve()
        with self.config_path.open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        if not isinstance(config, Mapping):
            raise TypeError("Benchmark configuration must be a mapping.")
        self.config = dict(config)
        self.root = self.config_path.parent
        self.device = _resolve_device(self.config.get("device", "auto"))
        configured_output = self.config.get("output", "./results/benchmark")
        self.output_dir = (
            Path(output_dir).resolve()
            if output_dir is not None
            else (self.root / configured_output).resolve()
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _task_config(self, task_name: str) -> dict[str, Any]:
        task_paths = self.config.get("tasks", {})
        if task_name not in task_paths:
            raise ValueError(f"Task {task_name!r} is not configured.")
        task_path = (self.root / task_paths[task_name]).resolve()
        with task_path.open(encoding="utf-8") as stream:
            task_config = yaml.safe_load(stream)
        if not isinstance(task_config, Mapping):
            raise TypeError(f"Task configuration {task_path} must be a mapping.")
        return dict(task_config)

    def _make_task(self, task_name: str):
        task_type = TASK_TYPES.get(task_name.casefold())
        if task_type is None:
            raise ValueError(f"Unsupported task {task_name!r}.")
        return task_type(self._task_config(task_name), device=self.device)

    def _make_simulator_callable(self, task):
        def simulate_and_summarize(theta: torch.Tensor) -> torch.Tensor:
            return task.summarize(task.simulate(theta))

        return simulate_and_summarize

    def _fit_sfmpe(
        self,
        task_name: str,
        task,
        observation: torch.Tensor,
        budget: int,
        meter: _SimulatorCallMeter,
    ):
        meter.set_phase("setup")
        sfmpe_config = self.root / self.config.get(
            "sfmpe_config",
            "benchmark_sfmpe.yaml",
        )
        pipeline = SFMPE(task, sfmpe_config)
        proposal_method = self.config.get(
            "sfmpe_proposal_methods",
            {},
        ).get(task_name)
        if proposal_method is not None:
            pipeline.proposal_params.method = proposal_method
            pipeline.manager.proposal_params.method = proposal_method
        pipeline.proposal_params.x_0 = observation.unsqueeze(0).to(self.device)
        remaining_budget = budget - meter.counts["setup"]
        if remaining_budget < 1:
            raise ValueError(
                "The simulation budget must leave at least one draw after "
                "SFMPE initialization."
            )
        round_budgets = _balanced_round_budgets(
            remaining_budget,
            self.config.get("rounds", 3),
        )
        training = dict(self.config.get("training", {}))
        epochs = training.get("epochs", 20)
        batch_size = training.get("batch_size", 64)
        meter.set_phase("inference")
        for round_id, round_budget in enumerate(round_budgets):
            result = pipeline.manager.run_round_experimental(
                round_id,
                round_budget,
                sims_per_sample=1,
                clean_sampling=True,
            )
            if result == -1:
                raise RuntimeError(
                    f"SFMPE stopped in round {round_id} for {task_name}."
                )
            pipeline.manager.train_estimator(
                [round_id],
                device=self.device,
                epochs=epochs,
                batch_size=min(batch_size, round_budget),
                shuffle=True,
                num_workers=0,
                show_every=None,
                validation_fraction=training.get("validation_fraction", 0.1),
                stop_after_epochs=training.get("stop_after_epochs", 20),
                min_delta=training.get("min_delta", 0.0),
                validation_seed=training.get("validation_seed"),
            )
            posterior = pipeline.manager.build_posterior(update_x=False)
            pipeline.manager.update_proposal(posterior)
        meter.set_phase("sampling")

        def sample_posterior(x: torch.Tensor, num_samples: int) -> torch.Tensor:
            pipeline.manager.proposal.params.x_0 = x.reshape(1, -1).to(self.device)
            samples = pipeline.manager.proposal.sample(
                (num_samples,),
                device=self.device,
            )
            if samples.ndim == 3 and samples.shape[1] == 1:
                samples = samples[:, 0, :]
            return samples

        return sample_posterior

    def _fit_snpe_c(
        self,
        task,
        observation: torch.Tensor,
        budget: int,
        meter: _SimulatorCallMeter,
    ):
        from sbi.inference import SNPE_C, simulate_for_sbi

        training = dict(self.config.get("training", {}))
        inference = SNPE_C(
            prior=task.prior,
            device=self.device,
            show_progress_bars=False,
        )
        simulator = self._make_simulator_callable(task)
        proposal = task.prior
        posterior = None
        density_estimator = None
        round_budgets = _balanced_round_budgets(
            budget,
            self.config.get("rounds", 3),
        )
        meter.set_phase("inference")
        for round_id, round_budget in enumerate(round_budgets):
            theta, x = simulate_for_sbi(
                simulator,
                proposal=proposal,
                num_simulations=round_budget,
                simulation_batch_size=min(64, round_budget),
                num_workers=1,
                show_progress_bar=False,
            )
            appended = inference.append_simulations(
                theta,
                x,
                proposal=None if round_id == 0 else cast(Any, proposal),
            )
            training_batch_size = min(
                training.get("batch_size", 64),
                max(2, round_budget),
            )
            density_estimator = appended.train(
                max_num_epochs=training.get("epochs", 20),
                stop_after_epochs=training.get("stop_after_epochs", 8),
                training_batch_size=training_batch_size,
                learning_rate=training.get("learning_rate", 5e-4),
                validation_fraction=training.get("validation_fraction", 0.1),
                show_train_summary=False,
            )
            posterior = inference.build_posterior(
                density_estimator,
                sample_with="rejection",
            )
            proposal = posterior.set_default_x(observation)

        if posterior is None or density_estimator is None:
            raise RuntimeError("SNPE-C did not complete any training rounds.")
        meter.set_phase("sampling")

        def sample_posterior(x: torch.Tensor, num_samples: int) -> torch.Tensor:
            result = posterior.sample(
                (num_samples,),
                x=x.to(self.device),
                show_progress_bars=False,
            )
            if isinstance(result, tuple):
                raise RuntimeError("SNPE-C returned an unexpected summary tuple.")
            return result

        return sample_posterior

    def _fit_smc_abc(
        self,
        task,
        observation: torch.Tensor,
        budget: int,
        meter: _SimulatorCallMeter,
    ):
        from sbi.inference import SMCABC

        settings = dict(self.config.get("smc_abc", {}))
        if budget < 20:
            raise ValueError(
                "SMC-ABC requires a budget of at least 20 simulations for "
                "the SBI KDE posterior."
            )
        particles = min(
            max(20, min(settings.get("particles", 100), budget // 5)),
            budget,
        )
        initial_population = min(
            budget,
            max(
                particles,
                min(settings.get("initial_population", 500), budget // 2),
            ),
        )
        simulator = self._make_simulator_callable(task)

        def infer(x: torch.Tensor):
            smc = SMCABC(
                simulator=simulator,
                prior=task.prior,
                num_workers=1,
                show_progress_bars=False,
            )
            return smc(
                x.to(self.device),
                num_particles=particles,
                num_initial_pop=initial_population,
                num_simulations=budget,
                epsilon_decay=settings.get("epsilon_decay", 0.5),
                kde=True,
                kde_sample_weights=True,
            )

        meter.set_phase("inference")
        posterior_result = infer(observation)
        if isinstance(posterior_result, tuple):
            raise RuntimeError("SMC-ABC returned an unexpected summary tuple.")
        posterior = posterior_result
        meter.set_phase("sampling")

        def sample_posterior(x: torch.Tensor, num_samples: int) -> torch.Tensor:
            distribution = (
                posterior
                if torch.equal(x, observation)
                else infer(x)
            )
            result = cast(Any, distribution).sample((num_samples,))
            if not isinstance(result, torch.Tensor):
                raise TypeError("SMC-ABC posterior sampling did not return a tensor.")
            return result.to(self.device)

        return sample_posterior

    def _fit(
        self,
        task_name: str,
        method: str,
        task,
        observation: torch.Tensor,
        budget: int,
        meter: _SimulatorCallMeter,
    ):
        if method == "SFMPE":
            return self._fit_sfmpe(task_name, task, observation, budget, meter)
        if method == "SNPE_C":
            return self._fit_snpe_c(task, observation, budget, meter)
        if method == "SMC_ABC":
            return self._fit_smc_abc(task, observation, budget, meter)
        raise ValueError(f"Unsupported method {method!r}.")

    def _evaluate_target(
        self,
        task_name: str,
        method: str,
        budget: int,
        replicate: int,
        seed: int,
        observation: torch.Tensor,
        observation_scale: torch.Tensor,
    ) -> tuple[dict[str, Any], torch.Tensor]:
        _set_seed(seed)
        task = self._make_task(task_name)
        meter = _SimulatorCallMeter(task.simulator)
        inference_start = time.perf_counter()
        sample_posterior = self._fit(
            task_name,
            method,
            task,
            observation,
            budget,
            meter,
        )
        inference_seconds = time.perf_counter() - inference_start
        inference_calls = meter.counts["setup"] + meter.counts["inference"]
        if inference_calls > budget:
            raise RuntimeError(
                f"{method} used {inference_calls} simulator draws, "
                f"exceeding budget {budget}."
            )

        num_posterior_samples = self.config.get("posterior_samples", 256)
        meter.set_phase("sampling")
        sampling_start = time.perf_counter()
        samples = sample_posterior(observation, num_posterior_samples)
        posterior_sampling_seconds = time.perf_counter() - sampling_start
        if samples.ndim != 2 or samples.shape[1] != task.theta_dim:
            raise ValueError(
                f"{method} returned posterior sample shape {tuple(samples.shape)}; "
                f"expected (samples, {task.theta_dim})."
            )

        meter.set_phase("ppc")
        ppc_samples = samples[
            : min(
                samples.shape[0],
                self.config.get("diagnostics", {}).get("ppc_samples", 128),
            )
        ]
        ppc = posterior_predictive_metrics(
            task,
            observation,
            ppc_samples,
            task.simulate,
        )

        metrics: dict[str, Any] = {}
        if hasattr(task, "true_posterior"):
            reference_samples = task.sample_true_posterior(
                observation,
                (num_posterior_samples,),
            )
            metrics.update(exact_posterior_metrics(samples, reference_samples))
        metrics.update(ppc)

        diagnostics = self.config.get("diagnostics", {})
        if diagnostics.get("sensitivity_enabled", True):
            meter.set_phase("sensitivity")
            metrics.update(
                sensitivity_metrics(
                    observation,
                    samples,
                    sample_posterior,
                    observation_scale,
                    diagnostics.get("sensitivity_samples", 128),
                    diagnostics.get("sensitivity_scale", 0.1),
                )
            )

        support_check = getattr(task, "check_support", None)
        if callable(support_check):
            support_values = cast(torch.Tensor, support_check(samples))
            metrics["posterior_support_fraction"] = float(
                support_values.float().mean().item()
            )
        row = {
            "task": task_name,
            "method": method,
            "budget": budget,
            "replicate": replicate,
            "seed": seed,
            "inference_simulator_draws": inference_calls,
            "setup_simulator_draws": meter.counts["setup"],
            "sfmpe_proposal_method": (
                self.config.get("sfmpe_proposal_methods", {}).get(task_name)
                if method == "SFMPE"
                else None
            ),
            "ppc_simulator_draws": meter.counts["ppc"],
            "sensitivity_simulator_draws": meter.counts["sensitivity"],
            "inference_seconds": inference_seconds,
            "posterior_sampling_seconds": posterior_sampling_seconds,
            "metrics": metrics,
        }
        return row, samples.detach().cpu()

    def run(self) -> list[dict[str, Any]]:
        task_names = list(self.config.get("tasks", {}).keys())
        methods = list(self.config.get("methods", []))
        budgets = [int(budget) for budget in self.config.get("budgets", [])]
        replicates = int(self.config.get("replicates", 1))
        posterior_rows: list[dict[str, Any]] = []
        summaries: list[dict[str, Any]] = []

        for task_index, task_name in enumerate(task_names):
            target_seed = int(self.config.get("seed", 0)) + task_index * 100_000
            _set_seed(target_seed)
            target_task = self._make_task(task_name)
            target_thetas = target_task.sample_prior((replicates,)).detach().cpu()
            target_observations = target_task.summarize(
                target_task.simulate(target_thetas.to(self.device))
            ).detach().cpu()
            observation_scale = target_observations.std(
                dim=0,
                unbiased=False,
            ).clamp_min(1e-6)

            for method_index, method in enumerate(methods):
                for budget_index, budget in enumerate(budgets):
                    run_rows = []
                    run_samples = []
                    for replicate in range(replicates):
                        seed = (
                            int(self.config.get("seed", 0))
                            + task_index * 100_000
                            + method_index * 10_000
                            + budget_index * 1_000
                            + replicate
                        )
                        row, samples = self._evaluate_target(
                            task_name,
                            method,
                            budget,
                            replicate,
                            seed,
                            target_observations[replicate],
                            observation_scale,
                        )
                        run_rows.append(row)
                        run_samples.append(samples)
                        posterior_rows.append(row)
                        print(
                            f"{task_name} {method} budget={budget} "
                            f"target={replicate + 1}/{replicates} "
                            f"simulations={row['inference_simulator_draws']}"
                        )

                    aggregate = self._aggregate(
                        task_name,
                        method,
                        budget,
                        run_rows,
                        run_samples,
                        target_thetas,
                    )
                    summaries.append(aggregate)

        self._write_jsonl("target_metrics.jsonl", posterior_rows)
        self._write_jsonl("benchmark_summary.jsonl", summaries)
        return summaries

    def _aggregate(
        self,
        task_name: str,
        method: str,
        budget: int,
        rows: list[dict[str, Any]],
        sample_sets: list[torch.Tensor],
        target_thetas: torch.Tensor,
    ) -> dict[str, Any]:
        posterior_samples = torch.stack(sample_sets, dim=1)
        aggregate: dict[str, Any] = {
            "task": task_name,
            "method": method,
            "budget": budget,
            "replicates": len(rows),
            "mean_inference_simulator_draws": float(
                np.mean([row["inference_simulator_draws"] for row in rows])
            ),
            "mean_setup_simulator_draws": float(
                np.mean([row["setup_simulator_draws"] for row in rows])
            ),
            "mean_ppc_simulator_draws": float(
                np.mean([row["ppc_simulator_draws"] for row in rows])
            ),
            "mean_sensitivity_simulator_draws": float(
                np.mean([row["sensitivity_simulator_draws"] for row in rows])
            ),
            "mean_inference_seconds": float(
                np.mean([row["inference_seconds"] for row in rows])
            ),
            "mean_posterior_sampling_seconds": float(
                np.mean([row["posterior_sampling_seconds"] for row in rows])
            ),
        }
        diagnostic_settings = self.config.get("diagnostics", {})
        aggregate.update(
            calibration_metrics(
                target_thetas,
                posterior_samples,
                include_c2st=diagnostic_settings.get("sbc_c2st", False),
            )
        )
        numeric_metric_names = {
            metric_name
            for row in rows
            for metric_name, value in row["metrics"].items()
            if isinstance(value, (int, float))
        }
        for metric_name in numeric_metric_names:
            values = [
                row["metrics"][metric_name]
                for row in rows
                if metric_name in row["metrics"]
            ]
            aggregate[f"mean_{metric_name}"] = float(np.mean(values))
            aggregate[f"std_{metric_name}"] = float(np.std(values))

        if self.config.get("diagnostics", {}).get("sensitivity_enabled", True):
            aggregate["sensitivity_perturbation_scale"] = self.config.get(
                "diagnostics", {}
            ).get("sensitivity_scale", 0.1)
        return aggregate

    def _write_jsonl(self, filename: str, rows: list[dict[str, Any]]) -> None:
        output_path = self.output_dir / filename
        with output_path.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, allow_nan=False) + "\n")


def run_benchmark(
    config_path: str | Path,
    output_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    return BenchmarkRunner(config_path, output_dir=output_dir).run()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a matched-budget target-specific SBI benchmark."
    )
    parser.add_argument(
        "--config",
        default="configs/benchmark.yaml",
        help="Path to the benchmark YAML configuration.",
    )
    args = parser.parse_args()
    results = run_benchmark(args.config)
    print(f"Completed {len(results)} task/method/budget combinations.")


if __name__ == "__main__":
    main()
