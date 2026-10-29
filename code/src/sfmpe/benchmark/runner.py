"""Run target-specific sequential SBI benchmarks with matched simulation budgets."""

import argparse
import json
import math
import random
import time
from collections import defaultdict
from collections.abc import Callable, Mapping
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


def _normalized_rounds(value: Any) -> list[int]:
    """Normalize the `rounds` config entry to a sorted list of positive ints.

    ``rounds: 3`` and ``rounds: [1, 2, 3]`` are both accepted; a list runs
    one experiment per value (the rounds axis) with the same total budget
    split evenly across that many rounds.
    """
    values = value if isinstance(value, (list, tuple)) else [value]
    try:
        rounds = sorted({int(item) for item in values})
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"rounds must be an integer or a list of integers, got {value!r}"
        ) from error
    if not rounds or rounds[0] < 1:
        raise ValueError(f"rounds values must be >= 1, got {value!r}")
    return rounds


def _target_key(row: Mapping[str, Any]) -> tuple:
    """Resume/dedup identity of one target-metrics row."""
    return (
        row.get("task"),
        row.get("method"),
        int(row.get("budget", 0)),
        row.get("rounds"),
        int(row.get("replicate", 0)),
    )


def _summary_key(row: Mapping[str, Any]) -> tuple:
    """Resume/dedup identity of one summary row."""
    return (
        row.get("task"),
        row.get("method"),
        int(row.get("budget", 0)),
        row.get("rounds"),
    )


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


def _resolve_policy_device(device: Any, fallback: str, label: str) -> str:
    """Resolve one 'device' entry from the ``devices`` config block.

    ``auto`` inherits the benchmark-level device; anything else must be
    'cpu' or 'cuda' (and 'cuda' is validated like the top-level ``device``
    setting).
    """
    if device is None or device == "auto":
        return fallback
    if device not in {"cpu", "cuda"}:
        raise ValueError(
            f"devices.{label} must be 'auto', 'cpu', or 'cuda', got {device!r}."
        )
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"devices.{label} requested CUDA but it is unavailable."
        )
    return device


class BenchmarkRunner:
    """Execute a configurable task × method × budget benchmark."""

    def __init__(
        self,
        config_path: str | Path,
        output_dir: str | Path | None = None,
        device: str | None = None,
        overrides: Mapping[str, Any] | None = None,
    ):
        self.config_path = Path(config_path).resolve()
        with self.config_path.open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        if not isinstance(config, Mapping):
            raise TypeError("Benchmark configuration must be a mapping.")
        self.config = dict(config)
        if overrides:
            # Shard/CLI overrides (tasks, methods, budgets, replicates,
            # rounds, time_limit_hours, output, ...).
            overrides = dict(overrides)
            requested_tasks = overrides.get("tasks")
            if isinstance(requested_tasks, (list, tuple, set)):
                # `tasks` is a name -> task-YAML mapping in the config; a
                # list override selects a subset of that mapping.
                base_tasks = dict(self.config.get("tasks", {}))
                missing = [
                    name for name in requested_tasks if name not in base_tasks
                ]
                if missing:
                    raise ValueError(
                        f"Unknown task(s) in override: {', '.join(missing)}; "
                        f"configured tasks: {', '.join(base_tasks)}"
                    )
                overrides["tasks"] = {
                    name: base_tasks[name] for name in requested_tasks
                }
            self.config.update(overrides)
        if device is not None:
            # CLI override wins over the YAML top-level `device` value.
            self.config["device"] = device
        self.root = self.config_path.parent
        self.device = _resolve_device(self.config.get("device", "auto"))
        configured_output = self.config.get("output", "./results/benchmark")
        self.output_dir = (
            Path(output_dir).resolve()
            if output_dir is not None
            else (self.root / configured_output).resolve()
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        time_limit_hours = self.config.get("time_limit_hours")
        self.time_limit_seconds = (
            float(time_limit_hours) * 3600.0
            if time_limit_hours is not None
            else None
        )

    # -- Device policy ---
    #
    # The 'device' config value selects where neural-network methods
    # (SFMPE, SNPE-C) and task simulation run. The optional 'devices' block
    # refines this explicitly instead of relying on hard-coded branches:
    #
    #   devices:
    #     nn: auto        # estimator + simulator for SFMPE/SNPE-C
    #     smc_abc: cpu    # sbi SMCABC computes kernel covariance via numpy -> CPU
    #     sbi_prior: auto # prior object for SNPE-C: sbi asserts it lives on the
    #                     # training device (check_if_prior_on_device), so
    #                     # 'auto' inherits the nn device; 'cpu' is allowed for
    #                     # CPU-only runs.
    def _nn_device(self) -> str:
        devices = self.config.get("devices", {})
        return _resolve_policy_device(
            devices.get("nn"),
            self.device,
            "nn",
        )

    def _task_device(self, method: str) -> str:
        """Device for the task (prior, simulator) used by a method."""
        devices = self.config.get("devices", {})
        if method == "SMC_ABC":
            device = devices.get("smc_abc", "cpu")
            if device not in {None, "auto", "cpu"}:
                raise ValueError(
                    f"devices.smc_abc must be 'auto' or 'cpu', got {device!r}; "
                    "sbi SMCABC uses numpy internally and needs CPU tensors."
                )
            return "cpu"
        return self._nn_device()

    def _sbi_prior_device(self) -> str:
        """Device for the sbi prior object handed to SNPE_C.

        sbi(>=0.21) requires the prior to live on the training device:
        ``check_if_prior_on_device`` asserts ``prior.sample((1,))`` lands on
        it. 'auto' therefore inherits the neural-network device. The earlier
        worry about CUDA tensors in ``simulate_for_sbi`` applies only to
        SMCABC (its numpy kernel machinery) which is covered by
        ``devices.smc_abc`` and stays CPU.
        """
        devices = self.config.get("devices", {})
        return _resolve_policy_device(
            devices.get("sbi_prior"),
            self._nn_device(),
            "sbi_prior",
        )

    # -- SFMPE configuration resolution ---
    #
    # 'sfmpe_configs' maps each task name to a per-task SFMPE YAML file
    # (flow/proposal hyperparameters live with the task). 'sfmpe_config' is
    # an explicit shared single-file alternative for custom configs; there is
    # no implicit default file, so a missing entry fails loudly.
    def _sfmpe_config_path(self, task_name: str) -> Path:
        sfmpe_configs = self.config.get("sfmpe_configs", {})
        if task_name in sfmpe_configs:
            return (self.root / sfmpe_configs[task_name]).resolve()
        shared = self.config.get("sfmpe_config")
        if shared is not None:
            return (self.root / shared).resolve()
        raise ValueError(
            f"No SFMPE pipeline config for task {task_name!r}: add it to "
            "'sfmpe_configs' in the benchmark YAML, or set a shared "
            "'sfmpe_config' file."
        )

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

    def _make_task(self, task_name: str, device: str | None = None):
        task_type = TASK_TYPES.get(task_name.casefold())
        if task_type is None:
            raise ValueError(f"Unsupported task {task_name!r}.")
        return task_type(
            self._task_config(task_name),
            device=device or self.device,
        )

    def _make_simulator_callable(self, task):
        def simulate_and_summarize(theta: torch.Tensor) -> torch.Tensor:
            return task.summarize(task.simulate(theta))

        return simulate_and_summarize

    def _training_block(self, name: str) -> dict:
        """Per-method training settings with legacy fallback.

        Canonical configs split estimator tuning into ``sfmpe_training``
        (SFMPE's ``fm_estimator.train`` kwargs) and ``snpe_c_training``
        (sbi SNPE-C ``train`` kwargs). Older configs that only declared the
        merged ``training`` block keep working: the block is read for both
        methods when the specific one is absent.
        """
        block = self.config.get(name, self.config.get("training", {}))
        return dict(block or {})

    def _fit_sfmpe(
        self,
        task_name: str,
        task,
        observation: torch.Tensor,
        budget: int,
        rounds: int,
        meter: _SimulatorCallMeter,
    ):
        meter.set_phase("setup")
        sfmpe_config = self._sfmpe_config_path(task_name)
        pipeline = SFMPE(task, sfmpe_config, device=self._nn_device())
        proposal_method = self.config.get(
            "sfmpe_proposal_methods",
            {},
        ).get(task_name)
        if proposal_method is not None:
            pipeline.proposal_params.method = proposal_method
            pipeline.manager.proposal_params.method = proposal_method
        pipeline.proposal_params.x_0 = observation.unsqueeze(0).to(task.device)
        remaining_budget = budget - meter.counts["setup"]
        if remaining_budget < 1:
            raise ValueError(
                "The simulation budget must leave at least one draw after "
                "SFMPE initialization."
            )
        round_budgets = _balanced_round_budgets(
            remaining_budget,
            rounds,
        )
        # Forwarded to fm_estimator.train (Phase 1 knobs only; `device` is
        # pinned by the device policy, not configurable here). `epochs` maps
        # to fm_estimator's max_epochs.
        training = self._training_block("sfmpe_training")
        train_kwargs = dict(
            epochs=training.get("epochs", 20),
            batch_size=training.get("batch_size", 64),
            shuffle=training.get("shuffle", True),
            num_workers=training.get("num_workers", 0),
            show_every=training.get("show_every"),
            cfg_mode=training.get("cfg_mode", False),
        )
        checkpoint_path = training.get("path")
        if checkpoint_path is not None:
            train_kwargs["path"] = checkpoint_path
        train_kwargs["validation_fraction"] = training.get(
            "validation_fraction",
            0.1,
        )
        train_kwargs["stop_after_epochs"] = training.get(
            "stop_after_epochs",
            20,
        )
        train_kwargs["min_delta"] = training.get("min_delta", 0.0)
        if training.get("validation_seed") is not None:
            train_kwargs["validation_seed"] = training["validation_seed"]
        nn_device = self._nn_device()
        # Clean sampling comes from the per-task SFMPE config
        # (run_sequential.clean_sampling). 'Truncated' proposals implementing
        # it compute proposal log-probs via the ODE path, which can underflow
        # ("underflow in dt") on early untrained flows; per-task configs may
        # disable it where that path is not reliable.
        clean_sampling = bool(
            pipeline.config.get("run_sequential", {}).get(
                "clean_sampling",
                True,
            )
        )
        # Same fast-fail as SFMPE.run: the benchmark bypasses run() and
        # drives manager.run_round_experimental directly, so a Truncated +
        # clean-sampling task without a quantile would otherwise die with an
        # opaque torch.quantile error mid-run.
        if (
            clean_sampling
            and pipeline.proposal_params.method == "Truncated"
            and "quantile" not in pipeline.proposal_params.method_params
        ):
            raise ValueError(
                f"'proposal.params.quantile' is required for clean-sampled "
                f"Truncated proposals (task {task_name}); provide "
                "proposal.params or set clean_sampling: false"
            )
        meter.set_phase("inference")
        for round_id, round_budget in enumerate(round_budgets):
            result = pipeline.manager.run_round_experimental(
                round_id,
                round_budget,
                sims_per_sample=1,
                clean_sampling=clean_sampling,
            )
            if result == -1:
                raise RuntimeError(
                    f"SFMPE stopped in round {round_id} for {task_name}."
                )
            # Cumulative training protocol: train on every simulation stored
            # so far (round 0 only, then rounds {0,1}, then {0,1,2}, ...).
            # Reusing already-simulated draws costs no budget, and it mirrors
            # how sbi's sequential SNPE-C trains on its accumulated dataset
            # at equal simulation budgets.
            round_kwargs = dict(train_kwargs)
            round_kwargs["batch_size"] = min(round_kwargs["batch_size"], round_budget)
            pipeline.manager.train_estimator(
                list(range(round_id + 1)),
                device=nn_device,
                **round_kwargs,
            )
            posterior = pipeline.manager.build_posterior(update_x=False)
            pipeline.manager.update_proposal(posterior)
        meter.set_phase("sampling")
        return (
            self._sfmpe_sampler(pipeline, task, nn_device),
            pipeline.proposal_params.method,
        )

    @staticmethod
    def _sfmpe_sampler(pipeline, task, nn_device: str):
        """Build a posterior sampler bound to the pipeline's current proposal."""

        def sample_posterior(x: torch.Tensor, num_samples: int) -> torch.Tensor:
            pipeline.manager.proposal.params.x_0 = x.reshape(1, -1).to(
                task.device
            )
            samples = pipeline.manager.proposal.sample(
                (num_samples,),
                device=nn_device,
            )
            if samples.ndim == 3 and samples.shape[1] == 1:
                samples = samples[:, 0, :]
            return samples

        return sample_posterior

    def _fit_snpe_c(
        self,
        task_name: str,
        task,
        observation: torch.Tensor,
        budget: int,
        rounds: int,
        meter: _SimulatorCallMeter,
    ):
        from sbi.inference import SNPE_C, simulate_for_sbi

        # Forwarded to sbi SNPE-C: `append_simulations(exclude_invalid_x=...)`
        # and `train(**kwargs)`. `max_num_epochs` honors the legacy `epochs`
        # key from the pre-split merged block; `training_batch_size` honors
        # the legacy `batch_size` key.
        training = self._training_block("snpe_c_training")
        max_num_epochs = training.get(
            "max_num_epochs",
            training.get("epochs", 20),
        )
        training_batch_size = training.get(
            "training_batch_size",
            training.get("batch_size", 64),
        )
        learning_rate = training.get("learning_rate", 5e-4)
        validation_fraction = training.get("validation_fraction", 0.1)
        stop_after_epochs = training.get("stop_after_epochs", 8)
        num_atoms = training.get("num_atoms", 10)
        clip_max_norm = training.get("clip_max_norm", 5.0)
        resume_training = training.get("resume_training", False)
        force_first_round_loss = training.get("force_first_round_loss", False)
        discard_prior_samples = training.get("discard_prior_samples", False)
        use_combined_loss = training.get("use_combined_loss", False)
        retrain_from_scratch = training.get("retrain_from_scratch", False)
        show_train_summary = training.get("show_train_summary", False)
        exclude_invalid_x = training.get("exclude_invalid_x")
        # Optional dict of extra DataLoader kwargs (e.g. num_workers) handed
        # to sbi's train() as-is; None keeps the sbi default.
        dataloader_kwargs = training.get("dataloader_kwargs")
        # calibration_kernel is a Python callable and deliberately stays out
        # of YAML configurability: sbi's calibration_kernel=None default is
        # used unless passed explicitly (null in YAML == sbi default).
        calibration_kernel = training.get("calibration_kernel")
        nn_device = self._nn_device()
        prior = self._make_task(
            task_name,
            device=self._sbi_prior_device(),
        ).prior
        inference = SNPE_C(
            prior=prior,
            device=nn_device,
            show_progress_bars=False,
        )
        task_simulator = self._make_simulator_callable(task)

        def simulator(theta: torch.Tensor) -> torch.Tensor:
            observations = task_simulator(theta.to(task.device))
            return observations.to("cpu")

        proposal = prior
        posterior = None
        density_estimator = None
        round_budgets = _balanced_round_budgets(
            budget,
            rounds,
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
                exclude_invalid_x=exclude_invalid_x,
            )
            round_training_batch_size = min(
                cast(int, training_batch_size),
                max(2, round_budget),
            )
            density_estimator = appended.train(
                num_atoms=cast(int, num_atoms),
                training_batch_size=round_training_batch_size,
                learning_rate=cast(float, learning_rate),
                validation_fraction=cast(float, validation_fraction),
                stop_after_epochs=cast(int, stop_after_epochs),
                max_num_epochs=cast(int, max_num_epochs),
                clip_max_norm=cast(Any, clip_max_norm),
                calibration_kernel=cast(Any, calibration_kernel),
                resume_training=cast(bool, resume_training),
                force_first_round_loss=cast(bool, force_first_round_loss),
                discard_prior_samples=cast(bool, discard_prior_samples),
                use_combined_loss=cast(bool, use_combined_loss),
                retrain_from_scratch=cast(bool, retrain_from_scratch),
                show_train_summary=cast(bool, show_train_summary),
                dataloader_kwargs=cast(Any, dataloader_kwargs),
            )
            posterior = inference.build_posterior(
                density_estimator,
                sample_with="rejection",
            )
            proposal = posterior.set_default_x(observation)

        if posterior is None or density_estimator is None:
            raise RuntimeError("SNPE-C did not complete any training rounds.")
        meter.set_phase("sampling")
        return self._snpe_c_sampler(posterior, nn_device)

    @staticmethod
    def _snpe_c_sampler(posterior, nn_device: str):
        """Build a posterior sampler bound to one SNPE-C round's posterior."""

        def sample_posterior(x: torch.Tensor, num_samples: int) -> torch.Tensor:
            result = posterior.sample(
                (num_samples,),
                x=x.to(nn_device),
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
        rounds: int,
        meter: _SimulatorCallMeter,
    ):
        from sbi.inference import SMCABC

        # Single-shot method: `rounds` is ignored and the runner records
        # rounds=None for its one evaluation row.

        settings = dict(self.config.get("smc_abc", {}))
        if budget < 20:
            raise ValueError(
                "SMC-ABC requires a budget of at least 20 simulations for "
                "the SBI KDE posterior."
            )
        # Full sbi SMCABC.__call__ knob set. num_simulations is pinned to the
        # metered budget and return_summary stays False (benchmark needs a
        # single posterior-distribution object), so those are not exposed.
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
        call_kwargs = dict(
            epsilon_decay=settings.get("epsilon_decay", 0.5),
            distance_based_decay=settings.get("distance_based_decay", False),
            ess_min=settings.get("ess_min"),
            kernel_variance_scale=settings.get("kernel_variance_scale", 1.0),
            use_last_pop_samples=settings.get("use_last_pop_samples", True),
            kde=settings.get("kde", True),
            kde_kwargs=settings.get("kde_kwargs", {}),
            kde_sample_weights=settings.get("kde_sample_weights", True),
            lra=settings.get("lra", False),
            lra_with_weights=settings.get("lra_with_weights", False),
            sass=settings.get("sass", False),
            sass_fraction=settings.get("sass_fraction", 0.25),
            sass_expansion_degree=settings.get("sass_expansion_degree", 1),
        )
        simulator = self._make_simulator_callable(task)

        def infer(x: torch.Tensor):
            smc = SMCABC(
                simulator=simulator,
                prior=task.prior,
                num_workers=settings.get("num_workers", 1),
                show_progress_bars=False,
            )
            return smc(
                x.to("cpu"),
                num_particles=particles,
                num_initial_pop=initial_population,
                num_simulations=budget,
                **call_kwargs,
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
            return result.to("cpu")

        return sample_posterior

    def _fit(
        self,
        task_name: str,
        method: str,
        task,
        observation: torch.Tensor,
        budget: int,
        rounds: int | None,
        meter: _SimulatorCallMeter,
    ):
        if method == "SFMPE":
            return self._fit_sfmpe(
                task_name, task, observation, budget, cast(int, rounds), meter
            )
        if method == "SNPE_C":
            return self._fit_snpe_c(
                task_name, task, observation, budget, cast(int, rounds), meter
            )
        if method == "SMC_ABC":
            return self._fit_smc_abc(task, observation, budget, 0, meter)
        raise ValueError(f"Unsupported method {method!r}.")

    def _evaluate_target(
        self,
        task_name: str,
        method: str,
        budget: int,
        rounds: int | None,
        replicate: int,
        seed: int,
        observation: torch.Tensor,
        observation_scale: torch.Tensor,
    ) -> tuple[dict[str, Any], torch.Tensor]:
        """Fit one target for a fixed number of rounds and score it.

        The simulation budget is split evenly across ``rounds`` rounds
        (e.g. 1000 -> [1000], [500, 500], [334, 333, 333] for 1/2/3 rounds),
        so every point on the rounds axis consumes the same total budget.
        """
        _set_seed(seed)
        task = self._make_task(
            task_name,
            device=self._task_device(method),
        )
        meter = _SimulatorCallMeter(task.simulator)
        inference_start = time.perf_counter()
        fit_result = self._fit(
            task_name,
            method,
            task,
            observation,
            budget,
            rounds,
            meter,
        )
        resolved_sfmpe_method: str | None = None
        if isinstance(fit_result, tuple):
            sample_posterior, resolved_sfmpe_method = fit_result
        else:
            sample_posterior = fit_result
        inference_seconds = time.perf_counter() - inference_start
        inference_calls = meter.counts["setup"] + meter.counts["inference"]
        if inference_calls > budget:
            raise RuntimeError(
                f"{method} used {inference_calls} simulator draws, "
                f"exceeding budget {budget}."
            )
        return self._score_posterior(
            task_name=task_name,
            method=method,
            budget=budget,
            replicate=replicate,
            seed=seed,
            task=task,
            observation=observation,
            observation_scale=observation_scale,
            meter=meter,
            sample_posterior=sample_posterior,
            resolved_sfmpe_method=resolved_sfmpe_method,
            rounds_value=rounds,
            inference_seconds=inference_seconds,
            inference_calls=inference_calls,
        )

    def _score_posterior(
        self,
        *,
        task_name: str,
        method: str,
        budget: int,
        replicate: int,
        seed: int,
        task,
        observation: torch.Tensor,
        observation_scale: torch.Tensor,
        meter: _SimulatorCallMeter,
        sample_posterior,
        resolved_sfmpe_method: str | None,
        rounds_value: int | None,
        inference_seconds: float,
        inference_calls: int,
    ) -> tuple[dict[str, Any], torch.Tensor]:
        counts_before = dict(meter.counts)
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
            # Number of sequential rounds this model was trained with
            # (None for single-shot SMC-ABC).
            "rounds": rounds_value,
            "seed": seed,
            "inference_simulator_draws": inference_calls,
            "setup_simulator_draws": meter.counts["setup"],
            "sfmpe_proposal_method": (
                resolved_sfmpe_method if method == "SFMPE" else None
            ),
            "ppc_simulator_draws": (
                meter.counts["ppc"] - counts_before.get("ppc", 0)
            ),
            "sensitivity_simulator_draws": (
                meter.counts["sensitivity"]
                - counts_before.get("sensitivity", 0)
            ),
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
        configured_rounds = _normalized_rounds(self.config.get("rounds", 3))
        posterior_rows: dict[tuple, dict[str, Any]] = {}
        summaries: dict[tuple, dict[str, Any]] = {}
        # Resume: keep rows from earlier (partial) runs; per-target skip marks
        # a replicate as already evaluated, so shards can be re-launched after
        # Kaggle's wall-clock cutoff.
        for row in self._load_jsonl("target_metrics.jsonl"):
            posterior_rows[_target_key(row)] = row
        for row in self._load_jsonl("benchmark_summary.jsonl"):
            summaries[_summary_key(row)] = row
        completed_targets = {
            (
                row["task"],
                row["method"],
                int(row["budget"]),
                row.get("rounds"),
                int(row["replicate"]),
            )
            for row in posterior_rows.values()
        }
        if completed_targets:
            print(
                f"Resuming: {len(completed_targets)} target(s) already recorded "
                f"in {self.output_dir}"
            )

        deadline = (
            time.perf_counter() + self.time_limit_seconds
            if self.time_limit_seconds is not None
            else None
        )
        stopped = False

        def time_exceeded() -> bool:
            return deadline is not None and time.perf_counter() >= deadline

        for task_index, task_name in enumerate(task_names):
            target_seed = int(self.config.get("seed", 0)) + task_index * 100_000
            _set_seed(target_seed)
            target_task = self._make_task(task_name, device=self._nn_device())
            target_thetas = target_task.sample_prior((replicates,)).detach().cpu()
            target_observations = target_task.summarize(
                target_task.simulate(target_thetas.to(self._nn_device()))
            ).detach().cpu()
            observation_scale = target_observations.std(
                dim=0,
                unbiased=False,
            ).clamp_min(1e-6)

            for method_index, method in enumerate(methods):
                if stopped:
                    break
                # SMC-ABC is single-shot: one experiment with rounds=None.
                method_rounds = (
                    [None] if method == "SMC_ABC" else configured_rounds
                )
                for budget_index, budget in enumerate(budgets):
                    if stopped:
                        break
                    for rounds_value in method_rounds:
                        if stopped:
                            break
                        rows: list[dict[str, Any]] = []
                        sample_sets: list[torch.Tensor] = []
                        evaluated = 0
                        skipped = 0
                        for replicate in range(replicates):
                            if time_exceeded():
                                stopped = True
                                break
                            if (
                                task_name,
                                method,
                                budget,
                                rounds_value,
                                replicate,
                            ) in completed_targets:
                                skipped += 1
                                continue
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
                                rounds_value,
                                replicate,
                                seed,
                                target_observations[replicate],
                                observation_scale,
                            )
                            evaluated += 1
                            rows.append(row)
                            sample_sets.append(samples)
                            posterior_rows[_target_key(row)] = row
                            self._write_jsonl(
                                "target_metrics.jsonl",
                                list(posterior_rows.values()),
                            )
                            print(
                                f"{task_name} {method} budget={budget} "
                                f"rounds={rounds_value} "
                                f"target={replicate + 1}/{replicates} "
                                f"simulations={row['inference_simulator_draws']}"
                            )

                        if evaluated == 0:
                            continue
                        if skipped:
                            # Mixed fresh/skipped replicates cannot be
                            # aggregated: calibration metrics need posterior
                            # samples that are not stored on disk.
                            print(
                                f"Skipped aggregation for {task_name} {method} "
                                f"budget={budget} rounds={rounds_value}: "
                                f"{evaluated} fresh + {skipped} resumed "
                                "replicates. Keep replicate counts fixed per "
                                "output directory."
                            )
                            continue
                        aggregate = self._aggregate(
                            task_name,
                            method,
                            budget,
                            rows,
                            sample_sets,
                            target_thetas,
                            rounds_value=rounds_value,
                        )
                        summaries[_summary_key(aggregate)] = aggregate
                        self._write_jsonl(
                            "benchmark_summary.jsonl",
                            list(summaries.values()),
                        )

        self._write_jsonl(
            "target_metrics.jsonl",
            list(posterior_rows.values()),
        )
        self._write_jsonl(
            "benchmark_summary.jsonl",
            list(summaries.values()),
        )
        if stopped:
            print(
                "Time limit reached; partial results were written. Re-run the "
                "same shard to resume (completed targets are skipped)."
            )
        return list(summaries.values())

    def _load_jsonl(self, filename: str) -> list[dict[str, Any]]:
        path = self.output_dir / filename
        if not path.is_file():
            return []
        rows = []
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    def _aggregate(
        self,
        task_name: str,
        method: str,
        budget: int,
        rows: list[dict[str, Any]],
        sample_sets: list[torch.Tensor],
        target_thetas: torch.Tensor,
        rounds_value: int | None = None,
    ) -> dict[str, Any]:
        posterior_samples = torch.stack(sample_sets, dim=1)
        aggregate: dict[str, Any] = {
            "task": task_name,
            "method": method,
            "budget": budget,
            "rounds": rounds_value,
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
    device: str | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    return BenchmarkRunner(
        config_path,
        output_dir=output_dir,
        device=device,
        overrides=overrides,
    ).run()


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
