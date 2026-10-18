"""YAML-configured setup and execution of a sequential flow-matching run."""

from collections.abc import Mapping
from pathlib import Path as FilePath
from typing import Any

import torch
import yaml

from sfmpe.core.distributions import Normal, Uniform
from sfmpe.flow.flow_model import FlowModel
from sfmpe.flow.path import AffinePath, GaussianPath
from sfmpe.flow.velocity import SimpleVelocityField
from sfmpe.inference.fm_estimator import FlowMatchingEstimator
from sfmpe.inference.sequential.proposal import ProposalParams
from sfmpe.inference.sequential.round_manager import RoundManager
from sfmpe.tasks.base_task import Task
from sfmpe.utils.logger import Logger


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"'{name}' must be a mapping")
    return dict(value)


def _component(value: Any, name: str, default_type: str) -> tuple[str, dict[str, Any]]:
    if value is None:
        return default_type, {}
    if isinstance(value, str):
        return value, {}
    config = _mapping(value, name)
    component_type = config.pop("type", None)
    if component_type is None:
        component_type = config.pop("name", default_type)
    params = config.pop("params", {})
    if config:
        raise ValueError(
            f"Unexpected keys in '{name}': {', '.join(sorted(config))}"
        )
    return component_type, _mapping(params, f"{name}.params")


def _resolve_device(config: Any) -> str:
    if config is None or config == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if config not in {"cpu", "cuda"}:
        raise ValueError("'device' must be 'auto', 'cpu', or 'cuda'")
    if config == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return config


def _build_logger(config: Mapping[str, Any]) -> Logger:
    settings = dict(config)
    level_name = settings.pop("level", "INFO")
    if not isinstance(level_name, str):
        raise TypeError("'logger.level' must be a string")
    level = getattr(Logger, level_name.upper(), None)
    if not isinstance(level, int):
        raise ValueError(f"Unknown logger level: {level_name}")
    allowed = {
        "name",
        "log_to_file",
        "log_file_path",
        "log_to_console",
        "format_string",
    }
    unexpected = set(settings) - allowed
    if unexpected:
        raise ValueError(
            f"Unexpected logger settings: {', '.join(sorted(unexpected))}"
        )
    log_file_path = settings.get("log_file_path")
    if settings.get("log_to_file", False) and log_file_path:
        FilePath(log_file_path).parent.mkdir(parents=True, exist_ok=True)
    return Logger(log_level=level, **settings)


class SFMPE:
    """Build SFMPE objects from a YAML file and run the configured rounds.

    The task is supplied by the caller. The YAML has ``flow``, ``optimizer``,
    ``proposal``, ``run_sequential``, and optional ``initial`` and ``logger`` sections.
    Component selectors can be strings or mappings with ``type`` and ``params``.

    Example:
        pipeline = SFMPE(task, "configs/sfmpe_sir.yaml")
        pipeline.run()
    """

    def __init__(self, task: Task, config_path: str | FilePath):
        self.config_path = FilePath(config_path)
        with self.config_path.open(encoding="utf-8") as config_file:
            loaded_config = yaml.safe_load(config_file)
        self.config = _mapping(loaded_config, "configuration")

        if "task" in self.config:
            raise ValueError(
                "Do not configure 'task' in the SFMPE YAML; pass the task "
                "instance to SFMPE(task, config_path)"
            )
        self.task = task
        configured_device = self.config.get("device")
        if configured_device is None:
            self.device = getattr(task, "device", "cpu")
        else:
            self.device = _resolve_device(configured_device)
            task.device = self.device
        self.task.summary.to(self.device)
        self.logger = _build_logger(
            _mapping(self.config.get("logger", {}), "logger")
        )
        self.task.logger = self.logger
        self.logger.info("Starting %s task", task.__class__.__name__)

        flow_config = _mapping(self.config.get("flow"), "flow")
        self.flow_model = self._build_flow(flow_config)
        self.flow_model.to(self.device)
        self.logger.info("Initialized flow model")

        optimizer = self._build_optimizer(self.config.get("optimizer", {}))
        loss_fn = self._build_loss(self.config.get("loss", "MSELoss"))
        self.estimator = FlowMatchingEstimator(
            self.flow_model,
            optimizer,
            loss_fn,
            logger=self.logger,
        )

        proposal_config = _mapping(self.config.get("proposal", {}), "proposal")
        self.proposal_params = self._build_proposal_params(proposal_config)
        self.manager = RoundManager(
            self.task,
            self.estimator,
            self.proposal_params,
            device=self.device,
            logger=self.logger,
        )

    def _build_flow(self, config: Mapping[str, Any]) -> FlowModel:
        init_type, init_params = _component(
            config.get("init_dist"), "flow.init_dist", "normal"
        )
        init_name = init_type.casefold()
        init_dimensions = init_params.pop("dim", self.task.theta_dim)
        if init_dimensions != self.task.theta_dim:
            raise ValueError(
                "'flow.init_dist.params.dim' must match task.theta_dim"
            )
        init_device = init_params.pop("device", self.device)
        if init_device != self.device:
            raise ValueError(
                "'flow.init_dist.params.device' must match the pipeline device"
            )
        if init_name == "normal":
            init_dist = Normal(
                dim=self.task.theta_dim,
                device=self.device,
                **init_params,
            )
        elif init_name == "uniform":
            init_dist = Uniform(dim=self.task.theta_dim, **init_params)
        else:
            raise NotImplementedError(
                f"Initial distribution '{init_type}' is not implemented"
            )

        velocity_type, velocity_params = _component(
            config.get("velocity_model"),
            "flow.velocity_model",
            "simple",
        )
        if velocity_type.casefold() not in {
            "simple",
            "simplevelocityfield",
        }:
            raise NotImplementedError(
                f"Velocity model '{velocity_type}' is not implemented"
            )
        velocity_model = SimpleVelocityField(
            self.task.theta_dim,
            self.task.data_dim,
            **velocity_params,
        )

        path_type, path_params = _component(
            config.get("path"), "flow.path", "affine"
        )
        time_dist_config = path_params.pop("time_dist", None)
        time_dist = self._build_time_distribution(time_dist_config)
        path_name = path_type.casefold()
        if path_name == "affine":
            path = AffinePath(time_dist=time_dist, **path_params)
        elif path_name == "gaussian":
            path = GaussianPath(time_dist=time_dist, **path_params)
        else:
            raise NotImplementedError(f"Path '{path_type}' is not implemented")
        return FlowModel(velocity_model, init_dist, path)

    @staticmethod
    def _build_time_distribution(config: Any) -> Any:
        if config is None:
            return Uniform()
        distribution_type, params = _component(
            config, "flow.path.params.time_dist", "uniform"
        )
        name = distribution_type.casefold()
        if name == "uniform":
            return Uniform(**params)
        if name == "beta":
            alpha = params.pop("alpha", 1.0)
            beta = params.pop("beta", 1.0)
            if params:
                raise ValueError(
                    "Unexpected beta time distribution settings: "
                    f"{', '.join(sorted(params))}"
                )
            return torch.distributions.Beta(alpha, beta)
        raise NotImplementedError(
            f"Time distribution '{distribution_type}' is not implemented"
        )

    def _build_optimizer(self, config: Any) -> torch.optim.Optimizer:
        optimizer_type, params = _component(config, "optimizer", "Adam")
        optimizers = {
            "adam": torch.optim.Adam,
            "adamw": torch.optim.AdamW,
            "sgd": torch.optim.SGD,
        }
        optimizer_class = optimizers.get(optimizer_type.casefold())
        if optimizer_class is None:
            raise NotImplementedError(
                f"Optimizer '{optimizer_type}' is not implemented"
            )
        task_params = _mapping(
            params.pop("task", {"summary": {"lr": 1e-4}}),
            "optimizer.params.task",
        )
        summary_params = _mapping(
            task_params.pop("summary", {"lr": 1e-4}),
            "optimizer.params.task.summary",
        )
        if task_params:
            raise ValueError(
                "Unexpected task optimizer settings: "
                f"{', '.join(sorted(task_params))}"
            )
        flow_params = _mapping(
            params.pop("flow_model", {"lr": 1e-4}),
            "optimizer.params.flow_model",
        )
        summary_parameters = list(self.task.summary.parameters())
        parameter_groups = []
        if summary_parameters:
            parameter_groups.append(
                {"params": summary_parameters, **summary_params}
            )
        parameter_groups.append(
            {"params": list(self.flow_model.parameters()), **flow_params}
        )
        return optimizer_class(parameter_groups, **params)

    @staticmethod
    def _build_loss(config: Any) -> torch.nn.Module:
        loss_type, params = _component(config, "loss", "MSELoss")
        losses = {
            "mseloss": torch.nn.MSELoss,
            "l1loss": torch.nn.L1Loss,
            "smoothl1loss": torch.nn.SmoothL1Loss,
        }
        loss_class = losses.get(loss_type.casefold())
        if loss_class is None:
            raise NotImplementedError(f"Loss '{loss_type}' is not implemented")
        return loss_class(**params)

    def _build_proposal_params(
        self, config: Mapping[str, Any]
    ) -> ProposalParams:
        method = config.get("method", "Truncated")
        supported_methods = {
            "NPE-A",
            "NPE-B",
            "fast-Truncated",
            "correct-fast-Truncated",
            "Truncated",
        }
        if method not in supported_methods:
            raise NotImplementedError(
                f"Proposal method '{method}' is not implemented"
            )
        method_params = _mapping(config.get("params", {}), "proposal.params")
        initial_value = self.config.get("initial")
        initial_config = _mapping(
            {} if initial_value is None else initial_value, "initial"
        )
        if "theta_0" in initial_config:
            theta_0 = torch.as_tensor(
                initial_config["theta_0"],
                dtype=torch.float32,
                device=self.device,
            )
            if theta_0.ndim == 1:
                theta_0 = theta_0.unsqueeze(0)
        else:
            theta_0 = self.task.sample_prior((1,)).to(self.device)
        if theta_0.shape[-1] != self.task.theta_dim:
            raise ValueError(
                "'initial.theta_0' final dimension must match task.theta_dim"
            )
        x_0 = self.task.summarize(self.task.simulate(theta_0)).to(self.device)

        params = ProposalParams()
        params.task = self.task
        params.method = method
        params.method_params = method_params
        params.theta_0 = theta_0
        params.x_0 = x_0
        params.n_steps = config.get("n_steps", 32)
        return params

    def run(self) -> Any:
        """Run the configured sequential training and return its result."""
        config = _mapping(
            self.config.get("run_sequential"), "run_sequential"
        )
        training = _mapping(config.pop("training", {}), "run_sequential.training")
        allowed = {
            "num_rounds",
            "sims_per_round",
            "clean_sampling",
            "upd_x",
            "sims_per_sample",
            "device",
        }
        unexpected = set(config) - allowed
        if unexpected:
            raise ValueError(
                "Unexpected run_sequential settings: "
                f"{', '.join(sorted(unexpected))}"
            )
        if "num_rounds" not in config or "sims_per_round" not in config:
            raise ValueError(
                "'run_sequential' requires 'num_rounds' and 'sims_per_round'"
            )
        run_device = config.setdefault("device", self.device)
        if run_device != self.device:
            raise ValueError(
                "'run_sequential.device' must match the initialized device"
            )
        if (
            config.get("clean_sampling", False)
            and self.proposal_params.method == "Truncated"
            and "quantile" not in self.proposal_params.method_params
        ):
            raise ValueError(
                "'proposal.params.quantile' is required for clean-sampled "
                "Truncated proposals"
            )
        checkpoint_path = training.get("path")
        if checkpoint_path:
            FilePath(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        return self.manager.run_sequential(**config, **training)
