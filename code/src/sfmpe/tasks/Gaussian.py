from collections.abc import Mapping
from typing import Any

import torch
from sbi.simulators.linear_gaussian import (
    linear_gaussian,
    true_posterior_linear_gaussian_mvn_prior,
)
from torch.distributions import MultivariateNormal

from sfmpe.core.simulator import Simulator
from sfmpe.core.summary import Summary
from sfmpe.tasks.base_task import Task
from sfmpe.utils.logger import Logger, setup_logging


class GaussianPrior(MultivariateNormal):
    """Multivariate Gaussian prior for the Gaussian-linear benchmark."""

    def __init__(self, mean: torch.Tensor, covariance: torch.Tensor):
        super().__init__(loc=mean, covariance_matrix=covariance)

    def sample(
        self,
        sample_shape: torch.Size | list[int] | tuple[int, ...] = torch.Size(),
        **kwargs,
    ) -> torch.Tensor:
        device = kwargs.pop("device", None)
        if kwargs:
            raise TypeError(
                f"sample() got unexpected keyword arguments: {', '.join(kwargs)}"
            )
        samples = super().sample(torch.Size(sample_shape))
        return samples if device is None else samples.to(device)


class GaussianSimulator(Simulator):
    """Wrapper around sbi's linear-Gaussian simulator."""

    def __init__(
        self,
        likelihood_shift: torch.Tensor,
        likelihood_covariance: torch.Tensor,
    ):
        self.likelihood_shift = likelihood_shift
        self.likelihood_covariance = likelihood_covariance
        self.data_dim = likelihood_shift.shape[-1]

    def simulate(self, theta: torch.Tensor, **kwargs) -> torch.Tensor:
        if kwargs:
            raise TypeError(
                f"simulate() got unexpected keyword arguments: {', '.join(kwargs)}"
            )
        if theta.shape[-1] < self.data_dim:
            raise ValueError("theta must have at least as many dimensions as x.")
        batch_shape = theta.shape[:-1]
        flat_theta = theta.reshape(-1, theta.shape[-1])
        shift = self.likelihood_shift.to(device=theta.device, dtype=theta.dtype)
        covariance = self.likelihood_covariance.to(
            device=theta.device, dtype=theta.dtype
        )
        observations = linear_gaussian(
            flat_theta,
            shift,
            covariance,
            num_discarded_dims=theta.shape[-1] - self.data_dim,
        )
        return observations.reshape(*batch_shape, self.data_dim)


class GaussianSummary(Summary):
    def __init__(self, data_dim: int):
        super().__init__()
        self.emb_dim = data_dim

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return data


class GaussianTask(Task):
    """Gaussian-linear SBI task with exact posterior sampling for validation."""

    summary: GaussianSummary

    def __init__(self, config: Mapping[str, Any], device: str = "cpu"):
        self.prior_parameters = dict(config.get("prior", {}))
        self.simulator_parameters = dict(config.get("simulator", {}))
        self.summary_parameters = config.get("summary", "identity")
        self.logger_config = config.get("logger")
        self.device = torch.device(device)

        prior_mean = torch.as_tensor(
            self.prior_parameters.get("mean", [0.0, 0.0]),
            dtype=torch.float32,
            device=self.device,
        )
        prior_covariance = torch.as_tensor(
            self.prior_parameters.get(
                "covariance", torch.eye(prior_mean.numel()).tolist()
            ),
            dtype=torch.float32,
            device=self.device,
        )
        if prior_mean.ndim != 1 or prior_covariance.shape != (
            prior_mean.numel(),
            prior_mean.numel(),
        ):
            raise ValueError(
                "prior.mean must be a vector and prior.covariance must be a "
                "square matrix of matching dimension."
            )
        self.theta_dim = prior_mean.numel()

        likelihood_shift = torch.as_tensor(
            self.simulator_parameters.get(
                "likelihood_shift", [0.0] * self.theta_dim
            ),
            dtype=torch.float32,
            device=self.device,
        )
        likelihood_covariance = torch.as_tensor(
            self.simulator_parameters.get(
                "likelihood_covariance",
                torch.eye(likelihood_shift.numel()).tolist(),
            ),
            dtype=torch.float32,
            device=self.device,
        )
        if likelihood_shift.ndim != 1 or likelihood_covariance.shape != (
            likelihood_shift.numel(),
            likelihood_shift.numel(),
        ):
            raise ValueError(
                "simulator.likelihood_shift must be a vector and "
                "simulator.likelihood_covariance must be a square matrix "
                "of matching dimension."
            )
        if likelihood_shift.numel() > self.theta_dim:
            raise ValueError(
                "The simulator output dimension cannot exceed theta_dim."
            )
        self.data_dim = likelihood_shift.numel()
        self.prior_mean = prior_mean
        self.prior_covariance = prior_covariance
        self.likelihood_shift = likelihood_shift
        self.likelihood_covariance = likelihood_covariance

        super().__init__(device=device)
        self.theta_dim = prior_mean.numel()
        self.data_dim = likelihood_shift.numel()
        self.summary.to(self.device)

    def build_prior(self) -> GaussianPrior:
        return GaussianPrior(self.prior_mean, self.prior_covariance)

    def build_simulator(self) -> GaussianSimulator:
        return GaussianSimulator(
            self.likelihood_shift,
            self.likelihood_covariance,
        )

    def build_summary(self) -> GaussianSummary:
        summary_type = (
            self.summary_parameters.get("type", "identity")
            if isinstance(self.summary_parameters, Mapping)
            else self.summary_parameters
        )
        if summary_type != "identity":
            raise NotImplementedError(
                f"Summary {summary_type!r} is not implemented."
            )
        return GaussianSummary(self.data_dim)

    def build_logger(self) -> Logger:
        if self.logger_config is None:
            return setup_logging(
                name="Gaussian",
                level=Logger.INFO,
                log_to_file=False,
            )
        return setup_logging(
            name=self.logger_config.get("name", "Gaussian"),
            level=getattr(
                Logger,
                self.logger_config.get("level", "INFO").upper(),
            ),
            log_to_file=self.logger_config.get("log_to_file", False),
            log_file_path=self.logger_config.get(
                "log_file_path", "logs/gaussian.log"
            ),
        )

    def sample_prior(self, size) -> torch.Tensor:
        return self.prior.sample(size)

    def simulate(self, theta: torch.Tensor) -> torch.Tensor:
        return self.simulator.simulate(theta)

    def summarize(self, x: torch.Tensor) -> torch.Tensor:
        return self.summary(x)

    def true_posterior(self, x: torch.Tensor) -> MultivariateNormal:
        """Return the exact posterior for one observation or repeated trials."""
        observation = x.to(device=self.device, dtype=self.prior_mean.dtype)
        if self.data_dim == self.theta_dim:
            return true_posterior_linear_gaussian_mvn_prior(
                observation,
                self.likelihood_shift,
                self.likelihood_covariance,
                self.prior_mean,
                self.prior_covariance,
            )

        repeated_observations = observation.reshape(-1, self.data_dim)
        design = torch.eye(
            self.theta_dim,
            device=self.device,
            dtype=self.prior_mean.dtype,
        )[: self.data_dim]
        cross_covariance = self.prior_covariance @ design.T
        innovation_covariance = (
            design @ cross_covariance
            + self.likelihood_covariance / repeated_observations.shape[0]
        )
        gain = torch.linalg.solve(
            innovation_covariance,
            cross_covariance.T,
        ).T
        posterior_mean = self.prior_mean + gain @ (
            repeated_observations.mean(dim=0)
            - self.likelihood_shift
            - design @ self.prior_mean
        )
        posterior_covariance = (
            self.prior_covariance - gain @ design @ self.prior_covariance
        )
        posterior_covariance = (
            posterior_covariance + posterior_covariance.T
        ) / 2
        return MultivariateNormal(posterior_mean, posterior_covariance)

    def sample_true_posterior(
        self,
        x: torch.Tensor,
        sample_shape: torch.Size | tuple[int, ...] = (1,),
    ) -> torch.Tensor:
        """Sample an exact posterior.

        A single observation returns ``(*sample_shape, theta_dim)``. A matrix
        of independent observations returns
        ``(num_observations, *sample_shape, theta_dim)``. For repeated trials
        of one parameter vector, call ``true_posterior(x).sample(...)``.
        """
        if x.ndim == 1:
            return self.true_posterior(x).sample(torch.Size(sample_shape))
        if x.ndim != 2:
            raise ValueError("x must have shape (data_dim,) or (batch, data_dim).")
        return torch.stack(
            [
                self.true_posterior(observation).sample(torch.Size(sample_shape))
                for observation in x
            ],
            dim=0,
        )

    def check_support(self, theta: torch.Tensor) -> torch.Tensor:
        return torch.isfinite(self.prior.log_prob(theta))
