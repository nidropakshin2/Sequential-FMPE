import torch

from sfmpe.core.distributions import Distribution
from sfmpe.core.simulator import Simulator
from sfmpe.core.summary import Summary
from sfmpe.tasks.base_task import Task
from sfmpe.utils.logger import Logger, setup_logging


class StochVolPrior(Distribution):
    def __init__(
        self,
        mu_range=(-0.5, 0.5),
        phi_range=(0.90, 0.99),
        sigma_range=(0.05, 0.5),
        m_range=(-3.0, -0.5),
    ):
        """Prior for Hull-White-style stochastic volatility.

        theta = [mu, phi, sigma, m]
          mu    - mean return
          phi   - persistence of log volatility
          sigma - volatility of log volatility
          m     - long-term mean of log volatility
        """
        super().__init__()
        self.mu_range = mu_range
        self.phi_range = phi_range
        self.sigma_range = sigma_range
        self.m_range = m_range

    def sample(self, size, **kwargs):
        """Sample parameters with independent uniform priors."""
        device = kwargs.pop("device", torch.device("cpu"))
        if kwargs:
            raise TypeError(
                f"sample() got unexpected keyword arguments: {', '.join(kwargs.keys())}"
            )

        def sample_uniform(low, high, size):
            return low + (high - low) * torch.rand(*size, device=device)

        mu = sample_uniform(self.mu_range[0], self.mu_range[1], (*size, 1))
        phi = sample_uniform(self.phi_range[0], self.phi_range[1], (*size, 1))
        sigma = sample_uniform(self.sigma_range[0], self.sigma_range[1], (*size, 1))
        m = sample_uniform(self.m_range[0], self.m_range[1], (*size, 1))

        return torch.cat([mu, phi, sigma, m], dim=-1).to(device)

    def log_prob(self, value: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError("log_prob() method is not implemented for StochVol model")


class StochVolSimulator(Simulator):
    def __init__(self, simulator_parameters={"T": 100, "v0": 0.0}, **kwargs):
        self.T = simulator_parameters["T"]
        self.v0 = simulator_parameters.get("v0", 0.0)

    def simulate(self, theta, **kwargs):
        def assign_value(x):
            if type(x) in {tuple, list}:
                return int(torch.randint(x[0], x[1], size=(1,)))
            else:
                return x

        T = kwargs.pop("T", assign_value(self.T))
        v0 = kwargs.pop("v0", assign_value(self.v0))

        mu = theta[..., 0]
        phi = theta[..., 1]
        sigma = theta[..., 2]
        m = theta[..., 3]

        batch_shape = theta.shape[:-1]
        x = torch.zeros((*batch_shape, T), device=theta.device)
        v = torch.zeros(batch_shape, device=theta.device) + v0

        for t in range(T):
            stdev = torch.clamp(torch.exp(v / 2.0), max=100)
            x[..., t] = mu + stdev * torch.randn(*batch_shape, device=theta.device)
            v = m + phi * (v - m) + sigma * torch.randn(*batch_shape, device=theta.device)

        return x


class HandmadeSummary(Summary):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.emb_dim = 5

    def forward(self, data):
        mean = data.mean(dim=-1, keepdim=True)
        std = data.std(dim=-1, unbiased=False, keepdim=True)
        log_var = torch.log(std.pow(2) + self.eps)

        centered = data - mean
        skew = (centered.pow(3).mean(dim=-1) / (std.squeeze(-1).pow(3) + self.eps)).unsqueeze(-1)
        kurt = (centered.pow(4).mean(dim=-1) / (std.squeeze(-1).pow(4) + self.eps)).unsqueeze(-1) - 3.0

        auto_cov = (centered[..., 1:] * centered[..., :-1]).mean(dim=-1)
        acf1 = (auto_cov / (std.squeeze(-1).pow(2) + self.eps)).unsqueeze(-1)

        summary = torch.cat([mean, log_var, skew, kurt, acf1], dim=-1).to(data.device)
        return summary


class StochVolTask(Task):
    def __init__(self, config, device="cpu"):
        self.device = device
        self.prior_parameters = config["prior"]
        self.simulator_parameters = config["simulator"]
        self.summary_parameters = config["summary"]
        self.logger_config = config.get("logger")
        super().__init__(device=device)

        def check_support(theta):
            # clamp_min = torch.log(torch.tensor([sir_task.prior.beta_range[0] + sir_task.prior.gamma_range[0], sir_task.prior.gamma_range[0]]))
            # clamp_max = torch.tensor([sir_task.prior.beta_range[1], sir_task.prior.gamma_range[1]])
            clamp_min = torch.tensor([self.prior_parameters[parameter][0] for parameter in self.prior_parameters.keys()])
            clamp_max = torch.tensor([self.prior_parameters[parameter][1] for parameter in self.prior_parameters.keys()])
            # WARNING: проблемы с размерностями
            mask = (theta >= clamp_min) & (theta <= clamp_max)
            while len(mask.shape) > 1:
                mask = mask.all(dim=-1)
            return mask

        self.check_support = check_support

        self.theta_dim = 4
        self.data_dim = self.summary.emb_dim

    def build_prior(self):
        return StochVolPrior(
            mu_range=self.prior_parameters.get("mu_range", (-0.5, 0.5)),
            phi_range=self.prior_parameters.get("phi_range", (0.90, 0.99)),
            sigma_range=self.prior_parameters.get("sigma_range", (0.05, 0.5)),
            m_range=self.prior_parameters.get("m_range", (-3.0, -0.5)),
        )

    def build_simulator(self):
        return StochVolSimulator(self.simulator_parameters, device=self.device)

    def build_summary(self):
        if self.summary_parameters == "handmade":
            return HandmadeSummary()
        raise NotImplementedError(f"Summary {self.summary_parameters} is not implemented")

    def build_logger(self):
        if self.logger_config is None:
            return setup_logging(name="StochVol", level=Logger.INFO, log_to_file=False)
        return setup_logging(
            name=self.logger_config["name"],
            level=getattr(Logger, self.logger_config["level"]),
            log_to_file=True,
            log_file_path=self.logger_config["log_file_path"],
        )

    def sample_prior(self, size):
        return self.prior.sample(size, device=self.device)

    def simulate(self, theta):
        return self.simulator.simulate(theta, device=self.device)

    def summarize(self, x):
        return self.summary(x)
