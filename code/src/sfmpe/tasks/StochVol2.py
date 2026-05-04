import torch
import torch.nn as nn

from sfmpe.core.distributions import Distribution
from sfmpe.core.simulator import Simulator
from sfmpe.core.summary import Summary
from sfmpe.tasks.base_task import Task
from sfmpe.utils.logger import Logger, setup_logging


class StochVolPrior(Distribution):
    def __init__(self, config: dict):
        """
        Prior for stochastic volatility model
        theta = [mu, phi, sigma, m]
        """
        super().__init__()

        # mu ~ Normal
        self.mu_loc = config.get("mu_loc", 0.0)
        self.mu_scale = config.get("mu_scale", 0.2)

        # phi* ~ Beta(alpha, beta); phi = 2*phi* - 1
        self.phi_alpha = config.get("phi_alpha", 20.0)
        self.phi_beta = config.get("phi_beta", 1.5)

        # sigma^2 ~ Inverse-Gamma(shape, scale)
        self.sigma_ig_shape = config.get("sigma_ig_shape", 2.5)
        self.sigma_ig_scale = config.get("sigma_ig_scale", 0.025)

        # m ~ Normal
        self.m_loc = config.get("m_loc", -1.5)
        self.m_scale = config.get("m_scale", 1.0)

    def sample(self, size, **kwargs):
        device = kwargs.pop("device", torch.device("cpu"))
        if kwargs:
            raise TypeError(
                f"sample() got unexpected keyword arguments: "
                f"{', '.join(kwargs.keys())}"
            )

        # mu ~ Normal
        mu = self.mu_loc + self.mu_scale * torch.randn(*size, 1, device=device)

        # phi* ~ Beta(alpha, beta), phi = 2*phi* - 1
        beta_dist = torch.distributions.Beta(self.phi_alpha, self.phi_beta)
        phi_star = beta_dist.sample((*size, 1)).to(device)
        phi = 2.0 * phi_star - 1.0

        # sigma^2 ~ Inverse-Gamma(shape, scale)
        gamma_dist = torch.distributions.Gamma(
            concentration=self.sigma_ig_shape,
            rate=self.sigma_ig_scale
        )
        sigma2 = 1.0 / gamma_dist.sample((*size, 1)).to(device)
        sigma = torch.sqrt(sigma2)

        # m ~ Normal
        m = self.m_loc + self.m_scale * torch.randn(*size, 1, device=device)

        return torch.cat([mu, phi, sigma, m], dim=-1)

    def log_prob(self, value: torch.Tensor, **kwargs):
        raise NotImplementedError("log_prob() is not implemented for StochVolPrior")


class StochVolSimulator(Simulator):
    def __init__(self, config: dict):
        self.T = config.get("T", 100)
        self.v0 = config.get("v0", 0.0)
        # Дополнительные параметры, которые могут быть переопределены при вызове simulate
        self._extra = config.get("extra", {})

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
    def __init__(self, config: dict):
        super().__init__()
        self.eps = config.get("eps", 1e-8)
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


class LSTMSummary(Summary):
    def __init__(self, config: dict):
        super().__init__()
        self.input_dim = config.get("input_dim", 1)
        self.hidden_dim = config.get("hidden_dim", 32)
        self.num_layers = config.get("num_layers", 2)
        self.output_dim = config.get("output_dim", 5)
        self.dropout = config.get("dropout", 0.1)
        self.emb_dim = self.output_dim

        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
        )

        self.fc = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.output_dim),
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Forward pass through LSTM summary network.

        Args:
            data: Input time series, shape (batch, T) or (batch, T, input_dim)

        Returns:
            Summary embeddings, shape (batch, output_dim)
        """
        if data.dim() == 2:
            data = data.unsqueeze(-1)
        lstm_out, (h_n, _) = self.lstm(data)
        last_hidden = h_n[-1]
        summary = self.fc(last_hidden)
        return summary


class StochVolTask(Task):
    def __init__(self, config, device="cpu"):
        self.device = device
        self.prior_parameters = config.get("prior", {})
        self.simulator_parameters = config.get("simulator", {})
        self.summary_parameters = config.get("summary", {})
        self.logger_config = config.get("logger")
        super().__init__(device=device)

        def check_support(theta):
            clamp_min = torch.tensor([self.prior_parameters.get(parameter, [0])[0] for parameter in self.prior_parameters.keys()])
            clamp_max = torch.tensor([self.prior_parameters.get(parameter, [0])[1] for parameter in self.prior_parameters.keys()])
            mask = (theta >= clamp_min) & (theta <= clamp_max)
            while len(mask.shape) > 1:
                mask = mask.all(dim=-1)
            return mask

        self.check_support = check_support
        self.theta_dim = 4
        self.data_dim = self.summary.emb_dim

    def build_prior(self):
        return StochVolPrior(self.prior_parameters)

    def build_simulator(self):
        return StochVolSimulator(self.simulator_parameters)

    def build_summary(self):
        if isinstance(self.summary_parameters, str) and self.summary_parameters == "handmade":
            return HandmadeSummary(self.summary_parameters if isinstance(self.summary_parameters, dict) else {})
        elif isinstance(self.summary_parameters, str) and self.summary_parameters == "lstm":
            return LSTMSummary(self.summary_parameters if isinstance(self.summary_parameters, dict) else {})
        elif isinstance(self.summary_parameters, dict):
            # Если в конфиге явно указан тип summary
            summary_type = self.summary_parameters.get("type", "handmade")
            if summary_type == "handmade":
                return HandmadeSummary(self.summary_parameters)
            elif summary_type == "lstm":
                return LSTMSummary(self.summary_parameters)
        raise NotImplementedError(f"Summary {self.summary_parameters} is not implemented")

    def build_logger(self):
        if self.logger_config is None:
            return setup_logging(name="StochVol", level=Logger.INFO, log_to_file=False)
        return setup_logging(
            name=self.logger_config.get("name", "StochVol"),
            level=getattr(Logger, self.logger_config.get("level", "INFO")),
            log_to_file=True,
            log_file_path=self.logger_config.get("log_file_path", "stochvol.log"),
        )

    def sample_prior(self, size):
        return self.prior.sample(size, device=self.device)

    def simulate(self, theta):
        return self.simulator.simulate(theta, device=self.device)

    def summarize(self, x):
        return self.summary(x)