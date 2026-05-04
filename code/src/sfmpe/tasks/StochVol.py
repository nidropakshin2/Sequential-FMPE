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

    def check_support(self, theta: torch.Tensor) -> torch.Tensor:
        
        clamp_min = torch.tensor([-1, 0.25, 0.05, -5])
        clamp_max = torch.tensor([1, 1, 0.8, 2.5])

        mask = (clamp_min <= theta) & (theta <= clamp_max)
        while len(mask.shape) > 1:
            mask = mask.all(dim=-1)

        return mask

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

        return x.unsqueeze(-1)


class HandmadeSummary(Summary):
    def __init__(self, config={}):
        super().__init__()
        self.eps = config.get("eps", 1e-8)
        self.emb_dim = 5

    def forward(self, data):
        data = data.squeeze(-1)
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
        original_shape = data.shape
        batch_dims = original_shape[:-2]
        T = original_shape[-2]
        input_dim = original_shape[-1]

        flat_data = data.view(-1, T, input_dim)   # (total_batch, T, input_dim)

        lstm_out, (h_n, _) = self.lstm(flat_data)   # lstm_out: (total_batch, T, hidden_dim)
        last_hidden = h_n[-1]   # (total_batch, hidden_dim)
        summary_flat = self.fc(last_hidden)   # (total_batch, output_dim)
        summary = summary_flat.view(*batch_dims, self.output_dim)

        return summary


class StochVolTask(Task):
    def __init__(self, config, device="cpu"):
        self.device = device
        self.prior_parameters = config.get("prior", {})
        self.simulator_parameters = config.get("simulator", {})
        self.summary_parameters = config.get("summary", {})
        self.logger_config = config.get("logger")
        super().__init__(device=device)

        # def check_support(theta):
        #     clamp_min = torch.tensor([self.prior_parameters.get(parameter, [0])[0] for parameter in self.prior_parameters.keys()])
        #     clamp_max = torch.tensor([self.prior_parameters.get(parameter, [0])[1] for parameter in self.prior_parameters.keys()])
        #     mask = (theta >= clamp_min) & (theta <= clamp_max)
        #     while len(mask.shape) > 1:
        #         mask = mask.all(dim=-1)
        #     return mask

        self.check_support = self.prior.check_support
        self.theta_dim = 4
        self.data_dim = self.summary.emb_dim

    def build_prior(self):
        return StochVolPrior(self.prior_parameters)

    def build_simulator(self):
        return StochVolSimulator(self.simulator_parameters)

    def build_summary(self):
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


# ---------------------- TESTING ----------------------


# import torch
# import torch.nn as nn

# from sfmpe.core.distributions import Distribution
# from sfmpe.core.simulator import Simulator
# from sfmpe.core.summary import Summary
# from sfmpe.tasks.base_task import Task
# from sfmpe.utils.logger import Logger, setup_logging


# # class StochVolPrior(Distribution):
# #     def __init__(
# #         self,
# #         mu_range=(-0.5, 0.5),
# #         phi_range=(0.90, 0.99),
# #         sigma_range=(0.05, 0.5),
# #         m_range=(-3.0, -0.5),
# #     ):
# #         """Prior for Hull-White-style stochastic volatility.

# #         theta = [mu, phi, sigma, m]
# #           mu    - mean return
# #           phi   - persistence of log volatility
# #           sigma - volatility of log volatility
# #           m     - long-term mean of log volatility
# #         """
# #         super().__init__()
# #         self.mu_range = mu_range
# #         self.phi_range = phi_range
# #         self.sigma_range = sigma_range
# #         self.m_range = m_range

# #     def sample(self, size, **kwargs):
# #         """Sample parameters with independent uniform priors."""
# #         device = kwargs.pop("device", torch.device("cpu"))
# #         if kwargs:
# #             raise TypeError(
# #                 f"sample() got unexpected keyword arguments: {', '.join(kwargs.keys())}"
# #             )

# #         def sample_uniform(low, high, size):
# #             return low + (high - low) * torch.rand(*size, device=device)

# #         mu = sample_uniform(self.mu_range[0], self.mu_range[1], (*size, 1))
# #         phi = sample_uniform(self.phi_range[0], self.phi_range[1], (*size, 1))
# #         sigma = sample_uniform(self.sigma_range[0], self.sigma_range[1], (*size, 1))
# #         m = sample_uniform(self.m_range[0], self.m_range[1], (*size, 1))

# #         return torch.cat([mu, phi, sigma, m], dim=-1).to(device)

# #     def log_prob(self, value: torch.Tensor, **kwargs) -> torch.Tensor:
# #         raise NotImplementedError("log_prob() method is not implemented for StochVol model")


# class StochVolPrior(Distribution):
#     def __init__(
#         self,
#         mu_loc=0.0,
#         mu_scale=0.2,
#         phi_alpha=20.0,
#         phi_beta=1.5,
#         sigma_ig_shape=2.5,   # corresponds to IG(5/2, ...)
#         sigma_ig_scale=0.025, # corresponds to scale = 0.05 / 2
#         m_loc=-1.5,
#         m_scale=1.0,
#     ):
#         """
#         Prior for stochastic volatility model

#         theta = [mu, phi, sigma, m]

#         mu    : mean return
#         phi   : persistence of log volatility
#         sigma : volatility of volatility
#         m     : long-run mean of log volatility
#         """
#         super().__init__()
        
#         self.mu_loc = mu_loc
#         self.mu_scale = mu_scale

#         self.phi_alpha = phi_alpha
#         self.phi_beta = phi_beta

#         self.sigma_ig_shape = sigma_ig_shape
#         self.sigma_ig_scale = sigma_ig_scale

#         self.m_loc = m_loc
#         self.m_scale = m_scale

#     def sample(self, size, **kwargs):
#         device = kwargs.pop("device", torch.device("cpu"))
#         if kwargs:
#             raise TypeError(
#                 f"sample() got unexpected keyword arguments: "
#                 f"{', '.join(kwargs.keys())}"
#             )

#         # mu ~ Normal
#         mu = (
#             self.mu_loc
#             + self.mu_scale * torch.randn(*size, 1, device=device)
#         )

#         # phi* ~ Beta(alpha, beta), phi = 2*phi* - 1
#         beta_dist = torch.distributions.Beta(
#             self.phi_alpha,
#             self.phi_beta
#         )
#         phi_star = beta_dist.sample((*size, 1)).to(device)
#         phi = 2.0 * phi_star - 1.0

#         # sigma^2 ~ Inverse-Gamma(shape, scale)
#         # implemented via:
#         # if X ~ Gamma(shape, rate=scale)
#         # then 1/X ~ InvGamma(shape, scale)
#         gamma_dist = torch.distributions.Gamma(
#             concentration=self.sigma_ig_shape,
#             rate=self.sigma_ig_scale
#         )
#         sigma2 = 1.0 / gamma_dist.sample((*size, 1)).to(device)
#         sigma = torch.sqrt(sigma2)

#         # m ~ Normal
#         m = (
#             self.m_loc
#             + self.m_scale * torch.randn(*size, 1, device=device)
#         )

#         return torch.cat([mu, phi, sigma, m], dim=-1)

#     def log_prob(self, value: torch.Tensor, **kwargs):
#         raise NotImplementedError("log_prob() is not implemented for StochVolPrior")



# class StochVolSimulator(Simulator):
#     def __init__(self, simulator_parameters={"T": 100, "v0": 0.0}, **kwargs):
#         self.T = simulator_parameters["T"]
#         self.v0 = simulator_parameters.get("v0", 0.0)

#     def simulate(self, theta, **kwargs):
#         def assign_value(x):
#             if type(x) in {tuple, list}:
#                 return int(torch.randint(x[0], x[1], size=(1,)))
#             else:
#                 return x

#         T = kwargs.pop("T", assign_value(self.T))
#         v0 = kwargs.pop("v0", assign_value(self.v0))

#         mu = theta[..., 0]
#         phi = theta[..., 1]
#         sigma = theta[..., 2]
#         m = theta[..., 3]

#         batch_shape = theta.shape[:-1]
#         x = torch.zeros((*batch_shape, T), device=theta.device)
#         v = torch.zeros(batch_shape, device=theta.device) + v0

#         for t in range(T):
#             stdev = torch.clamp(torch.exp(v / 2.0), max=100)
#             x[..., t] = mu + stdev * torch.randn(*batch_shape, device=theta.device)
#             v = m + phi * (v - m) + sigma * torch.randn(*batch_shape, device=theta.device)

#         return x

# class HandmadeSummary(Summary):
#     def __init__(self, eps=1e-8):
#         super().__init__()
#         self.eps = eps
#         self.emb_dim = 5

#     def forward(self, data):
#         mean = data.mean(dim=-1, keepdim=True)
#         std = data.std(dim=-1, unbiased=False, keepdim=True)
#         log_var = torch.log(std.pow(2) + self.eps)

#         centered = data - mean
#         skew = (centered.pow(3).mean(dim=-1) / (std.squeeze(-1).pow(3) + self.eps)).unsqueeze(-1)
#         kurt = (centered.pow(4).mean(dim=-1) / (std.squeeze(-1).pow(4) + self.eps)).unsqueeze(-1) - 3.0

#         auto_cov = (centered[..., 1:] * centered[..., :-1]).mean(dim=-1)
#         acf1 = (auto_cov / (std.squeeze(-1).pow(2) + self.eps)).unsqueeze(-1)

#         summary = torch.cat([mean, log_var, skew, kurt, acf1], dim=-1).to(data.device)
#         return summary


# class LSTMSummary(Summary):
#     """LSTM-based learned summary network for stochastic volatility time series."""

#     def __init__(
#         self,
#         input_dim: int = 1,
#         hidden_dim: int = 32,
#         num_layers: int = 2,
#         output_dim: int = 5,
#         dropout: float = 0.1,
#     ):
#         super().__init__()
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.num_layers = num_layers
#         self.output_dim = output_dim
#         self.emb_dim = output_dim

#         self.lstm = nn.LSTM(
#             input_size=input_dim,
#             hidden_size=hidden_dim,
#             num_layers=num_layers,
#             batch_first=True,
#             dropout=dropout if num_layers > 1 else 0.0,
#         )

#         self.fc = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, output_dim),
#         )

#     def forward(self, data: torch.Tensor) -> torch.Tensor:
#         """Forward pass through LSTM summary network.

#         Args:
#             data: Input time series, shape (batch, T) or (batch, T, input_dim)

#         Returns:
#             Summary embeddings, shape (batch, output_dim)
#         """
#         if data.dim() == 2:
#             data = data.unsqueeze(-1)

#         lstm_out, (h_n, _) = self.lstm(data)

#         last_hidden = h_n[-1]

#         summary = self.fc(last_hidden)

#         return summary


# class StochVolTask(Task):
#     def __init__(self, config, device="cpu"):
#         self.device = device
#         self.prior_parameters = config["prior"]
#         self.simulator_parameters = config["simulator"]
#         self.summary_parameters = config["summary"]
#         self.logger_config = config.get("logger")
#         super().__init__(device=device)

#         def check_support(theta):
#             clamp_min = torch.tensor([self.prior_parameters[parameter][0] for parameter in self.prior_parameters.keys()])
#             clamp_max = torch.tensor([self.prior_parameters[parameter][1] for parameter in self.prior_parameters.keys()])
#             # WARNING: проблемы с размерностями
#             mask = (theta >= clamp_min) & (theta <= clamp_max)
#             while len(mask.shape) > 1:
#                 mask = mask.all(dim=-1)
#             return mask

#         self.check_support = check_support

#         self.theta_dim = 4
#         self.data_dim = self.summary.emb_dim

#     def build_prior(self):
#         return StochVolPrior(
#             mu_range=self.prior_parameters.get("mu_range", (-0.5, 0.5)),
#             phi_range=self.prior_parameters.get("phi_range", (0.90, 0.99)),
#             sigma_range=self.prior_parameters.get("sigma_range", (0.05, 0.5)),
#             m_range=self.prior_parameters.get("m_range", (-3.0, -0.5)),
#         )

#     def build_simulator(self):
#         return StochVolSimulator(self.simulator_parameters, device=self.device)

#     def build_summary(self):
#         if self.summary_parameters == "handmade":
#             return HandmadeSummary()
#         elif self.summary_parameters == "lstm":
#             lstm_config = self.summary_parameters if isinstance(self.summary_parameters, dict) else {}
#             return LSTMSummary(
#                 input_dim=lstm_config.get("input_dim", 1),
#                 hidden_dim=lstm_config.get("hidden_dim", 32),
#                 num_layers=lstm_config.get("num_layers", 2),
#                 output_dim=lstm_config.get("output_dim", self.data_dim),
#                 dropout=lstm_config.get("dropout", 0.1),
#             )
#         raise NotImplementedError(f"Summary {self.summary_parameters} is not implemented")

#     def build_logger(self):
#         if self.logger_config is None:
#             return setup_logging(name="StochVol", level=Logger.INFO, log_to_file=False)
#         return setup_logging(
#             name=self.logger_config["name"],
#             level=getattr(Logger, self.logger_config["level"]),
#             log_to_file=True,
#             log_file_path=self.logger_config["log_file_path"],
#         )

#     def sample_prior(self, size):
#         return self.prior.sample(size, device=self.device)

#     def simulate(self, theta):
#         return self.simulator.simulate(theta, device=self.device)

#     def summarize(self, x):
#         return self.summary(x)



# import torch
# import matplotlib.pyplot as plt

# def simulate_hull_white(r0, a, sigma, theta_t, T, n_steps, n_paths, device='cpu'):
#     """
#     Симулирует процесс Халла-Уайта, предполагая, что theta_t — это константа.

#     Args:
#         r0 (float): Начальная процентная ставка.
#         a (float): Скорость возврата к среднему.
#         sigma (float): Волатильность.
#         theta_t (float): Долгосрочное среднее.
#         T (float): Временной горизонт.
#         n_steps (int): Количество временных шагов.
#         n_paths (int): Количество симулированных траекторий.
#         device (str): 'cpu' или 'cuda'.

#     Returns:
#         torch.Tensor: Тензор формы (n_paths, n_steps+1).
#     """
#     dt = torch.tensor(T / n_steps)
#     # Инициализируем тензор для хранения всех траекторий
#     rates = torch.zeros(n_paths, n_steps + 1, device=device)
#     rates[:, 0] = r0

#     # Генерируем все случайные шумы за один раз
#     z = torch.randn(n_paths, n_steps, device=device)

#     for t in range(1, n_steps + 1):
#         r_prev = rates[:, t-1]
#         # Дискретизация методом Эйлера-Маруямы
#         dr = (theta_t - a * r_prev) * dt + sigma * torch.sqrt(dt) * z[:, t-1]
#         rates[:, t] = r_prev + dr

#     return rates

# # Параметры модели
# r0 = 0.02
# a = 0.1
# sigma = 0.01
# theta_t = 0.03
# T = 1.0
# n_steps = 252
# n_paths = 500
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# # Запуск симуляции
# rates = simulate_hull_white(r0, a, sigma, theta_t, T, n_steps, n_paths, device)

# # Визуализация
# plt.figure(figsize=(10, 6))
# for i in range(n_paths):
#     plt.plot(rates[i].cpu().numpy(), lw=0.8)
# plt.title("Симуляция процентных ставок (Модель Халла-Уайта)")
# plt.xlabel("Временные шаги")
# plt.ylabel("Процентная ставка")
# plt.show()