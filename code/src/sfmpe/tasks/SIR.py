import torch
from torch.distributions import Transform, biject_to, constraints, Distribution


# from sfmpe.core.distributions import Distribution
from sfmpe.core.simulator import Simulator
from sfmpe.core.summary import Summary
from sfmpe.tasks.base_task import Task
from sfmpe.utils.logger import Logger, setup_logging


class SIRSupport(constraints.Constraint):
    event_dim = 1
    is_discrete = False

    def __init__(self, gamma_range, r0_range):
        self.gamma_range = gamma_range
        self.r0_range = r0_range

    def check(self, value):
        beta, gamma = value.unbind(dim=-1)
        gamma_min, gamma_max = self.gamma_range
        r0_min, r0_max = self.r0_range
        valid_gamma = (gamma >= gamma_min) & (gamma <= gamma_max)
        safe_gamma = gamma.clamp_min(torch.finfo(value.dtype).tiny)
        r0 = beta / safe_gamma
        return valid_gamma & (r0 >= r0_min) & (r0 <= r0_max) & (beta > 0)

    def __repr__(self):
        return f"SIRSupport(gamma_range={self.gamma_range}, r0_range={self.r0_range})"


class SIRTransform(Transform):
    domain = constraints.independent(constraints.real, 1)
    bijective = True
    sign = 1

    def __init__(self, gamma_range, r0_range, cache_size=1):
        super().__init__(cache_size=cache_size)
        self.gamma_range = gamma_range
        self.r0_range = r0_range

    def with_cache(self, cache_size=1):
        return SIRTransform(self.gamma_range, self.r0_range, cache_size)

    @property
    def codomain(self):
        return SIRSupport(self.gamma_range, self.r0_range)

    def _call(self, value):
        r0_min, r0_max = self.r0_range
        gamma_min, gamma_max = self.gamma_range
        r0 = r0_min + (r0_max - r0_min) * value[..., 0].sigmoid()
        gamma = gamma_min + (gamma_max - gamma_min) * value[..., 1].sigmoid()
        return torch.stack((r0 * gamma, gamma), dim=-1)

    def _inverse(self, value):
        beta, gamma = value.unbind(dim=-1)
        r0_min, r0_max = self.r0_range
        gamma_min, gamma_max = self.gamma_range
        r0_fraction = ((beta / gamma) - r0_min) / (r0_max - r0_min)
        gamma_fraction = (gamma - gamma_min) / (gamma_max - gamma_min)
        return torch.stack(
            (torch.logit(r0_fraction), torch.logit(gamma_fraction)), dim=-1
        )

    def log_abs_det_jacobian(self, value, transformed):
        r0_min, r0_max = self.r0_range
        gamma_min, gamma_max = self.gamma_range
        log_r0_derivative = (
            torch.log(torch.as_tensor(r0_max - r0_min, device=value.device, dtype=value.dtype))
            + torch.nn.functional.logsigmoid(value[..., 0])
            + torch.nn.functional.logsigmoid(-value[..., 0])
        )
        log_gamma_derivative = (
            torch.log(torch.as_tensor(gamma_max - gamma_min, device=value.device, dtype=value.dtype))
            + torch.nn.functional.logsigmoid(value[..., 1])
            + torch.nn.functional.logsigmoid(-value[..., 1])
        )
        return log_r0_derivative + torch.log(transformed[..., 1]) + log_gamma_derivative


@biject_to.register(SIRSupport)
def _biject_to_sir_support(constraint):
    return SIRTransform(constraint.gamma_range, constraint.r0_range)


class SIRPrior(Distribution):
    arg_constraints = {}

    def __init__(self, gamma_range=(0.05, 0.5), beta_range=(0.5, 5.0)):
        """
        gamma_range: tuple, default=(0.05, 0.5)
        beta_range: tuple, default=(0.5, 5.0)
        """
        super().__init__()
        self.gamma_range = gamma_range
        self.beta_range = beta_range

    @property
    def support(self):
        return SIRSupport(self.gamma_range, self.beta_range)

    def sample(self, sample_shape=(1, ), **kwargs):
        """
        size: tuple, e.g. (B, N, dim)
        kwargs
            device: str; defaults to the device this prior was assigned by
            its task (see base_task), falling back to 'cpu'.
        """
        device = kwargs.pop("device", getattr(self, "device", torch.device("cpu")))

        def sample_log_uniform(low, high, size):
            u = torch.rand(size, device=device)
            log_low = torch.log(torch.tensor(low, device=device, dtype=torch.float32))
            log_high = torch.log(torch.tensor(high, device=device, dtype=torch.float32))
            return torch.exp(log_low + u * (log_high - log_low))

        if kwargs:
            raise TypeError(f"sample() got unexpected keyword arguments: {', '.join(kwargs.keys())}")

        gamma = sample_log_uniform(self.gamma_range[0], self.gamma_range[1], (*sample_shape, 1))
        R0 = sample_log_uniform(self.beta_range[0], self.beta_range[1], (*sample_shape, 1))
        beta = R0 * gamma
        return torch.cat([beta, gamma], dim=-1).to(device)

    def log_prob(self, value: torch.Tensor, **kwargs) -> torch.Tensor:
        import math
        """
        value: тензор формы (..., 2) с [beta, gamma]
        Возвращает: тензор формы (...)
        """
        beta = value[..., 0]
        gamma = value[..., 1]

        beta_lo, beta_hi = self.beta_range
        gamma_lo, gamma_hi = self.gamma_range

        # log-нормировочные константы log-log-uniform (Python floats, device-agnostic)
        log_norm_beta = math.log(math.log(beta_hi / beta_lo))
        log_norm_gamma = math.log(math.log(gamma_hi / gamma_lo))

        tiny = torch.finfo(value.dtype).tiny
        safe_beta = beta.clamp_min(tiny)
        safe_gamma = gamma.clamp_min(tiny)

        # log p(beta, gamma) = log p_R0(R0) + log p_gamma(gamma) - log(gamma)
        log_p = (
            -torch.log(safe_beta)
            - torch.log(safe_gamma)
            - log_norm_beta
            - log_norm_gamma
        )

        valid = self.support.check(value)
        return torch.where(valid, log_p, torch.full_like(log_p, float("-inf")))


class SIRSimulator(Simulator):
    def __init__(self, simulator_parameters={"S": 1000, "I": 100, "R": 0, "T": 90}, **kwargs):
        """
        simulation_parameters: dict,
            {
             "S": int   - number of Susceptible individuals
                  tuple - bounds for bounds for following uniform distribution
             "I": int   - number of Infectious individuals
                  tuple - bounds for bounds for following uniform distribution
             "R": int   - number of Recovered individuals
                  tuple - bounds for bounds for following uniform distribution
             "T": int   - simulation duration in days
                  tuple - bounds for bounds for following uniform distribution
             }
        """
        self.S = simulator_parameters["S"]
        self.I = simulator_parameters["I"]
        self.R = simulator_parameters["R"]
        self.T = simulator_parameters["T"]


    def simulate(self, theta, **kwargs):
        def assign_value(x):
            if type(x) in {tuple, list}:
                return int(torch.randint(x[0], x[1], size=(1,)))
            else:
                return x
        
        S = kwargs.pop("S", assign_value(self.S)) 
        I = kwargs.pop("I", assign_value(self.I))
        R = kwargs.pop("R", assign_value(self.R)) 
        T = kwargs.pop("T", assign_value(self.T))

        S = torch.zeros(theta.shape[:-1], device=theta.device) + S
        I = torch.zeros(theta.shape[:-1], device=theta.device) + I
        R = torch.zeros(theta.shape[:-1], device=theta.device) + R
        N = S + I + R

        states = torch.zeros((*theta.shape[:-1], T, 3), device=theta.device)
        for t in range(0, T):
            states[..., t, 0] = S
            states[..., t, 1] = I
            states[..., t, 2] = R

            p_inf = 1 - torch.exp(-theta[..., 0] * I / N)
            p_rec = 1 - torch.exp(-theta[..., 1])

            new_inf = torch.binomial(S, p_inf)
            new_rec = torch.binomial(I, p_rec)

            S = S - new_inf
            I = I + new_inf - new_rec
            R = R + new_rec

        states[..., T-1, 0] = S
        states[..., T-1, 1] = I
        states[..., T-1, 2] = R
        return states        


class HandmadeSummary(Summary):
    def __init__(self, config, eps=1e-8):
        super().__init__()
        self.eps = eps
        # self.emb_dim = 5
        self.emb_dim = 5

    def forward(self, data):
        def safe_div(a, b):
            return a / (b + self.eps)

        S = data[..., 0]
        I = data[..., 1]
        R = data[..., 2]
        # batch, T = I.shape
        T = I.shape[-1]

        N = S[..., 0] + I[..., 0] + R[..., 0]
        N = N.clamp(min=self.eps)

        
        I_peak, t_peak = I.max(dim=-1)            
        t_peak = t_peak.to(torch.float32) / float(max(1, T - 1))
        t_peak = t_peak.unsqueeze(-1)             

        I_peak_frac = safe_div(I_peak, N).unsqueeze(-1)
        
        I_mean_frac = safe_div(I.mean(dim=-1), N).unsqueeze(-1)

        R_final_frac = safe_div(R[..., -1], N).unsqueeze(-1)

        S_final_frac = safe_div(S[..., -1], N).unsqueeze(-1)

        
        s = torch.cat([t_peak, I_peak_frac, I_mean_frac, R_final_frac, S_final_frac], dim=-1).to(data.device)
        # s = torch.cat([I_peak_frac, I_mean_frac, R_final_frac, S_final_frac], dim=-1).to(data.device)
        
        B = torch.tensor([[-0.6543, -1.0543,  0.4823, -0.4612,  0.7340],
                          [ 2.6280, -1.6988, -0.6246, -0.0949, -0.3497],
                          [ 0.4116, -0.8507, -0.1116, -0.5067,  1.4959],
                          [-0.2124, -0.1211, -0.8651, -1.2937,  1.2938],
                          [ 0.0257, -0.9242, -1.4133,  0.3826, -1.0719]]).to(data.device)
        # B = torch.tensor([[ -1.6988, -0.6246, -0.0949, -0.3497],
        #                   [ -0.8507, -0.1116, -0.5067,  1.4959],
        #                   [ -0.1211, -0.8651, -1.2937,  1.2938],
        #                   [ -0.9242, -1.4133,  0.3826, -1.0719]]).to(data.device)
        s = torch.matmul(s, B)
        
        return s
    

import torch.nn as nn


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



class SIRTask(Task):
    def __init__(self, config, device='cpu'):
        self.device = device
        self.prior_paramters = config["prior"]
        self.simulator_parameters = config["simulator"]
        self.summary_parameters = config["summary"]
        self.logger_config = config.get("logger")
        super().__init__(device=device)

        self.summary.to(self.device)

        self.theta_dim = 2
        self.data_dim = self.summary.emb_dim
    
    def check_support(self, theta):
        return self.prior.support.check(theta)


    def build_prior(self):
        return SIRPrior(gamma_range=self.prior_paramters["gamma_range"],
                        beta_range=self.prior_paramters["beta_range"])

    def build_simulator(self):
        return SIRSimulator(self.simulator_parameters, device=self.device)

    def build_summary(self):
        summary_type = self.summary_parameters.get("type", "handmade")
        if summary_type == "handmade":
            return HandmadeSummary(self.summary_parameters)
        elif summary_type == "lstm":
            return LSTMSummary(self.summary_parameters)
        else:
            raise NotImplementedError(f"Summary {self.summary_parameters} is not implemented")
    
    def build_logger(self):
        if self.logger_config is None:
            return setup_logging(name="SIR", 
                                 level=Logger.INFO, 
                                 log_to_file=False)
        return setup_logging(name=self.logger_config["name"],
                             level=getattr(Logger, self.logger_config["level"]),
                             log_to_file=True,
                             log_file_path=self.logger_config["log_file_path"])

    def sample_prior(self, size):
        return self.prior.sample(size, device=self.device)

    def simulate(self, theta):
        return self.simulator.simulate(theta, device=self.device)

    def summarize(self, x):
        return self.summary(x)