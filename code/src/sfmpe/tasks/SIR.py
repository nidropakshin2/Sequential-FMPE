import torch
import numpy as np

from sfmpe.core.distributions import Distribution
from sfmpe.core.simulator import Simulator
from sfmpe.core.summary import Summary
from sfmpe.tasks.base_task import Task


class SIRPrior(Distribution):
    def __init__(self, gamma_range=(0.05, 0.5), beta_range=(0.5, 5.0)):
        """
            gamma_range: tuple, default=(0.05, 0.5)
            beta_range: tuple, default=(0.5, 5.0)
        """
        self.gamma_range = gamma_range
        self.beta_range = beta_range
    
    def sample(self, size, **kwargs):
        """
            size: tuple, e.g. (B, N, dim)
            kwargs
                device: str, default='cpu'
        """

        def sample_log_uniform(low, high, size):
            """
                low, high: scalars (floats), low > 0
                size: tuple, e.g. (B,)
            """
            u = torch.rand(size)
            log_low = torch.log(torch.tensor(low))
            log_high = torch.log(torch.tensor(high))
            return torch.exp(log_low + u * (log_high - log_low))
        
        device = kwargs.pop('device', torch.device('cpu'))

        if kwargs:
            raise TypeError(f"sample() got unexpected keyword arguments: {', '.join(kwargs.keys())}")


        gamma = sample_log_uniform(self.gamma_range[0], self.gamma_range[1], (*size, 1))
        R0 = sample_log_uniform(self.beta_range[0], self.beta_range[1], (*size, 1))
        beta = R0 * gamma
        return torch.cat([beta, gamma], dim=-1).to(device)
    
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("log_prob() method is not implemented for SIR model")
        return torch.tensor([])


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
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.emb_dim = 5

    def forward(self, data):
        def safe_div(a, b):
            return a / (b + self.eps)

        S = data[..., 0]
        I = data[..., 1]
        R = data[..., 2]
        # batch, T = I.shape
        T = I.shape[-2]

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
        
        B = torch.tensor([[-0.6543, -1.0543,  0.4823, -0.4612,  0.7340],
                        [ 2.6280, -1.6988, -0.6246, -0.0949, -0.3497],
                        [ 0.4116, -0.8507, -0.1116, -0.5067,  1.4959],
                        [-0.2124, -0.1211, -0.8651, -1.2937,  1.2938],
                        [ 0.0257, -0.9242, -1.4133,  0.3826, -1.0719]]).to(data.device)
        
        s = torch.matmul(s, B)
        
        return s


class SIRTask(Task):
    def __init__(self, config, device=torch.device('cpu')):
        self.device = device
        self.simulator_parameters = config["simulator"]
        self.summary = config["summary"]
        super().__init__()
        
        self.theta_dim = 2
        self.data_dim = self.summary.emb_dim
    
    def build_prior(self):
        return SIRPrior()

    def build_simulator(self):
        return SIRSimulator(self.simulator_parameters, device=self.device)

    def build_summary(self):
        if self.summary == "handmade":
            return HandmadeSummary()
        else:
            raise NotImplementedError

    def sample_prior(self, size):
        return self.prior.sample(size, device=self.device)

    def simulate(self, theta):
        return self.simulator.simulate(theta, device=self.device)

    def summarize(self, x):
        return self.summary(x)