import torch

from sfmpe.core.distributions import Distribution
from sfmpe.flow.sampler import ODESampler


class ProposalParams:
    """
        method: None | str, one of None, 'NPE-A', 'NPE-B', 'NPE-C'
        x_0: None | torch.Tensor, if sequential mode you have to specify x_0
        weight: None | float, weight used for 'NPE-B' method
        n_steps: None | int, steps of ODESampler
    """
    method: None | str 
    theta_dim: int
    x_0: None | torch.Tensor 
    weight: None | float = 0.5
    n_steps: int = 8



class Proposal(Distribution):
    def __init__(self, flow_model, params: ProposalParams):
        self.flow_model = flow_model
        self.params = params
        self.sampler = ODESampler(flow_model) 
    

    def sample(self, size, **kwargs) -> torch.Tensor:
        if self.params.method == "NPE-A": 
            x_0 = self.params.x_0
            # TODO: проблемы с размерностями
            # print(x_0.shape, size)
            x_0 = x_0.unsqueeze(0).expand(*size, *x_0.shape)
            # print(x_0.shape)
            return self.sampler.sample(x_0=x_0, n_steps=self.params.n_steps, theta_dim=self.params.theta_dim)

        elif self.params.method == 'NPE-B':
            return NotImplementedError(f"Method {self.params.method} is not implemented") # type: ignore
        
        elif self.params.method == 'NPE-C':
            return NotImplementedError(f"Method {self.params.method} is not implemented") # type: ignore
        
        else:
            return NotImplementedError(f"Method {self.params.method} is not implemented") # type: ignore
