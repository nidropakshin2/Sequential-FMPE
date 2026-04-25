import torch
from abc import ABC, abstractmethod


class Distribution(ABC):
    def __init__(self):
        self.params = None

    @abstractmethod
    def sample(self, size, **kwargs) -> torch.Tensor:
        pass

    def sample_like(self, tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        return tensor
    
    def log_prob(self, value: torch.Tensor, **kwargs) -> torch.Tensor:
        return torch.Tensor([-torch.inf])
    
    def to(self, device: torch.device) -> None:
        pass

class Uniform(Distribution):
    def __init__(self, low=0, high=1):
        self.low = low
        self.high = high
        self.dist = torch.distributions.Uniform(low, high) 

    def sample(self, size, **kwargs) -> torch.Tensor:
        return self.dist.sample(size)
    
    def sample_like(self, tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.low + (self.high - self.low) * torch.rand_like(tensor)

    def log_prob(self, value: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.dist.log_prob(value)


class Normal(Distribution):

    def sample(self, size, **kwargs) -> torch.Tensor:
        return torch.randn(size)



