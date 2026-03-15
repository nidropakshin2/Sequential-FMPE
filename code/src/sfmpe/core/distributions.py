import torch
from abc import ABC, abstractmethod


class Distribution(ABC):
    @abstractmethod
    def sample(self, size, **kwargs) -> torch.Tensor:
        pass

    def sample_like(self, tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        return tensor

    def to(self, device: torch.device) -> None:
        pass

class Uniform(Distribution):
    def __init__(self, low=0, high=1):
        self.low = low
        self.high = high

    def sample(self, size, **kwargs) -> torch.Tensor:
        return self.low + (self.high - self.low) * torch.rand(size)
    
    def sample_like(self, tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.low + (self.high - self.low) * torch.rand_like(tensor)
    


class Normal(Distribution):

    def sample(self, size: tuple, **kwargs) -> torch.Tensor:
        return torch.randn(size)



