import torch
from abc import ABC, abstractmethod
from sfmpe.core.distributions import Distribution, Uniform

class Path(ABC):
    def __init__(self, time_dist: Distribution):
        self.time_dist = time_dist

    @abstractmethod
    def sample(self, x0, x1, t) -> torch.Tensor:
        pass

    @abstractmethod
    def velocity(self, x0, x1, t) -> torch.Tensor:
        pass


class AffinePath(Path):
    def __init__(self, time_dist=Uniform()):
        self.time_dist = time_dist
    
    def sample(self, x0, x1, t):
        return (1 - t) * x0 + t * x1

    def velocity(self, x0, x1, t):
        return x1 - x0

class GaussianPath(Path):
    def __init__(self, time_dist=Uniform()):
        self.time_dist = time_dist
        self.alpha = lambda t: torch.exp(-t/2)
        self.sigma = lambda t: torch.min(torch.sqrt(1 - torch.exp(-t)))
        self.beta  = lambda t: 100 * (1 - t) 

    def sample(self, x0, x1, t):
        return self.sigma(self.beta(t)) * x0 + self.alpha(self.beta(t)) * x1

    def velocity(self, x0, x1, t):
        return 0.5 * 100 * self.alpha(self.beta(t)) * x1 - 0.5 * 100 * self.alpha(2 * self.beta(t)) / self.sigma(self.beta(t)) * x0
