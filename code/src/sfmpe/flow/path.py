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
    def velocity(self, x0, x1) -> torch.Tensor:
        pass


class AffinePath(Path):
    def __init__(self, time_dist=Uniform()):
        self.time_dist = time_dist
    
    def sample(self, x0, x1, t):
        return (1 - t) * x0 + t * x1

    def velocity(self, x0, x1):
        return x1 - x0

