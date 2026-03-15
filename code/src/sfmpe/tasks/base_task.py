import torch
from abc import ABC, abstractmethod
from sfmpe.core.distributions import Distribution
from sfmpe.core.simulator import Simulator


class Task(ABC):
    """
    Base class for SBI task
    """

    def __init__(self, device="cpu"):
        self.device = device

        self.prior = self.build_prior()
        self.simulator = self.build_simulator()
        self.summary = self.build_summary()

    # -------- components --------
    
    @abstractmethod
    def build_prior(self) -> Distribution:
        pass
    
    @abstractmethod
    def build_simulator(self) -> Simulator:
        pass
    
    @abstractmethod
    def build_summary(self):
        return lambda x: x

    # -------- helper methods --------

    def sample_prior(self, size):
        return self.prior.sample(size)
    
    def simulate(self, theta):
        return self.simulator.simulate(theta)
    
    def summarize(self, x):
        return self.summary(x)

    # -------- convenience --------

    def simulate_dataset(self, size):

        theta = self.sample_prior(size)
        x = self.simulate(theta)
        x = self.summarize(x)

        return theta, x