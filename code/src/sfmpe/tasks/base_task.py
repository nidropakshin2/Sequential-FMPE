import torch
from abc import ABC, abstractmethod
from sfmpe.core.distributions import Distribution
from sfmpe.core.simulator import Simulator
from sfmpe.utils.logger import get_default_logger


class Task(ABC):
    """
    Base class for SBI task
    """

    def __init__(self, device="cpu"):
        self.device = device

        self.prior = self.build_prior()
        self.check_support = None
        self.simulator = self.build_simulator()
        self.summary = self.build_summary()
        self.logger = self.build_logger()
        
        self.theta_dim = -1
        self.data_dim = -1

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
    
    def build_logger(self):
        return get_default_logger()

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
    
    # -------- convenience --------
    