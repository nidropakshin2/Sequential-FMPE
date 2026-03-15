from abc import ABC, abstractmethod
import torch


class Simulator(ABC):
    @abstractmethod
    def __init__(self, simulator_parameters, **kwargs):
        """
        simulator_parameters: dict, dictionary with simulator parameters values 
        """
        pass

    @abstractmethod
    def simulate(self, theta, **kwargs) -> torch.Tensor:
        """
        theta: torch.Tensor, (batch, dim), simulation parameters 
        return simulated data
        """
        pass

