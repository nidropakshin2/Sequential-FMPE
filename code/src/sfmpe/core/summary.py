import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class Summary(nn.Module, ABC):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return data

