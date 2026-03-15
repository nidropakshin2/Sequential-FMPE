import torch
import torch.nn as nn
from sfmpe.core.distributions import Distribution
from sfmpe.flow.path import Path

class FlowModel(nn.Module):
    def __init__(self, 
                 velocity_model,
                 init_dist: Distribution,
                 path: Path):
        super().__init__()
        self.velocity_model = velocity_model
        self.init_dist      = init_dist
        self.path           = path 

    def velocity(self, t, theta, x):
        return self.velocity_model(t, theta, x)


