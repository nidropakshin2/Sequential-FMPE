import torch
from torchdiffeq import odeint


class ODESampler:
    def __init__(self, flow_model):
        self.flow_model = flow_model

    def sample(self, x0, context, t_span=torch.tensor([0.0, 1.0])):
        def ode_func(t, x):
            t_batch = torch.ones(x.shape[0], device=x.device) * t
            return self.flow_model.velocity(x, t_batch, context)

        return odeint(ode_func, x0, t_span)[-1]

