import torch
from torchdiffeq import odeint


class ODESampler:
    def __init__(self, flow_model):
        self.flow_model = flow_model

    def sample_obsolete(self, x_0, n_steps=8, **kwargs) -> torch.Tensor:
        # WARNING: возможны проблемы с размерностями
        theta_0 = self.flow_model.init_dist.sample((*x_0.shape[:-1], 2)).to(x_0.device)
        theta_0 = theta_0
        self.flow_model.velocity_model.eval()
        self.flow_model.velocity_model.to(x_0.device)
        def ode_func(t, y):
            theta, x = y
            t_batch = torch.ones(x.shape[:-1], device=x.device) * t
            return self.flow_model.velocity(t_batch, theta, x)

        t = torch.linspace(0, 1, steps=n_steps).to(x_0.device)
        y = (theta_0, x_0)
        return odeint(ode_func, y, t)[-1] # type: ignore

    def sample(self, x_0, n_steps=32, **kwargs) -> torch.Tensor:
        # WARNING: возможны проблемы с размерностями
        scale = kwargs.get("scale", None)

        self.flow_model.velocity_model.eval()
        self.flow_model.velocity_model.to(x_0.device)

        theta_0 = scale * self.flow_model.init_dist.sample((*x_0.shape[:-1],)).to(x_0.device)

        t = torch.linspace(0, 1, steps=n_steps + 1).to(x_0.device)
        
        for i in range(n_steps):
            theta_0 = self.flow_model.velocity_model.step(theta=theta_0, x=x_0, t_start=t[i], t_end=t[i+1])
        return theta_0.detach()