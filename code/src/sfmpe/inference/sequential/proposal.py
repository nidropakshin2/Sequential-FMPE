import torch
from torchdiffeq import odeint

from sfmpe.core.distributions import Distribution
from sfmpe.flow.sampler import ODESampler
from sfmpe.tasks.base_task import Task

class ProposalParams:
    """
        method: None | str, one of None, 'NPE-A', 'NPE-B', 'NPE-C'
        x_0: None | torch.Tensor, if sequential mode you have to specify x_0
        weight: None | float, weight used for 'NPE-B' method
        n_steps: None | int, steps of ODESampler
    """
    task: None | Task
    method: None | str 
    method_params: None | dict
    theta_0: torch.Tensor 
    x_0: None | torch.Tensor 
    # weight: None | float = 0.5
    n_steps: int = 8



class Proposal(Distribution):
    def __init__(self, flow_model, params: ProposalParams):
        self.flow_model = flow_model
        self.params = params
        self.sampler = ODESampler(flow_model)
    

    def sample(self, size, **kwargs) -> torch.Tensor:
        # TODO: убрать это дерьмо и сделать просто Truncated
        if self.params.method == "NPE-A": 
            x_0 = self.params.x_0
            x_0_expanded = x_0.unsqueeze(0).expand(*size, *x_0.shape)
            return self.sampler.sample(x_0=x_0_expanded, n_steps=self.params.n_steps)

        elif self.params.method == 'NPE-B':
            total = 1
            for dim in size:
                total *= dim
            assert self.params.method_params is not None 
            weight = self.params.method_params.get("weight", None)
            assert weight is not None

            num_proposals   = int(weight * total)
            num_priors      = total - num_proposals

            x_0 = self.params.x_0
            x_0_expanded = x_0.unsqueeze(0).expand(num_proposals, *x_0.shape)
            proposals = self.sampler.sample(x_0=x_0_expanded, n_steps=self.params.n_steps)
            
            assert self.params.task is not None
            priors = self.params.task.prior.sample((num_priors, *x_0.shape[:-1], ))
            samples = torch.cat([proposals, priors], dim=0)
            samples = samples[torch.randperm(samples.shape[0])]
            samples = samples.view(*size, *x_0.shape[:-1], -1)
            return samples
        
        elif self.params.method == 'NPE-C':
            raise NotImplementedError(f"Method {self.params.method} is not implemented") # type: ignore

        elif self.params.method == 'Truncated':
            assert self.params.method_params is not None
            scale = self.params.method_params.get("scale", None)
            assert scale is not None

            x_0 = self.params.x_0
            x_0_expanded = x_0.unsqueeze(0).expand(*size, *x_0.shape)
            return self.sampler.sample(x_0=x_0_expanded, 
                                       n_steps=self.params.n_steps, 
                                       scale=scale)


        else:
            raise NotImplementedError(f"Method {self.params.method} is not implemented") # type: ignore


    def log_prob(self, value: torch.Tensor, **kwargs):
        """
        value: (*batch, d)
        returns: (*batch,)
        """

        device = value.device
        batch_size, d = (value.shape[:-1]), value.shape[-1]
        
        x_0 = self.params.x_0
        # WARNING: проблемы с размерностями
        x_0_expanded = x_0.expand(*batch_size, self.params.task.data_dim).to(device)
            
        # --- initial state (at t = 0) ---
        theta0 = value

        logp = torch.zeros(batch_size, device=device)

        # Hutchinson noise
        eps = torch.randn_like(theta0)

        # --- ODE function ---
        def ode_func(t, state):
            theta, logp = state

            theta.requires_grad_(True)
            t_expanded = t.unsqueeze(0).expand(*batch_size, 1)
            # print(t_expanded.shape, theta.shape, x_0_expanded.shape)
            u = self.flow_model.velocity_model(t=t_expanded, theta=theta, x=x_0_expanded)

            jvp = torch.autograd.grad(
                (u * eps).sum(),
                theta,
                create_graph=False,
                retain_graph=False,
            )[0]

            div = (jvp * eps).sum(dim=-1)

            dtheta = u
            dlogp = -div

            return dtheta, dlogp

        # --- integrate BACKWARD: t=1 -> t=0 ---
        _, logp_correction = odeint(
            ode_func,
            y0=(theta0, logp),
            t=torch.linspace(0, 1, steps=self.params.n_steps)
        )

        # --- init dist log prob ---
        if self.params.task is not None:
            base_logp = self.flow_model.init_dist.log_prob(theta0)
        else:
            raise NotImplementedError("Base distribution not defined")

        # --- итог ---
        logp = base_logp + logp_correction[-1]

        return logp