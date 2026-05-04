from matplotlib.pylab import indices
import torch
import torch.nn as nn
import os

from sfmpe.flow.flow_model import FlowModel
from sfmpe.data.round_dataset import RoundDataset
from sfmpe.core.distributions import Distribution
from sfmpe.flow.sampler import ODESampler
from sfmpe.inference.sequential.proposal import Proposal, ProposalParams
from sfmpe.utils.logger import Logger, get_default_logger


class FlowMatchingEstimator:
    def __init__(self, 
                 flow_model: FlowModel,
                 optimizer: torch.optim.Optimizer, 
                 loss_fn: torch.nn.Module,
                 logger: Logger | None = None,
                 dataset_prepocessor = None):

        self.flow_model = flow_model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        if logger is None:
            self.logger = get_default_logger()
        else:
            self.logger = logger
        if dataset_prepocessor is None:
            self.dataset_prepocessor = lambda theta, x, *args: (theta, x)
        else:
            self.dataset_prepocessor = dataset_prepocessor
        

    def train(self, dataset: RoundDataset, **kwargs):


        self.device       = kwargs.pop('device', 'cpu')
        self.epochs       = kwargs.pop('epochs', 1000)
        self.path         = kwargs.pop('path', None)
        self.batch_size   = kwargs.pop('batch_size', 256)
        self.show_every   = kwargs.pop('show_every', None)

        if kwargs:
            raise TypeError(f"train() got unexpected keyword arguments: {', '.join(kwargs.keys())}")
        
        def check_nan(x, parameter_name):
            if torch.isnan(x).any():
                self.logger.error(f"{parameter_name} is nan")
                return True
            return False 

        self.flow_model.to(self.device)
        self.flow_model.train()

        loss_stats = []
        min_loss   = torch.inf

        for epoch in range(self.epochs + 1):
            
            
            theta_1, x = dataset.theta, dataset.x
            if check_nan(theta_1, "dataset theta_1"):
                raise ValueError("Captured None")
            if check_nan(x, "dataset x"):
                raise ValueError("Captured None")
            theta_0 = self.flow_model.init_dist.sample_like(theta_1).to(self.device)
            if check_nan(theta_0, "theta_0"):
                raise ValueError("Captured None")

            t = self.flow_model.path.time_dist.sample((*theta_0.shape[:-1], 1))
            if check_nan(t, "t"):
                raise ValueError("Captured None")
            theta_t = self.flow_model.path.sample(theta_0, theta_1, t)
            if check_nan(theta_t, "theta_t"):
                raise ValueError("Captured None")
            dtheta_t = self.flow_model.path.velocity(theta_0, theta_1)
            if check_nan(dtheta_t, "dtheta_t"):
                raise ValueError("Captured None")

            mask = (torch.rand(*x.shape[:-1]) > 0.1)
            x_masked =  x * mask.unsqueeze(-1).expand(*mask.shape, x.shape[-1])
            v = 0.999 * self.flow_model.velocity_model(t=t, theta=theta_t, x=x_masked) +\
                0.001 * self.flow_model.velocity_model(t=t, theta=theta_t, x=torch.zeros_like(x)) 
            if check_nan(v, "v"):
                raise ValueError("Captured None")
            loss = self.loss_fn(v, dtheta_t)
            if check_nan(loss, "loss"):
                raise ValueError("Captured None")
            loss_stats.append(loss.detach().item())

            if loss < min_loss and self.path is not None:
                torch.save(self.flow_model.state_dict(), self.path)
            
            if self.show_every != None: 
                if epoch % self.show_every == 0:
                    if self.logger:
                        self.logger.info(f"Epoch: {epoch}, Loss: {loss_stats[epoch]:.4f}")
                    else:
                        print(f"Epoch: {epoch}, Loss: {loss_stats[epoch]:.4f}")
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        return loss_stats

    def load(self):
        assert os.path.exists(self.path)
        if self.logger:
            self.logger.info(f"Loading model from {self.path}...")
        self.flow_model.load_state_dict(torch.load(self.path, weights_only=True, map_location=self.device))
        
        return self.flow_model

    def build_posterior(self, params=None):
        if self.logger:
            self.logger.info("Building posterior...")
        if params is None:
            return ODESampler(self.flow_model)
        else: 
            return Proposal(self.flow_model, params)
