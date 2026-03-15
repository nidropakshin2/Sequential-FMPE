import torch
import torch.nn as nn
import os

from sfmpe.flow.flow_model import FlowModel
from sfmpe.data.round_dataset import RoundDataset
from sfmpe.core.distributions import Distribution


class FlowMatchingEstimator:
    def __init__(self, 
                 flow_model: FlowModel,
                 optimizer: torch.optim.Optimizer, 
                 loss_fn: torch.nn.Module):

        self.flow_model = flow_model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
    

    def train(self, dataset: RoundDataset, **kwargs):

        self.device       = kwargs.pop('device', 'cpu')
        self.epochs       = kwargs.pop('epochs', 1000)
        self.path         = kwargs.pop('path', None)
        self.show_every   = kwargs.pop('show_every', None)

        if kwargs:
            raise TypeError(f"train() got unexpected keyword arguments: {', '.join(kwargs.keys())}")

        self.flow_model.to(self.device)
        self.flow_model.train()

        loss_stats = []
        min_loss   = torch.inf

        for epoch in range(1, self.epochs + 1):
            
            theta_1, x = dataset[:]
            theta_0 = self.flow_model.init_dist.sample(theta_1.shape).to(self.device)

            t = self.flow_model.path.time_dist.sample((*theta_0.shape[:-1], 1))
            theta_t = self.flow_model.path.sample(theta_0, theta_1, t)
            dtheta_t = self.flow_model.path.velocity(theta_0, theta_1)

            v = self.flow_model.velocity_model(t=t, theta=theta_t, x=x)

            loss = self.loss_fn(v, dtheta_t)
            loss_stats.append(loss.detach().item())

            if loss < min_loss and self.path is not None:
                torch.save(self.flow_model.state_dict(), self.path)
            
            if self.show_every != None: 
                if epoch % self.show_every == 0:
                    print(f"Epoch: {epoch}, Loss: {loss_stats[epoch]:.4f}")
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        return loss_stats

    def load(self):
        assert os.path.exists(self.path)
        self.flow_model.load_state_dict(torch.load(self.path, weights_only=True, map_location=self.device))
        
        return self.flow_model

    def build_posterior(self) -> Distribution:
        
        return 
