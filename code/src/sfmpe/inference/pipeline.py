import torch
import numpy as np 
import matplotlib.pyplot as plt 
from torch import nn, Tensor
import time
import os
from abc import ABC, abstractmethod
from sfmpe.flow.flow_model import FlowModel
from sfmpe.tasks.base_task import Task


class Pipeline(ABC):
    def __init__(self, 
                 task: Task, 
                 flow_model: FlowModel, 
                 optimizer: torch.optim.Optimizer, 
                 loss_fn: torch.nn.Module
                 ):
        
        self.task      = task
        self.flow_model = flow_model
        self.optimizer  = optimizer
        self.loss_fn    = loss_fn

    def train(self, **kwargs):

        self.device       = kwargs.pop('device', 'cpu')
        self.epochs       = kwargs.pop('epochs', 1000)
        self.path         = kwargs.pop('path', None)
        self.show_every   = kwargs.pop('show_every', None)
        self.dataset_size = kwargs.pop('dataset_size', None)

        if kwargs:
            raise TypeError(f"sample() got unexpected keyword arguments: {', '.join(kwargs.keys())}")

        self.flow_model.to(self.device)
        self.flow_model.train()

        loss_stats = []
        min_loss   = torch.inf

        for epoch in range(self.epochs + 1):
            
            # TODO
            X_1, x = self.task.sample_from_dataset(self.dataset_size).to(self.device)
            X_0 = self.flow_model.init_dist.sample(X_1.shape).to(self.device)

            # TODO: возможны проблемы при обработке батчей

            t = self.flow_model.path.time_dist.sample((X_0.shape[0], 1))
            X_t = self.flow_model.path.sample(X_0, X_1, t)
            dX_t = self.flow_model.path.velocity(X_0, X_1)

            v = self.flow_model.velocity_model(x=x, t=t, context=X_t)
            loss = self.loss_fn(v, dX_t)
            loss_stats.append(loss.detach().item())

            if loss < min_loss:
                torch.save(self.flow_model.state_dict(), self.path)
            
            if self.show_every != None: 
                if epoch % self.show_every == 0:
                    print(f"Epoch: {epoch}, Loss: {loss_stats[epoch]:.4f}")
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        return loss_stats


    # TODO: переписать пайплайн
    def pipeline(self,
                 model,
                summary,
                optimizer,
                sim_parameters,
                train_model=True,
                epochs=2000, 
                show_every=200,
                device='cpu'):
        if train_model:
            model.train()
            
            loss = torch.nn.MSELoss()
            
            datetime = time.strftime("%Y-%m-%d_%H_%M_%S")
            path = f"./models/SIR_{datetime}.pth.tar.gz" 
        
            losses = self.train()
            plt.plot(losses)
            return losses
        
        else:
            assert os.path.exists(self.path)
            model.load_state_dict(torch.load(self.path, weights_only=True, map_location=torch.device(device)))
            model.eval()
