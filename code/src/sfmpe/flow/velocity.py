import torch
import torch.nn as nn


class SimpleVelocityField(nn.Module):
    def __init__(self, theta_dim, x_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1 + x_dim + theta_dim, hidden_dim), nn.ReLU(),
            
        )
        for _ in range(num_layers):
            self.net.append(nn.Linear(hidden_dim, hidden_dim))
            self.net.append(nn.ReLU())
        self.net.append(nn.Linear(hidden_dim, theta_dim))


    def forward(self, t, theta, x):
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        inp = torch.cat([t, x, theta], dim=-1)
        return self.net(inp)
    
    
    def step(self, theta, x, t_start, t_end):
        t_start = t_start.expand(*theta.shape[:-1], 1)
        t_end = t_end.expand(*theta.shape[:-1], 1)
        
        
        t_mid = (t_end + t_start) / 2
        theta_mid = theta + self.forward(t=t_start, theta=theta, x=x) * (t_mid - t_start)
        return theta + self.forward(t=t_mid, theta=theta_mid, x=x) * (t_end - t_mid)

