import torch
import torch.nn as nn


class SimpleVelocityField(nn.Module):
    def __init__(self, theta_dim, x_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1 + theta_dim + x_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, theta_dim),
        )

    def forward(self, t, theta, x):
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        inp = torch.cat([t, theta, x], dim=-1)
        return self.net(inp)

