import torch
from torch.utils.data import Dataset


class SimulationDataset(Dataset):
    def __init__(self):
        self.theta = []
        self.data = []

    def add(self, theta, x):
        self.theta.append(theta.detach())
        self.data.append(x.detach())

    def __len__(self):
        return sum(t.shape[0] for t in self.theta)

    def __getitem__(self, idx):
        raise NotImplementedError(
            "Simple implementation assumes custom loader construction"
        )

    def get_all(self):
        return (
            torch.cat(self.theta),
            torch.cat(self.data),
        )

