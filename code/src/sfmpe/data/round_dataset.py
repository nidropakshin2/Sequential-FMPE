import torch
from torch.utils.data import Dataset


class RoundDataset(Dataset):
    """
    PyTorch dataset built from SimulationStore
    """

    def __init__(self, store, rounds=None):
        """
        rounds:
            None - use all rounds
            list[int] - selected rounds
        """

        theta, x, round_id = store.get_all()

        if rounds is not None:
            mask = torch.zeros_like(round_id, dtype=torch.bool)
            for r in rounds:
                mask |= round_id == r

            theta = theta[mask]
            x = x[mask]

        self.theta = theta
        self.x = x

    def __len__(self):
        return self.theta.shape[0]

    def __getitem__(self, idx):

        return (self.theta[idx], self.x[idx])