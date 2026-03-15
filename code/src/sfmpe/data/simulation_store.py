import torch


class SimulationStore:
    """
    Storage for all simulations across sequential rounds
    """

    def __init__(self):
        self.theta = []
        self.x = []
        self.round_id = []


    def add(self, theta, x, round_id):
        """
        Add simulations to store

        theta : (B, N, theta_dim) or (N, theta_dim) 
        x : (B, N, x_dim) or (N, x_dim)
        """
        if len(theta.shape) > 2:
            theta = theta.view(-1, theta.shape[-1])
            x = x.view(-1, x.shape[-1])
        self.theta.append(theta.detach())
        self.x.append(x.detach())
        self.round_id.append(
            torch.full((theta.shape[0],), round_id, device=theta.device)
        )

    def get_all(self):
        """
        Return all stored simulations
        """

        return (
            torch.cat(self.theta, dim=0),
            torch.cat(self.x, dim=0),
            torch.cat(self.round_id, dim=0),
        )

    def get_round(self, round_id):
        """
        Return simulations from specific round
        """

        theta, x, r = self.get_all()

        mask = r == round_id

        return theta[mask], x[mask]

    def size(self):
        if len(self.theta) == 0:
            return 0
        return sum(t.shape[0] for t in self.theta)