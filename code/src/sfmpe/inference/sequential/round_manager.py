# TODO: поревьюить код нейронки
# -----------------------------

import torch

from sfmpe.data.simulation_store import SimulationStore
from sfmpe.data.round_dataset import RoundDataset


class RoundManager:
    """
    Controls sequential SBI rounds.
    """

    def __init__(
        self,
        task,
        estimator,
        proposal,
        device="cpu",
    ):

        self.task = task
        self.estimator = estimator
        self.proposal = proposal
        self.device = device

        self.store = SimulationStore()

    def run_round(self, round_id, num_simulations):

        # sample parameters
        theta = self.proposal.sample((num_simulations, ), device=self.device)

        # simulate data
        x = self.task.simulator.simulate(theta)

        # summary statistics
        features = self.task.summary(x)

        # store simulations
        self.store.add(theta, features, round_id)

    def train_estimator(self, rounds=None, **train_kwargs):

        dataset = RoundDataset(self.store, rounds)

        self.estimator.train(dataset, **train_kwargs)

    def update_proposal(self, posterior):

        self.proposal = posterior

    def run_sequential(
        self,
        num_rounds,
        sims_per_round,
        **train_kwargs,
    ):

        for r in range(num_rounds):

            print(f"Round {r}")

            self.run_round(r, sims_per_round)

            self.train_estimator([r], **train_kwargs)

            # TODO: решить как определить proposal либо как распределение, либо как датасет
            posterior = self.estimator.build_posterior()

            self.update_proposal(posterior)