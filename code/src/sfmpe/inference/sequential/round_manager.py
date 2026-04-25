import torch

from sfmpe.data.simulation_store import SimulationStore
from sfmpe.data.round_dataset import RoundDataset
from sfmpe.utils.logger import Logger, get_default_logger


class RoundManager:
    """
    Controls sequential SBI rounds.
    """

    def __init__(
        self,
        task,
        estimator,
        proposal_params,
        validator=None,
        device="cpu",
        logger=None,
    ):

        self.task = task
        self.estimator = estimator
        self.proposal = task.prior
        self.proposal_params = proposal_params
        self.device = device
        
        
        if logger is None:
            self.logger = self.task.logger
            self.estimator.logger = self.logger
        else:
            self.logger = logger

        self.store = SimulationStore()
        self.losses = []
        
        # Log initialization
        self.logger.info(f"RoundManager initialized with device: {device}")
        self.logger.info(f"Task: {task.__class__.__name__}")
        self.logger.info(f"Estimator: {estimator.__class__.__name__}")

    def run_round(self, round_id, sims_per_round):

        # Log round start
        self.logger.info(f"Starting round {round_id} with {sims_per_round} simulations")
        
        # sample parameters
        self.logger.debug(f"Proposal distribution: {self.proposal}")
        if round_id == 0:
            theta = self.proposal.sample((sims_per_round, *self.proposal_params.x_0.shape[:-1]), device=self.device)
        else:
            theta = self.proposal.sample((sims_per_round, ), device=self.device)
        self.logger.debug(f"x_0 shape {self.proposal_params.x_0.shape[:-1]}")
        
        self.logger.debug(f"Sampled {sims_per_round} parameters with shape {theta.shape}")

        # simulate data
        x = self.task.simulator.simulate(theta)
        self.logger.debug(f"Simulated data with shape: {x.shape}")

        # summary statistics
        features = self.task.summary(x)
        self.logger.debug(f"Computed summary statistics with shape: {features.shape}")

        # store simulations
        self.store.add(theta, features, round_id)
        self.logger.info(f"Round {round_id} completed - stored {sims_per_round} simulations")

    def train_estimator(self, rounds=None, **train_kwargs):

        dataset = RoundDataset(self.store, rounds)
        
        num_samples = len(dataset)
        self.logger.info(f"Training estimator on {num_samples} samples from rounds: {rounds}")
        
        # Extract training parameters for logging
        epochs = train_kwargs.get('epochs', 'default')
        self.logger.info(f"Training parameters: epochs={epochs}")
        
        # Train the estimator
        loss_stats = self.estimator.train(dataset, **train_kwargs)
        
        self.logger.info(f"Estimator training completed")
        return loss_stats

    def update_proposal(self, posterior):

        self.logger.debug(f"Updating proposal distribution")
        self.logger.debug(f"Old proposal: {self.proposal}")
        self.proposal = posterior
        self.logger.debug(f"New proposal: {posterior}")

    def run_sequential(
        self,
        num_rounds,
        sims_per_round,
        **train_kwargs,
    ):
        
        self.logger.info(f"Starting sequential training with {num_rounds} rounds")
        self.logger.info(f"Simulations per round: {sims_per_round}")
        self.logger.info(f"Training kwargs: {train_kwargs}")

        for r in range(num_rounds):

            self.logger.info(f"--- Round {r}/{num_rounds} ---")

            self.run_round(r, sims_per_round)

            self.losses += self.train_estimator([r], **train_kwargs)
            # if torch.isnan(torch.tensor(self.losses[-1])):
            #     self.logger.error(f"Loss is nan, stopping execution...")
            #     return -1
            posterior = self.estimator.build_posterior(self.proposal_params)
            self.logger.debug(f"Built posterior: {posterior}")

            # self.validator.

            self.update_proposal(posterior)
            
            # Log progress
            progress = (r + 1) / num_rounds * 100
            self.logger.info(f"Round {r} completed - Progress: {progress:.1f}%")

        self.logger.info(f"Sequential training completed successfully")
        self.logger.info(f"Total simulations: {num_rounds * sims_per_round}")