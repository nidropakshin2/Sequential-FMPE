from copy import deepcopy
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from sfmpe.data.round_dataset import RoundDataset
from sfmpe.flow.flow_model import FlowModel
from sfmpe.flow.sampler import ODESampler
from sfmpe.inference.sequential.proposal import Proposal
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
            self.dataset_prepocessor = lambda theta, x, *_: (theta, x)
        else:
            self.dataset_prepocessor = dataset_prepocessor

    def train(self, dataset: RoundDataset, **kwargs):
        """Train to the best validation loss, stopping after a patience window.

        ``epochs`` is the maximum number of epochs. ``stop_after_epochs`` is
        the number of consecutive epochs without validation improvement before
        early stopping, following the convention used by ``sbi``.
        """
        settings = {
            "device": kwargs.pop("device", "cpu"),
            "max_epochs": kwargs.pop("epochs", 1000),
            "path": kwargs.pop("path", None),
            "batch_size": kwargs.pop("batch_size", 256),
            "shuffle": kwargs.pop("shuffle", False),
            "show_every": kwargs.pop("show_every", None),
            "cfg_mode": kwargs.pop("cfg_mode", False),
            "num_workers": kwargs.pop("num_workers", 0),
            "validation_fraction": kwargs.pop("validation_fraction", 0.1),
            "stop_after_epochs": kwargs.pop("stop_after_epochs", 100),
            "min_delta": kwargs.pop("min_delta", 0.0),
            "validation_seed": kwargs.pop("validation_seed", None),
        }
        if kwargs:
            raise TypeError(
                f"train() got unexpected keyword arguments: {', '.join(kwargs)}"
            )
        if settings["max_epochs"] < 1:
            raise ValueError("epochs must be at least 1.")
        if settings["batch_size"] < 1:
            raise ValueError("batch_size must be at least 1.")
        if settings["stop_after_epochs"] < 1:
            raise ValueError("stop_after_epochs must be at least 1.")
        if settings["min_delta"] < 0:
            raise ValueError("min_delta must be non-negative.")
        validation_fraction = settings["validation_fraction"]
        if not 0.0 < validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between 0 and 1.")
        if len(dataset) < 2:
            raise ValueError(
                "At least two simulations are required for a train/validation split."
            )

        generator = None
        if settings["validation_seed"] is not None:
            generator = torch.Generator().manual_seed(settings["validation_seed"])
        training_size = max(1, int(len(dataset) * (1.0 - validation_fraction)))
        validation_size = len(dataset) - training_size
        if training_size < 1:
            raise ValueError(
                "validation_fraction leaves no training simulations; reduce it."
            )
        train_dataset, validation_dataset = random_split(
            dataset,
            [training_size, validation_size],
            generator=generator,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=min(settings["batch_size"], training_size),
            shuffle=settings["shuffle"],
            num_workers=settings["num_workers"],
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=min(settings["batch_size"], validation_size),
            shuffle=False,
            num_workers=settings["num_workers"],
        )

        self.device = settings["device"]
        self.flow_model.to(self.device)
        self.optimizer.zero_grad(set_to_none=True)
        self.train_loss_history: list[float] = []
        self.validation_loss_history: list[float] = []
        self.best_epoch = 0
        self.best_validation_loss = float("inf")
        self.epochs_trained = 0
        best_validation_loss = float("inf")
        best_model_state = None
        epochs_since_improvement = 0
        checkpoint_path = settings["path"]
        if checkpoint_path is not None:
            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

        validation_batches = []
        self.flow_model.eval()
        with torch.no_grad():
            for theta_batch, x_batch in validation_loader:
                theta, x = self.dataset_prepocessor(
                    theta_batch.to(self.device),
                    x_batch.to(self.device),
                )
                theta_0 = self.flow_model.init_dist.sample_like(theta).to(self.device)
                time = self.flow_model.path.time_dist.sample(
                    (*theta_0.shape[:-1], 1)
                ).to(self.device)
                cfg_mask = (
                    torch.rand(*x.shape[:-1], device=x.device) > 0.1
                    if settings["cfg_mode"]
                    else None
                )
                validation_batches.append((theta, x, theta_0, time, cfg_mask))

        for epoch in range(1, settings["max_epochs"] + 1):
            self.flow_model.train()
            epoch_loss_sum = 0.0
            training_examples = 0
            for theta_batch, x_batch in train_loader:
                theta, x = self.dataset_prepocessor(
                    theta_batch.to(self.device),
                    x_batch.to(self.device),
                )
                theta_0 = self.flow_model.init_dist.sample_like(theta).to(self.device)
                time = self.flow_model.path.time_dist.sample(
                    (*theta_0.shape[:-1], 1)
                ).to(self.device)
                loss = self._flow_matching_loss(
                    theta,
                    x,
                    theta_0,
                    time,
                    settings["cfg_mode"],
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Non-finite training loss at epoch {epoch}."
                    )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                batch_size = theta.shape[0]
                epoch_loss_sum += loss.detach().item() * batch_size
                training_examples += batch_size
            train_loss = epoch_loss_sum / training_examples
            self.train_loss_history.append(train_loss)

            self.flow_model.eval()
            validation_loss_sum = 0.0
            validation_examples = 0
            with torch.no_grad():
                for theta, x, theta_0, time, cfg_mask in validation_batches:
                    loss = self._flow_matching_loss(
                        theta,
                        x,
                        theta_0,
                        time,
                        settings["cfg_mode"],
                        cfg_mask,
                    )
                    if not torch.isfinite(loss):
                        raise FloatingPointError(
                            f"Non-finite validation loss at epoch {epoch}."
                        )
                    batch_size = theta.shape[0]
                    validation_loss_sum += loss.item() * batch_size
                    validation_examples += batch_size
            validation_loss = validation_loss_sum / validation_examples
            self.validation_loss_history.append(validation_loss)
            self.epochs_trained = epoch

            if validation_loss < best_validation_loss - settings["min_delta"]:
                best_validation_loss = validation_loss
                self.best_epoch = epoch
                best_model_state = deepcopy(self.flow_model.state_dict())
                epochs_since_improvement = 0
                if checkpoint_path is not None:
                    torch.save(best_model_state, checkpoint_path)
            else:
                epochs_since_improvement += 1

            show_every = settings["show_every"]
            if show_every is not None and epoch % show_every == 0:
                self.logger.info(
                    "Epoch %d: train loss %.6f, validation loss %.6f",
                    epoch,
                    train_loss,
                    validation_loss,
                )

            if epochs_since_improvement >= settings["stop_after_epochs"]:
                self.logger.info(
                    "Early stopping at epoch %d; best validation loss %.6f "
                    "at epoch %d.",
                    epoch,
                    best_validation_loss,
                    self.best_epoch,
                )
                break

        if best_model_state is None:
            raise RuntimeError("Training completed without a finite validation loss.")
        self.flow_model.load_state_dict(best_model_state)
        self.best_validation_loss = best_validation_loss
        self.flow_model.eval()
        self.logger.info(
            "Training completed after %d epochs; best validation loss %.6f "
            "at epoch %d.",
            self.epochs_trained,
            best_validation_loss,
            self.best_epoch,
        )
        return self.train_loss_history

    def train_experimental(self, dataset: RoundDataset, **kwargs):
        """Backward-compatible entry point for the minibatch training loop."""
        return self.train(dataset, **kwargs)

    def _flow_matching_loss(
        self,
        theta_1: torch.Tensor,
        x: torch.Tensor,
        theta_0: torch.Tensor,
        time: torch.Tensor,
        cfg_mode: bool,
        cfg_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        theta_t = self.flow_model.path.sample(theta_0, theta_1, time).to(self.device)
        target_velocity = self.flow_model.path.velocity(theta_0, theta_1, time)
        if cfg_mode:
            if cfg_mask is None:
                cfg_mask = torch.rand(*x.shape[:-1], device=x.device) > 0.1
            masked_x = x * cfg_mask.unsqueeze(-1).to(x.dtype)
            velocity = (
                0.999
                * self.flow_model.velocity_model(t=time, theta=theta_t, x=masked_x)
                + 0.001
                * self.flow_model.velocity_model(
                    t=time,
                    theta=theta_t,
                    x=torch.zeros_like(x),
                )
            )
        else:
            velocity = self.flow_model.velocity_model(t=time, theta=theta_t, x=x)
        return self.loss_fn(velocity, target_velocity)

    def load(self, path):
        assert os.path.exists(path)
        if self.logger:
            self.logger.info(f"Loading model from {path}...")
        # WARNING: понять откуда брать device
        self.device = 'cpu'
        self.flow_model.load_state_dict(torch.load(path, weights_only=True, map_location=self.device))
        
        return self.flow_model

    def build_posterior(self, params=None):
        if self.logger:
            self.logger.info("Building posterior...")
        if params is None:
            return ODESampler(self.flow_model)
        else: 
            return Proposal(self.flow_model, params)
