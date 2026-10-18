import torch
from torch import nn

from sfmpe.core.distributions import Uniform
from sfmpe.data.round_dataset import RoundDataset
from sfmpe.flow.flow_model import FlowModel
from sfmpe.flow.path import AffinePath
from sfmpe.inference.fm_estimator import FlowMatchingEstimator


class _ScalarVelocity(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        t: torch.Tensor,
        theta: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        del t, x
        return torch.zeros_like(theta) + self.bias


class _ConstantValidationLoss(nn.Module):
    def __init__(self, velocity_model: _ScalarVelocity):
        super().__init__()
        self.velocity_model = velocity_model

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        if not self.velocity_model.training:
            return prediction.new_tensor(1.0)
        return (prediction - target).square().mean()


class _MemoryStore:
    def __init__(self, theta: torch.Tensor, observations: torch.Tensor):
        self.theta = theta
        self.observations = observations

    def get_all(self):
        round_ids = torch.zeros(self.theta.shape[0], dtype=torch.long)
        return self.theta, self.observations, round_ids


def _make_dataset(theta: torch.Tensor, observations: torch.Tensor) -> RoundDataset:
    return RoundDataset(_MemoryStore(theta, observations))


def test_training_early_stops_and_restores_best_validation_checkpoint(tmp_path):
    torch.manual_seed(7)
    velocity_model = _ScalarVelocity()
    flow_model = FlowModel(
        velocity_model=velocity_model,
        init_dist=Uniform(dim=1),
        path=AffinePath(),
    )
    estimator = FlowMatchingEstimator(
        flow_model,
        torch.optim.SGD(flow_model.parameters(), lr=0.05),
        _ConstantValidationLoss(velocity_model),
    )
    theta = torch.linspace(-1.0, 1.0, 20).unsqueeze(-1)
    observations = torch.zeros(20, 1)
    dataset = _make_dataset(theta, observations)
    checkpoint_path = tmp_path / "best-model.pth"

    train_losses = estimator.train_experimental(
        dataset,
        device="cpu",
        epochs=30,
        batch_size=4,
        shuffle=True,
        validation_fraction=0.25,
        stop_after_epochs=2,
        validation_seed=3,
        path=str(checkpoint_path),
    )

    assert len(train_losses) == 3
    assert estimator.epochs_trained == 3
    assert estimator.best_epoch == 1
    assert estimator.best_validation_loss == 1.0
    assert estimator.validation_loss_history == [1.0, 1.0, 1.0]
    assert checkpoint_path.is_file()
    saved_state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    for name, parameter in flow_model.state_dict().items():
        torch.testing.assert_close(parameter, saved_state[name])


def test_training_requires_nonempty_train_and_validation_sets():
    velocity_model = _ScalarVelocity()
    flow_model = FlowModel(
        velocity_model=velocity_model,
        init_dist=Uniform(dim=1),
        path=AffinePath(),
    )
    estimator = FlowMatchingEstimator(
        flow_model,
        torch.optim.SGD(flow_model.parameters(), lr=0.05),
        nn.MSELoss(),
    )

    try:
        estimator.train(
            _make_dataset(torch.zeros(1, 1), torch.zeros(1, 1)),
            epochs=2,
        )
    except ValueError as error:
        assert "At least two simulations" in str(error)
    else:
        raise AssertionError("A one-row dataset must not be split for validation.")
