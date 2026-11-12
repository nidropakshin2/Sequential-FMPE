import torch
import yaml
import time 
import pytest
import os
from types import SimpleNamespace


from sfmpe.tasks.SIR import SIRTask

@pytest.fixture(scope='session')
def sir_task():
    config_path = './code/tests/test_configs/sir_config.yaml'
    with open(config_path) as f:
        config = yaml.safe_load(f)
    device = 'cpu'
    return SIRTask(config, device)


def test_sir_config_structure():
    config_path = './code/tests/test_configs/sir_config.yaml'
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    expected = {
        "prior": {
            "gamma_range": [0.05, 0.5],
            "beta_range": [0.5, 5.0]},
        "simulator": {
            "S": [1000, 2000],
            "I": [100, 300],
            "R": [0, 100],
            "T": [60, 120]},
        "summary": "handmade",
        "logger": {
            "name": "SIR",
            "level": "INFO",
            "log_file_path": "./code/tests/logs/sir.log"
        }
    }
    assert config == expected


def test_simulator(sir_task):
    assert sir_task is not None

    theta, x = sir_task.simulate_dataset((10,))

    assert (theta.shape, x.shape) == (torch.Size([10, sir_task.theta_dim]), torch.Size([10, sir_task.data_dim]))


from sfmpe.flow.flow_model import FlowModel
from sfmpe.flow.velocity import SimpleVelocityField
from sfmpe.flow.path import AffinePath
from sfmpe.core.distributions import Uniform

@pytest.fixture
def flow_model(sir_task):
    assert sir_task is not None

    init_dist = Uniform()
    velocity_model = SimpleVelocityField(sir_task.theta_dim, sir_task.data_dim)
    path = AffinePath()
    flow = FlowModel(velocity_model, init_dist, path)
    return flow


from sfmpe.inference.fm_estimator import FlowMatchingEstimator

@pytest.fixture
def fm_estimator(flow_model):
    assert flow_model is not None

    optimizer = torch.optim.Adam(flow_model.parameters())
    loss_fn = torch.nn.MSELoss()
    FM_Estimator = FlowMatchingEstimator(flow_model, optimizer, loss_fn)

    return FM_Estimator


from sfmpe.data.simulation_store import SimulationStore

# TODO: проблемы с размерностями при добавлении батчей разного размера, нужно привести к единому виду, 
# например (B, N, dim) и потом объединять по первому измерению
def test_simulation_store(sir_task):
    assert sir_task is not None

    store = SimulationStore("./code/tests/test_datasets")
    theta, x = sir_task.simulate_dataset((1, 2,))
    store.add(theta, x, 0)
    theta, x = sir_task.simulate_dataset((1, 1,))
    store.add(theta, x, 1)

    theta_all, x_all, rounds = store.get_all()
    assert (theta_all.shape, x_all.shape, rounds.shape) == (torch.Size([3, sir_task.theta_dim]), torch.Size([3, sir_task.data_dim]), torch.Size([3]))


from sfmpe.data.round_dataset import RoundDataset

def test_train(sir_task, fm_estimator):
    assert sir_task is not None

    store = SimulationStore("./code/tests/test_datasets")
    theta, x = sir_task.simulate_dataset((100, 2,))
    store.add(theta, x, 0)
    theta, x = sir_task.simulate_dataset((100, 2,))
    store.add(theta, x, 1)

    dataset = RoundDataset(store, rounds=[0, 1])

    datetime = time.strftime("%Y-%m-%d_%H_%M_%S")
    path = f"./code/tests/test_models/test_SIR_{datetime}.pth.tar.gz"

    losses = fm_estimator.train(dataset, path=path, epochs=10)
    assert losses is not None
    assert os.path.exists(path)


from sfmpe.inference.sequential.round_manager import RoundManager
from sfmpe.inference.sequential.proposal import Proposal, ProposalParams

def test_round_manager(sir_task, fm_estimator):

    # TODO проблемы с размерностями при переходе к большему числу
    theta, x = sir_task.simulate_dataset((2,))

    params = ProposalParams()
    params.method = "NPE-A"
    params.x_0 = x

    datetime = time.strftime("%Y-%m-%d_%H_%M_%S")
    path = f"./code/tests/test_models/test_SIR_{datetime}.pth.tar.gz"
    manager = RoundManager(sir_task, fm_estimator, params)
    manager.run_sequential(3, 100, path=path, epochs=10)


def _truncated_test_manager(max_prior_draws=1000, condition_count=1):
    class UnitBoxSupport:
        def check(self, samples):
            return ((samples >= 0) & (samples <= 1)).all(dim=-1)

    class UnitBoxPrior:
        support = UnitBoxSupport()

        def sample(self, shape, **kwargs):
            device = kwargs.get("device", "cpu")
            return torch.rand(*shape, 2, device=device)

    class Logger:
        def debug(self, *_args, **_kwargs):
            pass

        def info(self, *_args, **_kwargs):
            pass

    task = SimpleNamespace(
        prior=UnitBoxPrior(),
        theta_dim=2,
        logger=Logger(),
    )
    params = ProposalParams()
    params.method = "Truncated"
    params.method_params = {
        "quantile": 0.8,
        "region_samples": 100,
        "padding": 0.0,
        "max_prior_draws": max_prior_draws,
    }
    params.theta_0 = torch.zeros(1, condition_count, 2)

    class Posterior:
        def sample(self, shape, device=None):
            centers = torch.linspace(0.2, 0.8, condition_count, device=device)
            return centers.reshape(1, condition_count, 1) + 0.3 * (
                torch.rand(*shape, condition_count, 2, device=device) - 0.5
            )

        def log_prob(self, _samples):
            raise AssertionError("Truncated prior sampling must not use log_prob")

    manager = RoundManager(task, object(), params, logger=task.logger)
    manager.update_proposal(Posterior())
    return manager


def test_truncated_sampling_draws_from_prior_inside_posterior_region():
    manager = _truncated_test_manager()

    samples = manager.clean_sample_experimental((8,), clean_sampling=False)

    assert manager.truncated_bounds is not None
    lower, upper = manager.truncated_bounds
    assert samples.shape == (8, manager.K, manager.task.theta_dim)
    assert torch.all(samples >= lower.unsqueeze(0))
    assert torch.all(samples <= upper.unsqueeze(0))
    assert torch.all(manager.task.prior.support.check(samples))


def test_legacy_truncated_sampler_uses_conditional_prior():
    manager = _truncated_test_manager()

    samples = manager.clean_sample((5, manager.K), clean_sampling=False)

    assert samples.shape == (5, manager.K, manager.task.theta_dim)
    assert manager.truncated_bounds is not None
    lower, upper = manager.truncated_bounds
    assert torch.all(samples >= lower.unsqueeze(0))
    assert torch.all(samples <= upper.unsqueeze(0))


def test_truncated_sampling_keeps_conditions_in_their_own_regions():
    manager = _truncated_test_manager(condition_count=2)

    samples = manager.clean_sample_experimental((8,), clean_sampling=False)

    assert samples.shape == (8, 2, manager.task.theta_dim)
    assert torch.all(samples[:, 0] < 0.5)
    assert torch.all(samples[:, 1] > 0.5)
    lower, upper = manager.truncated_bounds
    assert torch.all(samples >= lower.unsqueeze(0))
    assert torch.all(samples <= upper.unsqueeze(0))


def test_truncated_sampling_fails_when_region_cannot_be_sampled():
    manager = _truncated_test_manager(max_prior_draws=4)
    manager.truncated_bounds = (
        torch.full((manager.K, manager.task.theta_dim), 2.0),
        torch.full((manager.K, manager.task.theta_dim), 3.0),
    )

    with pytest.raises(RuntimeError, match="accepted 0/"):
        manager._sample_from_truncated_prior((1,))
