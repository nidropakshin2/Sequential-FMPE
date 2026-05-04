import torch
import yaml
import time 
import pytest
import os


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
    


