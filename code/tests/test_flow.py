import torch
import yaml
import time 

from sfmpe.tasks.SIR import SIRTask
from sfmpe.flow.flow_model import FlowModel
from sfmpe.flow.velocity import SimpleVelocityField
from sfmpe.flow.path import AffinePath
from sfmpe.core.distributions import Uniform
from sfmpe.inference.fm_estimator import FlowMatchingEstimator
from sfmpe.data.round_dataset import RoundDataset
from sfmpe.data.simulation_store import SimulationStore
from sfmpe.inference.sequential.round_manager import RoundManager


with open('./code/tests/test_configs/sir_config.yaml', 'r') as file:  
    config = yaml.safe_load(file)  

device = torch.device('cpu')
sir = SIRTask(config, device)

theta, x = sir.simulate_dataset((10,))

print(theta.shape, x.shape, sep="\n")

init_dist = Uniform()
velocity_model = SimpleVelocityField(sir.theta_dim, sir.data_dim)
path = AffinePath()
flow = FlowModel(velocity_model, init_dist, path)


optimizer = torch.optim.Adam(flow.parameters())
loss_fn = torch.nn.MSELoss()
FM_Estimator = FlowMatchingEstimator(flow, optimizer, loss_fn)

store = SimulationStore()
theta, x = sir.simulate_dataset((2, 2,))
store.add(theta, x, 0)
theta, x = sir.simulate_dataset((3, 1,))
store.add(theta, x, 1)

dataset = RoundDataset(store, rounds=[0, 1])

datetime = time.strftime("%Y-%m-%d_%H_%M_%S")
path = f"./code/tests/test_models/test_SIR_{datetime}.pth.tar.gz"
# losses = FM_Estimator.train(dataset, path=path, epochs=2)

manager = RoundManager(sir, FM_Estimator, sir.prior)
manager.run_sequential(3, 3, path=path, epochs=2)

