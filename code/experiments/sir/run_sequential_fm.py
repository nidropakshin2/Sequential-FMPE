import yaml
import torch
import time
import matplotlib.pyplot as plt


from sfmpe.tasks.SIR import SIRTask


config_path = './config.yaml'
with open(config_path) as f:
    config = yaml.safe_load(f)
setup = config['setup']
task_config = config['task']

import os

if not os.path.exists("./logs"):
    os.mkdir("./logs")
if not os.path.exists("./models"):
    os.mkdir("./models")


device = setup['device']
task = SIRTask(task_config, device)
task.summary.to(device)
logger = task.logger
logger.info("Starting SIR task")

from sfmpe.inference.fm_estimator import FlowMatchingEstimator
from sfmpe.flow.flow_model import FlowModel
from sfmpe.flow.velocity import SimpleVelocityField
from sfmpe.flow.path import AffinePath
from sfmpe.core.distributions import Uniform, Normal


init_dist = Normal(dim=task.theta_dim)
velocity_model = SimpleVelocityField(task.theta_dim, task.data_dim)
path = AffinePath()
flow_model = FlowModel(velocity_model, init_dist, path)
logger.info("Initialized flow model")

optimizer = torch.optim.Adam([
                        # {"params": task.summary.parameters(), "lr":1e-4},
                        {"params": flow_model.parameters(), "lr":1e-4}
                        ])
loss_fn = torch.nn.MSELoss()

def preprocessor(theta, x):
    s = task.simulate(theta).to(device)
    x = task.summarize(s).to(device)
    return theta, x

flow_model.to(device)
fm_estimator = FlowMatchingEstimator(flow_model, optimizer, loss_fn)#, dataset_prepocessor=preprocessor)
logger.info("Initialized flow matching estimator")

from sfmpe.inference.sequential.round_manager import RoundManager
from sfmpe.inference.sequential.proposal import Proposal, ProposalParams

theta_0, x_0 = task.simulate_dataset((1,))
# theta_0 = torch.tensor([[1.6667, 0.4016]]).to(device)
# x_0 = task.summarize(task.simulate(theta_0))
# theta_0, x_0 = theta_0.expand(5, -1), x_0.expand(5, -1)
task.logger.info(f"Starting SFMPE for theta {theta_0}")


# TODO: сделать парсинг из конфига
params = ProposalParams()
params.task = task
params.method = "NPE-A"
params.theta_0 = theta_0
params.x_0 = x_0
params.method_params = {'scale': 0.8, 'quantile': 0.3}


datetime = time.strftime("%Y-%m-%d_%H_%M_%S")
path = f"./models/SIR_latest.pth.tar.gz"

num_rounds      = setup['num_rounds']
sims_per_round  = setup['sims_per_round']
epochs          = setup['epochs']
show_every      = setup['show_every']
clean_sampling  = setup['clean_sampling']
upd_x           = setup['upd_x']

manager = RoundManager(task, fm_estimator, params,
                       device=device,
                    #    storage_dir=f"./datasets/SIR_{datetime}"
                       )

# TODO: инициализация из конфига
manager.run_sequential(num_rounds=num_rounds,
                       sims_per_round=sims_per_round,
                       path=path, epochs=epochs, 
                       show_every=show_every,
                       clean_sampling=clean_sampling,
                       upd_x=upd_x)
loss = manager.losses
_, ax = plt.subplots()
ax.plot(loss)