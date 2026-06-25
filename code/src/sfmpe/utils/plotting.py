import matplotlib.pyplot as plt

def dist_plot(*dists, true_param=None):
    # assert len(dist1.shape) == len(dist2.shape) and dist1.shape[-1] == dist2.shape[-1]
    theta_dim = dists[0].shape[-1]
    num_bins  = [int(dists[i].shape[0] ** 0.5) for i in range(len(dists))]
    
    _, ax = plt.subplots(len(dists), theta_dim, sharex='col', figsize=(3*theta_dim, 3*len(dists)))
    for j in range(theta_dim):
        for thetas, i in zip(dists, range(len(dists))):
            counts, bins, patches = ax[i][j].hist(thetas[:, :, j], bins=num_bins[i], alpha=0.5, density=True)

            bin_centers = (bins[:-1] + bins[1:]) / 2

            ax[i][j].plot(bin_centers, counts, 'b-', linewidth=1)

            if true_param is not None:
                ax[i][j].axvline(x=true_param[:, j], linestyle='--', color='green')


class Validator:
    def __init__(self, manager):
        self.manager = manager
        self.task = manager.task
        self.logger = manager.logger
        self.posterior = manager.build_posterior()
    

    def plot_comparison(self, size=(10000, 1), path=None):
        if path is not None:
            return NotImplementedError("saving plots is not implemented")
        else:
            self.logger.debug("Plotting comparison...")
            dist_plot(self.manager.store.theta[-1], 
                      self.posterior.sample(size[:-1]),
                      true_param=self.manager.proposal_params.theta_0 if self.manager.proposal_params is not None else None)