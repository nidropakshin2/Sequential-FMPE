import torch
from torch.distributions import Beta, InverseGamma, Normal

from sfmpe.tasks.StochVol import StochVolPrior, StochVolSimulator, StochVolTask


def make_prior():
    return StochVolPrior(
        {
            "mu_loc": 0.0,
            "mu_scale": 0.2,
            "phi_alpha": 20.0,
            "phi_beta": 1.5,
            "sigma_ig_shape": 2.5,
            "sigma_ig_scale": 0.025,
            "m_loc": -1.5,
            "m_scale": 1.0,
        }
    )


def test_stochvol_prior_sample_log_prob_and_support_are_consistent():
    prior = make_prior()
    theta = prior.sample((128,))

    assert theta.shape == (128, 4)
    assert prior.support.check(theta).all()
    assert torch.isfinite(prior.log_prob(theta)).all()

    mu, phi, sigma, mean_log_variance = theta.unbind(dim=-1)
    expected_log_prob = (
        Normal(prior.mu_loc, prior.mu_scale).log_prob(mu)
        + Beta(prior.phi_alpha, prior.phi_beta).log_prob((phi + 1.0) / 2.0)
        - torch.log(torch.tensor(2.0))
        + InverseGamma(
            concentration=prior.sigma_ig_shape,
            rate=prior.sigma_ig_scale,
        ).log_prob(sigma.square())
        + torch.log(2.0 * sigma)
        + Normal(prior.m_loc, prior.m_scale).log_prob(mean_log_variance)
    )
    torch.testing.assert_close(prior.log_prob(theta), expected_log_prob)

    invalid = theta[:2].clone()
    invalid[0, 1] = 1.0
    invalid[1, 2] = 0.0
    assert torch.isneginf(prior.log_prob(invalid)).all()


def test_stochvol_simulator_starts_at_stationary_log_variance(monkeypatch):
    simulator = StochVolSimulator({"T": 1})
    theta = torch.tensor(
        [
            [0.0, 0.5, 0.2, -2.0],
            [0.0, -0.5, 0.4, 1.0],
        ]
    )
    initial_noise = torch.tensor([1.0, -1.0])
    return_noise = torch.ones(2)
    state_noise = torch.zeros(2)
    random_values = iter((initial_noise, return_noise, state_noise))

    def fixed_randn(size, *, device=None, dtype=None):
        assert torch.Size(size) == torch.Size((2,))
        return next(random_values).to(device=device, dtype=dtype)

    monkeypatch.setattr(torch, "randn", fixed_randn)
    observations = simulator.simulate(theta)

    expected_initial_log_variance = theta[:, 3] + (
        theta[:, 2] / torch.sqrt(1.0 - theta[:, 1].square())
    ) * initial_noise
    expected = torch.exp(expected_initial_log_variance / 2.0)
    torch.testing.assert_close(observations[:, 0, 0], expected)


def test_stochvol_task_support_matches_prior_support():
    config = {
        "prior": {
            "mu_loc": 0.0,
            "mu_scale": 0.2,
            "phi_alpha": 20.0,
            "phi_beta": 1.5,
            "sigma_ig_shape": 2.5,
            "sigma_ig_scale": 0.025,
            "m_loc": -1.5,
            "m_scale": 1.0,
        },
        "simulator": {"T": 10},
        "summary": {"type": "handmade"},
    }
    task = StochVolTask(config)
    theta = task.prior.sample((16,))
    invalid = theta.clone()
    invalid[0, 1] = 1.1
    invalid[1, 2] = 0.0

    torch.testing.assert_close(
        task.check_support(theta), task.prior.support.check(theta)
    )
    torch.testing.assert_close(
        task.check_support(invalid), task.prior.support.check(invalid)
    )
    assert task.check_support(theta).all()
    assert not task.check_support(invalid).all()
