from typing import cast

import torch

from sfmpe.tasks.Gaussian import GaussianTask
from sfmpe.utils.metrics import check_tarp, run_sbc, run_tarp


def make_task():
    return GaussianTask(
        {
            "prior": {
                "mean": [0.2, -0.1],
                "covariance": [[1.5, 0.3], [0.3, 1.0]],
            },
            "simulator": {
                "likelihood_shift": [0.1, -0.2],
                "likelihood_covariance": [[0.4, 0.1], [0.1, 0.6]],
            },
            "summary": {"type": "identity"},
        }
    )


def test_gaussian_task_simulator_prior_and_exact_posterior():
    torch.manual_seed(17)
    task = make_task()
    theta, x = task.simulate_dataset((12,))

    assert theta.shape == (12, 2)
    assert x.shape == (12, 2)
    torch.testing.assert_close(task.summarize(x), x)
    assert task.check_support(theta).all()

    repeated_x = task.simulate(theta[:1]).repeat(4, 1)
    posterior = task.true_posterior(repeated_x)
    samples = task.sample_true_posterior(repeated_x[:1], (7,))
    assert samples.shape == (1, 7, 2)
    repeated_covariance = cast(torch.Tensor, posterior.covariance_matrix)
    single_covariance = cast(
        torch.Tensor,
        task.true_posterior(repeated_x[0]).covariance_matrix,
    )
    assert torch.all(
        torch.diagonal(repeated_covariance) < torch.diagonal(single_covariance)
    )
    assert torch.isfinite(task.prior.log_prob(theta)).all()


def test_gaussian_task_supports_arbitrary_leading_batch_dimensions():
    task = make_task()
    theta = task.sample_prior((2, 3))
    observations = task.simulate(theta)

    assert observations.shape == (2, 3, 2)
    assert task.summarize(observations).shape == (2, 3, 2)


def test_gaussian_exact_posterior_supports_discarded_output_dimensions():
    task = GaussianTask(
        {
            "prior": {
                "mean": [0.1, -0.2, 0.3],
                "covariance": [
                    [1.0, 0.2, 0.1],
                    [0.2, 0.8, -0.1],
                    [0.1, -0.1, 0.7],
                ],
            },
            "simulator": {
                "likelihood_shift": [0.0, 0.1],
                "likelihood_covariance": [[0.4, 0.0], [0.0, 0.5]],
            },
        }
    )
    theta = task.sample_prior((5,))
    observations = task.simulate(theta)
    samples = task.sample_true_posterior(observations, (7,))

    assert observations.shape == (5, 2)
    assert samples.shape == (5, 7, 3)
    assert torch.isfinite(samples).all()


def test_sbc_and_tarp_accept_sbi_layout_posterior_samples():
    torch.manual_seed(23)
    task = make_task()
    num_observations = 32
    num_posterior_samples = 128
    thetas, observations = task.simulate_dataset((num_observations,))
    posterior_samples = task.sample_true_posterior(
        observations, (num_posterior_samples,)
    ).transpose(0, 1)

    ranks, dap_samples = run_sbc(
        thetas,
        posterior_samples,
        num_posterior_samples=num_posterior_samples,
    )
    assert ranks.shape == (num_observations, task.theta_dim)
    assert dap_samples.shape == (num_observations, task.theta_dim)
    assert torch.all((ranks >= 0) & (ranks <= num_posterior_samples))

    ecp, alpha = run_tarp(
        thetas,
        posterior_samples,
        num_bins=8,
    )
    assert ecp.shape == alpha.shape == (9,)
    assert torch.isfinite(ecp).all()
    assert torch.isfinite(alpha).all()
    area_to_curve, p_value = check_tarp(ecp, alpha)
    assert torch.isfinite(torch.tensor(area_to_curve))
    assert 0.0 <= p_value <= 1.0
