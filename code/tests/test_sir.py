import torch
from torch.distributions import biject_to

from sfmpe.tasks.SIR import SIRPrior, SIRTask


def test_sir_prior_sample_log_prob_and_support_are_consistent():
    prior = SIRPrior(gamma_range=(0.05, 0.5), beta_range=(0.5, 5.0))
    theta = prior.sample((256,))

    assert prior.support.check(theta).all()
    assert torch.isfinite(prior.log_prob(theta)).all()

    invalid = torch.tensor(
        [
            [1.0, 0.05],
            [0.025, 0.5],
        ]
    )
    assert not prior.support.check(invalid).any()
    assert torch.isneginf(prior.log_prob(invalid)).all()

    transform = biject_to(prior.support)
    unconstrained = torch.randn(32, 2)
    transformed = transform(unconstrained)
    assert prior.support.check(transformed).all()
    torch.testing.assert_close(transform.inv(transformed), unconstrained)
    assert torch.isfinite(
        transform.log_abs_det_jacobian(unconstrained, transformed)
    ).all()
    point = torch.tensor([0.3, -0.4], requires_grad=True)
    jacobian = torch.autograd.functional.jacobian(transform, point)
    expected_logdet = torch.linalg.slogdet(jacobian).logabsdet
    torch.testing.assert_close(
        transform.log_abs_det_jacobian(point, transform(point)),
        expected_logdet,
    )


def test_sir_task_support_matches_prior_support():
    task = SIRTask(
        {
            "prior": {
                "gamma_range": [0.05, 0.5],
                "beta_range": [0.5, 5.0],
            },
            "simulator": {"S": 100, "I": 10, "R": 0, "T": 10},
            "summary": {"type": "handmade"},
        }
    )
    theta = task.prior.sample((16,))
    invalid = torch.tensor([[1.0, 0.05], [0.025, 0.5]])

    torch.testing.assert_close(
        task.check_support(theta), task.prior.support.check(theta)
    )
    torch.testing.assert_close(
        task.check_support(invalid), task.prior.support.check(invalid)
    )
    assert task.check_support(theta).all()
    assert not task.check_support(invalid).any()
