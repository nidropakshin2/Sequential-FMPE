from collections.abc import Callable

import torch

from sfmpe.utils.metrics import check_sbc, check_tarp, run_sbc, run_tarp


def _rbf_mmd(first: torch.Tensor, second: torch.Tensor) -> float:
    first = first.to(dtype=torch.float64)
    second = second.to(dtype=torch.float64)
    combined = torch.cat((first, second))
    distances = torch.pdist(combined).square()
    positive_distances = distances[distances > 0]
    if positive_distances.numel() == 0:
        return 0.0
    bandwidth_squared = positive_distances.median().clamp_min(1e-12)

    def kernel(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        squared_distance = torch.cdist(left, right).square()
        return torch.exp(-squared_distance / (2.0 * bandwidth_squared))

    return float(
        (
            kernel(first, first).mean()
            + kernel(second, second).mean()
            - 2.0 * kernel(first, second).mean()
        ).item()
    )


def _matrix_sqrt(matrix: torch.Tensor) -> torch.Tensor:
    eigenvalues, eigenvectors = torch.linalg.eigh(
        (matrix + matrix.T) / 2
    )
    return (
        eigenvectors
        * eigenvalues.clamp_min(0).sqrt().unsqueeze(0)
    ) @ eigenvectors.T


def _gaussian_wasserstein2(
    first: torch.Tensor,
    second: torch.Tensor,
) -> float:
    first = first.to(dtype=torch.float64)
    second = second.to(dtype=torch.float64)
    first_mean = first.mean(dim=0)
    second_mean = second.mean(dim=0)
    first_covariance = torch.cov(first.T)
    second_covariance = torch.cov(second.T)
    first_covariance_sqrt = _matrix_sqrt(first_covariance)
    middle = (
        first_covariance_sqrt
        @ second_covariance
        @ first_covariance_sqrt
    )
    covariance_term = (
        torch.trace(first_covariance)
        + torch.trace(second_covariance)
        - 2.0 * torch.trace(_matrix_sqrt(middle))
    )
    squared_distance = (
        (first_mean - second_mean).square().sum()
        + covariance_term.clamp_min(0)
    )
    return float(squared_distance.clamp_min(0).sqrt().item())


def exact_posterior_metrics(
    samples: torch.Tensor,
    reference_samples: torch.Tensor,
) -> dict[str, float]:
    """Compare samples to a known posterior using sample-based discrepancies."""
    if samples.ndim != 2 or reference_samples.ndim != 2:
        raise ValueError("Both sample sets must have shape (samples, dimensions).")
    if samples.shape[1] != reference_samples.shape[1]:
        raise ValueError("Sample sets must have the same parameter dimension.")
    if min(samples.shape[0], reference_samples.shape[0]) < 5:
        raise ValueError("At least five samples per distribution are required.")

    from sbi.utils.metrics import c2st

    mean_error = torch.linalg.vector_norm(
        samples.mean(0) - reference_samples.mean(0)
    )
    covariance_error = torch.linalg.matrix_norm(
        torch.cov(samples.T) - torch.cov(reference_samples.T),
        ord="fro",
    )
    return {
        "mean_error": float(mean_error.item()),
        "covariance_frobenius_error": float(covariance_error.item()),
        "gaussian_wasserstein2": _gaussian_wasserstein2(
            samples, reference_samples
        ),
        "mmd_rbf": _rbf_mmd(samples, reference_samples),
        "c2st": float(
            c2st(
                samples.detach().to(dtype=torch.float32, device="cpu"),
                reference_samples.detach().to(dtype=torch.float32, device="cpu"),
            ).item()
        ),
    }


def calibration_metrics(
    thetas: torch.Tensor,
    posterior_samples: torch.Tensor,
    num_bins: int | None = None,
    include_c2st: bool = False,
) -> dict[str, float | list[float]]:
    """Aggregate SBC/TARP over independent target-specific inference runs."""
    if posterior_samples.ndim != 3:
        raise ValueError(
            "posterior_samples must have shape (samples, targets, parameters)."
        )
    if posterior_samples.shape[1:] != thetas.shape:
        raise ValueError("Posterior samples and truths have incompatible shapes.")
    num_samples = posterior_samples.shape[0]
    ranks, dap_samples = run_sbc(
        thetas,
        posterior_samples,
        num_posterior_samples=num_samples,
    )
    sbc = None
    ks_pvalues = None
    if include_c2st:
        sbc = check_sbc(
            ranks,
            thetas,
            dap_samples,
            num_posterior_samples=num_samples,
        )
        rank_c2st = sbc["c2st_ranks"].tolist()
        dap_c2st = sbc["c2st_dap"].tolist()
    else:
        from scipy.stats import kstest, uniform

        ks_pvalues = torch.tensor(
            [
                kstest(
                    values.detach().cpu().numpy(),
                    uniform(loc=0, scale=num_samples).cdf,
                ).pvalue
                for values in ranks.T
            ]
        )
        rank_c2st = []
        dap_c2st = []
    bins = num_bins or min(20, max(4, thetas.shape[0] // 10))
    ecp, alpha = run_tarp(
        thetas,
        posterior_samples,
        num_bins=bins,
    )
    atc, tarp_ks_pvalue = check_tarp(ecp, alpha)
    ks_values = (
        sbc["ks_pvals"].tolist()
        if sbc is not None
        else ks_pvalues.tolist() if ks_pvalues is not None else []
    )
    return {
        "sbc_ks_pvalues": ks_values,
        "sbc_rank_c2st": rank_c2st,
        "sbc_dap_c2st": dap_c2st,
        "tarp_area_to_curve": atc,
        "tarp_ks_pvalue": tarp_ks_pvalue,
        "tarp_ecp": ecp.tolist(),
        "tarp_alpha": alpha.tolist(),
    }


def posterior_predictive_metrics(
    task,
    observation: torch.Tensor,
    posterior_samples: torch.Tensor,
    simulate: Callable[[torch.Tensor], torch.Tensor],
) -> dict[str, float | str | list[float]]:
    """Compare observed summary features with posterior predictive summaries."""
    support_check = getattr(task, "check_support", None)
    valid_fraction = 1.0
    if callable(support_check):
        valid_samples = torch.as_tensor(support_check(posterior_samples)).bool()
        if valid_samples.shape != posterior_samples.shape[:-1]:
            raise ValueError(
                "Task support check must return one boolean per posterior sample."
            )
        valid_fraction = float(valid_samples.float().mean().item())
        posterior_samples = posterior_samples[valid_samples]
        if posterior_samples.shape[0] == 0:
                return {
                    "ppc_available": 0.0,
                    "ppc_valid_sample_fraction": valid_fraction,
                    "ppc_unavailable_reason": (
                        "All posterior samples are outside task support."
                    ),
                }
    with torch.no_grad():
        predictive_data = simulate(posterior_samples)
        predictive_summaries = task.summarize(predictive_data)
    observation = observation.reshape(1, -1).to(predictive_summaries)
    predictive_mean = predictive_summaries.mean(dim=0)
    predictive_std = predictive_summaries.std(dim=0, unbiased=False).clamp_min(1e-8)
    z_scores = (observation.squeeze(0) - predictive_mean).abs() / predictive_std
    tail_probabilities = (
        (predictive_summaries - predictive_mean).abs()
        >= (observation - predictive_mean).abs()
    ).float().mean(dim=0)
    return {
        "ppc_available": 1.0,
        "ppc_mean_abs_z": float(z_scores.mean().item()),
        "ppc_max_abs_z": float(z_scores.max().item()),
        "ppc_two_sided_tail_probabilities": tail_probabilities.tolist(),
        "ppc_valid_sample_fraction": valid_fraction,
    }


def sensitivity_metrics(
    observation: torch.Tensor,
    base_samples: torch.Tensor,
    sample_posterior: Callable[[torch.Tensor, int], torch.Tensor],
    observation_scale: torch.Tensor,
    num_samples: int,
    perturbation_scale: float,
) -> dict[str, float]:
    """Measure posterior-mean displacement under a standardized x perturbation."""
    if perturbation_scale <= 0:
        raise ValueError("perturbation_scale must be positive.")
    direction = torch.ones_like(observation)
    perturbation = direction * observation_scale.to(observation) * perturbation_scale
    positive_samples = sample_posterior(observation + perturbation, num_samples)
    negative_samples = sample_posterior(observation - perturbation, num_samples)
    baseline_mean = base_samples.mean(dim=0)
    baseline_scale = base_samples.std(dim=0, unbiased=False).clamp_min(1e-8)
    positive_shift = (
        (positive_samples.mean(0) - baseline_mean) / baseline_scale
    ).norm()
    negative_shift = (
        (negative_samples.mean(0) - baseline_mean) / baseline_scale
    ).norm()
    return {
        "sensitivity_plus_posterior_sd": float(positive_shift.item()),
        "sensitivity_minus_posterior_sd": float(negative_shift.item()),
    }
