from typing import cast

import torch


def nrmse(y_true, y_pred):
    mse = torch.mean((y_true - y_pred) ** 2)
    rmse = torch.sqrt(mse)
    norm_factor = torch.max(y_true) - torch.min(y_true)
    nrmse_value = rmse / norm_factor
    return nrmse_value


def r2_score(y_true, y_pred):
    mean_y_true = torch.mean(y_true)
    ss_tot = torch.sum((y_true - mean_y_true) ** 2)
    ss_res = torch.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2


class Validator:
    def __init__(self):
        pass


def _posterior_samples(
    xs: torch.Tensor,
    posterior,
    num_posterior_samples: int,
    show_progress_bar: bool = False,
) -> torch.Tensor:
    if isinstance(posterior, torch.Tensor):
        samples = posterior
    else:
        if not hasattr(posterior, "sample"):
            raise TypeError(
                "posterior must be a tensor of samples or expose sample()."
            )
        observations = xs
        if show_progress_bar:
            from tqdm.auto import tqdm

            observations = tqdm(observations)
        samples = torch.stack(
            [
                posterior.sample((num_posterior_samples,), x=x)
                for x in observations
            ],
            dim=1,
        )
    if samples.ndim != 3 or samples.shape[1] != xs.shape[0]:
        raise ValueError(
            "Posterior samples must have shape "
            "(num_posterior_samples, num_observations, theta_dim)."
        )
    if samples.shape[0] != num_posterior_samples:
        raise ValueError(
            "The requested number of posterior samples does not match the "
            "provided sample tensor."
        )
    return samples


def run_sbc(
    thetas: torch.Tensor,
    xs: torch.Tensor,
    posterior=None,
    num_posterior_samples: int = 1000,
    reduce_fns="marginals",
    num_workers: int = 1,
    show_progress_bar: bool = False,
    use_batched_sampling: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute marginal SBC ranks from posterior samples or an SBI posterior.

    For direct SFMPE use, pass posterior samples in ``xs`` with shape
    ``(num_posterior_samples, num_observations, theta_dim)`` and omit
    ``posterior``. With an SBI posterior, pass observations in ``xs`` and the
    posterior object in ``posterior``; it is sampled once for each observation.

    The return value follows current ``sbi.diagnostics.run_sbc``: ranks with
    shape ``(num_observations, theta_dim)`` and data-averaged posterior samples
    with shape ``(num_observations, theta_dim)``.
    """
    if num_workers != 1:
        raise NotImplementedError("SBC posterior sampling currently uses one worker.")
    del use_batched_sampling
    if reduce_fns != "marginals":
        raise NotImplementedError("Only marginal SBC ranks are currently supported.")
    if thetas.ndim != 2:
        raise ValueError("thetas must have shape (num_observations, theta_dim).")
    if posterior is None and xs.ndim == 3:
        samples = xs
    else:
        if xs.ndim != 2 or xs.shape[0] != thetas.shape[0]:
            raise ValueError("xs must have one observation for each theta.")
        sampling_posterior = posterior if posterior is not None else xs
        samples = _posterior_samples(
            xs,
            sampling_posterior,
            num_posterior_samples,
            show_progress_bar,
        )
    if samples.ndim != 3:
        raise ValueError(
            "Posterior samples must have shape "
            "(num_posterior_samples, num_observations, theta_dim)."
        )
    if samples.shape[1:] != thetas.shape:
        raise ValueError(
            f"Posterior samples have shape {tuple(samples.shape[1:])}; "
            f"expected {tuple(thetas.shape)}."
        )
    if num_posterior_samples != 1000 and samples.shape[0] != num_posterior_samples:
        raise ValueError(
            "The requested number of posterior samples does not match the "
            "provided sample tensor."
        )
    ranks = (samples < thetas.unsqueeze(0)).sum(dim=0)
    return ranks, samples[0]


def check_sbc(
    ranks: torch.Tensor,
    prior_samples: torch.Tensor,
    dap_samples: torch.Tensor,
    num_posterior_samples: int = 1000,
    num_c2st_repetitions: int = 1,
) -> dict[str, torch.Tensor]:
    """Return SBI-compatible KS uniformity and C2ST checks for SBC ranks."""
    from scipy.stats import kstest, uniform
    from sbi.utils.metrics import c2st

    if ranks.ndim != 2:
        raise ValueError("ranks must have shape (num_observations, theta_dim).")
    if prior_samples.shape != dap_samples.shape or prior_samples.shape != (
        ranks.shape[0],
        ranks.shape[1],
    ):
        raise ValueError("prior_samples and dap_samples must match ranks.")
    if num_c2st_repetitions < 1:
        raise ValueError("num_c2st_repetitions must be at least 1.")

    ks_pvals = torch.tensor(
        [
            kstest(
                values.detach().cpu().numpy(),
                uniform(loc=0, scale=num_posterior_samples).cdf,
            ).pvalue
            for values in ranks.T
        ],
        dtype=torch.float32,
    )
    rank_c2st = []
    for values in ranks.T:
        repetition_scores = [
            c2st(
                values.detach().to(dtype=torch.float32, device="cpu").reshape(-1, 1),
                torch.rand(
                    values.shape[0],
                    1,
                    device="cpu",
                )
                * num_posterior_samples,
                seed=repetition,
            )
            for repetition in range(num_c2st_repetitions)
        ]
        rank_c2st.append(torch.stack(repetition_scores).mean().reshape(()))
    dap_c2st = torch.stack(
        [
            c2st(
                prior_samples[:, dimension : dimension + 1].detach().cpu(),
                dap_samples[:, dimension : dimension + 1].detach().cpu(),
            ).reshape(-1)
            for dimension in range(ranks.shape[1])
        ]
    )
    return {
        "ks_pvals": ks_pvals,
        "c2st_ranks": torch.stack(rank_c2st),
        "c2st_dap": dap_c2st,
    }


def _l2_distance(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(first - second, dim=-1)


def run_tarp(
    thetas: torch.Tensor,
    xs: torch.Tensor,
    posterior=None,
    references: torch.Tensor | None = None,
    num_posterior_samples: int = 1000,
    num_workers: int = 1,
    show_progress_bar: bool = False,
    distance=None,
    num_bins: int | None = None,
    z_score_theta: bool = True,
    use_batched_sampling: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculate TARP expected coverage from samples or an SBI posterior.

    Direct sample input follows sbi's low-level layout
    ``(num_posterior_samples, num_observations, theta_dim)``.
    """
    if num_workers != 1:
        raise NotImplementedError("TARP posterior sampling currently uses one worker.")
    del use_batched_sampling
    if thetas.ndim != 2:
        raise ValueError("thetas must have shape (num_observations, theta_dim).")
    if posterior is None and xs.ndim == 3:
        samples = xs
    else:
        if xs.ndim != 2 or xs.shape[0] != thetas.shape[0]:
            raise ValueError("xs must have one observation for each theta.")
        samples = _posterior_samples(
            xs,
            posterior if posterior is not None else xs,
            num_posterior_samples,
            show_progress_bar,
        )
    if samples.shape[1:] != thetas.shape:
        raise ValueError(
            "Posterior samples must have shape "
            "(num_posterior_samples, num_observations, theta_dim)."
        )
    if references is None:
        low = thetas.min(dim=0).values
        high = thetas.max(dim=0).values
        if torch.any(high <= low):
            raise ValueError(
                "TARP requires variation in every theta dimension when "
                "references are not supplied."
            )
        references = torch.distributions.Uniform(low, high).sample(
            (thetas.shape[0],)
        )
    if references.shape != thetas.shape:
        raise ValueError("references must have the same shape as thetas.")
    if z_score_theta:
        low = thetas.min(dim=0, keepdim=True).values
        scale = (thetas.max(dim=0, keepdim=True).values - low).clamp_min(1e-10)
        samples = (samples - low.unsqueeze(0)) / scale.unsqueeze(0)
        thetas = (thetas - low) / scale
        references = (references - low) / scale

    distance_fn = _l2_distance if distance is None else distance
    sample_distances = distance_fn(references.unsqueeze(0), samples)
    theta_distances = distance_fn(references, thetas)
    coverage = (sample_distances < theta_distances.unsqueeze(0)).float().mean(0)
    bins = max(1, thetas.shape[0] // 10) if num_bins is None else num_bins
    if bins < 1:
        raise ValueError("num_bins must be at least 1.")
    histogram, alpha = torch.histogram(coverage.cpu(), bins=bins, density=True)
    histogram = histogram.to(device=thetas.device)
    alpha = alpha.to(device=thetas.device)
    ecp = torch.cat(
        (
            torch.zeros(1, device=thetas.device, dtype=histogram.dtype),
            histogram.cumsum(0) / histogram.sum(),
        )
    )
    return ecp, alpha


def check_tarp(
    ecp: torch.Tensor,
    alpha: torch.Tensor,
) -> tuple[float, float]:
    """Return SBI's area-to-curve and KS p-value for TARP."""
    from scipy.stats import kstest

    if ecp.ndim != 1 or alpha.ndim != 1 or ecp.shape != alpha.shape:
        raise ValueError("ecp and alpha must be one-dimensional with equal shape.")
    if alpha.numel() < 2:
        raise ValueError("ecp and alpha must contain at least two values.")
    midpoint = alpha.shape[0] // 2
    step = alpha[1] - alpha[0]
    area_to_curve = (
        (ecp[midpoint:] - alpha[midpoint:]) * step
    ).sum().item()
    p_value = cast(
        float,
        kstest(
            ecp.detach().cpu().numpy(),
            alpha.detach().cpu().numpy(),
        )[1],
    )
    return area_to_curve, p_value