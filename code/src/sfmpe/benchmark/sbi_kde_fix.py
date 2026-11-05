"""Runtime fix for the KDE bandwidth cross-validation bug in sbi 0.21.

sbi's ``get_kde(..., bandwidth="cv")`` selects the KDE bandwidth with a
"zoom-in" grid search over ``sklearn.GridSearchCV``. When the best bandwidth
lands on the first grid point, sbi's zoom step uses the grid *index* as a
bandwidth and a negative step:

    diff = (lower - upper) / steps   # negative because lower < upper
    lower = grid.best_index_ - diff  # grid index used as a bandwidth
    upper = grid.best_index_ + diff  # -> negative upper bound

so the next ``np.linspace`` range contains negative bandwidths (with the
initial range ``[0.1 * std, 0.5 * std]`` and ``steps = 10`` it becomes
``linspace(+0.04 * std, -0.04 * std)``: 5 of 10 candidates are negative).
sklearn's ``KernelDensity`` rejects every negative candidate per CV fold
("The 'bandwidth' parameter of KernelDensity must be a float in the range
(0.0, inf) ... Got np.float64(-0.003986...)"), which floods the log with
``FitFailedWarning`` tracebacks (5 negative candidates x 20 cv partitions =
the observed "100 fits failed out of a total of 200") and collapses the zoom
instead of narrowing it. The final bandwidth then comes from a corrupted
search, so the KDE posterior of SMC-ABC is smoothed with an essentially
arbitrary sampled bandwidth.

``install_sbi_kde_fix()`` replaces ``get_kde`` with a corrected copy of the
same algorithm (the boundary zoom uses the bandwidth *value*, stays inside a
positive, ordered range; the dead ``best_index_ == steps`` sentinel of the
original is dropped) and rebinds every sbi namespace that had already
imported the original. On sbi versions where the bandwidth selection was
rewritten without ``GridSearchCV`` the installer is a no-op.
"""

import inspect
import sys
import types
from typing import Optional, Union

import numpy as np
import torch
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from torch.distributions.transforms import IndependentTransform, identity_transform

from sbi.types import transform_types

# Keyword defaults hardcoded inside sbi 0.21's get_kde.
_KDE_KWARGS = dict(
    kernel="gaussian",
    algorithm="auto",
    metric="euclidean",
    atol=0,
    rtol=0,
    breadth_first=True,
    leaf_size=40,
    metric_params=None,
)


def _make_sbi_kde_wrapper(kde, transform):
    """Build sbi's own KDEWrapper around the fitted KDE.

    A late import so the patched ``get_kde`` returns exactly the object type
    sbi's SMCABC expects (``sbi.utils.KDEWrapper`` re-exports the same class).
    """
    from sbi.utils.kde import KDEWrapper

    return KDEWrapper(kde, transform)


def _cv_grid_search(
    transformed_samples: np.ndarray,
    bandwidth_range: np.ndarray,
    num_cv_partitions: int,
):
    """The sklearn grid search of sbi's bandwidth zoom (kept as a hook so
    tests can drive the zoom decisions without fitting anything)."""
    return GridSearchCV(
        KernelDensity(**_KDE_KWARGS),
        {"bandwidth": bandwidth_range},
        cv=num_cv_partitions,
    )


def _fixed_cv_bandwidth(
    transformed_samples: np.ndarray,
    num_cv_partitions: int,
    num_cv_repetitions: int,
) -> float:
    """Cross-validated bandwidth: the same zoom algorithm as sbi 0.21 with
    the boundary zoom fixed (bandwidth values instead of grid indices, and
    the search never leaves a positive, ordered range)."""
    scale = float(transformed_samples.std())
    if scale <= 0:
        # All-identical particles: keep the grid positive but degenerate.
        scale = 1e-6
    steps = 10
    lower = 0.1 * scale
    upper = 0.5 * scale
    current_best = -1e10
    bandwidth_selected = lower

    for _ in range(num_cv_repetitions):
        bandwidth_range = np.linspace(lower, upper, steps)
        grid = _cv_grid_search(transformed_samples, bandwidth_range, num_cv_partitions)
        grid.fit(transformed_samples)
        bandwidth_selected = float(grid.best_params_["bandwidth"])

        if abs(current_best - grid.best_score_) <= 0.001:
            # The score stopped improving; keep the last best bandwidth.
            break
        current_best = grid.best_score_

        best_bandwidth = float(bandwidth_range[grid.best_index_])
        best_index = int(grid.best_index_)
        if best_index <= 0:
            # Zoom around the boundary bandwidth VALUE instead of its index.
            step = (upper - lower) / (2 * (steps - 1))
            lower, upper = best_bandwidth - step, best_bandwidth + step
        else:
            second_best_index = list(
                grid.cv_results_["rank_test_score"]
            ).index(2)
            upper = float(bandwidth_range[second_best_index])
            lower = float(bandwidth_range[best_index])
            if upper < lower:
                upper, lower = lower, upper
        # Never search non-positive or inverted bandwidths.
        lower = max(lower, 0.001 * scale)
        upper = max(upper, lower)
    return bandwidth_selected


def fixed_get_kde(
    samples: torch.Tensor,
    bandwidth: Union[float, str] = "cv",
    transform: transform_types = None,
    sample_weights: Optional[np.ndarray] = None,
    num_cv_partitions: int = 20,
    num_cv_repetitions: int = 5,
):
    # Faithful copy of sbi 0.21's get_kde (Apache-2.0, sbi-dev/sbi) with only
    # the ``bandwidth="cv"`` zoom corrected; keep in sync with that version.
    transform_ = identity_transform if transform is None else transform

    # Make sure transform has event dimension and returns scalar log_prob.
    if transform_.event_dim == 0:
        transform_ = IndependentTransform(transform_, reinterpreted_batch_ndims=1)
    if isinstance(bandwidth, str):
        assert bandwidth in ["cv", "scott", "silvermann"], "invalid kde bandwidth name."

    transformed_samples = transform_(samples).numpy()
    num_samples, dim_samples = transformed_samples.shape

    if bandwidth == "scott":
        bandwidth_selected = num_samples ** (-1.0 / (dim_samples + 4))
    elif bandwidth == "silvermann":
        bandwidth_selected = (num_samples * (dim_samples + 2) / 4.0) ** (
            -1.0 / (dim_samples + 4)
        )
    elif bandwidth == "cv":
        bandwidth_selected = _fixed_cv_bandwidth(
            transformed_samples, num_cv_partitions, num_cv_repetitions
        )
    elif float(bandwidth) > 0:
        bandwidth_selected = float(bandwidth)
    else:
        raise ValueError("bandwidth must be positive, 'scott', 'silvermann' or 'cv'")

    # Run final fit with selected bandwidth.
    kde = KernelDensity(bandwidth=bandwidth_selected, **_KDE_KWARGS)
    kde.fit(transformed_samples, sample_weight=sample_weights)

    return _make_sbi_kde_wrapper(kde, transform_)


def install_sbi_kde_fix() -> bool:
    """Patch sbi's buggy ``get_kde`` in every sbi namespace.

    Returns whether the patch was applied this call (False when sbi is new
    enough to not need it, sbi is missing, or the fix is already installed).
    """
    try:
        from sbi.utils import kde as kde_module
    except ImportError:
        return False
    current = getattr(kde_module, "get_kde", None)
    if current is None or getattr(current, "_sfmpe_fixed", False):
        return False
    try:
        source = inspect.getsource(current)
    except (OSError, TypeError):
        source = ""
    if "GridSearchCV" not in source:
        # A newer sbi already rewrote the bandwidth selection.
        return False

    fixed_get_kde._sfmpe_fixed = True
    kde_module.get_kde = fixed_get_kde
    # Rebind aliases that sbi modules imported before the patch (sbi.utils
    # re-exports get_kde, SMCABC imported it at module scope, ...).
    for name, module in list(sys.modules.items()):
        if not name.startswith("sbi") or not isinstance(module, types.ModuleType):
            continue
        try:
            module_dict = vars(module)
        except TypeError:
            continue
        if module_dict.get("get_kde") is current:
            module_dict["get_kde"] = fixed_get_kde
    return True
