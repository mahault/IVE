"""
Parameter fitting pipeline for the IVE active inference model.

Supports:
- Grid search (fast, coarse)
- MLE via scipy.optimize (refined)
- Summary-statistic matching (when only means/SDs are available)

Target data: Moche et al. (2024) or any dataset providing
help rates / donation amounts by condition (identified vs statistical).
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution

from .agent import get_help_probability, DEFAULTS
from .utils import cohens_d_from_proportions


def model_predictions(params: dict, n_samples: int = 300) -> dict:
    """Get model-predicted help rates for both contexts.

    Args:
        params: Dict with keys matching agent.build_agent args.
        n_samples: Monte Carlo samples per condition.

    Returns:
        Dict with p_help_stat, p_help_id, delta, cohens_h.
    """
    common = {k: v for k, v in params.items() if k not in ("context",)}

    p_stat = get_help_probability(**common, context="stat", n_samples=n_samples)
    p_id = get_help_probability(**common, context="id", n_samples=n_samples)

    return {
        "p_help_stat": p_stat,
        "p_help_id": p_id,
        "delta": p_id - p_stat,
        "cohens_h": cohens_d_from_proportions(p_id, p_stat, n_samples, n_samples),
    }


def squared_error_loss(
    params: dict,
    target_stat: float,
    target_id: float,
    n_samples: int = 300,
) -> float:
    """Compute squared error between model and target help rates."""
    preds = model_predictions(params, n_samples=n_samples)
    return (preds["p_help_stat"] - target_stat) ** 2 + (preds["p_help_id"] - target_id) ** 2


def grid_search(
    target_stat: float = 0.30,
    target_id: float = 0.55,
    param_grids: dict = None,
    n_samples: int = 200,
    verbose: bool = True,
) -> tuple:
    """Grid search over parameter space to minimize squared error.

    Args:
        target_stat: Target P(Help) for statistical context.
        target_id: Target P(Help) for identified context.
        param_grids: Dict mapping param names to arrays of values to try.
            Defaults to a reasonable grid over delta_C, delta_p, cost_penalty.
        n_samples: MC samples per evaluation.
        verbose: Print progress.

    Returns:
        (best_params, best_result) where best_result includes predictions and error.
    """
    if param_grids is None:
        param_grids = {
            "delta_C": np.linspace(0.0, 3.0, 6),
            "delta_p": np.linspace(0.0, 0.7, 6),
            "cost_penalty": np.linspace(0.1, 2.0, 6),
        }

    # Fixed params (not swept)
    fixed = {
        "p_success_base": DEFAULTS["p_success_base"],
        "util_saved_base": DEFAULTS["util_saved_base"],
        "gamma_base": DEFAULTS["gamma_base"],
        "delta_gamma": DEFAULTS["delta_gamma"],
    }

    grid_keys = list(param_grids.keys())
    grid_arrays = [param_grids[k] for k in grid_keys]

    from itertools import product
    combos = list(product(*grid_arrays))
    total = len(combos)

    best_error = np.inf
    best_params = None
    best_result = None

    for idx, combo in enumerate(combos):
        params = dict(fixed)
        for k, v in zip(grid_keys, combo):
            params[k] = v

        preds = model_predictions(params, n_samples=n_samples)
        error = (preds["p_help_stat"] - target_stat) ** 2 + (preds["p_help_id"] - target_id) ** 2

        if error < best_error:
            best_error = error
            best_params = dict(params)
            best_result = {**preds, "error": error}

        if verbose and (idx + 1) % 20 == 0:
            print(f"  [{idx + 1}/{total}] best error = {best_error:.4f}")

    if verbose:
        print(f"Grid search complete. Best error = {best_error:.4f}")
        print(f"  P(Help|stat) = {best_result['p_help_stat']:.3f} (target {target_stat})")
        print(f"  P(Help|id)   = {best_result['p_help_id']:.3f} (target {target_id})")

    return best_params, best_result


def fit_mle(
    target_stat: float = 0.30,
    target_id: float = 0.55,
    n_samples: int = 300,
    bounds: dict = None,
    method: str = "differential_evolution",
) -> tuple:
    """Fit model parameters via maximum likelihood / minimum squared error.

    Uses scipy optimizers to find the parameter set that best matches
    target help rates.

    Args:
        target_stat: Target P(Help) for statistical context.
        target_id: Target P(Help) for identified context.
        n_samples: MC samples per evaluation.
        bounds: Dict mapping param names to (low, high) tuples.
        method: "differential_evolution" (global) or "nelder-mead" (local).

    Returns:
        (best_params, best_result).
    """
    if bounds is None:
        bounds = {
            "delta_C": (0.0, 4.0),
            "delta_p": (0.0, 0.7),
            "cost_penalty": (0.1, 3.0),
        }

    fixed = {
        "p_success_base": DEFAULTS["p_success_base"],
        "util_saved_base": DEFAULTS["util_saved_base"],
        "gamma_base": DEFAULTS["gamma_base"],
        "delta_gamma": DEFAULTS["delta_gamma"],
    }

    param_names = list(bounds.keys())
    param_bounds = [bounds[k] for k in param_names]

    def objective(x):
        params = dict(fixed)
        for k, v in zip(param_names, x):
            params[k] = v
        return squared_error_loss(params, target_stat, target_id, n_samples=n_samples)

    if method == "differential_evolution":
        result = differential_evolution(
            objective,
            bounds=param_bounds,
            maxiter=30,
            tol=1e-4,
            seed=42,
            polish=False,
        )
    else:
        x0 = np.array([(b[0] + b[1]) / 2 for b in param_bounds])
        result = minimize(objective, x0, method="Nelder-Mead",
                          options={"maxiter": 200, "xatol": 0.01})

    best_params = dict(fixed)
    for k, v in zip(param_names, result.x):
        best_params[k] = float(v)

    preds = model_predictions(best_params, n_samples=max(n_samples, 500))
    best_result = {**preds, "error": result.fun, "optimizer_result": result}

    return best_params, best_result


def fit_to_study_summary(
    mean_donation_stat: float,
    mean_donation_id: float,
    max_donation: float,
    sd_stat: float = None,
    sd_id: float = None,
    n_stat: int = 100,
    n_id: int = 100,
    n_samples: int = 300,
) -> tuple:
    """Fit model to summary statistics from a single study.

    Converts donation amounts to help rates by normalizing by max_donation,
    then fits the model to match those rates.

    Args:
        mean_donation_stat: Mean donation in statistical condition.
        mean_donation_id: Mean donation in identified condition.
        max_donation: Maximum possible donation.
        sd_stat, sd_id: Standard deviations (for effect size computation).
        n_stat, n_id: Sample sizes per condition.
        n_samples: MC samples for model evaluation.

    Returns:
        (best_params, result) where result includes target rates and fit quality.
    """
    target_stat = mean_donation_stat / max_donation
    target_id = mean_donation_id / max_donation

    best_params, best_result = grid_search(
        target_stat=target_stat,
        target_id=target_id,
        n_samples=n_samples,
        verbose=True,
    )

    # Add empirical effect size if SDs provided
    if sd_stat is not None and sd_id is not None:
        pooled_sd = np.sqrt(
            ((n_stat - 1) * sd_stat ** 2 + (n_id - 1) * sd_id ** 2) / (n_stat + n_id - 2)
        )
        empirical_d = (mean_donation_id - mean_donation_stat) / pooled_sd if pooled_sd > 0 else 0
        best_result["empirical_cohens_d"] = empirical_d

    best_result["target_stat"] = target_stat
    best_result["target_id"] = target_id

    return best_params, best_result
