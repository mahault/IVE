"""Visualization functions for IVE model results."""

import numpy as np
import matplotlib.pyplot as plt


def plot_help_rates_bar(
    help_stat: float,
    help_id: float,
    target_stat: float = None,
    target_id: float = None,
    title: str = "Help rates: Statistical vs Identified",
    ax: plt.Axes = None,
) -> plt.Axes:
    """Bar plot comparing model (and optionally target) help rates."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))

    contexts = ["Statistical", "Identified"]
    model_vals = [help_stat, help_id]
    x = np.arange(len(contexts))
    width = 0.35

    if target_stat is not None and target_id is not None:
        target_vals = [target_stat, target_id]
        ax.bar(x - width / 2, target_vals, width, label="Empirical", alpha=0.7, color="C0")
        ax.bar(x + width / 2, model_vals, width, label="Model", alpha=0.7, color="C1")
        ax.legend()
    else:
        ax.bar(x, model_vals, width * 2, alpha=0.7, color=["C0", "C1"])

    ax.set_xticks(x)
    ax.set_xticklabels(contexts)
    ax.set_ylim(0, 1)
    ax.set_ylabel("P(Help)")
    ax.set_title(title)
    return ax


def plot_help_vs_parameter(
    param_values: np.ndarray,
    help_stat_curve: np.ndarray,
    help_id_curve: np.ndarray,
    param_name: str = "Parameter",
    tuned_value: float = None,
    ax: plt.Axes = None,
) -> plt.Axes:
    """Line plot of P(Help) vs a swept parameter for both contexts."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    ax.plot(param_values, help_stat_curve, label="Statistical", color="C0")
    ax.plot(param_values, help_id_curve, label="Identified", color="C1", linestyle="--")

    if tuned_value is not None:
        ax.axvline(tuned_value, color="k", linestyle=":", alpha=0.5, label="Tuned")

    ax.set_xlabel(param_name)
    ax.set_ylabel("P(Help)")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def plot_ive_delta(
    param_values: np.ndarray,
    help_stat_curve: np.ndarray,
    help_id_curve: np.ndarray,
    param_name: str = "Parameter",
    ax: plt.Axes = None,
) -> plt.Axes:
    """Plot the IVE magnitude (delta = P_help_id - P_help_stat) vs parameter."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    delta = help_id_curve - help_stat_curve
    ax.plot(param_values, delta, color="C2")
    ax.axhline(0.0, color="k", linestyle=":", alpha=0.3)
    ax.set_xlabel(param_name)
    ax.set_ylabel("IVE magnitude (P_id - P_stat)")
    ax.grid(True, alpha=0.3)
    return ax


def plot_sweep_heatmap(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    values: np.ndarray,
    x_label: str = "x",
    y_label: str = "y",
    title: str = "Heatmap",
    ax: plt.Axes = None,
) -> plt.Axes:
    """Heatmap of a 2D parameter sweep."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(
        values,
        origin="lower",
        aspect="auto",
        extent=[x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]],
        vmin=0.0,
        vmax=1.0,
    )
    plt.colorbar(im, ax=ax, label=title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    return ax


def plot_effect_sizes(
    conditions: list,
    empirical_d: list,
    model_d: list,
    ax: plt.Axes = None,
) -> plt.Axes:
    """Compare empirical vs model effect sizes across conditions."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    x = np.arange(len(conditions))
    width = 0.35

    ax.bar(x - width / 2, empirical_d, width, label="Empirical", alpha=0.7, color="C0")
    ax.bar(x + width / 2, model_d, width, label="Model", alpha=0.7, color="C1")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=30, ha="right")
    ax.set_ylabel("Cohen's d")
    ax.set_title("Effect sizes: Empirical vs Model")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    return ax
