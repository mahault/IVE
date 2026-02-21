"""Utility functions for state indexing and effect size computation."""

import numpy as np


def state_index(context: int, outcome: int, cost: int) -> int:
    """Encode (context, outcome, cost) into a single state index.

    Args:
        context: 0=Statistical, 1=Identified
        outcome: 0=NotSaved, 1=Saved
        cost:    0=NoCost, 1=Cost

    Returns:
        Integer state index in [0, 7].
    """
    return context * 4 + outcome * 2 + cost


def decode_state(idx: int) -> dict:
    """Decode a state index back into (context, outcome, cost)."""
    context = idx // 4
    rem = idx % 4
    outcome = rem // 2
    cost = rem % 2
    return {"context": context, "outcome": outcome, "cost": cost}


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d for two independent groups."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def cohens_d_from_proportions(p1: float, p2: float, n1: int, n2: int) -> float:
    """Compute Cohen's d from two proportions (help rates).

    Uses the arcsine transformation: h = 2 * arcsin(sqrt(p)).
    Cohen's h is then converted to approximate d.
    """
    h1 = 2.0 * np.arcsin(np.sqrt(p1))
    h2 = 2.0 * np.arcsin(np.sqrt(p2))
    return h1 - h2  # Cohen's h (approximately equal to d for proportions)


def log_odds_ratio(p1: float, p2: float, eps: float = 1e-8) -> float:
    """Compute log odds ratio between two help rates."""
    odds1 = (p1 + eps) / (1.0 - p1 + eps)
    odds2 = (p2 + eps) / (1.0 - p2 + eps)
    return np.log(odds1 / odds2)
