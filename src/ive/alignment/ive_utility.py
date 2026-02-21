"""
IVE-weighted utility function for AI alignment analysis.

Standard utilitarian aggregation treats all individuals as interchangeable:
    U_total = sum(u_i) for all individuals i

IVE-weighted aggregation gives identified individuals non-substitutable weight:
    U_ive = sum(w_i * u_i) where w_i depends on identification level

The key philosophical insight: IVE-weighting makes welfare non-fungible
for identified individuals, avoiding certain repugnant conclusions but
introducing identifiability bias.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
import pandas as pd


@dataclass
class Individual:
    """A person in a moral scenario.

    Attributes:
        utility: Their welfare level (cardinal utility).
        identity_level: Graded identification (0=anonymous, 1=fully identified).
        group_size: How many people this individual represents.
        label: Descriptive label for display.
    """
    utility: float
    identity_level: float = 0.0
    group_size: int = 1
    label: str = ""

    @property
    def identified(self):
        return self.identity_level > 0.5


@dataclass
class Scenario:
    """A moral scenario comparing options over populations.

    Each option maps to a list of Individuals representing the state of
    the world if that option is chosen.
    """
    name: str
    description: str
    options: dict = field(default_factory=dict)  # name -> list[Individual]


def utilitarian_utility(individuals: List[Individual]) -> float:
    """Standard utilitarian aggregation: sum of individual utilities.

    U = sum(u_i * group_size_i)
    """
    return sum(ind.utility * ind.group_size for ind in individuals)


def ive_weighted_utility(
    individuals: List[Individual],
    coupling: float = 0.65,
    nonsubstitutability: float = 1.0,
) -> float:
    """IVE-weighted utility aggregation.

    The weight for each individual depends on their identification level:
        w_i = 1 + coupling * identity_level_i * nonsubstitutability

    For identified individuals (identity_level near 1), w_i > 1, making
    their welfare count more. The nonsubstitutability parameter controls
    how resistant identified welfare is to trade-offs.

    Args:
        individuals: List of Individual instances.
        coupling: IVE coupling strength (maps to identity_affect_coupling).
            0 = utilitarian (no IVE weighting).
        nonsubstitutability: How much extra weight identified individuals get.
            0 = fully substitutable (reduces to utilitarian).
            1 = moderate non-substitutability.

    Returns:
        Weighted utility sum.
    """
    total = 0.0
    for ind in individuals:
        weight = 1.0 + coupling * ind.identity_level * nonsubstitutability
        total += weight * ind.utility * ind.group_size
    return total


def ive_weighted_utility_with_floor(
    individuals: List[Individual],
    coupling: float = 0.65,
    floor_weight: float = 0.5,
) -> float:
    """IVE-weighted utility with a minimum weight floor.

    Ensures anonymous individuals are not ignored entirely. The floor_weight
    sets the minimum weight for completely anonymous individuals, preventing
    degenerate cases.

    Args:
        individuals: List of Individual instances.
        coupling: IVE coupling strength.
        floor_weight: Minimum weight for anonymous individuals (default 0.5).

    Returns:
        Weighted utility sum.
    """
    total = 0.0
    for ind in individuals:
        base = floor_weight + (1.0 - floor_weight) * ind.identity_level
        weight = base + coupling * ind.identity_level
        total += weight * ind.utility * ind.group_size
    return total


def compare_aggregations(
    scenario: Scenario,
    coupling_values: Optional[List[float]] = None,
) -> pd.DataFrame:
    """Compare utilitarian vs IVE-weighted outcomes for a scenario.

    For each option, computes both aggregation methods and shows which
    option each method prefers.

    Args:
        scenario: A Scenario with named options.
        coupling_values: List of coupling strengths to test.
            Default: [0.0, 0.3, 0.65, 1.0, 1.5]

    Returns:
        DataFrame with columns: option, utilitarian, ive_{coupling}, ...
    """
    if coupling_values is None:
        coupling_values = [0.0, 0.3, 0.65, 1.0, 1.5]

    rows = []
    for opt_name, individuals in scenario.options.items():
        row = {"option": opt_name}
        row["utilitarian"] = utilitarian_utility(individuals)
        for c in coupling_values:
            row[f"ive_c={c}"] = ive_weighted_utility(individuals, coupling=c)
        rows.append(row)

    return pd.DataFrame(rows)
