"""AI alignment extensions: IVE-weighted utility functions.

This module implements IVE-weighted aggregation for moral philosophy
scenarios, comparing standard utilitarian reasoning with
identification-sensitive weighting.

Key insight: standard utilitarianism treats welfare as perfectly
fungible and persons as interchangeable. The IVE suggests that
identified individuals receive non-substitutable weight, which
avoids certain repugnant conclusions but introduces identifiability bias.
"""

from .ive_utility import (
    Individual,
    Scenario,
    utilitarian_utility,
    ive_weighted_utility,
    ive_weighted_utility_with_floor,
    compare_aggregations,
)
from .parfit_scenarios import (
    repugnant_conclusion,
    trolley_identified_statistical,
    resource_allocation,
    scope_insensitivity,
    run_all_scenarios,
)
