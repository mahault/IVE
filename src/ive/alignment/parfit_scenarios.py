"""
Classic moral philosophy scenarios parameterized for IVE analysis.

Implements:
1. Repugnant Conclusion (Parfit, 1984)
2. Trolley Problem variants with identified vs statistical victims
3. Resource Allocation (triage) with identifiability
4. Scope Insensitivity test
"""

import numpy as np
import pandas as pd
from .ive_utility import (
    Individual, Scenario,
    utilitarian_utility, ive_weighted_utility, compare_aggregations,
)


def repugnant_conclusion(
    small_pop_size: int = 10,
    small_pop_utility: float = 100.0,
    large_pop_multiplier: float = 2.0,
    large_pop_utility: float = 1.0,
    identified_fraction_small: float = 1.0,
    identified_fraction_large: float = 0.0,
) -> Scenario:
    """Parfit's Repugnant Conclusion.

    Option A: small population, each with high utility, identified.
    Option B: large population (enough to exceed A's total utility under
              utilitarianism), each with barely-worth-living utility, anonymous.

    Standard utilitarianism always prefers B if the population is large enough.
    IVE-weighting can prefer A because identified welfare weighs more.

    Args:
        small_pop_size: Number of people in Option A.
        small_pop_utility: Welfare level per person in A.
        large_pop_multiplier: Option B pop = multiplier * A_total_util / B_per_person.
        large_pop_utility: Welfare level per person in B (barely worth living).
        identified_fraction_small: Fraction of A that is identified (0-1).
        identified_fraction_large: Fraction of B that is identified (0-1).

    Returns:
        Scenario with options "A_small_happy" and "B_large_barely".
    """
    total_a = small_pop_size * small_pop_utility
    large_pop_size = int(total_a * large_pop_multiplier / max(large_pop_utility, 0.01))

    option_a = [
        Individual(
            utility=small_pop_utility,
            identity_level=identified_fraction_small,
            group_size=small_pop_size,
            label=f"{small_pop_size} identified, happy people",
        )
    ]
    option_b = [
        Individual(
            utility=large_pop_utility,
            identity_level=identified_fraction_large,
            group_size=large_pop_size,
            label=f"{large_pop_size} anonymous, barely-living people",
        )
    ]

    return Scenario(
        name="Repugnant Conclusion",
        description=(
            f"Option A: {small_pop_size} people with utility {small_pop_utility} "
            f"(total={total_a}). "
            f"Option B: {large_pop_size} people with utility {large_pop_utility} "
            f"(total={large_pop_size * large_pop_utility}). "
            "Utilitarianism prefers B. Does IVE-weighting prefer A?"
        ),
        options={"A_small_happy": option_a, "B_large_barely": option_b},
    )


def trolley_identified_statistical(
    n_track: int = 5,
    victim_identified: bool = True,
    victim_identity_level: float = 1.0,
    track_identity_level: float = 0.0,
) -> Scenario:
    """Trolley problem with identifiability manipulation.

    Option A (do nothing): n_track people die on main track (anonymous).
    Option B (divert): 1 person dies on side track (identified or not).

    Standard utilitarianism: always divert (save n > 1).
    IVE-weighted: if side-track victim is identified, diversion costs more.

    Args:
        n_track: Number on the main track.
        victim_identified: Whether side-track victim is identified.
        victim_identity_level: Identity level of side-track victim.
        track_identity_level: Identity level of main-track people.

    Returns:
        Scenario with options "do_nothing" and "divert".
    """
    id_level = victim_identity_level if victim_identified else 0.0

    # Utility = lives saved (positive) or lost (negative)
    option_do_nothing = [
        Individual(
            utility=-1.0,
            identity_level=track_identity_level,
            group_size=n_track,
            label=f"{n_track} anonymous people die",
        ),
        Individual(
            utility=0.0,
            identity_level=id_level,
            group_size=1,
            label="Side-track person survives",
        ),
    ]
    option_divert = [
        Individual(
            utility=0.0,
            identity_level=track_identity_level,
            group_size=n_track,
            label=f"{n_track} anonymous people saved",
        ),
        Individual(
            utility=-1.0,
            identity_level=id_level,
            group_size=1,
            label=f"{'Identified' if victim_identified else 'Anonymous'} person dies",
        ),
    ]

    return Scenario(
        name=f"Trolley ({'identified' if victim_identified else 'anonymous'} victim)",
        description=(
            f"Main track: {n_track} anonymous. Side track: 1 "
            f"{'identified' if victim_identified else 'anonymous'} person. "
            f"Divert the trolley?"
        ),
        options={"do_nothing": option_do_nothing, "divert": option_divert},
    )


def resource_allocation(
    option_a_beneficiaries: int = 1,
    option_a_benefit: float = 80.0,
    option_a_identified: bool = True,
    option_b_beneficiaries: int = 100,
    option_b_benefit: float = 5.0,
    option_b_identified: bool = False,
) -> Scenario:
    """Resource allocation: concentrate vs distribute.

    Option A: Help few identified people significantly.
    Option B: Help many anonymous people modestly.

    Standard utilitarianism prefers B if total_B > total_A.
    IVE-weighted may prefer A if identified weight is high enough.

    Args:
        option_a_beneficiaries: Number helped in Option A.
        option_a_benefit: Benefit per person in A.
        option_a_identified: Whether A beneficiaries are identified.
        option_b_beneficiaries: Number helped in Option B.
        option_b_benefit: Benefit per person in B.
        option_b_identified: Whether B beneficiaries are identified.
    """
    option_a = [
        Individual(
            utility=option_a_benefit,
            identity_level=1.0 if option_a_identified else 0.0,
            group_size=option_a_beneficiaries,
            label=f"{option_a_beneficiaries} identified, large benefit",
        )
    ]
    option_b = [
        Individual(
            utility=option_b_benefit,
            identity_level=1.0 if option_b_identified else 0.0,
            group_size=option_b_beneficiaries,
            label=f"{option_b_beneficiaries} anonymous, small benefit",
        )
    ]

    total_a = option_a_beneficiaries * option_a_benefit
    total_b = option_b_beneficiaries * option_b_benefit

    return Scenario(
        name="Resource Allocation",
        description=(
            f"Option A: help {option_a_beneficiaries} identified person(s), "
            f"benefit={option_a_benefit} (total={total_a}). "
            f"Option B: help {option_b_beneficiaries} anonymous, "
            f"benefit={option_b_benefit} (total={total_b})."
        ),
        options={"A_concentrated": option_a, "B_distributed": option_b},
    )


def scope_insensitivity(
    group_sizes=None,
    utility_per_person: float = 10.0,
    identity_level: float = 0.0,
) -> list:
    """Scope insensitivity test.

    Standard IVE prediction: willingness to help is relatively insensitive
    to the number of anonymous victims (scope insensitivity), but sensitive
    to identification level.

    Args:
        group_sizes: List of group sizes to test.
        utility_per_person: Benefit per person if helped.
        identity_level: Identification level of victims.

    Returns:
        List of Scenarios with increasing group sizes.
    """
    if group_sizes is None:
        group_sizes = [1, 10, 100, 1000, 10000]

    scenarios = []
    for n in group_sizes:
        option_help = [
            Individual(
                utility=utility_per_person,
                identity_level=identity_level,
                group_size=n,
                label=f"{n} people helped",
            )
        ]
        option_no_help = [
            Individual(
                utility=0.0,
                identity_level=identity_level,
                group_size=n,
                label=f"{n} people not helped",
            )
        ]
        scenarios.append(Scenario(
            name=f"Scope (n={n}, id={identity_level:.1f})",
            description=f"Help {n} people (identity={identity_level:.1f})?",
            options={"help": option_help, "no_help": option_no_help},
        ))

    return scenarios


def run_all_scenarios(coupling_values=None):
    """Run all Parfit scenarios across coupling values.

    Returns:
        Dict mapping scenario_name -> DataFrame of results.
    """
    if coupling_values is None:
        coupling_values = [0.0, 0.3, 0.65, 1.0, 1.5, 2.0]

    results = {}

    # Repugnant conclusion
    rc = repugnant_conclusion()
    results["repugnant_conclusion"] = compare_aggregations(rc, coupling_values)

    # Trolley: identified victim
    trolley_id = trolley_identified_statistical(victim_identified=True)
    results["trolley_identified"] = compare_aggregations(trolley_id, coupling_values)

    # Trolley: anonymous victim
    trolley_anon = trolley_identified_statistical(victim_identified=False)
    results["trolley_anonymous"] = compare_aggregations(trolley_anon, coupling_values)

    # Resource allocation
    ra = resource_allocation()
    results["resource_allocation"] = compare_aggregations(ra, coupling_values)

    # Scope insensitivity: anonymous
    scope_anon = scope_insensitivity(identity_level=0.0)
    scope_rows = []
    for sc in scope_anon:
        for c in coupling_values:
            help_u = ive_weighted_utility(sc.options["help"], coupling=c)
            scope_rows.append({"n": sc.options["help"][0].group_size,
                              "coupling": c, "ive_utility": help_u})
    results["scope_anonymous"] = pd.DataFrame(scope_rows)

    # Scope insensitivity: identified
    scope_id = scope_insensitivity(identity_level=1.0)
    scope_rows = []
    for sc in scope_id:
        for c in coupling_values:
            help_u = ive_weighted_utility(sc.options["help"], coupling=c)
            scope_rows.append({"n": sc.options["help"][0].group_size,
                              "coupling": c, "ive_utility": help_u})
    results["scope_identified"] = pd.DataFrame(scope_rows)

    return results
