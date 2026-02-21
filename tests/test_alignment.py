"""Tests for the alignment module."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ive.alignment import (
    Individual,
    Scenario,
    utilitarian_utility,
    ive_weighted_utility,
    ive_weighted_utility_with_floor,
    compare_aggregations,
    repugnant_conclusion,
    trolley_identified_statistical,
    resource_allocation,
    scope_insensitivity,
    run_all_scenarios,
)


class TestIndividual:
    def test_basic_creation(self):
        ind = Individual(utility=10.0, identity_level=0.8, group_size=5)
        assert ind.utility == 10.0
        assert ind.identity_level == 0.8
        assert ind.group_size == 5

    def test_identified_threshold(self):
        assert Individual(utility=1, identity_level=0.6).identified is True
        assert Individual(utility=1, identity_level=0.4).identified is False
        assert Individual(utility=1, identity_level=0.5).identified is False


class TestUtilitarianUtility:
    def test_single_person(self):
        inds = [Individual(utility=10.0)]
        assert utilitarian_utility(inds) == 10.0

    def test_group(self):
        inds = [Individual(utility=5.0, group_size=3)]
        assert utilitarian_utility(inds) == 15.0

    def test_multiple_groups(self):
        inds = [
            Individual(utility=10.0, group_size=2),
            Individual(utility=5.0, group_size=3),
        ]
        assert utilitarian_utility(inds) == 35.0

    def test_ignores_identity(self):
        """Utilitarian utility should not depend on identity_level."""
        inds_anon = [Individual(utility=10.0, identity_level=0.0)]
        inds_id = [Individual(utility=10.0, identity_level=1.0)]
        assert utilitarian_utility(inds_anon) == utilitarian_utility(inds_id)


class TestIVEWeightedUtility:
    def test_coupling_zero_equals_utilitarian(self):
        """With coupling=0, IVE-weighted should equal utilitarian."""
        inds = [
            Individual(utility=10.0, identity_level=1.0, group_size=2),
            Individual(utility=5.0, identity_level=0.0, group_size=3),
        ]
        util = utilitarian_utility(inds)
        ive = ive_weighted_utility(inds, coupling=0.0)
        assert abs(util - ive) < 1e-10

    def test_identified_weighted_more(self):
        """Identified individuals should count more with positive coupling."""
        inds_id = [Individual(utility=10.0, identity_level=1.0)]
        inds_anon = [Individual(utility=10.0, identity_level=0.0)]
        ive_id = ive_weighted_utility(inds_id, coupling=0.65)
        ive_anon = ive_weighted_utility(inds_anon, coupling=0.65)
        assert ive_id > ive_anon

    def test_weight_formula(self):
        """Check weight = 1 + coupling * identity * nonsub."""
        ind = Individual(utility=1.0, identity_level=0.8, group_size=1)
        result = ive_weighted_utility([ind], coupling=0.5, nonsubstitutability=1.0)
        expected = (1.0 + 0.5 * 0.8 * 1.0) * 1.0
        assert abs(result - expected) < 1e-10

    def test_higher_coupling_stronger_ive(self):
        """Higher coupling should give identified individuals more weight."""
        inds = [Individual(utility=10.0, identity_level=1.0)]
        low = ive_weighted_utility(inds, coupling=0.3)
        high = ive_weighted_utility(inds, coupling=1.0)
        assert high > low


class TestIVEWeightedFloor:
    def test_floor_prevents_zero_weight(self):
        """Even with identity_level=0, weight should be at least floor_weight."""
        inds = [Individual(utility=10.0, identity_level=0.0)]
        result = ive_weighted_utility_with_floor(inds, coupling=0.0, floor_weight=0.5)
        assert result == 5.0  # 0.5 * 10


class TestCompareAggregations:
    def test_returns_dataframe(self):
        scenario = repugnant_conclusion()
        df = compare_aggregations(scenario, coupling_values=[0.0, 0.65])
        assert len(df) == 2  # Two options
        assert "utilitarian" in df.columns
        assert "ive_c=0.65" in df.columns


class TestRepugnantConclusion:
    def test_creates_scenario(self):
        sc = repugnant_conclusion()
        assert sc.name == "Repugnant Conclusion"
        assert "A_small_happy" in sc.options
        assert "B_large_barely" in sc.options

    def test_utilitarian_prefers_b(self):
        """Standard utilitarianism should prefer B (large barely-living)."""
        sc = repugnant_conclusion()
        util_a = utilitarian_utility(sc.options["A_small_happy"])
        util_b = utilitarian_utility(sc.options["B_large_barely"])
        assert util_b > util_a

    def test_ive_can_prefer_a(self):
        """IVE-weighting with high coupling should prefer A (identified happy)."""
        sc = repugnant_conclusion()
        ive_a = ive_weighted_utility(sc.options["A_small_happy"], coupling=2.0)
        ive_b = ive_weighted_utility(sc.options["B_large_barely"], coupling=2.0)
        assert ive_a > ive_b


class TestTrolley:
    def test_creates_scenario(self):
        sc = trolley_identified_statistical()
        assert "do_nothing" in sc.options
        assert "divert" in sc.options

    def test_utilitarian_prefers_divert(self):
        """Utilitarianism should prefer diverting (save 5, lose 1)."""
        sc = trolley_identified_statistical(n_track=5)
        util_nothing = utilitarian_utility(sc.options["do_nothing"])
        util_divert = utilitarian_utility(sc.options["divert"])
        assert util_divert > util_nothing

    def test_identified_victim_costs_more_ive(self):
        """Identified side-track victim should make diversion costlier under IVE."""
        sc_id = trolley_identified_statistical(victim_identified=True)
        sc_anon = trolley_identified_statistical(victim_identified=False)
        # IVE-weighted cost of diversion is higher when victim is identified
        ive_divert_id = ive_weighted_utility(sc_id.options["divert"], coupling=1.0)
        ive_divert_anon = ive_weighted_utility(sc_anon.options["divert"], coupling=1.0)
        # Identified victim makes diversion worse (more negative)
        assert ive_divert_id < ive_divert_anon


class TestResourceAllocation:
    def test_creates_scenario(self):
        sc = resource_allocation()
        assert "A_concentrated" in sc.options
        assert "B_distributed" in sc.options

    def test_utilitarian_prefers_distributed(self):
        """By default, B (100*5=500) > A (1*80=80)."""
        sc = resource_allocation()
        util_a = utilitarian_utility(sc.options["A_concentrated"])
        util_b = utilitarian_utility(sc.options["B_distributed"])
        assert util_b > util_a


class TestScopeInsensitivity:
    def test_returns_list(self):
        scenarios = scope_insensitivity()
        assert isinstance(scenarios, list)
        assert len(scenarios) == 5

    def test_group_sizes_increase(self):
        scenarios = scope_insensitivity()
        sizes = [sc.options["help"][0].group_size for sc in scenarios]
        assert sizes == sorted(sizes)


class TestRunAllScenarios:
    def test_returns_dict(self):
        results = run_all_scenarios(coupling_values=[0.0, 0.65])
        assert "repugnant_conclusion" in results
        assert "trolley_identified" in results
        assert "trolley_anonymous" in results
        assert "resource_allocation" in results
        assert "scope_anonymous" in results
        assert "scope_identified" in results
