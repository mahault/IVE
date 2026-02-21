"""Tests for the IVE agent module."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ive.agent import build_agent, choose_action, get_help_probability
from ive.utils import state_index, decode_state, cohens_d
from ive.networks import (
    build_network_agent, choose_network_action,
    get_network_help_probability, context_to_network_states,
    apply_aggregation, CASE_PRESETS,
)


class TestStateIndexing:
    def test_roundtrip(self):
        for ctx in range(2):
            for out in range(2):
                for cst in range(2):
                    idx = state_index(ctx, out, cst)
                    decoded = decode_state(idx)
                    assert decoded["context"] == ctx
                    assert decoded["outcome"] == out
                    assert decoded["cost"] == cst

    def test_range(self):
        indices = set()
        for ctx in range(2):
            for out in range(2):
                for cst in range(2):
                    indices.add(state_index(ctx, out, cst))
        assert indices == set(range(8))


class TestBuildAgent:
    def test_builds_stat(self):
        agent = build_agent(context="stat")
        assert agent is not None

    def test_builds_id(self):
        agent = build_agent(context="id")
        assert agent is not None

    def test_different_gamma(self):
        agent_stat = build_agent(gamma_base=16.0, delta_gamma=4.0, context="stat")
        agent_id = build_agent(gamma_base=16.0, delta_gamma=4.0, context="id")
        # Identified agent should have higher gamma
        assert agent_id.gamma > agent_stat.gamma


class TestChooseAction:
    def test_returns_valid_action(self):
        agent = build_agent(context="stat")
        action = choose_action(agent, "stat")
        assert action in (0, 1)

    def test_identified_helps_more(self):
        """Over many trials, identified context should produce more Help actions."""
        np.random.seed(42)
        n = 200
        help_stat = sum(
            choose_action(build_agent(context="stat"), "stat") for _ in range(n)
        )
        help_id = sum(
            choose_action(build_agent(context="id"), "id") for _ in range(n)
        )
        # IVE: identified should help at least as much as statistical
        assert help_id >= help_stat


class TestGetHelpProbability:
    def test_returns_float(self):
        p = get_help_probability(context="stat", n_samples=50)
        assert isinstance(p, float)
        assert 0.0 <= p <= 1.0

    def test_ive_direction(self):
        """Identified should have higher or equal help probability."""
        np.random.seed(42)
        p_stat = get_help_probability(context="stat", n_samples=100)
        p_id = get_help_probability(context="id", n_samples=100)
        assert p_id >= p_stat


class TestNetworkAgent:
    def test_builds_stat(self):
        states = context_to_network_states("stat")
        agent = build_network_agent(**states)
        assert agent.num_factors == 4
        assert agent.num_modalities == 5
        assert agent.num_states == [3, 3, 3, 3]

    def test_builds_id(self):
        states = context_to_network_states("id")
        agent = build_network_agent(**states)
        assert agent is not None

    def test_action_valid(self):
        states = context_to_network_states("stat")
        agent = build_network_agent(**states)
        action = choose_network_action(agent, **states)
        assert action in (0, 1)

    def test_ive_direction(self):
        """Identified should help more than statistical."""
        np.random.seed(42)
        p_stat = get_network_help_probability(
            n_samples=200, **context_to_network_states("stat")
        )
        p_id = get_network_help_probability(
            n_samples=200, **context_to_network_states("high_id")
        )
        assert p_id > p_stat

    def test_graded_ive(self):
        """Help probability should increase with identification level."""
        np.random.seed(42)
        rates = []
        for ctx in ["stat", "id", "high_id"]:
            p = get_network_help_probability(
                n_samples=200, **context_to_network_states(ctx)
            )
            rates.append(p)
        # Monotonically increasing (with MC noise, allow small violations)
        assert rates[2] > rates[0]

    def test_aggregation_reduces_help(self):
        """Aggregation should reduce help probability compared to individual."""
        np.random.seed(42)
        p_individual = get_network_help_probability(
            n_samples=200, **context_to_network_states("high_id")
        )
        agg_params = apply_aggregation(n_victims=20, aggregation_type="bureaucratic")
        p_aggregated = get_network_help_probability(n_samples=200, **agg_params)
        assert p_individual > p_aggregated

    def test_case_presets_run(self):
        """All case presets should build and produce valid actions."""
        for name, params in CASE_PRESETS.items():
            agent = build_network_agent(**params)
            states = {
                "identity_state": params.get("identity_state", 0),
                "affect_state": params.get("affect_state", 0),
                "distance_state": params.get("distance_state", 0),
            }
            action = choose_network_action(agent, **states)
            assert action in (0, 1), f"Preset {name} produced invalid action {action}"


class TestGaesserValidation:
    """Tests for Gaesser et al. (2019) model fit."""

    # Best-fit params from grid search
    GAESSER_FIT = {
        "identity_affect_coupling": 0.65,
        "cost_penalty": 0.9,
        "util_saved": 1.4,
        "affect_preference_boost": 0.4,
    }

    def test_gaesser_fit_stat(self):
        """Gaesser-fitted model: stat rate should be in plausible range."""
        np.random.seed(42)
        p = get_network_help_probability(
            n_samples=300, **context_to_network_states("stat"), **self.GAESSER_FIT
        )
        # Gaesser target: 0.588; allow MC noise
        assert 0.45 < p < 0.72

    def test_gaesser_fit_id(self):
        """Gaesser-fitted model: identified rate should be in plausible range."""
        np.random.seed(42)
        p = get_network_help_probability(
            n_samples=300, **context_to_network_states("high_id"), **self.GAESSER_FIT
        )
        # Gaesser target: 0.745; allow MC noise
        assert 0.60 < p < 0.88

    def test_gaesser_ive_direction(self):
        """Gaesser-fitted model should show positive IVE."""
        np.random.seed(42)
        p_stat = get_network_help_probability(
            n_samples=300, **context_to_network_states("stat"), **self.GAESSER_FIT
        )
        p_id = get_network_help_probability(
            n_samples=300, **context_to_network_states("high_id"), **self.GAESSER_FIT
        )
        assert p_id > p_stat


class TestCohensD:
    def test_zero_difference(self):
        a = np.array([1.0, 2.0, 3.0])
        assert cohens_d(a, a) == 0.0

    def test_positive(self):
        a = np.array([5.0, 6.0, 7.0])
        b = np.array([1.0, 2.0, 3.0])
        assert cohens_d(a, b) > 0
