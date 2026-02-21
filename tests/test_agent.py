"""Tests for the IVE agent module."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ive.agent import build_agent, choose_action, get_help_probability
from ive.utils import state_index, decode_state, cohens_d


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


class TestCohensD:
    def test_zero_difference(self):
        a = np.array([1.0, 2.0, 3.0])
        assert cohens_d(a, a) == 0.0

    def test_positive(self):
        a = np.array([5.0, 6.0, 7.0])
        b = np.array([1.0, 2.0, 3.0])
        assert cohens_d(a, b) > 0
