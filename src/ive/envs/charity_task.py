"""
Charity/donation task environment for IVE experiments.

Models a one-step charitable giving scenario where an agent observes a
victim (identified or statistical) and decides whether to help.

This environment decouples the world dynamics from the agent, making it
easy to swap in different scenarios and run Monte Carlo simulations.
"""

import numpy as np
from ..utils import state_index, decode_state


class CharityTask:
    """One-step charitable giving environment.

    The agent sees a victim (context = stat or id), decides Help or NoHelp,
    and the environment resolves the outcome probabilistically.

    Parameters
    ----------
    p_success_stat : float
        P(saved | Help, statistical victim).
    p_success_id : float
        P(saved | Help, identified victim).
    """

    def __init__(self, p_success_stat: float = 0.3, p_success_id: float = 0.9):
        self.p_success_stat = p_success_stat
        self.p_success_id = p_success_id

    def reset(self, context: str = "stat") -> dict:
        """Reset environment to initial state.

        Returns:
            Initial observation dict.
        """
        self.context = context
        self.ctx_val = 1 if context == "id" else 0
        self.state = state_index(self.ctx_val, outcome=0, cost=0)
        return {
            "context_obs": self.ctx_val,
            "outcome_obs": 0,
            "cost_obs": 0,
        }

    def step(self, action: int) -> tuple:
        """Execute one action.

        Args:
            action: 0=NoHelp, 1=Help.

        Returns:
            (observation, reward_info) where observation is a dict and
            reward_info contains outcome details.
        """
        decoded = decode_state(self.state)
        ctx = decoded["context"]

        if action == 0:
            # NoHelp: stay NotSaved, NoCost
            self.state = state_index(ctx, outcome=0, cost=0)
            obs = {"context_obs": ctx, "outcome_obs": 0, "cost_obs": 0}
            info = {"helped": False, "saved": False, "cost": False}
        else:
            # Help: probabilistic outcome, always incurs cost
            p = self.p_success_id if ctx == 1 else self.p_success_stat
            saved = int(np.random.rand() < p)
            self.state = state_index(ctx, outcome=saved, cost=1)
            obs = {"context_obs": ctx, "outcome_obs": saved, "cost_obs": 1}
            info = {"helped": True, "saved": bool(saved), "cost": True}

        return obs, info


def run_monte_carlo(
    n_trials: int = 1000,
    context: str = "stat",
    p_success_stat: float = 0.3,
    p_success_id: float = 0.9,
    agent_builder=None,
    agent_params: dict = None,
) -> dict:
    """Run Monte Carlo simulation of the charity task.

    Args:
        n_trials: Number of trials to simulate.
        context: "stat" or "id".
        p_success_stat: Environment success rate for statistical victims.
        p_success_id: Environment success rate for identified victims.
        agent_builder: Callable that returns a pymdp Agent. If None,
            uses the default from agent.py.
        agent_params: Dict of params to pass to agent_builder.

    Returns:
        Dict with help_count, success_count, trials, help_rate, success_rate.
    """
    from ..agent import build_agent, choose_action

    env = CharityTask(p_success_stat=p_success_stat, p_success_id=p_success_id)

    params = agent_params or {}
    help_count = 0
    success_count = 0

    for _ in range(n_trials):
        obs = env.reset(context=context)

        if agent_builder is not None:
            agent = agent_builder(**params, context=context)
        else:
            agent = build_agent(**params, context=context)

        action = choose_action(agent, context)

        _, info = env.step(action)
        if info["helped"]:
            help_count += 1
        if info["saved"]:
            success_count += 1

    return {
        "context": context,
        "trials": n_trials,
        "help_count": help_count,
        "success_count": success_count,
        "help_rate": help_count / n_trials,
        "success_rate": success_count / n_trials,
    }
