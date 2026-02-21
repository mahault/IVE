"""
IVE Active Inference agent with disentangled mechanisms.

The Identified Victim Effect is decomposed into three separable parameters:

    delta_C     : Preference shift — identified victims are valued more
                  (maps to affective salience / insula activation)
    delta_gamma : Precision shift — identified victims produce more urgent
                  policy selection (maps to striatal / dACC gain)
    delta_p     : Controllability shift — perceived efficacy of helping is
                  higher for identified victims (maps to perceived agency)

These can be independently manipulated, fitted, and eventually mapped to
distinct neural circuits (Phase 2).
"""

import numpy as np
from pymdp import utils
from pymdp.agent import Agent

from .utils import state_index


# Default parameter values
# gamma_base=1.0 keeps action selection stochastic (higher = more deterministic).
# cost_penalty must be comparable to util_saved to create a meaningful
# help/no-help tradeoff; if cost << util, agent always helps.
DEFAULTS = {
    "p_success_base": 0.3,
    "delta_p": 0.2,          # p_success_id = base + delta_p
    "util_saved_base": 1.0,
    "delta_C": 0.5,          # C_saved_id = base + delta_C
    "cost_penalty": 1.5,
    "gamma_base": 1.0,       # low precision -> stochastic; high -> deterministic
    "delta_gamma": 0.0,      # precision boost for identified
}


def build_agent(
    p_success_base: float = DEFAULTS["p_success_base"],
    delta_p: float = DEFAULTS["delta_p"],
    util_saved_base: float = DEFAULTS["util_saved_base"],
    delta_C: float = DEFAULTS["delta_C"],
    cost_penalty: float = DEFAULTS["cost_penalty"],
    gamma_base: float = DEFAULTS["gamma_base"],
    delta_gamma: float = DEFAULTS["delta_gamma"],
    context: str = "stat",
) -> Agent:
    """Build an active inference agent for a specific context.

    Unlike the toy model which encoded both contexts in one agent, here we
    build a context-specific agent. This makes it straightforward to give
    identified agents different precision (gamma) — something pymdp sets
    at agent construction time.

    Args:
        p_success_base: Base probability that Help saves the victim.
        delta_p: Added success probability for identified context.
        util_saved_base: Base log-preference for "victim saved" outcome.
        delta_C: Added preference for identified context.
        cost_penalty: Disutility of cost observation (positive value).
        gamma_base: Base policy precision (action temperature).
        delta_gamma: Added precision for identified context.
        context: "stat" or "id" — determines which parameter set to use.

    Returns:
        A pymdp Agent configured for the given context.
    """
    is_id = context == "id"

    p_success = p_success_base + (delta_p if is_id else 0.0)
    p_success = np.clip(p_success, 0.0, 1.0)

    util_saved = util_saved_base + (delta_C if is_id else 0.0)

    gamma = gamma_base + (delta_gamma if is_id else 0.0)
    gamma = max(gamma, 0.01)  # must be positive

    # ---- Dimensions ----
    num_states = [8]       # context x outcome x cost
    num_controls = [2]     # NoHelp, Help
    num_obs = [2, 2, 2]   # context_cue, outcome, cost

    # ---- A: Likelihood (deterministic observation mapping) ----
    A = utils.obj_array(3)
    A_context = np.zeros((2, 8))
    A_outcome = np.zeros((2, 8))
    A_cost = np.zeros((2, 8))

    for s in range(8):
        ctx = s // 4
        rem = s % 4
        out = rem // 2
        cst = rem % 2
        A_context[ctx, s] = 1.0
        A_outcome[out, s] = 1.0
        A_cost[cst, s] = 1.0

    A[0] = A_context
    A[1] = A_outcome
    A[2] = A_cost

    # ---- B: Transitions ----
    B = utils.obj_array(1)
    B_factor = np.zeros((8, 8, 2))

    ctx_val = 1 if is_id else 0

    # Action 0: NoHelp — stay in (context, NotSaved, NoCost)
    for s in range(8):
        c = s // 4
        s_next = state_index(c, outcome=0, cost=0)
        B_factor[s_next, s, 0] = 1.0

    # Action 1: Help
    for s in range(8):
        c = s // 4
        rem = s % 4
        out = rem // 2
        if out == 0:  # currently NotSaved
            s_saved = state_index(c, outcome=1, cost=1)
            s_not = state_index(c, outcome=0, cost=1)
            # Use context-appropriate success probability
            p = p_success if c == ctx_val else p_success_base
            B_factor[s_saved, s, 1] = p
            B_factor[s_not, s, 1] = 1.0 - p
        else:
            s_saved_cost = state_index(c, outcome=1, cost=1)
            B_factor[s_saved_cost, s, 1] = 1.0

    B[0] = B_factor

    # ---- D: Prior (start in NotSaved, NoCost, known context) ----
    D = utils.obj_array(1)
    D_factor = np.zeros(8)
    D_factor[state_index(ctx_val, 0, 0)] = 1.0
    D[0] = D_factor

    # ---- C: Preferences ----
    C = utils.obj_array(3)
    C[0] = np.array([0.0, 0.0])                    # neutral over context cue
    C[1] = np.array([0.0, util_saved])              # prefer saved
    C[2] = np.array([0.0, -cost_penalty])           # dislike cost

    # ---- Build Agent ----
    # action_selection="stochastic" is critical: without it, pymdp uses
    # argmax (deterministic) action selection, which collapses all
    # probabilistic structure. alpha controls the action selection
    # temperature (separate from gamma which controls policy precision).
    agent = Agent(
        A=A, B=B, C=C, D=D,
        gamma=gamma,
        action_selection="stochastic",
        alpha=gamma,  # use same precision for action selection
    )
    return agent


def choose_action(agent: Agent, context: str) -> int:
    """Run one-step inference and return the chosen action.

    Args:
        agent: A pymdp Agent (built for the given context).
        context: "stat" or "id".

    Returns:
        0 = NoHelp, 1 = Help.
    """
    agent.reset()
    ctx_obs = 1 if context == "id" else 0
    observation = [ctx_obs, 0, 0]  # see context cue, no one saved, no cost

    agent.infer_states(observation)
    agent.infer_policies()
    action = agent.sample_action()
    return int(action.item()) if hasattr(action, 'item') else int(action)


def get_help_probability(
    p_success_base: float = DEFAULTS["p_success_base"],
    delta_p: float = DEFAULTS["delta_p"],
    util_saved_base: float = DEFAULTS["util_saved_base"],
    delta_C: float = DEFAULTS["delta_C"],
    cost_penalty: float = DEFAULTS["cost_penalty"],
    gamma_base: float = DEFAULTS["gamma_base"],
    delta_gamma: float = DEFAULTS["delta_gamma"],
    context: str = "stat",
    n_samples: int = 500,
) -> float:
    """Estimate P(Help) for given parameters and context via Monte Carlo.

    Builds an agent, runs n_samples trials, returns the fraction that chose Help.
    """
    help_count = 0
    for _ in range(n_samples):
        agent = build_agent(
            p_success_base=p_success_base,
            delta_p=delta_p,
            util_saved_base=util_saved_base,
            delta_C=delta_C,
            cost_penalty=cost_penalty,
            gamma_base=gamma_base,
            delta_gamma=delta_gamma,
            context=context,
        )
        action = choose_action(agent, context)
        if action == 1:
            help_count += 1
    return help_count / n_samples
