"""
Factorized neural-circuit model of the IVE.

Upgrades the Phase 1 single-factor agent to a multi-factor generative model
where each hidden state factor maps to a distinct neural circuit:

    Factor          States                                          Neural proxy
    ------          ------                                          -----------
    S_identity      {anonymous, partial, full}                      TPJ (mentalizing)
    S_affect        {low, medium, high}                             Insula (affective salience)
    S_distance      {proximal, distal, abstract}                    mPFC (self-other distance)
    S_outcome       {not_saved_nocost, not_saved_cost, saved_cost}  Striatum / vmPFC (valuation)

Observation modalities:
    obs_identity    3 levels — maps to S_identity with adjustable precision
    obs_affect      3 levels — maps to S_affect, modulated by identity and distance
    obs_distance    3 levels — maps to S_distance
    obs_outcome     2 levels {not_saved, saved} — maps to S_outcome
    obs_cost        2 levels {no_cost, cost} — maps to S_outcome

The IVE emerges from identity -> affect coupling: when identity precision is
high (fully identified victim), the affective response is amplified, biasing
policy selection toward Help.

Aggregation (Phase 2 case studies) operates by:
    - Reducing identity precision (A matrix becomes noisier)
    - Shifting distance prior toward abstract
    - Diluting identity-affect coupling
"""

import numpy as np
from pymdp import utils
from pymdp.agent import Agent


# ---------------------------------------------------------------------------
# State / observation indices
# ---------------------------------------------------------------------------

# Hidden state factors
IDENTITY = 0    # S_identity: TPJ
AFFECT = 1      # S_affect: Insula
DISTANCE = 2    # S_distance: mPFC
OUTCOME = 3     # S_outcome: Striatum — includes cost encoding

NUM_STATES = [3, 3, 3, 3]
# outcome states: 0=not_saved_nocost, 1=not_saved_cost, 2=saved_cost

# State labels
IDENTITY_LABELS = ["anonymous", "partial", "full"]
AFFECT_LABELS = ["low", "medium", "high"]
DISTANCE_LABELS = ["proximal", "distal", "abstract"]
OUTCOME_LABELS = ["not_saved_nocost", "not_saved_cost", "saved_cost"]

# Observation modalities
OBS_IDENTITY = 0   # 3 levels
OBS_AFFECT = 1     # 3 levels
OBS_DISTANCE = 2   # 3 levels
OBS_OUTCOME = 3    # 2 levels: {not_saved, saved}
OBS_COST = 4       # 2 levels: {no_cost, cost}

NUM_OBS = [3, 3, 3, 2, 2]

# Actions: only outcome factor is controllable
# Action 0 = NoHelp, Action 1 = Help
NUM_CONTROLS = [1, 1, 1, 2]
CONTROL_FAC_IDX = [3]


# ---------------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------------

NETWORK_DEFAULTS = {
    # Identity -> affect coupling strength
    "identity_affect_coupling": 0.7,

    # Distance -> affect attenuation
    "distance_affect_attenuation": 0.5,

    # Outcome preferences
    "util_saved": 1.5,
    "cost_penalty": 1.5,

    # Affect -> preference boost
    "affect_preference_boost": 0.8,

    # Observation precision
    "identity_precision": 0.9,
    "affect_precision": 0.8,

    # Base P(saved | Help)
    "p_success_base": 0.3,

    # Bonus success probability when identity = full
    "p_success_identity_bonus": 0.2,

    # Policy / action precision
    "gamma": 1.5,
    "alpha": 1.5,

    # Planning horizon
    "policy_len": 1,
}


# ---------------------------------------------------------------------------
# A matrix builders
# ---------------------------------------------------------------------------

def _build_A_identity(num_states, precision=0.9):
    """Identity observation depends primarily on S_identity."""
    n_id, n_aff, n_dist, n_out = num_states
    A = np.zeros((3, n_id, n_aff, n_dist, n_out))

    noise = (1.0 - precision) / 2
    for o_id in range(3):
        for s_id in range(3):
            val = precision if o_id == s_id else noise
            A[o_id, s_id, :, :, :] = val
    return A


def _build_A_affect(num_states, identity_affect_coupling=0.7,
                    distance_affect_attenuation=0.5, precision=0.8):
    """Affect observation depends on S_affect, modulated by S_identity and S_distance.

    This is the core IVE mechanism: identification increases the precision
    of affect representations (insula activation), while distance dampens it.
    """
    n_id, n_aff, n_dist, n_out = num_states
    A = np.zeros((3, n_id, n_aff, n_dist, n_out))

    for s_id in range(n_id):
        for s_aff in range(n_aff):
            for s_dist in range(n_dist):
                id_boost = s_id / (n_id - 1)       # 0..1
                dist_damp = s_dist / (n_dist - 1)   # 0..1

                eff_prec = precision * (
                    1.0
                    + identity_affect_coupling * id_boost
                    - distance_affect_attenuation * dist_damp
                )
                eff_prec = np.clip(eff_prec, 0.1, 0.99)

                noise = (1.0 - eff_prec) / 2
                for o_aff in range(3):
                    val = eff_prec if o_aff == s_aff else noise
                    A[o_aff, s_id, s_aff, s_dist, :] = val
    return A


def _build_A_distance(num_states, precision=0.9):
    """Distance observation depends primarily on S_distance."""
    n_id, n_aff, n_dist, n_out = num_states
    A = np.zeros((3, n_id, n_aff, n_dist, n_out))

    noise = (1.0 - precision) / 2
    for o_dist in range(3):
        for s_dist in range(3):
            val = precision if o_dist == s_dist else noise
            A[o_dist, :, :, s_dist, :] = val
    return A


def _build_A_outcome(num_states):
    """Outcome observation: {not_saved=0, saved=1} from S_outcome.

    S_outcome states: 0=not_saved_nocost, 1=not_saved_cost, 2=saved_cost
    -> obs 0 (not_saved) for states 0,1; obs 1 (saved) for state 2
    """
    n_id, n_aff, n_dist, n_out = num_states
    A = np.zeros((2, n_id, n_aff, n_dist, n_out))

    # not_saved observation for outcome states 0 and 1
    A[0, :, :, :, 0] = 1.0  # not_saved_nocost -> not_saved
    A[0, :, :, :, 1] = 1.0  # not_saved_cost -> not_saved
    # saved observation for outcome state 2
    A[1, :, :, :, 2] = 1.0  # saved_cost -> saved
    return A


def _build_A_cost(num_states):
    """Cost observation: {no_cost=0, cost=1} from S_outcome.

    S_outcome states: 0=not_saved_nocost, 1=not_saved_cost, 2=saved_cost
    -> obs 0 (no_cost) for state 0; obs 1 (cost) for states 1,2
    """
    n_id, n_aff, n_dist, n_out = num_states
    A = np.zeros((2, n_id, n_aff, n_dist, n_out))

    A[0, :, :, :, 0] = 1.0  # not_saved_nocost -> no_cost
    A[1, :, :, :, 1] = 1.0  # not_saved_cost -> cost
    A[1, :, :, :, 2] = 1.0  # saved_cost -> cost
    return A


# ---------------------------------------------------------------------------
# B matrix builders
# ---------------------------------------------------------------------------

def _build_B_static(n_states):
    """Uncontrollable factor — state persists (identity matrix)."""
    B = np.zeros((n_states, n_states, 1))
    B[:, :, 0] = np.eye(n_states)
    return B


def _build_B_outcome(p_success=0.3):
    """Outcome factor transitions: 3 states, 2 actions.

    States: 0=not_saved_nocost, 1=not_saved_cost, 2=saved_cost

    Action 0 (NoHelp): always -> state 0 (not_saved, no cost)
    Action 1 (Help):   from state 0 -> state 2 with p_success (saved+cost),
                                     -> state 1 with 1-p_success (not_saved+cost)
    """
    B = np.zeros((3, 3, 2))

    # NoHelp: always not_saved_nocost
    B[0, :, 0] = 1.0

    # Help: from not_saved_nocost
    B[2, 0, 1] = p_success          # -> saved_cost
    B[1, 0, 1] = 1.0 - p_success    # -> not_saved_cost
    # Help: from already-cost states, stay
    B[1, 1, 1] = 1.0  # not_saved_cost stays
    B[2, 2, 1] = 1.0  # saved_cost stays

    return B


# ---------------------------------------------------------------------------
# C and D builders
# ---------------------------------------------------------------------------

def _build_C(num_obs, util_saved=1.5, cost_penalty=1.5,
             affect_preference_boost=0.8):
    """Build preference vectors across all observation modalities."""
    C = utils.obj_array(len(num_obs))

    # Identity: neutral
    C[OBS_IDENTITY] = np.zeros(num_obs[OBS_IDENTITY])

    # Affect: graded — higher affect is slightly preferred (empathic motivation)
    C[OBS_AFFECT] = np.array([
        0.0,
        affect_preference_boost * 0.3,
        affect_preference_boost * 0.6,
    ])

    # Distance: neutral
    C[OBS_DISTANCE] = np.zeros(num_obs[OBS_DISTANCE])

    # Outcome: prefer saved
    C[OBS_OUTCOME] = np.array([0.0, util_saved])

    # Cost: dislike cost
    C[OBS_COST] = np.array([0.0, -cost_penalty])

    return C


def _build_D(num_states, identity_state=0, affect_state=0, distance_state=0):
    """Build prior beliefs. Outcome always starts at not_saved_nocost (state 0)."""
    D = utils.obj_array(len(num_states))

    D[IDENTITY] = np.zeros(num_states[IDENTITY])
    D[IDENTITY][identity_state] = 1.0

    D[AFFECT] = np.zeros(num_states[AFFECT])
    D[AFFECT][affect_state] = 1.0

    D[DISTANCE] = np.zeros(num_states[DISTANCE])
    D[DISTANCE][distance_state] = 1.0

    D[OUTCOME] = np.zeros(num_states[OUTCOME])
    D[OUTCOME][0] = 1.0  # start in not_saved_nocost

    return D


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_network_agent(
    identity_state: int = 0,
    affect_state: int = 0,
    distance_state: int = 0,
    identity_affect_coupling: float = NETWORK_DEFAULTS["identity_affect_coupling"],
    distance_affect_attenuation: float = NETWORK_DEFAULTS["distance_affect_attenuation"],
    identity_precision: float = NETWORK_DEFAULTS["identity_precision"],
    affect_precision: float = NETWORK_DEFAULTS["affect_precision"],
    util_saved: float = NETWORK_DEFAULTS["util_saved"],
    cost_penalty: float = NETWORK_DEFAULTS["cost_penalty"],
    affect_preference_boost: float = NETWORK_DEFAULTS["affect_preference_boost"],
    p_success_base: float = NETWORK_DEFAULTS["p_success_base"],
    p_success_identity_bonus: float = NETWORK_DEFAULTS["p_success_identity_bonus"],
    gamma: float = NETWORK_DEFAULTS["gamma"],
    alpha: float = NETWORK_DEFAULTS["alpha"],
    policy_len: int = NETWORK_DEFAULTS["policy_len"],
) -> Agent:
    """Build a factorized active inference agent with neural-circuit mapping.

    The agent has 4 hidden state factors (identity, affect, distance, outcome)
    and 5 observation modalities (identity, affect, distance, outcome, cost).
    Only the outcome factor is controllable (Help/NoHelp).

    The IVE mechanism: identity modulates the precision of affect observations
    (A_affect matrix). When a victim is fully identified (identity_state=2),
    the affect signal is sharper, producing stronger preferences for helping.
    Distance attenuates this coupling.

    Args:
        identity_state: 0=anonymous, 1=partial, 2=full
        affect_state: 0=low, 1=medium, 2=high
        distance_state: 0=proximal, 1=distal, 2=abstract
        identity_affect_coupling: Strength of identity -> affect precision boost
        distance_affect_attenuation: Strength of distance -> affect precision reduction
        identity_precision: How well the agent perceives identity
        affect_precision: Base affect observation precision
        util_saved: Preference strength for saved outcome
        cost_penalty: Disutility of cost
        affect_preference_boost: How much high affect biases toward helping
        p_success_base: Base P(saved | Help)
        p_success_identity_bonus: Added success prob for identified victims
        gamma: Policy precision
        alpha: Action selection precision
        policy_len: Planning horizon

    Returns:
        A pymdp Agent configured with the factorized generative model.
    """
    num_states = list(NUM_STATES)
    num_obs = list(NUM_OBS)

    # Scale p_success by identity state
    p_success = p_success_base + p_success_identity_bonus * (identity_state / 2.0)
    p_success = np.clip(p_success, 0.0, 1.0)

    # Build A matrices (5 modalities)
    A = utils.obj_array(5)
    A[OBS_IDENTITY] = _build_A_identity(num_states, precision=identity_precision)
    A[OBS_AFFECT] = _build_A_affect(
        num_states,
        identity_affect_coupling=identity_affect_coupling,
        distance_affect_attenuation=distance_affect_attenuation,
        precision=affect_precision,
    )
    A[OBS_DISTANCE] = _build_A_distance(num_states, precision=0.9)
    A[OBS_OUTCOME] = _build_A_outcome(num_states)
    A[OBS_COST] = _build_A_cost(num_states)

    # Build B matrices (4 factors)
    B = utils.obj_array(4)
    B[IDENTITY] = _build_B_static(num_states[IDENTITY])
    B[AFFECT] = _build_B_static(num_states[AFFECT])
    B[DISTANCE] = _build_B_static(num_states[DISTANCE])
    B[OUTCOME] = _build_B_outcome(p_success=p_success)

    # Build C preferences (5 modalities)
    C = _build_C(
        num_obs,
        util_saved=util_saved,
        cost_penalty=cost_penalty,
        affect_preference_boost=affect_preference_boost,
    )

    # Build D priors (4 factors)
    D = _build_D(
        num_states,
        identity_state=identity_state,
        affect_state=affect_state,
        distance_state=distance_state,
    )

    agent = Agent(
        A=A, B=B, C=C, D=D,
        control_fac_idx=CONTROL_FAC_IDX,
        policy_len=policy_len,
        gamma=gamma,
        alpha=alpha,
        action_selection="stochastic",
    )
    return agent


def choose_network_action(agent: Agent, identity_state: int = 0,
                          affect_state: int = 0, distance_state: int = 0) -> int:
    """Run one-step inference and return action (0=NoHelp, 1=Help)."""
    agent.reset()
    # 5 observations: identity, affect, distance, outcome=not_saved, cost=no_cost
    observation = [identity_state, affect_state, distance_state, 0, 0]
    agent.infer_states(observation)
    agent.infer_policies()
    action = agent.sample_action()

    # action is array with one entry per factor; extract outcome action
    if hasattr(action, '__len__'):
        act = action[OUTCOME]
    else:
        act = action
    return int(act.item()) if hasattr(act, 'item') else int(act)


def get_network_help_probability(n_samples: int = 500, **kwargs) -> float:
    """Estimate P(Help) via Monte Carlo. All kwargs -> build_network_agent."""
    help_count = 0
    id_state = kwargs.get("identity_state", 0)
    aff_state = kwargs.get("affect_state", 0)
    dist_state = kwargs.get("distance_state", 0)

    for _ in range(n_samples):
        agent = build_network_agent(**kwargs)
        action = choose_network_action(agent, id_state, aff_state, dist_state)
        if action == 1:
            help_count += 1
    return help_count / n_samples


# ---------------------------------------------------------------------------
# Phase 1 context mapping
# ---------------------------------------------------------------------------

def context_to_network_states(context: str) -> dict:
    """Map Phase 1 context labels to factorized state configurations.

    Args:
        context: "stat", "id", or "high_id"

    Returns:
        Dict of identity_state, affect_state, distance_state.
    """
    mappings = {
        "stat": {"identity_state": 0, "affect_state": 0, "distance_state": 1},
        "id": {"identity_state": 1, "affect_state": 1, "distance_state": 0},
        "high_id": {"identity_state": 2, "affect_state": 2, "distance_state": 0},
    }
    return mappings.get(context, mappings["stat"])


# ---------------------------------------------------------------------------
# Aggregation operator
# ---------------------------------------------------------------------------

def apply_aggregation(
    identity_precision: float = 0.9,
    identity_affect_coupling: float = 0.7,
    distance_affect_attenuation: float = 0.5,
    n_victims: int = 1,
    aggregation_type: str = "bureaucratic",
) -> dict:
    """Compute parameter modifications for institutional aggregation.

    Aggregation degrades identity information and increases psychological
    distance, reducing the IVE. Returns a dict of modified params that can
    be passed to build_network_agent.

    Args:
        identity_precision: Base identity precision.
        identity_affect_coupling: Base coupling strength.
        distance_affect_attenuation: Base distance attenuation.
        n_victims: Number of victims being aggregated (dilution factor).
        aggregation_type: "bureaucratic", "statistical", or "military".

    Returns:
        Dict of modified parameters.
    """
    # Dilution: identity precision and coupling scale inversely with n_victims
    dilution = 1.0 / np.sqrt(max(n_victims, 1))

    if aggregation_type == "bureaucratic":
        # Bureaucratic: pool identities, increase distance, reduce coupling
        return {
            "identity_state": 0,     # victims become anonymous
            "affect_state": 0,       # affect dampened
            "distance_state": 2,     # abstract (institutional) distance
            "identity_precision": max(identity_precision * dilution * 0.4, 0.1),
            "identity_affect_coupling": identity_affect_coupling * dilution * 0.3,
            "distance_affect_attenuation": min(distance_affect_attenuation + 0.3, 0.95),
        }
    elif aggregation_type == "statistical":
        # Statistical: replace individuals with summary statistics
        return {
            "identity_state": 0,
            "affect_state": 0,
            "distance_state": 1,     # distal but not fully abstract
            "identity_precision": max(identity_precision * dilution * 0.5, 0.1),
            "identity_affect_coupling": identity_affect_coupling * dilution * 0.4,
            "distance_affect_attenuation": distance_affect_attenuation + 0.15,
        }
    elif aggregation_type == "military":
        # Military: maximal distance, minimal identity precision
        return {
            "identity_state": 0,
            "affect_state": 0,
            "distance_state": 2,
            "identity_precision": max(identity_precision * dilution * 0.2, 0.1),
            "identity_affect_coupling": identity_affect_coupling * dilution * 0.1,
            "distance_affect_attenuation": 0.95,
        }
    else:
        raise ValueError(f"Unknown aggregation type: {aggregation_type}")


# ---------------------------------------------------------------------------
# Case study presets
# ---------------------------------------------------------------------------

CASE_PRESETS = {
    # Charity: standard IVE comparison
    "charity_stat": {
        "identity_state": 0, "affect_state": 0, "distance_state": 1,
    },
    "charity_id": {
        "identity_state": 2, "affect_state": 2, "distance_state": 0,
    },

    # Francis Inquiry: pre- and post-aggregation
    "francis_individual": {
        "identity_state": 2, "affect_state": 2, "distance_state": 0,
    },
    "francis_aggregated": {
        "identity_state": 0, "affect_state": 0, "distance_state": 2,
        "identity_precision": 0.3, "identity_affect_coupling": 0.2,
        "distance_affect_attenuation": 0.8,
    },

    # RADAR trial: individual vs aggregated patient
    "radar_individual": {
        "identity_state": 2, "affect_state": 1, "distance_state": 0,
    },
    "radar_aggregated": {
        "identity_state": 0, "affect_state": 0, "distance_state": 1,
        "identity_precision": 0.35, "identity_affect_coupling": 0.25,
    },

    # Military: ground encounter vs drone vs chain of command
    "military_ground": {
        "identity_state": 2, "affect_state": 2, "distance_state": 0,
    },
    "military_drone": {
        "identity_state": 0, "affect_state": 0, "distance_state": 2,
        "identity_precision": 0.4, "identity_affect_coupling": 0.3,
        "distance_affect_attenuation": 0.8,
    },
    "military_command": {
        "identity_state": 0, "affect_state": 0, "distance_state": 2,
        "identity_precision": 0.2, "identity_affect_coupling": 0.1,
        "distance_affect_attenuation": 0.95,
    },

    # Psychopathy analog: reduced affect coupling
    "psychopathy": {
        "identity_state": 2, "affect_state": 0, "distance_state": 0,
        "identity_affect_coupling": 0.1,
        "affect_preference_boost": 0.1,
    },

    # Burnout analog: flattened affect sensitivity
    "burnout": {
        "identity_state": 2, "affect_state": 0, "distance_state": 0,
        "identity_affect_coupling": 0.2,
        "affect_precision": 0.3,
    },
}
