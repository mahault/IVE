import numpy as np
from pymdp import utils
from pymdp.agent import Agent

"""
Toy active inference model of the Identified Victim Effect (IVE)
using the pymdp.Agent API, following the official object-array pattern.

Hidden state factor (single factor):
    4 states that combine context and outcome:
        0 = (Statistical, NotSaved)
        1 = (Statistical, Saved)
        2 = (Identified, NotSaved)
        3 = (Identified, Saved)

Observation modalities:
    Modality 0 (context cue): 0 = stat-cue, 1 = id-cue
    Modality 1 (outcome):     0 = no one saved, 1 = victim saved

Actions (one control factor):
    0 = Do Nothing
    1 = Help

IVE is encoded by:
    - Higher success probability of Help in the Identified context
      than in the Statistical context.
    - Strong preference for seeing "victim saved".
"""

def build_agent():
    # -----------------------------
    # Dimensions
    # -----------------------------
    num_obs = [2, 2]      # [context_modality, outcome_modality]
    num_states = [4]      # single hidden factor with 4 combined states
    num_controls = [2]    # single control factor, 2 actions: NoHelp, Help

    # -----------------------------
    # Likelihood A (object array)
    # -----------------------------
    A = utils.obj_array(len(num_obs))    # A[0], A[1]

    # Modality 0: context cue
    A_context = np.zeros((num_obs[0], num_states[0]))
    # Modality 1: outcome feedback
    A_outcome = np.zeros((num_obs[1], num_states[0]))

    # Fill context likelihood:
    # states 0,1 -> Statistical -> obs 0 (stat-cue)
    # states 2,3 -> Identified  -> obs 1 (id-cue)
    for s in range(num_states[0]):
        context_state = 0 if s < 2 else 1
        A_context[context_state, s] = 1.0

    # Fill outcome likelihood:
    # even indices -> NotSaved -> obs 0
    # odd indices  -> Saved    -> obs 1
    for s in range(num_states[0]):
        outcome_state = 0 if s % 2 == 0 else 1
        A_outcome[outcome_state, s] = 1.0

    A[0] = A_context
    A[1] = A_outcome

    # -----------------------------
    # Transitions B (object array)
    # -----------------------------
    B = utils.obj_array(len(num_states))    # one hidden factor

    B_factor = np.zeros((num_states[0], num_states[0], num_controls[0]))

    # Action 0: Do Nothing -> identity transition
    B_factor[:, :, 0] = np.eye(num_states[0])

    # Action 1: Help
    p_success_stat = 0.3   # success probability in Statistical context
    p_success_id   = 0.9   # success probability in Identified context

    for s in range(num_states[0]):
        # Statistical context: states 0 (NotSaved), 1 (Saved)
        if s in (0, 1):
            if s == 0:
                # (Statistical, NotSaved) -> maybe Saved (state 1)
                B_factor[1, s, 1] = p_success_stat
                B_factor[0, s, 1] = 1.0 - p_success_stat
            else:
                # (Statistical, Saved) stays Saved
                B_factor[1, s, 1] = 1.0

        # Identified context: states 2 (NotSaved), 3 (Saved)
        else:
            if s == 2:
                # (Identified, NotSaved) -> maybe Saved (state 3)
                B_factor[3, s, 1] = p_success_id
                B_factor[2, s, 1] = 1.0 - p_success_id
            else:
                # (Identified, Saved) stays Saved
                B_factor[3, s, 1] = 1.0

    B[0] = B_factor

    # -----------------------------
    # Priors D (object array)
    # -----------------------------
    D = utils.obj_array(len(num_states))
    D_factor = np.zeros(num_states[0])

    # Start state: NotSaved, unknown context (50/50 between stat and id)
    D_factor[0] = 0.5   # (Statistical, NotSaved)
    D_factor[2] = 0.5   # (Identified, NotSaved)

    D[0] = D_factor

    # -----------------------------
    # Preferences C (object array)
    # -----------------------------
    C = utils.obj_array(len(num_obs))

    # No preference over seeing stat vs id cue
    C_context = np.array([0.0, 0.0])

    # Strong preference for "victim saved" outcome
    # C_outcome is log-preferences: [no_saved, saved]
    C_outcome = np.array([0.0, 2.0])

    C[0] = C_context
    C[1] = C_outcome

    # -----------------------------
    # Build Agent
    # -----------------------------
    agent = Agent(
        A=A,
        B=B,
        C=C,
        D=D,
        inference_horizon=1
    )

    return agent

def choose_action(agent, context_obs):
    """
    Let the agent see:
      - the context cue (0 = stat-cue, 1 = id-cue)
      - that currently 'no one is saved' (outcome_obs = 0)
    and infer the best action.
    """
    agent.reset()

    outcome_obs = 0  # currently: no one saved
    observation = [context_obs, outcome_obs]

    # Perception
    agent.infer_states(observation)

    # Policy inference
    agent.infer_policies()

    # Action selection
    action = agent.sample_action()  # 0 = Do Nothing, 1 = Help
    return action

def main():
    agent = build_agent()

    # Scenario 1: Statistical victim
    action_stat = choose_action(agent, context_obs=0)  # stat-cue
    print("Scenario: Statistical victim")
    print("Chosen action:", "Help" if action_stat == 1 else "Do Nothing")
    print()

    # Scenario 2: Identified victim
    action_id = choose_action(agent, context_obs=1)    # id-cue
    print("Scenario: Identified victim")
    print("Chosen action:", "Help" if action_id == 1 else "Do Nothing")

if __name__ == "__main__":
    main()
