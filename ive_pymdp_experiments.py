"""
ive_pymdp_experiments.py

Toy active inference model of the Identified Victim Effect (IVE) using pymdp,
plus analytic expected-free-energy sweeps and logging.

Requirements:
    pip install pymdp numpy matplotlib
"""

import os
import csv
import numpy as np
from pymdp import utils
from pymdp.agent import Agent
import matplotlib.pyplot as plt

# --------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

DEFAULT_P_SUCCESS_STAT = 0.3
DEFAULT_P_SUCCESS_ID = 0.9
DEFAULT_UTIL_SAVED = 2.0
DEFAULT_COST_HELP = 0.5  # used in analytic EFE (not in pymdp agent)


# --------------------------------------------------------------------
# PYMDF ACTIVE INFERENCE MODEL
# --------------------------------------------------------------------

def build_agent(p_success_stat=0.3, p_success_id=0.9, util_saved=2.0):
    """
    Build an active inference agent with given success probabilities and
    preference strength. Compatible with older pymdp API:
        Agent(A=A, B=B, C=C, D=D)
    """
    # Dimensions
    num_obs = [2, 2]      # [context_modality, outcome_modality]
    num_states = [4]      # single hidden factor with 4 combined states:
                          # 0=(Stat,NotSaved), 1=(Stat,Saved),
                          # 2=(Id,NotSaved),   3=(Id,Saved)
    num_controls = [2]    # one control factor: 0=NoHelp, 1=Help

    # Likelihood A (object array)
    A = utils.obj_array(len(num_obs))

    A_context = np.zeros((num_obs[0], num_states[0]))
    A_outcome = np.zeros((num_obs[1], num_states[0]))

    # Context likelihood:
    # states 0,1 -> Statistical -> obs 0 (stat-cue)
    # states 2,3 -> Identified  -> obs 1 (id-cue)
    for s in range(num_states[0]):
        context_state = 0 if s < 2 else 1
        A_context[context_state, s] = 1.0

    # Outcome likelihood:
    # even indices -> NotSaved -> obs 0
    # odd indices  -> Saved    -> obs 1
    for s in range(num_states[0]):
        outcome_state = 0 if s % 2 == 0 else 1
        A_outcome[outcome_state, s] = 1.0

    A[0] = A_context
    A[1] = A_outcome

    # Transitions B (object array)
    B = utils.obj_array(len(num_states))
    B_factor = np.zeros((num_states[0], num_states[0], num_controls[0]))

    # Action 0: Do Nothing -> identity transition
    B_factor[:, :, 0] = np.eye(num_states[0])

    # Action 1: Help
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

    # Priors D (object array)
    D = utils.obj_array(len(num_states))
    D_factor = np.zeros(num_states[0])
    D_factor[0] = 0.5   # (Stat,NotSaved)
    D_factor[2] = 0.5   # (Id,NotSaved)
    D[0] = D_factor

    # Preferences C (object array)
    C = utils.obj_array(len(num_obs))
    C_context = np.array([0.0, 0.0])           # no preference over context cue
    C_outcome = np.array([0.0, util_saved])    # utility for "victim saved"
    C[0] = C_context
    C[1] = C_outcome

    agent = Agent(A=A, B=B, C=C, D=D)
    return agent


def choose_action(agent, context_obs):
    """
    Show the agent:
      - context cue (0=stat, 1=id)
      - current outcome obs 'no one saved' (0)
    and return the action (0=NoHelp, 1=Help).
    """
    agent.reset()
    outcome_obs = 0
    observation = [context_obs, outcome_obs]

    agent.infer_states(observation)
    agent.infer_policies()
    action = agent.sample_action()
    return int(action)


def run_experiment(
    num_trials=1000,
    p_success_stat=DEFAULT_P_SUCCESS_STAT,
    p_success_id=DEFAULT_P_SUCCESS_ID,
    util_saved=DEFAULT_UTIL_SAVED
):
    """
    Monte Carlo experiment using pymdp Agent.

    For each context (Statistical, Identified), we:
      - build an agent with given parameters
      - let it choose an action
      - sample the environment's next state from B
      - record whether it chose Help and whether the victim was Saved
    """

    stats = {
        "stat": {"help": 0, "trials": 0, "success": 0},
        "id":   {"help": 0, "trials": 0, "success": 0},
    }

    # Start from NotSaved in the appropriate context:
    state_indices = {"stat": 0, "id": 2}   # (Stat,NotSaved), (Id,NotSaved)
    context_obs_map = {"stat": 0, "id": 1}

    for context_label in ["stat", "id"]:
        current_state = state_indices[context_label]
        context_obs = context_obs_map[context_label]

        for _ in range(num_trials):
            agent = build_agent(
                p_success_stat=p_success_stat,
                p_success_id=p_success_id,
                util_saved=util_saved
            )

            action = choose_action(agent, context_obs)
            stats[context_label]["trials"] += 1
            if action == 1:
                stats[context_label]["help"] += 1

            B_factor = agent.B[0]   # shape: [next_state, current_state, action]
            current_state = int(current_state)
            action = int(action)
            probs_next = B_factor[:, current_state, action]
            next_state = np.random.choice(np.arange(len(probs_next)), p=probs_next)

            # Saved if outcome-part of state is "Saved" (odd index)
            if next_state % 2 == 1:
                stats[context_label]["success"] += 1

            # For simplicity, keep starting state fixed each trial
            current_state = state_indices[context_label]

    return stats


def log_mc_results(stats, filename="mc_results.csv",
                   p_success_stat=DEFAULT_P_SUCCESS_STAT,
                   p_success_id=DEFAULT_P_SUCCESS_ID,
                   util_saved=DEFAULT_UTIL_SAVED):
    """
    Save Monte Carlo summary stats to CSV.
    """
    path = os.path.join(LOG_DIR, filename)
    header = [
        "context",
        "p_success_stat",
        "p_success_id",
        "util_saved",
        "trials",
        "help_count",
        "success_count",
        "help_rate",
        "success_rate"
    ]

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for ctx in ["stat", "id"]:
            trials = stats[ctx]["trials"]
            help_ = stats[ctx]["help"]
            succ = stats[ctx]["success"]
            help_rate = help_ / trials if trials > 0 else np.nan
            succ_rate = succ / trials if trials > 0 else np.nan
            writer.writerow([
                ctx,
                p_success_stat,
                p_success_id,
                util_saved,
                trials,
                help_,
                succ,
                help_rate,
                succ_rate
            ])
    print(f"[LOG] Monte Carlo results written to {path}")


# --------------------------------------------------------------------
# ANALYTIC EXPECTED FREE ENERGY MODEL (NO pymdp)
# --------------------------------------------------------------------

def softmax(x):
    x = np.array(x, dtype=float)
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


def preference_distribution(u_saved=DEFAULT_UTIL_SAVED):
    """
    Preference distribution over outcomes [no_saved, saved]
    from utilities [0, u_saved].
    """
    logits = np.array([0.0, u_saved])
    return softmax(logits)


def efe_for_actions(p_success, u_saved=DEFAULT_UTIL_SAVED,
                    cost_help=DEFAULT_COST_HELP):
    """
    Compute simple 'expected free energy' for Help vs NoHelp, given:
      - p_success: P(saved | Help)
      - u_saved: utility of 'saved'
      - cost_help: extra cost of choosing Help (in G units)

    Outcomes: o in {no_saved, saved}
    """
    P_pref = preference_distribution(u_saved=u_saved)

    # Predicted outcomes
    P_help = np.array([1 - p_success, p_success])
    P_no   = np.array([1.0, 0.0])

    eps = 1e-12

    # Risk ~ cross-entropy between predicted outcomes and preference distribution
    risk_help = np.sum(P_help * (-np.log(P_pref + eps)))
    risk_no   = np.sum(P_no   * (-np.log(P_pref + eps)))

    # Ambiguity ~ entropy of predicted outcomes
    amb_help = -np.sum(P_help * np.log(P_help + eps))
    amb_no   = -np.sum(P_no   * np.log(P_no   + eps))

    # Total G: risk + ambiguity (+ cost for Help)
    G_help = risk_help + amb_help + cost_help
    G_no   = risk_no   + amb_no

    return G_help, G_no


def prob_help(p_success, u_saved=DEFAULT_UTIL_SAVED,
              cost_help=DEFAULT_COST_HELP, beta=5.0):
    """
    Probability of choosing Help over NoHelp using softmax(-G).
    """
    G_help, G_no = efe_for_actions(p_success, u_saved, cost_help)
    logits = np.array([-G_no, -G_help]) * beta  # [NoHelp, Help]
    return softmax(logits)[1]                   # P(Help)


def plot_and_log_prob_help_curves(
    filename="analytic_curve.csv",
    u_saved=DEFAULT_UTIL_SAVED,
    cost_stat=DEFAULT_COST_HELP,
    cost_id=DEFAULT_COST_HELP * 0.3,
    beta=5.0
):
    """
    1D curves: P(Help) vs p_success for statistical vs identified victims.
    Save results to CSV and show plot.
    """
    p_vals = np.linspace(0.0, 1.0, 101)

    p_help_stat = [prob_help(p, u_saved, cost_stat, beta) for p in p_vals]
    p_help_id   = [prob_help(p, u_saved, cost_id,   beta) for p in p_vals]

    # Log to CSV
    path = os.path.join(LOG_DIR, filename)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "p_success",
            "P_help_statistical",
            "P_help_identified",
            "u_saved",
            "cost_stat",
            "cost_id",
            "beta"
        ])
        for p, phs, phi in zip(p_vals, p_help_stat, p_help_id):
            writer.writerow([p, phs, phi, u_saved, cost_stat, cost_id, beta])
    print(f"[LOG] Analytic curve results written to {path}")

    # Plot
    plt.figure()
    plt.plot(p_vals, p_help_stat, label="Statistical victim")
    plt.plot(p_vals, p_help_id,   label="Identified victim", linestyle="--")
    plt.xlabel("Success probability of helping (p_success)")
    plt.ylabel("P(Help)")
    plt.ylim(0, 1)
    plt.title("P(Help) vs success probability\n(statistical vs identified)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_and_log_heatmaps(
    filename="analytic_heatmap.csv",
    u_saved=DEFAULT_UTIL_SAVED,
    cost_stat=DEFAULT_COST_HELP,
    cost_id=DEFAULT_COST_HELP * 0.3,
    beta=5.0
):
    """
    2D grid of P(Help) for statistical vs identified victims
    as a function of p_success_stat and p_success_id.
    """
    p_stat_grid = np.linspace(0.0, 0.9, 25)
    p_id_grid   = np.linspace(0.3, 0.99, 25)

    help_stat = np.zeros((len(p_stat_grid), len(p_id_grid)))
    help_id   = np.zeros((len(p_stat_grid), len(p_id_grid)))

    for i, p_stat in enumerate(p_stat_grid):
        for j, p_id in enumerate(p_id_grid):
            help_stat[i, j] = prob_help(p_stat, u_saved, cost_stat, beta)
            help_id[i, j]   = prob_help(p_id,   u_saved, cost_id,   beta)

    # Log to CSV
    path = os.path.join(LOG_DIR, filename)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "p_success_stat",
            "p_success_id",
            "P_help_statistical",
            "P_help_identified",
            "u_saved",
            "cost_stat",
            "cost_id",
            "beta"
        ])
        for i, p_stat in enumerate(p_stat_grid):
            for j, p_id in enumerate(p_id_grid):
                writer.writerow([
                    p_stat,
                    p_id,
                    help_stat[i, j],
                    help_id[i, j],
                    u_saved,
                    cost_stat,
                    cost_id,
                    beta
                ])
    print(f"[LOG] Analytic heatmap results written to {path}")

    # Heatmap for statistical context
    plt.figure()
    plt.imshow(
        help_stat,
        origin="lower",
        aspect="auto",
        extent=[p_id_grid[0], p_id_grid[-1], p_stat_grid[0], p_stat_grid[-1]],
        vmin=0.0, vmax=1.0
    )
    plt.colorbar(label="P(Help) (statistical victim)")
    plt.xlabel("p_success_id (x-axis unused here)")
    plt.ylabel("p_success_stat")
    plt.title("P(Help) for statistical victim")
    plt.tight_layout()

    # Heatmap for identified context
    plt.figure()
    plt.imshow(
        help_id,
        origin="lower",
        aspect="auto",
        extent=[p_id_grid[0], p_id_grid[-1], p_stat_grid[0], p_stat_grid[-1]],
        vmin=0.0, vmax=1.0
    )
    plt.colorbar(label="P(Help) (identified victim)")
    plt.xlabel("p_success_id")
    plt.ylabel("p_success_stat (y-axis unused here)")
    plt.title("P(Help) for identified victim")
    plt.tight_layout()

    plt.show()


# --------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------

def main():
    # 1) Quick pymdp demo: single decisions
    agent = build_agent()
    a_stat = choose_action(agent, context_obs=0)
    a_id   = choose_action(agent, context_obs=1)

    print("Single-step demo (pymdp Agent)")
    print("------------------------------")
    print("Statistical victim ->", "Help" if a_stat == 1 else "Do Nothing")
    print("Identified victim  ->", "Help" if a_id == 1 else "Do Nothing")
    print()

    # 2) Monte Carlo experiment with default parameters
    N = 1000
    stats = run_experiment(
        num_trials=N,
        p_success_stat=DEFAULT_P_SUCCESS_STAT,
        p_success_id=DEFAULT_P_SUCCESS_ID,
        util_saved=DEFAULT_UTIL_SAVED
    )

    print(f"Monte Carlo over {N} trials per context (pymdp)")
    print("------------------------------------------------")
    for ctx in ["stat", "id"]:
        trials = stats[ctx]["trials"]
        help_ = stats[ctx]["help"]
        succ = stats[ctx]["success"]
        help_rate = help_ / trials
        succ_rate = succ / trials
        print(f"Context: {ctx}")
        print(f"  Help rate:    {help_rate:.3f}")
        print(f"  Success rate: {succ_rate:.3f}")
        print()
    log_mc_results(
        stats,
        filename="mc_results.csv",
        p_success_stat=DEFAULT_P_SUCCESS_STAT,
        p_success_id=DEFAULT_P_SUCCESS_ID,
        util_saved=DEFAULT_UTIL_SAVED
    )

    # 3) Analytic EFE curves + logs
    plot_and_log_prob_help_curves(
        filename="analytic_curve.csv",
        u_saved=DEFAULT_UTIL_SAVED,
        cost_stat=DEFAULT_COST_HELP,
        cost_id=DEFAULT_COST_HELP * 0.3,
        beta=5.0
    )

    # 4) Analytic heatmaps + logs
    plot_and_log_heatmaps(
        filename="analytic_heatmap.csv",
        u_saved=DEFAULT_UTIL_SAVED,
        cost_stat=DEFAULT_COST_HELP,
        cost_id=DEFAULT_COST_HELP * 0.3,
        beta=5.0
    )


if __name__ == "__main__":
    main()
