"""
ive_pymdp_full_active.py

Fully active-inference model of the Identified Victim Effect (IVE) using pymdp.

Hidden states (single factor, 8 states) encode:
    context: 0 = Statistical, 1 = Identified
    outcome: 0 = NotSaved,   1 = Saved
    cost:    0 = NoCost,     1 = Cost

State index:
    idx = context * 4 + outcome * 2 + cost

Observations (3 modalities, each binary):
    Modality 0 (context cue): 0 = stat-cue, 1 = id-cue
    Modality 1 (outcome):     0 = no saved, 1 = victim saved
    Modality 2 (cost):        0 = no cost,  1 = cost

Actions:
    0 = NoHelp
    1 = Help

Generative model:
    - A: deterministic mapping from hidden state to observations.
    - B: action-dependent transitions for (context, outcome, cost).
    - C: preferences over observations:
        * like seeing "saved" in outcome modality
        * dislike seeing "cost" in cost modality
        * neutral over context cue
    - D: prior over hidden states: NotSaved, NoCost, unknown context (50/50)

Under active inference, the agent:
    - infers hidden state from observations (infer_states)
    - evaluates policies via expected free energy (infer_policies)
    - selects an action (sample_action)

The script:
    1) Runs a single-step demo.
    2) Runs Monte Carlo experiments for default parameters.
    3) Optionally runs a parameter sweep and plots heatmaps.
    4) Tunes parameters to match target help rates (e.g. 40% vs 80%).

Requires:
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

DEFAULT_P_SUCCESS_STAT = 0.3   # P(saved | Help, Statistical)
DEFAULT_P_SUCCESS_ID   = 0.9   # P(saved | Help, Identified)
DEFAULT_UTIL_SAVED     = 2.0   # utility for seeing "victim saved"
DEFAULT_COST_PENALTY   = 0.5   # dislike of seeing cost (log-preference penalty)

# Target help rates to match (e.g. from empirical study)
TARGET_HELP_STAT = 0.40
TARGET_HELP_ID   = 0.80


# --------------------------------------------------------------------
# STATE INDEXING
# --------------------------------------------------------------------

# context: 0=Stat, 1=Id
# outcome: 0=NotSaved, 1=Saved
# cost:    0=NoCost,   1=Cost
def state_index(context, outcome, cost):
    return context * 4 + outcome * 2 + cost


# --------------------------------------------------------------------
# BUILD FULL ACTIVE INFERENCE AGENT
# --------------------------------------------------------------------

def build_agent(p_success_stat=DEFAULT_P_SUCCESS_STAT,
                p_success_id=DEFAULT_P_SUCCESS_ID,
                util_saved=DEFAULT_UTIL_SAVED,
                cost_penalty=DEFAULT_COST_PENALTY):
    """
    Build an active inference agent with a full generative model:

    - A: likelihood for 3 modalities (context, outcome, cost)
    - B: transitions for 8 hidden states, 2 actions
    - C: preferences over observations (like saved, dislike cost)
    - D: prior over initial hidden state (NotSaved, unknown context, NoCost)
    """

    # Hidden state factor: 8 combined states
    num_states = [8]
    num_controls = [2]  # NoHelp, Help
    # Observation modalities: [context, outcome, cost], each with 2 outcomes
    num_obs = [2, 2, 2]

    # ---------------- A: Likelihood ----------------
    A = utils.obj_array(len(num_obs))

    # Modality 0: context cue (0=stat-cue, 1=id-cue)
    A_context = np.zeros((num_obs[0], num_states[0]))
    # Modality 1: outcome (0=no saved, 1=saved)
    A_outcome = np.zeros((num_obs[1], num_states[0]))
    # Modality 2: cost (0=no cost, 1=cost)
    A_cost = np.zeros((num_obs[2], num_states[0]))

    for s in range(num_states[0]):
        # decode state index into (context, outcome, cost)
        context = s // 4           # 0 or 1
        rem = s % 4
        outcome = rem // 2         # 0 or 1
        cost_state = rem % 2       # 0 or 1

        # context cue
        A_context[context, s] = 1.0
        # outcome cue
        A_outcome[outcome, s] = 1.0
        # cost cue
        A_cost[cost_state, s] = 1.0

    A[0] = A_context
    A[1] = A_outcome
    A[2] = A_cost

    # ---------------- B: Transitions ----------------
    B = utils.obj_array(len(num_states))
    B_factor = np.zeros((num_states[0], num_states[0], num_controls[0]))

    # Action 0: NoHelp
    # - context unchanged
    # - outcome forced to NotSaved
    # - cost forced to NoCost
    for s in range(num_states[0]):
        context = s // 4
        # next state always (context, NotSaved, NoCost)
        s_next = state_index(context, outcome=0, cost=0)
        B_factor[s_next, s, 0] = 1.0

    # Action 1: Help
    for s in range(num_states[0]):
        context = s // 4
        rem = s % 4
        outcome = rem // 2

        if outcome == 0:  # currently NotSaved
            if context == 0:   # Statistical
                p_succ = p_success_stat
            else:              # Identified
                p_succ = p_success_id

            # NotSaved -> Saved with probability p_succ; cost -> Cost
            s_saved = state_index(context, outcome=1, cost=1)
            s_notsaved = state_index(context, outcome=0, cost=1)
            B_factor[s_saved, s, 1] = p_succ
            B_factor[s_notsaved, s, 1] = 1.0 - p_succ
        else:
            # Already Saved: remain Saved but incur cost when acting
            s_saved_cost = state_index(context, outcome=1, cost=1)
            B_factor[s_saved_cost, s, 1] = 1.0

    B[0] = B_factor

    # ---------------- D: Prior over states ----------------
    D = utils.obj_array(len(num_states))
    D_factor = np.zeros(num_states[0])
    # Start with NotSaved, NoCost, unknown context: 50/50 stat vs id
    D_factor[state_index(0, 0, 0)] = 0.5  # (Stat, NotSaved, NoCost)
    D_factor[state_index(1, 0, 0)] = 0.5  # (Id,   NotSaved, NoCost)
    D[0] = D_factor

    # ---------------- C: Preferences ----------------
    C = utils.obj_array(len(num_obs))

    # Modality 0 (context cue): neutral
    C_context = np.array([0.0, 0.0])

    # Modality 1 (outcome): like "saved"
    C_outcome = np.array([0.0, util_saved])

    # Modality 2 (cost): dislike "cost"
    C_cost = np.array([0.0, -cost_penalty])

    C[0] = C_context
    C[1] = C_outcome
    C[2] = C_cost

    # ---------------- Build Agent ----------------
    agent = Agent(A=A, B=B, C=C, D=D)
    return agent


# --------------------------------------------------------------------
# SINGLE-STEP DECISION
# --------------------------------------------------------------------

def choose_action(agent, context_obs):
    """
    Show the agent:
      - context cue (0=stat, 1=id)
      - outcome: no one saved (0)
      - cost: no cost yet (0)
    and return the chosen action (0=NoHelp, 1=Help).
    """
    agent.reset()
    observation = [context_obs, 0, 0]  # [context, outcome, cost]

    agent.infer_states(observation)
    agent.infer_policies()
    action = agent.sample_action()
    return int(action)


# --------------------------------------------------------------------
# MONTE CARLO EXPERIMENTS
# --------------------------------------------------------------------

def run_experiment(
    num_trials=1000,
    p_success_stat=DEFAULT_P_SUCCESS_STAT,
    p_success_id=DEFAULT_P_SUCCESS_ID,
    util_saved=DEFAULT_UTIL_SAVED,
    cost_penalty=DEFAULT_COST_PENALTY
):
    """
    Run many simulated trials for statistical vs identified contexts.

    For each trial:
      - Build or reuse an agent with given parameters.
      - Provide context, outcome, cost observations.
      - Let agent choose action via active inference.
      - Sample environment next state from B.
      - Record help and success.
    """

    stats = {
        "stat": {"help": 0, "trials": 0, "success": 0},
        "id":   {"help": 0, "trials": 0, "success": 0},
    }

    for context_label in ["stat", "id"]:
        context_obs = 0 if context_label == "stat" else 1
        initial_state = state_index(context_obs, 0, 0)  # (ctx, NotSaved, NoCost)

        agent = build_agent(
            p_success_stat=p_success_stat,
            p_success_id=p_success_id,
            util_saved=util_saved,
            cost_penalty=cost_penalty
        )
        B_factor = agent.B[0]

        for _ in range(num_trials):
            agent.reset()
            observation = [context_obs, 0, 0]
            agent.infer_states(observation)
            agent.infer_policies()
            action = int(agent.sample_action())

            stats[context_label]["trials"] += 1
            if action == 1:
                stats[context_label]["help"] += 1

            # Environment transition: from initial_state with chosen action
            probs_next = B_factor[:, initial_state, action]
            next_state = np.random.choice(np.arange(len(probs_next)), p=probs_next)

            # Decode outcome from next_state
            outcome = (next_state % 4) // 2  # 0=NotSaved, 1=Saved
            if outcome == 1:
                stats[context_label]["success"] += 1

    return stats


def log_mc_results(
    stats,
    filename="mc_full_active.csv",
    p_success_stat=DEFAULT_P_SUCCESS_STAT,
    p_success_id=DEFAULT_P_SUCCESS_ID,
    util_saved=DEFAULT_UTIL_SAVED,
    cost_penalty=DEFAULT_COST_PENALTY
):
    """
    Save Monte Carlo summary stats to CSV.
    """
    path = os.path.join(LOG_DIR, filename)
    header = [
        "context",
        "p_success_stat",
        "p_success_id",
        "util_saved",
        "cost_penalty",
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
                cost_penalty,
                trials,
                help_,
                succ,
                help_rate,
                succ_rate
            ])
    print(f"[LOG] Monte Carlo results written to {path}")


# --------------------------------------------------------------------
# PARAMETER SWEEP (USING FULL AGENT)
# --------------------------------------------------------------------

def parameter_sweep(
    num_trials=200,
    p_stat_grid=None,
    cost_grid=None,
    p_success_id=DEFAULT_P_SUCCESS_ID,
    util_saved=DEFAULT_UTIL_SAVED
):
    """
    Sweep over:
      - p_success_stat (x-axis)
      - cost_penalty   (y-axis)
    and compute help rates for stat vs id contexts using the full agent.

    Returns:
        p_stat_grid, cost_grid, help_stat, help_id
    """
    if p_stat_grid is None:
        p_stat_grid = np.linspace(0.0, 0.9, 8)
    if cost_grid is None:
        cost_grid = np.linspace(0.0, 1.5, 8)

    help_stat = np.zeros((len(cost_grid), len(p_stat_grid)))
    help_id   = np.zeros((len(cost_grid), len(p_stat_grid)))

    total = len(cost_grid) * len(p_stat_grid)
    idx = 0

    for i, cost_penalty in enumerate(cost_grid):
        for j, p_stat in enumerate(p_stat_grid):
            idx += 1
            stats = run_experiment(
                num_trials=num_trials,
                p_success_stat=p_stat,
                p_success_id=p_success_id,
                util_saved=util_saved,
                cost_penalty=cost_penalty
            )
            help_stat[i, j] = stats["stat"]["help"] / stats["stat"]["trials"]
            help_id[i, j]   = stats["id"]["help"] / stats["id"]["trials"]

            if idx % 20 == 0:
                print(f"[SWEEP] {idx}/{total} combos done...")

    return p_stat_grid, cost_grid, help_stat, help_id


def log_sweep_results(p_stat_grid, cost_grid, help_stat, help_id,
                      filename="sweep_full_active.csv",
                      p_success_id=DEFAULT_P_SUCCESS_ID,
                      util_saved=DEFAULT_UTIL_SAVED):
    """
    Save parameter sweep results to CSV.
    """
    path = os.path.join(LOG_DIR, filename)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "p_success_stat",
            "cost_penalty",
            "p_success_id",
            "util_saved",
            "help_rate_stat",
            "help_rate_id"
        ])
        for i, cost_penalty in enumerate(cost_grid):
            for j, p_stat in enumerate(p_stat_grid):
                writer.writerow([
                    p_stat,
                    cost_penalty,
                    p_success_id,
                    util_saved,
                    help_stat[i, j],
                    help_id[i, j]
                ])
    print(f"[LOG] Sweep results written to {path}")


def plot_sweep(p_stat_grid, cost_grid, help_stat, help_id):
    """
    Plot heatmaps of help rates for stat vs id contexts over (p_success_stat, cost_penalty).
    """
    # Statistical context
    plt.figure()
    plt.imshow(
        help_stat,
        origin="lower",
        aspect="auto",
        extent=[p_stat_grid[0], p_stat_grid[-1], cost_grid[0], cost_grid[-1]],
        vmin=0.0, vmax=1.0
    )
    plt.colorbar(label="Help rate (statistical victim)")
    plt.xlabel("p_success_stat")
    plt.ylabel("cost_penalty")
    plt.title("Help rate for statistical victim\n(full active-inference agent)")
    plt.tight_layout()

    # Identified context
    plt.figure()
    plt.imshow(
        help_id,
        origin="lower",
        aspect="auto",
        extent=[p_stat_grid[0], p_stat_grid[-1], cost_grid[0], cost_grid[-1]],
        vmin=0.0, vmax=1.0
    )
    plt.colorbar(label="Help rate (identified victim)")
    plt.xlabel("p_success_stat")
    plt.ylabel("cost_penalty")
    plt.title("Help rate for identified victim\n(full active-inference agent)")
    plt.tight_layout()

    plt.show()


# --------------------------------------------------------------------
# PARAMETER TUNING TO MATCH TARGET HELP RATES
# --------------------------------------------------------------------

def tune_parameters(
    target_stat=TARGET_HELP_STAT,
    target_id=TARGET_HELP_ID,
    util_saved=DEFAULT_UTIL_SAVED,
    p_stat_candidates=None,
    p_id_candidates=None,
    cost_candidates=None,
    num_trials=300
):
    """
    Grid search over (p_success_stat, p_success_id, cost_penalty) to approximate
    target help rates for statistical and identified contexts.

    Returns:
        best_params (dict), best_stats (dict)
    """

    if p_stat_candidates is None:
        # subjective success for stat victims: low to medium
        p_stat_candidates = np.linspace(0.05, 0.6, 8)
    if p_id_candidates is None:
        # subjective success for identified victims: medium to high
        p_id_candidates = np.linspace(0.5, 0.99, 8)
    if cost_candidates is None:
        # cost of helping: low to fairly strong
        cost_candidates = np.linspace(0.1, 1.5, 8)

    best_E = np.inf
    best_params = None
    best_stats = None

    total_combos = len(p_stat_candidates) * len(p_id_candidates) * len(cost_candidates)
    combo_idx = 0

    print("Tuning parameters...")
    print(f"Total combinations: {total_combos}")

    for p_stat in p_stat_candidates:
        for p_id in p_id_candidates:
            for cost_penalty in cost_candidates:
                combo_idx += 1
                stats = run_experiment(
                    num_trials=num_trials,
                    p_success_stat=p_stat,
                    p_success_id=p_id,
                    util_saved=util_saved,
                    cost_penalty=cost_penalty
                )
                H_stat = stats["stat"]["help"] / stats["stat"]["trials"]
                H_id   = stats["id"]["help"]   / stats["id"]["trials"]

                # squared error vs target
                E = (H_stat - target_stat)**2 + (H_id - target_id)**2

                if E < best_E:
                    best_E = E
                    best_params = {
                        "p_success_stat": p_stat,
                        "p_success_id": p_id,
                        "util_saved": util_saved,
                        "cost_penalty": cost_penalty,
                    }
                    best_stats = {
                        "H_stat": H_stat,
                        "H_id": H_id,
                        "E": E,
                        "stats": stats
                    }

                if combo_idx % 50 == 0:
                    print(f"  Checked {combo_idx}/{total_combos} combos... Current best E={best_E:.4f}")

    print("Tuning complete.")
    print("Best parameters:")
    for k, v in best_params.items():
        print(f"  {k} = {v}")
    print("Best help rates:")
    print(f"  H_stat ~ {best_stats['H_stat']:.3f} (target {target_stat})")
    print(f"  H_id   ~ {best_stats['H_id']:.3f} (target {target_id})")
    print(f"  Total squared error E = {best_stats['E']:.5f}")

    return best_params, best_stats

def plot_model_vs_target(best_params, best_stats,
                         target_stat=TARGET_HELP_STAT,
                         target_id=TARGET_HELP_ID):
    """
    Simple bar plot comparing target vs model help rates for each context.
    """
    H_stat_model = best_stats["H_stat"]
    H_id_model   = best_stats["H_id"]

    contexts = ["Statistical", "Identified"]
    target_vals = [target_stat, target_id]
    model_vals  = [H_stat_model, H_id_model]

    x = np.arange(len(contexts))
    width = 0.35

    plt.figure()
    plt.bar(x - width/2, target_vals, width, label="Target", alpha=0.7)
    plt.bar(x + width/2, model_vals,  width, label="Model", alpha=0.7)
    plt.xticks(x, contexts)
    plt.ylim(0, 1)
    plt.ylabel("Help rate")
    plt.title("Target vs model help rates")
    plt.legend()
    plt.tight_layout()
    plt.show()


def log_tuning(best_params, best_stats, filename="tuning_full_active.csv"):
    """
    Save tuning result (best parameter set and resulting help rates) to CSV.
    """
    path = os.path.join(LOG_DIR, filename)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "p_success_stat",
            "p_success_id",
            "util_saved",
            "cost_penalty",
            "H_stat",
            "H_id",
            "E"
        ])
        writer.writerow([
            best_params["p_success_stat"],
            best_params["p_success_id"],
            best_params["util_saved"],
            best_params["cost_penalty"],
            best_stats["H_stat"],
            best_stats["H_id"],
            best_stats["E"]
        ])
    print(f"[LOG] Tuning result written to {path}")

def sweep_cost_1d(
    best_params,
    util_saved=DEFAULT_UTIL_SAVED,
    cost_min=0.0,
    cost_max=1.5,
    n_points=20,
    num_trials=300
):
    """
    For fixed p_success_stat and p_success_id (from best_params),
    sweep over cost_penalty and compute help rates for stat & id.
    """
    p_stat = best_params["p_success_stat"]
    p_id   = best_params["p_success_id"]

    cost_values = np.linspace(cost_min, cost_max, n_points)
    help_stat_vals = []
    help_id_vals   = []

    for cost_penalty in cost_values:
        stats = run_experiment(
            num_trials=num_trials,
            p_success_stat=p_stat,
            p_success_id=p_id,
            util_saved=util_saved,
            cost_penalty=cost_penalty
        )
        H_stat = stats["stat"]["help"] / stats["stat"]["trials"]
        H_id   = stats["id"]["help"]   / stats["id"]["trials"]
        help_stat_vals.append(H_stat)
        help_id_vals.append(H_id)

    return cost_values, np.array(help_stat_vals), np.array(help_id_vals)


def plot_help_vs_cost(best_params, best_stats,
                      target_stat=TARGET_HELP_STAT,
                      target_id=TARGET_HELP_ID):
    """
    Plot P(Help) vs cost_penalty for both contexts, marking the tuned cost.
    Also plot the difference curve ΔP = P_help_id - P_help_stat.
    """
    tuned_cost = best_params["cost_penalty"]

    cost_values, H_stat_curve, H_id_curve = sweep_cost_1d(
        best_params,
        util_saved=best_params["util_saved"],
        cost_min=0.0,
        cost_max=1.5,
        n_points=25,
        num_trials=300
    )

    # ---- Plot P(Help) vs cost ----
    plt.figure()
    plt.plot(cost_values, H_stat_curve, label="Statistical victim")
    plt.plot(cost_values, H_id_curve,   label="Identified victim", linestyle="--")
    plt.axvline(tuned_cost, color="k", linestyle=":", label="Tuned cost")
    # Mark the tuned point
    plt.scatter([tuned_cost], [best_stats["H_stat"]], color="C0")
    plt.scatter([tuned_cost], [best_stats["H_id"]],   color="C1")
    plt.xlabel("Cost penalty")
    plt.ylabel("P(Help)")
    plt.ylim(0, 1)
    plt.title("Help probability vs cost penalty")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # ---- Plot ΔP = P_help_id - P_help_stat ----
    delta = H_id_curve - H_stat_curve
    plt.figure()
    plt.plot(cost_values, delta)
    plt.axhline(0.0, color="k", linestyle=":")
    plt.axvline(tuned_cost, color="k", linestyle=":")
    plt.xlabel("Cost penalty")
    plt.ylabel("ΔP(Help) = P_id - P_stat")
    plt.title("Identified – Statistical help probability")
    plt.grid(True)
    plt.tight_layout()

    plt.show()


# --------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------

def main():
    # 1) Single-step demo
    agent = build_agent()
    a_stat = choose_action(agent, context_obs=0)
    a_id   = choose_action(agent, context_obs=1)

    print("Single-step demo (full active-inference agent)")
    print("---------------------------------------------")
    print("Statistical victim ->", "Help" if a_stat == 1 else "Do Nothing")
    print("Identified victim  ->", "Help" if a_id == 1 else "Do Nothing")
    print()

    # 2) Monte Carlo with default parameters
    N = 1000
    stats = run_experiment(
        num_trials=N,
        p_success_stat=DEFAULT_P_SUCCESS_STAT,
        p_success_id=DEFAULT_P_SUCCESS_ID,
        util_saved=DEFAULT_UTIL_SAVED,
        cost_penalty=DEFAULT_COST_PENALTY
    )
    print(f"Monte Carlo over {N} trials per context (full agent)")
    print("----------------------------------------------------")
    for ctx in ["stat", "id"]:
        trials = stats[ctx]["trials"]
        help_  = stats[ctx]["help"]
        succ   = stats[ctx]["success"]
        help_rate = help_ / trials
        succ_rate = succ / trials
        print(f"Context: {ctx}")
        print(f"  Help rate:    {help_rate:.3f}")
        print(f"  Success rate: {succ_rate:.3f}")
        print()
    log_mc_results(stats)

    # 3) Parameter sweep (optional; can comment out if too slow)
    p_stat_grid, cost_grid, help_stat, help_id = parameter_sweep(
        num_trials=200,
        p_stat_grid=None,
        cost_grid=None,
        p_success_id=DEFAULT_P_SUCCESS_ID,
        util_saved=DEFAULT_UTIL_SAVED
    )
    log_sweep_results(p_stat_grid, cost_grid, help_stat, help_id)
    plot_sweep(p_stat_grid, cost_grid, help_stat, help_id)

    # 4) Parameter tuning for target help rates (e.g., 40% vs 80%)
    
    best_params, best_stats = tune_parameters(
    target_stat=TARGET_HELP_STAT,
    target_id=TARGET_HELP_ID,
    util_saved=DEFAULT_UTIL_SAVED,
    num_trials=300
    )
    log_tuning(best_params, best_stats)
    plot_model_vs_target(best_params, best_stats,
                         target_stat=TARGET_HELP_STAT,
                         target_id=TARGET_HELP_ID)
    plot_help_vs_cost(best_params, best_stats,
                      target_stat=TARGET_HELP_STAT,
                      target_id=TARGET_HELP_ID)



if __name__ == "__main__":
    main()
