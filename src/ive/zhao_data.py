"""
Zhao et al. (2024) published summary statistics.

Source: Zhao H, Xu Y, Li L, Liu J, Cui F (2024). The neural mechanisms of
identifiable victim effect in prosocial decision-making. Human Brain Mapping,
45(2), e26609. https://doi.org/10.1002/hbm.26609

N = 31 (15 female, mean age 20.26 +/- 1.52)
Design: 2 (IV vs UIV) x 2 (Money vs Effort) within-subjects
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Behavioral data
# ---------------------------------------------------------------------------

ZHAO_BEHAVIORAL = {
    "n": 31,
    "age_mean": 20.26,
    "age_sd": 1.52,
    "n_female": 15,
    # Money task: donation in monetary units (MUs), range 0-10
    "money_iv_mean": 5.43,
    "money_iv_sd": 1.89,
    "money_uiv_mean": 4.35,
    "money_uiv_sd": 1.92,
    "money_t": 6.96,
    "money_df": 30,
    "money_d": 0.57,
    # Effort task: squeezes
    "effort_iv_mean": 10.17,
    "effort_iv_sd": 4.43,
    "effort_uiv_mean": 8.16,
    "effort_uiv_sd": 4.72,
    "effort_t": 6.42,
    "effort_df": 30,
    "effort_d": 0.44,
    # Cross-task correlation
    "cross_task_r": 0.74,
    "cross_task_p": 0.001,
}


# ---------------------------------------------------------------------------
# fMRI contrasts (conjunction of Money and Effort tasks)
# ---------------------------------------------------------------------------

# IV > UIV: identified > unidentifiable
ZHAO_IV_GT_UIV = pd.DataFrame([
    {"region": "Left mPFC", "BA": 9, "x": -10, "y": 44, "z": 48, "volume": 2010, "t": 8.45},
    {"region": "Left MTG", "BA": 21, "x": -60, "y": -6, "z": -18, "volume": 402, "t": 7.28},
    {"region": "Left TP/STG", "BA": 38, "x": -42, "y": 22, "z": -16, "volume": 957, "t": 7.05},
    {"region": "Posterior cingulate", "BA": 30, "x": -4, "y": -50, "z": 24, "volume": 648, "t": 6.86},
    {"region": "Right MTG", "BA": 21, "x": 60, "y": -2, "z": -22, "volume": 146, "t": 6.29},
    {"region": "Right orbIFG", "BA": 47, "x": 32, "y": 30, "z": -16, "volume": 153, "t": 6.18},
    {"region": "Right dorSFG", "BA": 9, "x": 20, "y": 40, "z": 52, "volume": 89, "t": 5.39},
])

# UIV > IV: unidentifiable > identified (more mentalizing demand for anonymous)
ZHAO_UIV_GT_IV = pd.DataFrame([
    {"region": "Right TPJ", "BA": 40, "x": 52, "y": -46, "z": 40, "volume": 2961, "t": 8.18},
    {"region": "Left TPJ", "BA": 40, "x": -52, "y": -44, "z": 44, "volume": 1081, "t": 6.41},
    {"region": "Right MFG", "BA": 9, "x": 30, "y": 36, "z": 36, "volume": 741, "t": 6.30},
    {"region": "Left MTG", "BA": 37, "x": -44, "y": -68, "z": 10, "volume": 515, "t": 5.99},
    {"region": "Right PCG", "BA": 6, "x": 38, "y": 6, "z": 48, "volume": 69, "t": 5.43},
    {"region": "Left dorSFG", "BA": 6, "x": -20, "y": 8, "z": 62, "volume": 45, "t": 5.02},
])

# PPI: rTPJ seed connectivity (IV - UIV contrast)
ZHAO_PPI_RTPJ = pd.DataFrame([
    {"region": "mPFC (Money)", "x": 4, "y": 32, "z": 2, "cluster": 31, "t": 4.34},
    {"region": "mPFC (Effort)", "x": 4, "y": 30, "z": 10, "cluster": 1407, "t": 7.19},
])

# MVPA classification accuracy (IV vs UIV)
ZHAO_MVPA = pd.DataFrame([
    {"region": "lTPJ", "money_acc": 64.32, "money_p": 0.001, "effort_acc": 65.85, "effort_p": 0.001},
    {"region": "rTPJ", "money_acc": 65.54, "money_p": 0.001, "effort_acc": 59.97, "effort_p": 0.005},
    {"region": "MFG", "money_acc": 55.34, "money_p": 0.01, "effort_acc": 59.04, "effort_p": 0.001},
    {"region": "lMTG", "money_acc": 61.26, "money_p": 0.001, "effort_acc": 60.99, "effort_p": 0.001},
    {"region": "rMTG", "money_acc": 62.01, "money_p": 0.001, "effort_acc": 56.84, "effort_p": 0.006},
    {"region": "TP", "money_acc": 56.50, "money_p": 0.001, "effort_acc": 58.78, "effort_p": 0.001},
])

# Brain-behavior correlations
ZHAO_CORRELATIONS = {
    "lTPJ_money_r": -0.38,
    "lTPJ_money_p": 0.035,
    "lTPJ_effort_r": -0.40,
    "lTPJ_effort_p": 0.024,
    "rTPJ_money_r": -0.32,
    "rTPJ_money_p": 0.077,
    "rTPJ_effort_r": -0.34,
    "rTPJ_effort_p": 0.058,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_zhao_behavioral_targets():
    """Return behavioral targets normalized for model fitting.

    Normalizes money task (0-10 MU scale) to 0-1 rates.

    Returns:
        Dict with 'iv_rate', 'uiv_rate', 'ive_magnitude', 'd'.
    """
    b = ZHAO_BEHAVIORAL
    return {
        "iv_rate": b["money_iv_mean"] / 10.0,
        "uiv_rate": b["money_uiv_mean"] / 10.0,
        "ive_magnitude": (b["money_iv_mean"] - b["money_uiv_mean"]) / 10.0,
        "d": b["money_d"],
        "n": b["n"],
    }


def get_zhao_fmri_contrasts():
    """Return fMRI contrast tables.

    Returns:
        Dict with 'iv_gt_uiv', 'uiv_gt_iv', 'ppi', 'mvpa' DataFrames.
    """
    return {
        "iv_gt_uiv": ZHAO_IV_GT_UIV,
        "uiv_gt_iv": ZHAO_UIV_GT_IV,
        "ppi": ZHAO_PPI_RTPJ,
        "mvpa": ZHAO_MVPA,
    }


def compare_model_to_zhao(model_predictions):
    """Compare model neural regressor predictions to Zhao fMRI contrasts.

    For each region, checks if the model predicts the correct direction:
      - TPJ: model should predict UIV > IV (mentalizing demand)
      - mPFC: model should predict IV > UIV (narrative processing)
      - Insula: model should predict IV > UIV (affect)

    Args:
        model_predictions: Dict with keys 'stat' and 'id', each containing
            dicts of {roi_name: regressor_value}.

    Returns:
        DataFrame with region, empirical direction, model direction, match.
    """
    comparisons = []

    # TPJ: empirical UIV > IV
    for region in ["rTPJ", "lTPJ"]:
        emp_direction = "UIV > IV"
        if region.lower().replace("_", "") in [k.lower() for k in model_predictions.get("stat", {})]:
            stat_val = model_predictions["stat"].get(region, 0)
            id_val = model_predictions["id"].get(region, 0)
            mod_direction = "UIV > IV" if stat_val > id_val else "IV > UIV"
            match = mod_direction == emp_direction
        else:
            mod_direction = "N/A"
            match = None
        comparisons.append({
            "region": region, "empirical": emp_direction,
            "model": mod_direction, "match": match,
        })

    # mPFC: empirical IV > UIV
    emp_direction = "IV > UIV"
    stat_val = model_predictions.get("stat", {}).get("mPFC", 0)
    id_val = model_predictions.get("id", {}).get("mPFC", 0)
    mod_direction = "IV > UIV" if id_val > stat_val else "UIV > IV"
    comparisons.append({
        "region": "mPFC", "empirical": emp_direction,
        "model": mod_direction, "match": mod_direction == emp_direction,
    })

    # Insula: empirical IV > UIV
    emp_direction = "IV > UIV"
    stat_val = model_predictions.get("stat", {}).get("insula", 0)
    id_val = model_predictions.get("id", {}).get("insula", 0)
    mod_direction = "IV > UIV" if id_val > stat_val else "UIV > IV"
    comparisons.append({
        "region": "Insula", "empirical": emp_direction,
        "model": mod_direction, "match": mod_direction == emp_direction,
    })

    return pd.DataFrame(comparisons)
