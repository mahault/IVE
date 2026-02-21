"""
Testable fMRI predictions from the IVE active inference model.

Five predictions derived from the factorized model, calibrated against
Zhao et al. (2024) empirical contrasts:

1. TPJ: UIV > IV — identity precision → inverse mentalizing demand
2. Insula: IV > UIV — affect update magnitude for identified victims
3. mPFC: IV > UIV — self-referential/narrative processing for identified
4. Aggregation increases TPJ — reducing identity precision raises mentalizing demand
5. TPJ-Insula coupling — identity_affect_coupling predicts PPI under IV condition

References:
    Zhao H et al. (2024). Human Brain Mapping, 45(2), e26609.
    Gaesser B et al. (2019). J Exp Psych: General, 149(8), 1455-1464.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional

from .networks import (
    build_network_agent, context_to_network_states, apply_aggregation,
    NETWORK_DEFAULTS, IDENTITY, AFFECT, DISTANCE,
    NUM_STATES, IDENTITY_LABELS,
)
from .neuroimaging import extract_neural_regressors, FACTOR_ROI_MAP


@dataclass
class Prediction:
    """A testable fMRI prediction from the model."""
    id: int
    title: str
    contrast: str
    direction: str
    mechanism: str
    rois: list
    model_regressor: str
    empirical_support: str
    effect_size: Optional[float] = None


# ---------------------------------------------------------------------------
# The five predictions
# ---------------------------------------------------------------------------

PREDICTIONS = [
    Prediction(
        id=1,
        title="TPJ reflects mentalizing demand (UIV > IV)",
        contrast="Unidentified > Identified",
        direction="UIV > IV",
        mechanism=(
            "High identity precision (identified victim) produces low prediction "
            "error in the identity factor, requiring less mentalizing effort. "
            "Anonymous victims require effortful inference → higher TPJ."
        ),
        rois=["rTPJ", "lTPJ"],
        model_regressor="tpj_proxy",
        empirical_support=(
            "Zhao et al. (2024): bilateral TPJ UIV > IV, rTPJ t=8.18, lTPJ t=6.41. "
            "Brain-behavior: lTPJ r=-0.38 (p=.035) — more TPJ activation for UIV "
            "predicts LESS IVE behaviorally."
        ),
        effect_size=0.57,
    ),
    Prediction(
        id=2,
        title="Insula encodes affect update (IV > UIV)",
        contrast="Identified > Unidentified",
        direction="IV > UIV",
        mechanism=(
            "Identity-affect coupling amplifies the affective response to identified "
            "victims. The posterior update on S_affect is larger → stronger insula "
            "BOLD. This is the core IVE mechanism in the model."
        ),
        rois=["rAnteriorInsula", "lAnteriorInsula"],
        model_regressor="insula_proxy",
        empirical_support=(
            "Zhao et al. (2024): IV > UIV in left TP/STG (BA 38, adjacent to insula), "
            "t=7.05. Consistent with meta-analyses linking anterior insula to "
            "empathic concern for identified others (FeldmanHall et al., 2015)."
        ),
    ),
    Prediction(
        id=3,
        title="mPFC encodes narrative/distance processing (IV > UIV)",
        contrast="Identified > Unidentified",
        direction="IV > UIV",
        mechanism=(
            "Identified victims activate episodic simulation / self-referential "
            "processing in mPFC. In the model, distance encoding is lower (proximal) "
            "for identified victims, corresponding to higher mPFC engagement in "
            "self-other overlap."
        ),
        rois=["mPFC"],
        model_regressor="mpfc_proxy",
        empirical_support=(
            "Zhao et al. (2024): Left mPFC IV > UIV, BA 9, t=8.45, volume=2010 mm³ "
            "(largest cluster). PPI: rTPJ-mPFC connectivity significant for IV-UIV."
        ),
        effect_size=None,
    ),
    Prediction(
        id=4,
        title="Aggregation increases TPJ demand",
        contrast="Aggregated > Individual (within identified condition)",
        direction="Higher TPJ after aggregation",
        mechanism=(
            "apply_aggregation() reduces identity precision by noise injection, "
            "shifting the identity factor toward anonymous. This increases "
            "mentalizing demand → higher TPJ. Predicts: news stories about "
            "'100 victims' produce MORE TPJ activation than '1 victim' despite "
            "less behavioral response (scope insensitivity + IVE)."
        ),
        rois=["rTPJ", "lTPJ"],
        model_regressor="tpj_proxy",
        empirical_support=(
            "Gaesser et al. (2019): rTPJ TMS disruption did NOT eliminate IVE "
            "(d=0.22, ns), consistent with TPJ reflecting demand rather than "
            "being the causal mechanism for identification."
        ),
    ),
    Prediction(
        id=5,
        title="TPJ-Insula functional connectivity tracks coupling",
        contrast="Identified > Unidentified (PPI)",
        direction="Stronger TPJ-Insula FC for IV",
        mechanism=(
            "identity_affect_coupling in the model creates functional coupling "
            "between the identity factor (TPJ) and affect factor (Insula). "
            "PPI with TPJ seed should show greater connectivity with insula "
            "during identified vs unidentified trials."
        ),
        rois=["rTPJ", "rAnteriorInsula"],
        model_regressor="tpj_insula_fc",
        empirical_support=(
            "Zhao et al. (2024): PPI with rTPJ seed shows mPFC connectivity "
            "for IV-UIV (t=7.19). Model predicts insula should also show "
            "task-dependent coupling (testable with extended PPI analysis)."
        ),
    ),
]


def get_predictions() -> list:
    """Return the list of all five predictions."""
    return PREDICTIONS


def predictions_to_table() -> pd.DataFrame:
    """Format predictions as a summary table.

    Returns:
        DataFrame with columns: id, title, direction, ROIs, regressor, empirical_d.
    """
    rows = []
    for p in PREDICTIONS:
        rows.append({
            "id": p.id,
            "title": p.title,
            "direction": p.direction,
            "ROIs": ", ".join(p.rois),
            "regressor": p.model_regressor,
            "empirical_d": p.effect_size,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Prediction testing functions
# ---------------------------------------------------------------------------

def _build_trial_grid():
    """Generate all 27 trial configurations (3×3×3 identity×affect×distance)."""
    configs = []
    for id_s in range(3):
        for aff_s in range(3):
            for dist_s in range(3):
                configs.append({
                    "identity_state": id_s,
                    "affect_state": aff_s,
                    "distance_state": dist_s,
                })
    return configs


def prediction_1_tpj_mentalizing_demand(model_params=None) -> dict:
    """Test Prediction 1: TPJ proxy should be HIGHER for anonymous (UIV > IV).

    Computes tpj_proxy for identified (identity_state=2) vs anonymous
    (identity_state=0) across affect and distance states.

    Returns:
        Dict with 'identified_mean', 'anonymous_mean', 'direction_correct',
        'effect_size' (Cohen's d).
    """
    configs = _build_trial_grid()
    regressors = extract_neural_regressors(configs, model_params)

    id_vals = regressors[regressors["identity_state"] == 2]["tpj_proxy"]
    anon_vals = regressors[regressors["identity_state"] == 0]["tpj_proxy"]

    id_mean = id_vals.mean()
    anon_mean = anon_vals.mean()
    pooled_sd = np.sqrt((id_vals.std()**2 + anon_vals.std()**2) / 2)
    d = (anon_mean - id_mean) / max(pooled_sd, 1e-10)

    return {
        "prediction": "TPJ: UIV > IV",
        "identified_mean": float(id_mean),
        "anonymous_mean": float(anon_mean),
        "direction_correct": anon_mean > id_mean,
        "effect_size_d": float(d),
    }


def prediction_2_insula_affect_update(model_params=None) -> dict:
    """Test Prediction 2: Insula proxy should be HIGHER for identified (IV > UIV).

    Returns:
        Dict with means, direction check, and effect size.
    """
    configs = _build_trial_grid()
    regressors = extract_neural_regressors(configs, model_params)

    id_vals = regressors[regressors["identity_state"] == 2]["insula_proxy"]
    anon_vals = regressors[regressors["identity_state"] == 0]["insula_proxy"]

    id_mean = id_vals.mean()
    anon_mean = anon_vals.mean()
    pooled_sd = np.sqrt((id_vals.std()**2 + anon_vals.std()**2) / 2)
    d = (id_mean - anon_mean) / max(pooled_sd, 1e-10)

    return {
        "prediction": "Insula: IV > UIV",
        "identified_mean": float(id_mean),
        "anonymous_mean": float(anon_mean),
        "direction_correct": id_mean > anon_mean,
        "effect_size_d": float(d),
    }


def prediction_3_mpfc_narrative(model_params=None) -> dict:
    """Test Prediction 3: mPFC proxy should be HIGHER for identified (IV > UIV).

    mPFC proxy combines proximity (low distance) and identity level to capture
    self-referential / narrative processing engagement.

    Returns:
        Dict with means, direction check, and effect size.
    """
    configs = _build_trial_grid()
    regressors = extract_neural_regressors(configs, model_params)

    id_vals = regressors[regressors["identity_state"] == 2]["mpfc_proxy"]
    anon_vals = regressors[regressors["identity_state"] == 0]["mpfc_proxy"]

    id_mean = id_vals.mean()
    anon_mean = anon_vals.mean()
    pooled_sd = np.sqrt((id_vals.std()**2 + anon_vals.std()**2) / 2)
    d = (id_mean - anon_mean) / max(pooled_sd, 1e-10)

    return {
        "prediction": "mPFC: IV > UIV",
        "identified_mean": float(id_mean),
        "anonymous_mean": float(anon_mean),
        "direction_correct": id_mean > anon_mean,
        "effect_size_d": float(d),
    }


def prediction_4_aggregation_tpj(
    base_params=None,
    n_victims=10,
) -> dict:
    """Test Prediction 4: Aggregation should increase TPJ proxy.

    Compares TPJ proxy for identified victims before and after
    apply_aggregation() reduces identity-affect coupling and shifts
    states toward anonymous/abstract.

    Returns:
        Dict with pre/post aggregation means and direction check.
    """
    # Before aggregation: identified + proximal + high affect
    pre_configs = [
        {"identity_state": 2, "affect_state": a, "distance_state": d}
        for a in range(3) for d in range(3)
    ]
    pre_regressors = extract_neural_regressors(pre_configs, base_params)
    pre_tpj = pre_regressors["tpj_proxy"].mean()

    # After aggregation: apply_aggregation modifies states and coupling
    agg_mods = apply_aggregation(
        n_victims=n_victims,
        aggregation_type="bureaucratic",
    )
    agg_params = dict(base_params or {})
    agg_params.update({
        k: v for k, v in agg_mods.items()
        if k not in ("identity_state", "affect_state", "distance_state")
    })
    # Aggregation shifts states toward anonymous/abstract
    post_configs = [
        {
            "identity_state": agg_mods.get("identity_state", 0),
            "affect_state": agg_mods.get("affect_state", 0),
            "distance_state": agg_mods.get("distance_state", 2),
        }
        for _ in range(9)  # same count as pre
    ]
    post_regressors = extract_neural_regressors(post_configs, agg_params)
    post_tpj = post_regressors["tpj_proxy"].mean()

    return {
        "prediction": "Aggregation increases TPJ",
        "pre_aggregation_tpj": float(pre_tpj),
        "post_aggregation_tpj": float(post_tpj),
        "n_victims": n_victims,
        "coupling_before": NETWORK_DEFAULTS["identity_affect_coupling"],
        "coupling_after": agg_mods.get("identity_affect_coupling"),
        "direction_correct": post_tpj > pre_tpj,
    }


def prediction_5_tpj_insula_coupling(model_params=None) -> dict:
    """Test Prediction 5: TPJ-Insula FC proxy should be stronger for identified.

    Returns:
        Dict with means, direction check, and effect size.
    """
    configs = _build_trial_grid()
    regressors = extract_neural_regressors(configs, model_params)

    id_vals = regressors[regressors["identity_state"] == 2]["tpj_insula_fc"]
    anon_vals = regressors[regressors["identity_state"] == 0]["tpj_insula_fc"]

    id_mean = id_vals.mean()
    anon_mean = anon_vals.mean()
    pooled_sd = np.sqrt((id_vals.std()**2 + anon_vals.std()**2) / 2)
    d = (id_mean - anon_mean) / max(pooled_sd, 1e-10)

    return {
        "prediction": "TPJ-Insula FC: IV > UIV",
        "identified_mean": float(id_mean),
        "anonymous_mean": float(anon_mean),
        "direction_correct": id_mean > anon_mean,
        "effect_size_d": float(d),
    }


def generate_all_predictions(model_params=None) -> pd.DataFrame:
    """Run all five prediction tests and return results as a DataFrame.

    Args:
        model_params: Override model parameters. Default uses NETWORK_DEFAULTS.

    Returns:
        DataFrame with one row per prediction, columns for means and direction.
    """
    results = [
        prediction_1_tpj_mentalizing_demand(model_params),
        prediction_2_insula_affect_update(model_params),
        prediction_3_mpfc_narrative(model_params),
        prediction_4_aggregation_tpj(model_params),
        prediction_5_tpj_insula_coupling(model_params),
    ]
    return pd.DataFrame(results)


def compare_predictions_to_zhao() -> pd.DataFrame:
    """Compare model prediction directions with Zhao et al. empirical contrasts.

    Uses Gaesser-fitted parameters (coupling=0.65, cost=0.9) and checks
    whether model predictions match Zhao contrast directions.

    Returns:
        DataFrame with prediction, model_direction, zhao_direction, match.
    """
    gaesser_params = {
        "identity_affect_coupling": 0.65,
        "cost_penalty": 0.9,
        "util_saved": 1.4,
        "affect_preference_boost": 0.4,
    }

    results = generate_all_predictions(gaesser_params)

    zhao_directions = {
        "TPJ: UIV > IV": "UIV > IV",
        "Insula: IV > UIV": "IV > UIV",
        "mPFC: IV > UIV": "IV > UIV",
        "Aggregation increases TPJ": "N/A (novel prediction)",
        "TPJ-Insula FC: IV > UIV": "IV > UIV (PPI: rTPJ-mPFC)",
    }

    rows = []
    for _, row in results.iterrows():
        pred = row["prediction"]
        rows.append({
            "prediction": pred,
            "model_correct_direction": row["direction_correct"],
            "zhao_empirical": zhao_directions.get(pred, ""),
        })

    return pd.DataFrame(rows)
