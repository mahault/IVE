"""
Neuroimaging analysis module for the IVE active inference model.

Provides:
  A. ROI definitions (MNI coordinates from Zhao et al. 2024 and standard atlases)
  B. Model-to-brain readout functions (extract neural regressors from the factorized model)
  C. fMRI data loading for Gaesser et al. (2019) BIDS dataset
  D. GLM and correlation analysis

Key insight from Zhao et al. (2024): TPJ shows UIV > IV (more activation for
unidentifiable victims), reflecting mentalizing DEMAND. In predictive coding
terms, high identity precision (identified victim) means low prediction error,
hence LOWER TPJ BOLD. IV > UIV is seen in mPFC, insula, and temporal pole.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

from .networks import (
    build_network_agent, choose_network_action,
    context_to_network_states, NETWORK_DEFAULTS,
    IDENTITY, AFFECT, DISTANCE, OUTCOME,
    OBS_IDENTITY, OBS_AFFECT, OBS_DISTANCE, OBS_OUTCOME, OBS_COST,
    NUM_STATES,
)


# ---------------------------------------------------------------------------
# Section A: ROI definitions
# ---------------------------------------------------------------------------

# MNI coordinates (peak voxels from published studies)
ROI_COORDS_MNI = {
    # From Zhao et al. (2024) UIV > IV peaks
    "rTPJ": (52, -46, 40),
    "lTPJ": (-52, -44, 44),
    # From Zhao et al. (2024) IV > UIV peaks
    "mPFC": (-10, 44, 48),
    "lTemporalPole": (-42, 22, -16),
    "posteriorCingulate": (-4, -50, 24),
    # Standard coordinates (empathy/affect literature)
    "rAnteriorInsula": (36, 18, 0),
    "lAnteriorInsula": (-36, 18, 0),
    # Decision / valuation
    "dACC": (0, 24, 36),
    "ventralStriatum": (10, 10, -6),
    # PPI connectivity target
    "mPFC_ppi": (4, 32, 2),  # From Zhao PPI analysis
}

ROI_RADIUS_MM = 10

# Mapping from model factors to ROIs
FACTOR_ROI_MAP = {
    "identity_precision": ["rTPJ", "lTPJ"],       # INVERSE: high precision → lower TPJ
    "affect_update": ["rAnteriorInsula", "lAnteriorInsula"],  # higher update → higher insula
    "distance_encoding": ["mPFC"],                  # abstract → higher mPFC
    "policy_precision": ["dACC", "ventralStriatum"],  # decisive → higher striatal
    "coupling_strength": ["rTPJ", "rAnteriorInsula"],  # FC proxy
}


def get_roi_masker(roi_name, radius_mm=ROI_RADIUS_MM, coords=None):
    """Create a nilearn NiftiSpheresMasker for a named ROI.

    Args:
        roi_name: Key from ROI_COORDS_MNI.
        radius_mm: Sphere radius in mm.
        coords: Override coordinate dict.

    Returns:
        nilearn.maskers.NiftiSpheresMasker instance (unfitted).
    """
    from nilearn.maskers import NiftiSpheresMasker
    coord_dict = coords or ROI_COORDS_MNI
    if roi_name not in coord_dict:
        raise ValueError(f"Unknown ROI: {roi_name}. Available: {list(coord_dict.keys())}")
    return NiftiSpheresMasker(
        seeds=[coord_dict[roi_name]],
        radius=radius_mm,
        standardize=True,
        detrend=True,
    )


def define_tpj_from_tom_localizer(
    subject_id,
    data_dir,
    smoothing_fwhm=6.0,
    threshold=3.0,
    radius_mm=10,
    search_radius_mm=20,
):
    """Define subject-specific TPJ ROI from task-tom belief>photo contrast.

    Uses the ToM localizer (belief vs photo trials) to find subject-specific
    TPJ peak, then creates a sphere masker around it.

    Args:
        subject_id: BIDS subject ID (e.g., 'sub-04').
        data_dir: Path to BIDS dataset root containing NIfTI files.
        smoothing_fwhm: Spatial smoothing in mm.
        threshold: z-score threshold for peak detection.
        radius_mm: Sphere radius around peak.
        search_radius_mm: Maximum distance from standard rTPJ to accept peak.

    Returns:
        NiftiSpheresMasker centered on subject-specific rTPJ peak,
        or None if no significant peak found.
    """
    from nilearn.glm.first_level import FirstLevelModel
    from nilearn.maskers import NiftiSpheresMasker

    nifti_paths = load_gaesser_fmri(subject_id, task="tom", data_dir=data_dir)
    events_list = load_gaesser_events(subject_id, task="tom", data_dir=data_dir)

    if not nifti_paths:
        return None

    # Run first-level GLM on ToM localizer
    fmri_model = FirstLevelModel(
        t_r=2.5,
        smoothing_fwhm=smoothing_fwhm,
        high_pass=0.01,
        hrf_model="spm",
    )
    fmri_model.fit(nifti_paths, events_list)

    # Compute belief > photo contrast
    contrast = fmri_model.compute_contrast("belief - photo", output_type="z_score")

    # Find peak near standard rTPJ
    from nilearn.image import coord_transform
    standard_rtpj = np.array(ROI_COORDS_MNI["rTPJ"])

    # Extract data and find peak
    contrast_data = contrast.get_fdata()
    affine = contrast.affine

    # Find voxels above threshold
    above_thresh = np.argwhere(contrast_data > threshold)
    if len(above_thresh) == 0:
        return None

    # Convert to MNI coordinates and find closest to standard rTPJ
    best_dist = float("inf")
    best_mni = None
    for voxel in above_thresh:
        mni = affine[:3, :3] @ voxel + affine[:3, 3]
        dist = np.linalg.norm(mni - standard_rtpj)
        if dist < best_dist and dist < search_radius_mm:
            best_dist = dist
            best_mni = tuple(mni.astype(float))

    if best_mni is None:
        return None

    return NiftiSpheresMasker(
        seeds=[best_mni],
        radius=radius_mm,
        standardize=True,
        detrend=True,
    )


# ---------------------------------------------------------------------------
# Section B: Model-to-brain readout functions
# ---------------------------------------------------------------------------

def _compute_identity_precision_proxy(agent, identity_state, affect_state,
                                      distance_state):
    """TPJ BOLD proxy (INVERSE mentalizing demand).

    Uses the effective precision of the A_affect matrix at the given
    identity/distance state. High identity → high affect precision →
    low mentalizing demand → LOWER TPJ BOLD.

    This is consistent with Zhao et al. (2024): UIV > IV in TPJ, because
    anonymous victims produce noisier affect representations, requiring
    more top-down mentalizing effort.

    Returns:
        Float: mentalizing demand proxy (higher = more TPJ activation).
    """
    A_aff = agent.A[OBS_AFFECT]
    # Effective precision: diagonal of A_affect at this identity/distance
    # Average over outcome states (which don't modulate affect precision)
    diag_prec = np.mean([
        A_aff[affect_state, identity_state, affect_state, distance_state, o]
        for o in range(NUM_STATES[OUTCOME])
    ])
    # INVERSE: high precision → low TPJ (less mentalizing demand)
    return float(1.0 - diag_prec)


def _compute_affect_update_magnitude(agent, identity_state, affect_state,
                                     distance_state):
    """Insula BOLD proxy: strength of affective signal.

    Uses the effective precision of the A_affect matrix as a measure of
    how strongly the affect signal is transmitted. Identified victims
    produce higher-precision affect representations via identity-affect
    coupling → stronger insula activation.

    Returns:
        Float: affect signal strength (higher = more insula, IV > UIV).
    """
    A_aff = agent.A[OBS_AFFECT]
    # Effective precision of affect observation at this configuration
    diag_prec = np.mean([
        A_aff[affect_state, identity_state, affect_state, distance_state, o]
        for o in range(NUM_STATES[OUTCOME])
    ])
    # Higher precision = stronger affect signal = more insula
    return float(diag_prec)


def _compute_distance_encoding(identity_state, distance_state):
    """mPFC BOLD proxy: self-referential / narrative engagement.

    Identified + proximal victims engage mPFC for episodic simulation.
    The proxy combines proximity (low distance) with identification level.

    Returns:
        Float: mPFC engagement proxy (higher = more mPFC, IV > UIV).
    """
    # Proximity: 0=proximal → 1.0, 2=abstract → 0.0
    proximity = 1.0 - distance_state / 2.0
    # Identity contribution
    identity = identity_state / 2.0
    # Combined: identified + proximal → highest mPFC
    return float(0.5 * proximity + 0.5 * identity)


def _compute_policy_precision_proxy(agent):
    """dACC/Striatal BOLD proxy: policy precision.

    High certainty about the correct action (low entropy of q_pi) →
    strong striatal signal. High conflict → strong dACC signal.

    Returns:
        Float: negative entropy of policy posterior (higher = more certain).
    """
    q_pi = agent.q_pi.copy()
    q_pi = np.clip(q_pi, 1e-10, 1.0)
    entropy = -np.sum(q_pi * np.log(q_pi))
    max_entropy = np.log(len(q_pi))
    # Normalized: 0 = maximum uncertainty, 1 = full certainty
    return float(1.0 - entropy / max(max_entropy, 1e-10))


def _compute_coupling_strength(identity_affect_coupling, identity_state,
                                distance_state, distance_affect_attenuation):
    """TPJ-Insula FC proxy: effective coupling strength.

    Models trial-by-trial modulation of TPJ-Insula functional connectivity.

    Returns:
        Float: effective coupling value.
    """
    id_boost = identity_state / 2.0
    dist_damp = distance_state / 2.0
    return identity_affect_coupling * (1.0 + id_boost) - distance_affect_attenuation * dist_damp


def extract_neural_regressors(trial_configs, model_params=None):
    """Extract model-derived neural regressors for a sequence of trials.

    For each trial configuration, runs the factorized model and extracts:
      - tpj_proxy: inverse identity precision (higher = more TPJ, UIV>IV)
      - insula_proxy: affect update magnitude (higher = more insula, IV>UIV)
      - mpfc_proxy: distance encoding (higher = more mPFC)
      - striatal_proxy: policy precision (higher = more striatal)
      - tpj_insula_fc: coupling strength (higher = stronger FC, IV>UIV)

    Args:
        trial_configs: List of dicts with keys
            {identity_state, affect_state, distance_state}.
        model_params: Override params for build_network_agent.

    Returns:
        DataFrame with one row per trial, columns for each regressor.
    """
    if model_params is None:
        model_params = {}

    rows = []
    coupling = model_params.get("identity_affect_coupling",
                                 NETWORK_DEFAULTS["identity_affect_coupling"])
    dist_atten = model_params.get("distance_affect_attenuation",
                                   NETWORK_DEFAULTS["distance_affect_attenuation"])

    for config in trial_configs:
        id_state = config["identity_state"]
        aff_state = config["affect_state"]
        dist_state = config["distance_state"]

        agent = build_network_agent(
            identity_state=id_state,
            affect_state=aff_state,
            distance_state=dist_state,
            **model_params,
        )
        agent.reset()

        observation = [id_state, aff_state, dist_state, 0, 0]
        agent.infer_states(observation)
        agent.infer_policies()

        rows.append({
            "identity_state": id_state,
            "affect_state": aff_state,
            "distance_state": dist_state,
            "tpj_proxy": _compute_identity_precision_proxy(
                agent, id_state, aff_state, dist_state),
            "insula_proxy": _compute_affect_update_magnitude(
                agent, id_state, aff_state, dist_state),
            "mpfc_proxy": _compute_distance_encoding(id_state, dist_state),
            "striatal_proxy": _compute_policy_precision_proxy(agent),
            "tpj_insula_fc": _compute_coupling_strength(
                coupling, id_state, dist_state, dist_atten),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Section C: fMRI data loading
# ---------------------------------------------------------------------------

# Events condition mapping for task-ieh
IEH_CONDITION_MAP = {
    "Imagine": "episodic",     # Episodic simulation → identified
    "Estimate": "control",     # Estimation control → statistical
    "Journal": "control",      # Journal control → statistical variant
    "Memory": "nuisance",      # Memory retrieval → nuisance regressor
    "difficultyTR": "nuisance",
    "storyTR": "nuisance",
}

# Mapping from IEH conditions to model states
IEH_MODEL_STATES = {
    "episodic": {"identity_state": 2, "affect_state": 2, "distance_state": 0},
    "control": {"identity_state": 0, "affect_state": 0, "distance_state": 1},
}


def _find_data_dir(data_dir=None):
    """Resolve data directory, checking both nifti and standard locations."""
    if data_dir is not None:
        return Path(data_dir)

    base = Path(__file__).parent.parent.parent / "data" / "gaesser"
    # Prefer NIfTI directory
    nifti_dir = base / "openneuro_nifti"
    if nifti_dir.exists():
        return nifti_dir
    return base / "openneuro"


def load_gaesser_fmri(subject_id, task="ieh", data_dir=None, runs=None):
    """Load NIfTI file paths for a Gaesser subject.

    Args:
        subject_id: BIDS subject ID (e.g., 'sub-04').
        task: 'ieh' (main task) or 'tom' (localizer).
        data_dir: Root of BIDS dataset.
        runs: Which runs to load. None = all.

    Returns:
        List of NIfTI file paths (strings). Empty if files not found.
    """
    data_dir = _find_data_dir(data_dir)
    func_dir = data_dir / subject_id / "func"
    if not func_dir.exists():
        return []

    max_runs = 8 if task == "ieh" else 2
    run_nums = runs or list(range(1, max_runs + 1))

    paths = []
    for run in run_nums:
        # Try common NIfTI extensions
        for ext in [".nii.gz", ".nii"]:
            p = func_dir / f"{subject_id}_task-{task}_run-{run:02d}_bold{ext}"
            if p.exists():
                paths.append(str(p))
                break
    return paths


def load_gaesser_events(subject_id, task="ieh", data_dir=None, runs=None):
    """Load events TSV files for a Gaesser subject.

    For task-ieh, adds a 'condition' column mapping trial types to
    episodic/control/nuisance.

    Args:
        subject_id: BIDS subject ID.
        task: 'ieh' or 'tom'.
        data_dir: Root of BIDS dataset.
        runs: Which runs to load. None = all.

    Returns:
        List of DataFrames (one per run).
    """
    data_dir = _find_data_dir(data_dir)
    func_dir = data_dir / subject_id / "func"
    if not func_dir.exists():
        return []

    max_runs = 8 if task == "ieh" else 2
    run_nums = runs or list(range(1, max_runs + 1))

    events = []
    for run in run_nums:
        p = func_dir / f"{subject_id}_task-{task}_run-{run:02d}_events.tsv"
        if p.exists():
            df = pd.read_csv(p, sep="\t")
            # Clean duration column (sometimes has brackets like [5])
            if "duration" in df.columns:
                df["duration"] = df["duration"].astype(str).str.strip("[]").astype(float)
            # Add condition mapping for ieh task
            if task == "ieh" and "trial_type" in df.columns:
                df["condition"] = df["trial_type"].map(IEH_CONDITION_MAP)
            events.append(df)

    return events


def load_gaesser_behavioral(experiment=1, data_dir=None):
    """Load Gaesser behavioral data from OSF Excel files.

    Args:
        experiment: 1 (fMRI) or 2 (TMS).
        data_dir: Path to osf_behavioral directory.

    Returns:
        DataFrame with behavioral data, or None if file not found.
    """
    if data_dir is None:
        data_dir = Path(__file__).parent.parent.parent / "data" / "gaesser" / "osf_behavioral"
    else:
        data_dir = Path(data_dir)

    if experiment == 1:
        fname = "GaesserWassermanHornYoung_Experiment1fMRI_Data.xlsx"
    else:
        fname = "GaesserWassermanHornYoung_Experiment2TMS_Data.xlsx"

    fpath = data_dir / fname
    if not fpath.exists():
        return None

    return pd.read_excel(fpath, header=1)


# ---------------------------------------------------------------------------
# Section D: GLM and correlation analysis
# ---------------------------------------------------------------------------

def run_first_level_glm(
    subject_id,
    task="ieh",
    data_dir=None,
    smoothing_fwhm=6.0,
    hrf_model="spm",
    high_pass=0.01,
    tr=2.5,
):
    """Run a first-level GLM on Gaesser fMRI data.

    For task-ieh: conditions = {episodic, control, nuisance}
    For task-tom: conditions = {belief, photo}

    Args:
        subject_id: BIDS subject ID.
        task: 'ieh' or 'tom'.
        data_dir: BIDS dataset root.
        smoothing_fwhm: Spatial smoothing in mm.
        hrf_model: HRF model for nilearn.
        high_pass: High-pass filter cutoff in Hz.
        tr: Repetition time in seconds.

    Returns:
        Fitted FirstLevelModel, or None if data not found.
    """
    from nilearn.glm.first_level import FirstLevelModel

    nifti_paths = load_gaesser_fmri(subject_id, task=task, data_dir=data_dir)
    events_list = load_gaesser_events(subject_id, task=task, data_dir=data_dir)

    if not nifti_paths or not events_list:
        return None

    # For task-ieh, use condition column as trial_type for GLM
    if task == "ieh":
        for df in events_list:
            if "condition" in df.columns:
                # Filter out nuisance for cleaner design matrix
                df_clean = df[df["condition"] != "nuisance"].copy()
                df_clean["trial_type"] = df_clean["condition"]
                events_list[events_list.index(df)] = df_clean[["onset", "duration", "trial_type"]]

    model = FirstLevelModel(
        t_r=tr,
        smoothing_fwhm=smoothing_fwhm,
        high_pass=high_pass,
        hrf_model=hrf_model,
    )
    model.fit(nifti_paths, events_list)
    return model


def extract_roi_timecourse(nifti_path, masker, confounds=None):
    """Extract mean BOLD timecourse from an ROI.

    Args:
        nifti_path: Path to NIfTI file.
        masker: NiftiSpheresMasker (fitted or unfitted).
        confounds: Optional confounds DataFrame.

    Returns:
        1D numpy array of BOLD signal values (one per TR).
    """
    signal = masker.fit_transform(nifti_path, confounds=confounds)
    return signal.ravel()


def correlate_model_regressors_with_roi(
    model_regressors,
    roi_timecourses,
    events,
    tr=2.5,
):
    """Correlate model-derived neural regressors with ROI timecourses.

    Aligns model regressors (one per trial) to fMRI timepoints using
    event onsets and computes Pearson correlations.

    Args:
        model_regressors: DataFrame from extract_neural_regressors().
        roi_timecourses: Dict mapping ROI name -> 1D BOLD array.
        events: Events DataFrame with 'onset' and 'condition' columns.
        tr: Repetition time in seconds.

    Returns:
        DataFrame with columns: roi, regressor, r, p_value.
    """
    from scipy.stats import pearsonr

    # Get trial onsets (only episodic and control trials)
    trial_events = events[events["condition"].isin(["episodic", "control"])].copy()
    trial_trs = (trial_events["onset"].values / tr).astype(int)

    results = []
    regressor_cols = [c for c in model_regressors.columns
                      if c not in ("identity_state", "affect_state", "distance_state")]

    for roi_name, bold in roi_timecourses.items():
        # Extract BOLD at trial onset TRs (with HRF delay ~5s = 2 TRs)
        hrf_delay_trs = 2
        trial_bold = []
        for t in trial_trs:
            idx = min(t + hrf_delay_trs, len(bold) - 1)
            trial_bold.append(bold[idx])
        trial_bold = np.array(trial_bold)

        if len(trial_bold) != len(model_regressors):
            continue

        for reg_col in regressor_cols:
            reg_vals = model_regressors[reg_col].values
            if np.std(reg_vals) < 1e-10 or np.std(trial_bold) < 1e-10:
                continue
            r, p = pearsonr(reg_vals, trial_bold)
            results.append({
                "roi": roi_name, "regressor": reg_col, "r": r, "p_value": p,
            })

    return pd.DataFrame(results)
