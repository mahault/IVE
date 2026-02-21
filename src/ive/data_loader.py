"""
Load and preprocess Moche et al. (2024) data for model fitting.

Data: https://osf.io/ukqs8
Paper: PLOS ONE 19(3): e0300863

Key variables:
    Identifiability: 1=Non-identified, 2=Identified, 3=Highly identified
    DV_20Children: donation amount (SEK)
    Sympathy_AE: sympathy rating (0-6 scale, Batson's adjectives)
    PersonalDistress_AE: personal distress rating (0-6 scale)
    EmpathicConcern_TK: empathic concern (1-7 scale, Toi & Batson)
"""

import os
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "moche2024")


def _load_sav(filename):
    """Load an SPSS .sav file, returning (DataFrame, metadata)."""
    import pyreadstat
    path = os.path.join(DATA_DIR, filename)
    return pyreadstat.read_sav(path)


def load_study1():
    """Study 1: 2x2 (Identifiability x Unit Asking), N=997.

    Returns filtered DataFrame (within 3SD) with columns:
        identifiability: 'non_id' or 'id'
        unit_asking: 'control' or 'ua'
        donation: float (SEK)
    """
    df, _ = _load_sav("Data_set_study_1.sav")
    df = df[df["Filter_3SD"] == 1.0].copy()

    df["identifiability"] = df["Identifiability"].map({1.0: "non_id", 2.0: "id"})
    df["unit_asking"] = df["Control_vs_UA"].map({1.0: "control", 2.0: "ua"})
    df["donation"] = df["DV_20Children"]

    return df[["identifiability", "unit_asking", "donation"]].dropna()


def load_study2b():
    """Study 2b: 3-level identifiability x 2-level UA + affect measures, N=596.

    Returns filtered DataFrame with columns:
        identifiability: 'non_id', 'id', or 'high_id'
        unit_asking: 'control' or 'ua'
        donation: float (SEK)
        sympathy: float (0-6)
        distress: float (0-6)
        empathic_concern: float (1-7)
    """
    df, _ = _load_sav("Data_set_study_2b_affective.sav")
    df = df[df["Filter_DV_3SD"] == 0.0].copy()

    df["identifiability"] = df["Identifiability"].map(
        {1.0: "non_id", 2.0: "id", 3.0: "high_id"}
    )
    df["unit_asking"] = df["Control_vs_UA"].map({1.0: "control", 2.0: "ua"})
    df["donation"] = df["DV_20Children"]
    df["sympathy"] = df["Sympathy_AE"]
    df["distress"] = df["PersonalDistress_AE"]
    df["empathic_concern"] = df["EmpathicConcern_TK"]

    cols = ["identifiability", "unit_asking", "donation", "sympathy", "distress", "empathic_concern"]
    return df[cols].dropna(subset=["identifiability", "donation"])


def load_study3():
    """Study 3: IVE and mental imagery, N=1500.

    Returns filtered DataFrame with columns:
        identifiability: 'no_ive', 'picture', or 'picture_text'
        emotion_order: 'emo_first' or 'don_first'
        donation: float (SEK)
        sympathy: float (0-6)
        distress: float (0-6)
    """
    df, _ = _load_sav("Dataset_study_3_IVE_mental_imagery.sav")
    df = df[df["Donation3SD"] == 0.0].copy() if "Donation3SD" in df.columns else df.copy()

    df["identifiability"] = df["Manipulation"].map(
        {1.0: "no_ive", 2.0: "picture", 3.0: "picture_text"}
    )
    df["emotion_order"] = df["Order"].map({1.0: "emo_first", 2.0: "don_first"})
    df["donation"] = df["Donation"]
    df["sympathy"] = df["Sympathy_AE"]
    df["distress"] = df["Personal_Distress_AE"]

    cols = ["identifiability", "emotion_order", "donation", "sympathy", "distress"]
    return df[cols].dropna(subset=["identifiability", "donation"])


def get_calibration_targets():
    """Extract key calibration targets from the data.

    Returns a dict with empirical means and effect sizes that the model
    should reproduce.
    """
    df2b = load_study2b()

    targets = {}

    for id_level in ["non_id", "id", "high_id"]:
        sub = df2b[df2b["identifiability"] == id_level]
        targets[f"donation_mean_{id_level}"] = sub["donation"].mean()
        targets[f"donation_sd_{id_level}"] = sub["donation"].std()
        targets[f"sympathy_mean_{id_level}"] = sub["sympathy"].mean()
        targets[f"distress_mean_{id_level}"] = sub["distress"].mean()
        targets[f"n_{id_level}"] = len(sub)

    # Effect sizes (Highly-ID vs Non-ID)
    non = df2b[df2b["identifiability"] == "non_id"]
    high = df2b[df2b["identifiability"] == "high_id"]

    for measure, col in [("sympathy", "sympathy"), ("distress", "distress"), ("donation", "donation")]:
        g1, g2 = non[col].dropna().values, high[col].dropna().values
        n1, n2 = len(g1), len(g2)
        pooled = np.sqrt(((n1 - 1) * g1.std() ** 2 + (n2 - 1) * g2.std() ** 2) / (n1 + n2 - 2))
        d = (g2.mean() - g1.mean()) / pooled if pooled > 0 else 0.0
        targets[f"cohens_d_{measure}"] = d

    return targets
