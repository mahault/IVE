"""Tests for the neuroimaging module."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ive.neuroimaging import (
    ROI_COORDS_MNI,
    FACTOR_ROI_MAP,
    extract_neural_regressors,
    IEH_CONDITION_MAP,
    IEH_MODEL_STATES,
)


class TestROIDefinitions:
    def test_all_rois_have_3d_coords(self):
        for name, coord in ROI_COORDS_MNI.items():
            assert len(coord) == 3, f"ROI {name} should have 3D coordinates"

    def test_key_rois_present(self):
        required = ["rTPJ", "lTPJ", "mPFC", "rAnteriorInsula", "lAnteriorInsula",
                     "dACC", "ventralStriatum"]
        for roi in required:
            assert roi in ROI_COORDS_MNI, f"Missing ROI: {roi}"

    def test_tpj_coords_match_zhao(self):
        """rTPJ and lTPJ should match Zhao et al. (2024) UIV>IV peaks."""
        assert ROI_COORDS_MNI["rTPJ"] == (52, -46, 40)
        assert ROI_COORDS_MNI["lTPJ"] == (-52, -44, 44)

    def test_factor_roi_map_keys(self):
        expected = ["identity_precision", "affect_update",
                    "distance_encoding", "policy_precision", "coupling_strength"]
        for key in expected:
            assert key in FACTOR_ROI_MAP


class TestNeuralRegressors:
    def test_basic_extraction(self):
        configs = [
            {"identity_state": 0, "affect_state": 1, "distance_state": 1},
            {"identity_state": 2, "affect_state": 1, "distance_state": 1},
        ]
        df = extract_neural_regressors(configs)
        assert len(df) == 2
        assert "tpj_proxy" in df.columns
        assert "insula_proxy" in df.columns
        assert "mpfc_proxy" in df.columns
        assert "striatal_proxy" in df.columns
        assert "tpj_insula_fc" in df.columns

    def test_tpj_direction_uiv_gt_iv(self):
        """TPJ proxy should be HIGHER for anonymous (UIV > IV)."""
        configs = [
            {"identity_state": 0, "affect_state": 1, "distance_state": 1},
            {"identity_state": 2, "affect_state": 1, "distance_state": 1},
        ]
        df = extract_neural_regressors(configs)
        anon_tpj = df[df["identity_state"] == 0]["tpj_proxy"].values[0]
        id_tpj = df[df["identity_state"] == 2]["tpj_proxy"].values[0]
        assert anon_tpj > id_tpj, f"TPJ should be UIV>IV: anon={anon_tpj}, id={id_tpj}"

    def test_insula_direction_iv_gt_uiv(self):
        """Insula proxy should be HIGHER for identified (IV > UIV)."""
        configs = [
            {"identity_state": 0, "affect_state": 1, "distance_state": 1},
            {"identity_state": 2, "affect_state": 1, "distance_state": 1},
        ]
        df = extract_neural_regressors(configs)
        anon_ins = df[df["identity_state"] == 0]["insula_proxy"].values[0]
        id_ins = df[df["identity_state"] == 2]["insula_proxy"].values[0]
        assert id_ins > anon_ins, f"Insula should be IV>UIV: id={id_ins}, anon={anon_ins}"

    def test_mpfc_direction_iv_gt_uiv(self):
        """mPFC proxy should be HIGHER for identified (IV > UIV)."""
        configs = [
            {"identity_state": 0, "affect_state": 1, "distance_state": 1},
            {"identity_state": 2, "affect_state": 1, "distance_state": 1},
        ]
        df = extract_neural_regressors(configs)
        anon_mpfc = df[df["identity_state"] == 0]["mpfc_proxy"].values[0]
        id_mpfc = df[df["identity_state"] == 2]["mpfc_proxy"].values[0]
        assert id_mpfc > anon_mpfc

    def test_fc_direction_iv_gt_uiv(self):
        """TPJ-Insula FC should be stronger for identified."""
        configs = [
            {"identity_state": 0, "affect_state": 1, "distance_state": 1},
            {"identity_state": 2, "affect_state": 1, "distance_state": 1},
        ]
        df = extract_neural_regressors(configs)
        anon_fc = df[df["identity_state"] == 0]["tpj_insula_fc"].values[0]
        id_fc = df[df["identity_state"] == 2]["tpj_insula_fc"].values[0]
        assert id_fc > anon_fc

    def test_coupling_zero_eliminates_affect_difference(self):
        """With coupling=0, insula proxy should not differ by identity."""
        configs = [
            {"identity_state": 0, "affect_state": 1, "distance_state": 0},
            {"identity_state": 2, "affect_state": 1, "distance_state": 0},
        ]
        df = extract_neural_regressors(configs, {"identity_affect_coupling": 0.0})
        anon_ins = df[df["identity_state"] == 0]["insula_proxy"].values[0]
        id_ins = df[df["identity_state"] == 2]["insula_proxy"].values[0]
        # With zero coupling, both should have same base precision
        assert abs(id_ins - anon_ins) < 0.01

    def test_full_grid_27_configs(self):
        """Full 3x3x3 grid should produce 27 rows."""
        configs = [
            {"identity_state": i, "affect_state": a, "distance_state": d}
            for i in range(3) for a in range(3) for d in range(3)
        ]
        df = extract_neural_regressors(configs)
        assert len(df) == 27
        assert df["tpj_proxy"].min() >= 0
        assert df["insula_proxy"].min() >= 0


class TestConditionMapping:
    def test_ieh_conditions(self):
        assert IEH_CONDITION_MAP["Imagine"] == "episodic"
        assert IEH_CONDITION_MAP["Estimate"] == "control"
        assert IEH_CONDITION_MAP["Journal"] == "control"

    def test_model_states_episodic(self):
        states = IEH_MODEL_STATES["episodic"]
        assert states["identity_state"] == 2  # identified
        assert states["distance_state"] == 0  # proximal

    def test_model_states_control(self):
        states = IEH_MODEL_STATES["control"]
        assert states["identity_state"] == 0  # anonymous
        assert states["distance_state"] == 1  # distal


@pytest.mark.nifti
class TestNiftiDependentFunctions:
    """Tests that require NIfTI data (skipped without data)."""

    def test_roi_masker_creation(self):
        """Test that ROI maskers can be created (requires nilearn)."""
        try:
            from ive.neuroimaging import get_roi_masker
            masker = get_roi_masker("rTPJ")
            assert masker is not None
        except ImportError:
            pytest.skip("nilearn not installed")
