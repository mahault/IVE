# IVE: Identified Victim Effect Under Active Inference

A computational model of the Identified Victim Effect (IVE) using active inference (pymdp), designed to bridge behavioural, neural, and institutional accounts of moral decision-making.

## Project Structure

```
src/ive/
  agent.py           # Disentangled IVE agent (delta_C, delta_gamma, delta_p)
  environment.py     # Base environment
  fitting.py         # Parameter fitting (grid search, MLE)
  plotting.py        # Visualization
  utils.py           # State indexing, effect size computation
  data_loader.py     # Load Moche et al. (2024) SPSS data
  envs/
    charity_task.py  # One-step donation task environment
    aggregation.py   # Aggregation environments (Phase 2)
  alignment/         # AI alignment extensions (Phase 3)
data/
  moche2024/         # Moche et al. (2024) behavioural data (6 studies, N=7,996)
  gaesser/           # Gaesser et al. (2019) fMRI prosocial task (OpenNeuro ds001439)
  zhao2024/          # Zhao et al. (2024) fMRI IVE (Phase 3, manual download)
tests/
  test_agent.py      # Unit tests (11 tests)
notebooks/
  01_disentangled_ive_demo.ipynb    # Parameter sweeps + interaction effects
  02_moche_data_fitting.ipynb       # Empirical data + model fitting + cross-validation
  03_effect_sizes_summary.ipynb     # Forest plot + bootstrap CIs + summary table
```

## IVE Mechanisms

The model decomposes the IVE into three separable parameters:

| Parameter | Interpretation | Neural proxy |
|-----------|---------------|-------------|
| `delta_C` | Preference shift: identified victims valued more | Insula (affective salience) |
| `delta_gamma` | Precision shift: more urgent policy selection | Striatum / dACC (gain) |
| `delta_p` | Controllability shift: higher perceived efficacy | Agency / mPFC |

## Current Results (Phase 1 complete)

Fitted to Moche et al. (2024) Study 2b:
- P(Help\|statistical) = 0.297, P(Help\|identified) = 0.390
- Cohen's h = 0.197 (within empirical IVE range)
- Primary driver: `delta_C = 0.9` (affective valuation shift)
- Key finding: IVE manifests as affect shift (sympathy d=0.155, distress d=0.471), not donation shift (d=-0.102)

Cross-validated across Studies 1-5:
- Model correctly predicts weak/variable donation IVE across studies
- Ablation: delta_C alone accounts for >90% of the IVE
- Study 3 (mental imagery): graded delta_C reproduces graded identification manipulation
- Study 4 shows reversed donation IVE, consistent with model's cost-regime sensitivity

## Setup

```bash
conda env create -f environment.yml
conda activate ive-pymdp
pip install -e .
```

## Tests

```bash
pytest tests/ -v
```

## Roadmap

See [ROADMAP.md](ROADMAP.md) for the three-phase plan:
1. Behavioural model + calibration (complete)
2. Neural network mapping + case studies (next)
3. Neuroimaging validation + AI alignment
