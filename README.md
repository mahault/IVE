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
notebooks/           # Analysis notebooks
```

## IVE Mechanisms

The model decomposes the IVE into three separable parameters:

| Parameter | Interpretation | Neural proxy |
|-----------|---------------|-------------|
| `delta_C` | Preference shift: identified victims valued more | Insula (affective salience) |
| `delta_gamma` | Precision shift: more urgent policy selection | Striatum / dACC (gain) |
| `delta_p` | Controllability shift: higher perceived efficacy | Agency / mPFC |

## Current Results

Fitted to Moche et al. (2024) Study 2b:
- P(Help\|statistical) = 0.297, P(Help\|identified) = 0.390
- Cohen's h = 0.197 (within empirical IVE range)
- Primary driver: `delta_C = 0.9` (affective valuation shift)
- Key finding: IVE manifests as affect shift (sympathy d=0.155, distress d=0.471), not donation shift (d=-0.102)

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
1. Behavioural model + calibration (current)
2. Neural network mapping + case studies
3. Neuroimaging validation + AI alignment
