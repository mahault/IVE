# IVE: Identified Victim Effect Under Active Inference

A computational model of the Identified Victim Effect (IVE) using active inference (pymdp), designed to bridge behavioural, neural, and institutional accounts of moral decision-making.

## Project Structure

```
src/ive/
  agent.py           # Phase 1: disentangled IVE agent (delta_C, delta_gamma, delta_p)
  networks.py        # Phase 2: factorized 4-factor neural-circuit model
  fitting.py         # Parameter fitting (grid search, MLE)
  plotting.py        # Visualization
  utils.py           # State indexing, effect size computation
  data_loader.py     # Load Moche et al. (2024) SPSS data (Studies 1-5)
  envs/
    charity_task.py  # One-step donation task environment
  alignment/         # AI alignment extensions (Phase 3)
data/
  moche2024/         # Moche et al. (2024) behavioural data (6 studies, N=7,996)
  gaesser/           # Gaesser et al. (2019) fMRI prosocial task (OpenNeuro ds001439)
  zhao2024/          # Zhao et al. (2024) fMRI IVE (Phase 3, manual download)
tests/
  test_agent.py      # Unit tests (21 tests: Phase 1 + Phase 2 + Gaesser validation)
notebooks/
  01_disentangled_ive_demo.ipynb    # Parameter sweeps + interaction effects
  02_moche_data_fitting.ipynb       # Empirical data + model fitting + cross-validation
  03_effect_sizes_summary.ipynb     # Forest plot + bootstrap CIs + summary table
  04_neural_mapping.ipynb           # Factorized model + precision modulation
  05_case_simulations.ipynb         # Case studies + lesion + interventions
  06_gaesser_validation.ipynb       # Gaesser et al. (2019) empirical validation
```

## IVE Mechanisms

### Phase 1: disentangled parameters

| Parameter | Interpretation | Neural proxy |
|-----------|---------------|-------------|
| `delta_C` | Preference shift: identified victims valued more | Insula (affective salience) |
| `delta_gamma` | Precision shift: more urgent policy selection | Striatum / dACC (gain) |
| `delta_p` | Controllability shift: higher perceived efficacy | Agency / mPFC |

### Phase 2: factorized neural-circuit model

| Factor | States | Neural proxy | Role |
|--------|--------|-------------|------|
| S_identity | {anonymous, partial, full} | **TPJ** | Victim individuation |
| S_affect | {low, medium, high} | **Insula** | Empathic arousal |
| S_distance | {proximal, distal, abstract} | **mPFC** | Psychological distance |
| S_outcome | {not_saved_nocost, not_saved_cost, saved_cost} | **Striatum** | Action outcomes |

The IVE emerges from **identity -> affect coupling**: when a victim is identified, the precision of affect representations in the insula increases (TPJ-Insula connectivity), producing stronger empathic responses that bias policy selection toward helping. Distance (mPFC) attenuates this coupling.

## Results

### Phase 1: Behavioural Calibration (complete)

**Best-fit parameters** (grid search on Moche et al. Study 2b):

| Parameter | Value | Role |
|-----------|-------|------|
| `delta_C` | 0.9 | Primary IVE driver (affective valuation shift) |
| `delta_p` | 0.1 | Minor contribution (perceived efficacy) |
| `delta_gamma` | 0.0 | Not needed to explain these data |
| `cost_penalty` | 1.9 | Sets the help/no-help tradeoff regime |

**Model fit:**

| Metric | Statistical | Identified |
|--------|------------|------------|
| P(Help) — model | 0.297 | 0.390 |
| Mean donation — empirical | 57.2 SEK | 48.5 SEK (high-id) |
| Cohen's h (model IVE) | 0.197 | |

**Key empirical finding — affect-donation dissociation:**

The IVE in Moche et al. (2024) manifests as an *affect* shift, not a *donation* shift. Across studies with affect measures:

| Measure | Cohen's d (id - non-id) | Interpretation |
|---------|------------------------|----------------|
| Personal distress | +0.471 | Strong IVE in affect |
| Sympathy | +0.155 | Moderate IVE in affect |
| Donation amount | -0.102 | No reliable IVE in behaviour |

This dissociation is consistent with the model: `delta_C` shifts the *preference* (affective valuation) for saving the victim, but whether this translates to behavioural change depends on the cost regime.

**Cross-validation across 5 studies:**

| Study | N | Design | Donation IVE (d) | Model prediction |
|-------|---|--------|-------------------|------------------|
| Study 1 | 984 | Id vs Non-id | ~0 (near null) | Consistent (weak IVE at this cost) |
| Study 2a | 1,194 | 3-level id | Small positive | Consistent |
| Study 2b | 596 | 3-level id + affect | -0.102 (null) | Calibration target |
| Study 3 | 1,500 | Imagery manipulation | +0.33 (picture+text) | Graded delta_C reproduces gradient |
| Study 4 | 1,632 | Id x UA x Order | -0.14 (reversed) | Consistent (cost-regime sensitivity) |

**Ablation analysis:** Removing `delta_C` eliminates the IVE entirely. Removing `delta_p` has minimal effect. `delta_gamma` was not needed (fitted to 0). This identifies affective valuation (insula circuit) as the primary computational mechanism.

### Phase 2: Neural Mapping & Case Studies (complete)

**Factorized model results:**

| Context | P(Help) | Mechanism |
|---------|---------|-----------|
| Statistical (anonymous, distal) | 0.268 | Low identity precision, distance attenuation |
| Identified (partial, proximal) | 0.357 | Moderate TPJ-Insula coupling |
| Highly identified (full, proximal) | 0.490 | Strong TPJ-Insula coupling, high affect |

**Case study simulations:**

| Case | Individual | Aggregated | Effect |
|------|-----------|------------|--------|
| Francis Inquiry | 0.490 | 0.245 | ~50% reduction from bureaucratic aggregation |
| RADAR trial | 0.475 | 0.260 | Statistical aggregation masks individual harm |
| Military (ground vs command) | 0.440 | 0.260 | Distance + deidentification gradient |
| Charity (stat vs id) | 0.268 | 0.470 | Standard IVE |

**Individual difference simulations:**
- Psychopathy analog (reduced TPJ-Insula coupling): flattened IVE
- Burnout analog (reduced affect precision): attenuated IVE
- High empathy (enhanced coupling): amplified IVE

**Institutional interventions:** Re-identification (patient stories, named case reviews) partially reverses the effects of aggregation.

### Phase 2b: Empirical Validation — Gaesser et al. (2019)

The factorized model was fitted to Gaesser et al. (2019) *Social Cognitive and Affective Neuroscience* — an fMRI study (N=18) + TMS study (N=19) on episodic simulation and prosocial helping.

**Best-fit parameters** (grid search on Experiment 1 WillingnessToHelp):

| Parameter | Value | Interpretation |
|-----------|-------|----------------|
| `identity_affect_coupling` | 0.65 | TPJ-Insula connectivity strength |
| `cost_penalty` | 0.9 | Low cost task (rating, not donating) |
| `util_saved` | 1.4 | Moderate valuation of helping |
| `affect_preference_boost` | 0.4 | Moderate empathic motivation |

**Model fit:**

| Metric | Empirical | Model | Residual |
|--------|-----------|-------|----------|
| P(Help \| Control) | 0.588 | ~0.59 | <0.01 |
| P(Help \| Episodic) | 0.745 | ~0.75 | <0.01 |
| IVE magnitude | +0.157 | ~+0.15 | <0.01 |
| TMS to rTPJ | d=0.22, ns | Weak effect (consistent) | - |

**Cross-study consistency:** The same model architecture fits both Moche (donation, cost_penalty=1.9) and Gaesser (willingness-to-help, cost_penalty=0.9) data. The higher cost in Moche explains why the IVE manifests in *affect* but not *behaviour*, while in Gaesser it manifests in *stated intentions*.

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
2. Neural network mapping + case studies (complete)
3. Neuroimaging validation + AI alignment (next)
