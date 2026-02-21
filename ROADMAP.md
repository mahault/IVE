# IVE Model Roadmap: From Toy Model to Neurally-Grounded Computational Neuroethics

**Project**: Identified Victim Effect (IVE) under Active Inference
**Collaborator**: David Foreman (david.foreman1@gmail.com)
**Lead**: Dr. Mahault Albarracin
**Created**: 2026-02-21
**Status**: Phase 2 complete + validated (Tasks 2.1-2.6 done, Gaesser validation done). Ready for Phase 3.

---

## Current State of the Codebase

The repo contains two working scripts:

| File | What it does |
|------|-------------|
| `ive_pymdp.py` | Minimal 4-state toy model. Single hidden factor (context x outcome). IVE = higher `p_success_id` (0.9) vs `p_success_stat` (0.3). Deterministic observation mapping. |
| `ive_pymdp_experiments.py` | Extended 8-state model adding a **cost** dimension. Monte Carlo experiments, parameter sweeps over `(p_success_stat, cost_penalty)`, grid-search tuning to match target help rates (40% stat / 80% id). Generates CSV logs + heatmap/bar/curve plots. |
| `environment.yml` | Conda env: python >= 3.8, numpy, inferactively-pymdp |

**Phase 1 results** (2026-02-21):
- Refactored into modular `src/ive/` package with disentangled IVE mechanisms
- Downloaded Moche et al. (2024) data from OSF: 6 SPSS files, 5 studies, N=7,996
- Downloaded Gaesser et al. (2019) data from OpenNeuro: 366 files (events + behavioral)
- Grid search fit: P(Help|stat)=0.297, P(Help|id)=0.390, Cohen's h=0.197
- Best-fit parameters: delta_C=0.9 (primary driver), delta_p=0.1, cost_penalty=1.9
- Key empirical finding: IVE appears in affect (distress d=0.471) not donations (d=-0.102)
- Fixed critical pymdp issue: must use `action_selection="stochastic"` (default is deterministic argmax)
- Cross-validated across Studies 1-5: model correctly predicts weak/variable donation IVE
- Ablation analysis: delta_C alone accounts for >90% of the model IVE
- Forest plot of effect sizes across all studies with 95% CIs
- Data loaders for all 6 studies (Studies 1, 2a, 2b, 3, 4, 5)
- 3 analysis notebooks: model demo, Moche data fitting, effect size summary
- 11 unit tests all passing

**Phase 2 results** (2026-02-21):
- Factorized 4-factor generative model: identity (TPJ), affect (Insula), distance (mPFC), outcome (Striatum)
- 5 observation modalities including separate cost channel
- IVE emerges from identity->affect precision modulation (TPJ-Insula coupling)
- Aggregation operator: bureaucratic, statistical, and military types
- Case study simulations: Francis Inquiry, RADAR trial, military abstraction, charity
- Individual difference profiles: psychopathy (reduced coupling), burnout (reduced precision)
- Institutional interventions: re-identification partially reverses aggregation effects
- 18 unit tests all passing
- 2 analysis notebooks: neural mapping demo, case study simulations

**Phase 2b: Gaesser validation** (2026-02-21):
- Fitted factorized model to Gaesser et al. (2019) Experiment 1 (fMRI, N=18)
- Target: P(Help|control)=0.588, P(Help|episodic)=0.745 (normalized WillingnessToHelp)
- Best-fit: coupling=0.65, cost_penalty=0.9, util_saved=1.4, affect_boost=0.4
- Model fit: stat=~0.59, id=~0.75, error<0.001 (near-perfect)
- TMS prediction: rTPJ disruption produces weak IVE change, consistent with empirical null (d=0.22, p=0.21)
- Cross-study: same architecture fits Moche (cost=1.9, donation) and Gaesser (cost=0.9, intention)
- Cost regime explains affect-behaviour dissociation: high cost→IVE in affect only; low cost→IVE in behaviour
- 21 unit tests all passing
- Notebook 06: Gaesser validation, grid search, TMS simulation, cross-study comparison

**Original toy model** (ive_pymdp.py, ive_pymdp_experiments.py) preserved for reference.

---

## What David Is Asking For

1. **Map to speculative neural networks** (TPJ, mPFC, insula, retrosplenial cortex)
2. **Apply to case studies** (Francis inquiry, RADAR trial, bureaucratic aggregation, military abstraction)
3. **Constrain by neurophysiology** (parameter ranges consistent with imaging literature, Cohen's d ~ 0.17)
4. **Optionally validate with neuroimaging** (predict fMRI contrasts, possibly collaborate with Drossi)
5. **Explore AI alignment angle** (does embedding IVE-like weighting reduce repugnant LLM outputs?)

---

## Three-Phase Execution Plan

---

### PHASE 1: Behavioural Model Refinement + Calibration to Open Data

**Goal**: Turn the toy model into a publishable, reproducible behavioural simulator calibrated against real IVE experiments.

**Estimated scope**: ~2-3 weeks of focused coding

#### Task 1.1: Refactor into modular architecture

**Current problem**: Agent and environment are tangled together in monolithic scripts.

**Target structure**:
```
src/
  ive/
    __init__.py
    agent.py            # build_agent() with parameterized IVE mechanism
    environment.py      # env.step(action, state) -> next_state, obs
    envs/
      __init__.py
      charity_task.py   # One-step donation/help task (maps to Moche et al.)
      aggregation.py    # Bureaucratic aggregation scenarios (Phase 2)
    fitting.py          # MLE / Bayesian parameter estimation
    plotting.py         # All visualization code
    utils.py            # State indexing, effect size computation
notebooks/
  01_reproduce_toy_model.ipynb
  02_fit_moche_data.ipynb
  03_effect_sizes.ipynb
tests/
  test_agent.py
  test_environment.py
```

#### Task 1.2: Disentangle IVE mechanisms

**Current**: IVE = `p_success_id > p_success_stat` (conflates valuation, precision, controllability).

**Target**: Three separable IVE parameters:

| Parameter | Interpretation | Active Inference mapping |
|-----------|---------------|------------------------|
| `delta_C` | Identified increases utility of "saved" outcome | Preference shift: `C_saved_id = C_saved_stat + delta_C` |
| `delta_gamma` | Identified increases policy precision / urgency | Policy precision: `gamma_id = gamma_stat + delta_gamma` |
| `delta_p` | Identified increases perceived controllability | Transition: `p_success_id = p_success_stat + delta_p` (optional, only if task framing justifies it) |

This disentanglement is critical because it allows Phase 2 to map each parameter to a distinct neural circuit.

#### Task 1.3: Implement fitting pipeline

- Download Moche et al. (2024) data from OSF (see Datasets section below)
- Implement likelihood function: P(observed_help_rate | params) using the agent's policy distribution
- Fit using either:
  - **MLE**: scipy.optimize.minimize on negative log-likelihood
  - **Bayesian**: emcee or PyMC to get posterior distributions over (delta_C, delta_gamma, delta_p)
- Compute model-predicted effect sizes (Cohen's d, odds ratios) and compare to empirical

#### Task 1.4: Reproduce empirical IVE patterns

Using Moche et al. (2024) 5-study data:
- Fit model to Study 1 (identifiability manipulation)
- Cross-validate on Studies 2-5 (singularity, unit asking, emotion rating)
- Show that `delta_C` and `delta_gamma` capture the core IVE
- Verify Cohen's d within empirical range (literature: d ~ 0.15-0.40)

#### Task 1.5: Generate Phase 1 deliverables

- [x] Notebook 01: Disentangled IVE demo (parameter sweeps, interaction effects)
- [x] Notebook 02: Moche data analysis, model fitting, cross-validation
- [x] Notebook 03: Effect size forest plot, bootstrap CIs, summary table
- [ ] CSV of fitted parameters + effect sizes (auto-generated by notebooks)
- [x] Bar plots: model vs empirical help rates per condition
- [ ] Write-up section: "Behavioural calibration of the IVE active inference model"

---

### PHASE 2: Neural Network Mapping + Case Study Simulations

**Goal**: Upgrade from "one hidden factor" to a factorized, mechanistic model where IVE emerges from interacting neural-circuit-like modules. Simulate the qualitative case studies from the manuscript.

**Estimated scope**: ~4-6 weeks

#### Task 2.1: Factorized generative model (neural mapping)

Replace the single 8-state factor with **3-4 interacting factors**:

| Factor | States | Neural proxy | Role |
|--------|--------|-------------|------|
| `S_identity` | {anonymous, partially_identified, fully_identified} | **TPJ** (mentalizing, individuation) | Encodes how identified/individuated the victim is |
| `S_affect` | {low, medium, high} | **Insula** (salience, interoception, empathic distress) | Encodes affective arousal / salience |
| `S_distance` | {proximal, distal, abstract} | **mPFC** (self-other distance, abstract reasoning) | Encodes psychological/institutional distance |
| `S_outcome` | {not_saved, saved} | **Striatum / vmPFC** (valuation) | Outcome of helping action |

**IVE mechanism as precision modulation**:

```
# Precision on identity -> affect coupling (A matrix)
# When identity = fully_identified, affect precision is HIGH
#   -> strong affective response -> biases toward Help
# When identity = anonymous, affect precision is LOW
#   -> weak affective response -> Help depends more on abstract reasoning

# Policy precision gamma is modulated:
gamma(context) = gamma_base + alpha * identity_salience(S_identity)

# Aggregation reduces identity_salience:
identity_salience = f(S_identity) / n_victims  # dilution under aggregation
```

#### Task 2.2: Implement aggregation operator

David's manuscript emphasizes that **aggregation destroys identifiability** (bureaucratic pooling, Simpson's paradox, institutional abstraction).

Implement aggregation as:
- **Marginalization** over `S_identity`: when institution pools victims, the agent loses token-level identity information
- **Reduced likelihood precision**: `A_identity` becomes noisier under aggregation (can't distinguish individuals)
- **Forced distance increase**: aggregation shifts `S_distance` toward `abstract`

```python
def apply_aggregation(agent, n_victims, aggregation_type="bureaucratic"):
    """
    Modify agent's generative model to simulate institutional aggregation.

    aggregation_type:
      - "bureaucratic": pool identities, increase distance
      - "statistical": replace individuals with summary statistics
      - "military": maximize distance, minimize identity precision
    """
```

#### Task 2.3: Build case study environments

Each case from the manuscript becomes a parameterized environment:

| Case | Key manipulation | Expected result |
|------|-----------------|----------------|
| **Francis Inquiry** (Mid Staffs) | Bureaucratic aggregation of patient complaints. Staff see statistics, not individuals. | Model predicts: aggregation reduces identity salience -> reduced Help probability -> systemic neglect emerges |
| **RADAR trial** | Medication trial aggregates patient outcomes. Individual adverse effects invisible in summary statistics. | Model predicts: statistical aggregation can mask harm to identifiable subgroups (Simpson's paradox analog) |
| **Military abstraction** | Drone operators, chain of command distance. Target becomes abstract. | Model predicts: maximal `S_distance` + reduced `S_identity` precision -> Help (or restraint) probability drops dramatically |
| **Charity/humanitarian** | NGO messaging: identified child vs "millions affected" | Model predicts: standard IVE curve; identified > statistical; singularity effect |

#### Task 2.4: Simulate and validate against qualitative predictions

For each case:
1. Set environment parameters (aggregation level, distance, identity precision)
2. Run Monte Carlo simulations
3. Compare model predictions to qualitative descriptions in manuscript
4. Show that the model **recovers** the moral distortions described
5. Compute sensitivity: which parameter has the largest effect?

#### Task 2.5: Lesion / individual-difference simulations

- **Psychopathy analog**: reduce empathic gain (lower `S_affect` precision or coupling)
- **Burnout analog**: reduce identity sensitivity over time (adaptation in `S_identity` precision)
- **Institutional design**: test whether "forcing identification" (e.g., patient stories in hospital dashboards) reverses aggregation effects

#### Task 2.6: Generate Phase 2 deliverables

- [x] `src/ive/networks.py`: network-mapped parameterization with factor graph
- [x] Aggregation operator in `networks.py`: bureaucratic, statistical, military
- [x] Case study presets in `networks.py`: Francis, RADAR, military, charity, psychopathy, burnout
- [x] `notebooks/04_neural_mapping.ipynb`: factor structure + precision modulation
- [x] `notebooks/05_case_simulations.ipynb`: all 4 case studies + lesion + interventions
- [ ] Write-up section: "Neural mapping and institutional moral failure simulations"

---

### PHASE 3: Neuroimaging Validation / Prediction + AI Alignment Extension

**Goal**: Produce testable neuroimaging predictions; optionally explore whether IVE-like weighting reduces morally repugnant AI outputs.

**Estimated scope**: ~6-8 weeks (can overlap with Phase 2)

#### Task 3.1: Define model-to-brain readouts

Map computational variables to fMRI-observable proxies:

| Model variable | Neural proxy | fMRI prediction |
|---------------|-------------|----------------|
| `S_identity` precision (gain) | TPJ activation | Higher BOLD in TPJ for identified vs statistical |
| `S_affect` update magnitude | Insula activation | Larger insula response for identified victims |
| `S_distance` state | mPFC activation | mPFC tracks psychological distance; higher for abstract/aggregated |
| Policy precision `gamma` | Striatal / dACC activation | Higher precision -> more decisive action -> stronger striatal signal |
| `S_identity` x `S_affect` coupling | TPJ-insula functional connectivity | Stronger coupling for identified conditions |

#### Task 3.2: Fit to Zhao et al. (2024) fMRI data

**Dataset**: Zhao et al. (2024) "Neural mechanisms of identifiable victim effect in prosocial decision-making" (see Datasets section)
- N=31, fMRI, identifiable vs unidentifiable victims, Money + Effort tasks
- Data on Science Data Bank: https://www.scidb.cn/s/YjY32q
- Regions: TPJ, mPFC, insula, MCC

**Procedure**:
1. Download behavioral data (donation amounts per trial, condition labels)
2. Fit factorized model (Phase 2) to trial-by-trial donation choices
3. Extract model-derived regressors (identity precision, affect update, policy precision)
4. If raw fMRI available: correlate model regressors with BOLD time series in ROIs
5. If only summary data: compare model-predicted regional activation patterns with reported contrasts

#### Task 3.3: Validate with Gaesser prosocial fMRI data

**Status: Behavioral validation complete (Phase 2b). fMRI validation pending.**

**Dataset**: OpenNeuro ds001439 (Gaesser et al.) - willingness to help task with ToM localizer

**Completed (behavioral)**:
- [x] Fitted factorized model to Experiment 1 WillingnessToHelp ratings
- [x] Grid search: coupling=0.65, cost=0.9, util=1.4, affect_boost=0.4
- [x] TMS simulation: model correctly predicts weak rTPJ disruption effect
- [x] Cross-study comparison with Moche data (different cost regimes)
- [x] Notebook 06: full Gaesser validation analysis

**Remaining (fMRI)**:
- [ ] Correlate model-predicted mentalizing demand with TPJ BOLD
- [ ] Use ToM localizer for ROI definition
- [ ] Test whether identity_affect_coupling maps to TPJ-Insula functional connectivity

#### Task 3.4: Generate testable predictions for future experiments

Even without fitting all imaging data, produce:
- **Prediction 1**: Identified > Statistical should produce TPJ > mPFC activation ratio
- **Prediction 2**: Aggregation should reduce TPJ activation and increase mPFC (abstract processing)
- **Prediction 3**: Individual differences in `delta_gamma` (precision shift) should correlate with trait empathy scores
- **Prediction 4**: Psychopathy-analog (reduced affect precision) should produce flat IVE (no help rate difference)

These predictions are testable by Drossi or any collaborator with an fMRI setup.

#### Task 3.5: AI alignment extension (David's "evil LLMs" suggestion)

Implement a proof-of-concept:
1. Define an "IVE-weighted utility function" that upweights identified individuals
2. Compare outputs of:
   - Standard utilitarian aggregation (maximizes total welfare -> repugnant conclusion)
   - IVE-weighted aggregation (identified individuals get non-substitutable weight)
3. Test on Parfit-style scenarios:
   - Repugnant conclusion (huge population with barely-worth-living lives vs small happy population)
   - Trolley problems with identified vs statistical victims
   - Resource allocation with identifiable vs abstract beneficiaries
4. Show that IVE-weighting avoids certain repugnant conclusions while introducing its own biases

This is a separate module, not dependent on the neural model:
```
src/ive/
  alignment/
    __init__.py
    ive_utility.py        # IVE-weighted utility function
    parfit_scenarios.py   # Repugnant conclusion, trolley, allocation
    llm_probe.py          # Optional: test if LLM outputs change with IVE prompting
```

#### Task 3.6: Generate Phase 3 deliverables

- [ ] `src/ive/neuroimaging.py`: model-to-brain readout definitions
- [ ] `notebooks/07_fmri_predictions.ipynb`: predicted activation patterns
- [ ] `notebooks/08_zhao_fit.ipynb`: fit to Zhao et al. fMRI behavioral data
- [ ] `notebooks/09_alignment.ipynb`: IVE utility vs standard utilitarian
- [ ] Write-up sections: "Neuroimaging predictions" + "AI alignment implications"

---

## Datasets

### Dataset 1: Moche et al. (2024) - Behavioural IVE [PHASE 1]

| Field | Value |
|-------|-------|
| **Citation** | Moche H, Karlsson H, Vastfjall D (2024). Victim identifiability, number of victims, and unit asking in charitable giving. PLOS ONE 19(3): e0300863 |
| **DOI** | https://doi.org/10.1371/journal.pone.0300863 |
| **Data** | https://osf.io/ukqs8 |
| **Pre-registration** | https://osf.io/v6947 |
| **N** | 7,996 across 5 studies |
| **Conditions** | Identifiable vs unidentifiable victim; single vs group; unit asking; emotion rating order |
| **Measures** | Donation amount, willingness to donate, emotion ratings (sympathy, distress) |
| **Open?** | Yes - all data, pre-registrations, supplementary files on OSF |
| **Use** | Calibrate `delta_C`, `delta_gamma`, `delta_p`; compute model effect sizes; cross-validate across 5 studies |

### Dataset 2: Zhao et al. (2024) - fMRI IVE [PHASE 3]

| Field | Value |
|-------|-------|
| **Citation** | Zhao H et al. (2024). The neural mechanisms of identifiable victim effect in prosocial decision-making. Human Brain Mapping 45(2): e26609 |
| **DOI** | https://doi.org/10.1002/hbm.26609 |
| **Data** | https://www.scidb.cn/s/YjY32q (Science Data Bank) |
| **N** | 31 (15 female, mean age 20.26) |
| **Conditions** | Identifiable victim (IV) vs Unidentifiable victim (UIV) |
| **Tasks** | Money task (donate monetary units) + Effort task (hand-grip for donations) |
| **Imaging** | fMRI: TPJ, mPFC, insula, MCC contrasts |
| **Open?** | Yes - data on Science Data Bank |
| **Use** | Fit factorized model to trial-by-trial donation choices; correlate model regressors with BOLD in TPJ/mPFC/insula; validate neural predictions |

### Dataset 3: Gaesser et al. - Prosocial fMRI [PHASE 3, secondary]

| Field | Value |
|-------|-------|
| **Citation** | Gaesser B et al. (2019). A role for the medial temporal lobe subsystem in guiding prosociality. Social Cognitive and Affective Neuroscience 14(4): 397-410 |
| **Data** | OpenNeuro ds001439 (https://openneuro.org/datasets/ds001439) |
| **N** | ~18 (after exclusions) |
| **Task** | Read stories about people in need, rate willingness to help; episodic simulation vs control |
| **Imaging** | fMRI with ToM localizer (TPJ definition), hippocampal/MTL engagement |
| **Open?** | Yes - BIDS format on OpenNeuro |
| **Use** | Secondary validation: does model-predicted mentalizing demand correlate with TPJ BOLD? ToM localizer useful for ROI definition |

---

## Dependency Graph

```
Phase 1                    Phase 2                     Phase 3
--------                   --------                    --------
1.1 Refactor          -->  2.1 Factor model       -->  3.1 Brain readouts
1.2 Disentangle IVE   -->  2.2 Aggregation op     -->  3.2 Zhao fMRI fit
1.3 Fitting pipeline  -->  2.3 Case envs          -->  3.3 Gaesser validation
1.4 Fit Moche data    -->  2.4 Simulate cases     -->  3.4 Predictions
1.5 Deliverables      -->  2.5 Lesion sims             3.5 AI alignment
                           2.6 Deliverables        -->  3.6 Deliverables
```

Phase 3.5 (AI alignment) can start independently after Phase 1 is done.

---

## Technical Requirements

### Dependencies to add

```yaml
# Updated environment.yml
name: ive-pymdp
channels:
  - conda-forge
  - defaults
dependencies:
  - python>=3.9
  - numpy
  - scipy
  - matplotlib
  - seaborn
  - pandas
  - jupyter
  - pytest
  - pip
  - pip:
    - inferactively-pymdp
    - emcee          # MCMC sampling for Bayesian fitting
    - arviz          # Bayesian visualization
    - nibabel        # NIfTI file handling (Phase 3)
    - nilearn        # Neuroimaging analysis (Phase 3)
```

### Immediate next steps (start here)

1. **Create `src/ive/` package structure** (Task 1.1)
2. **Download Moche et al. data** from https://osf.io/ukqs8
3. **Implement disentangled IVE** with `delta_C`, `delta_gamma`, `delta_p` (Task 1.2)
4. **Write fitting pipeline** targeting Moche Study 1 (Task 1.3)

---

## Questions to Clarify with David

Before going deep into Phase 2, ask David:

1. **What are the "speculative networks" precisely?** Does he have a specific wiring diagram (TPJ -> insula -> mPFC) or is this to be inferred from the manuscript?
2. **Parameter constraints**: Does he want specific neurophysiological bounds (e.g., gain modulation ranges from EEG/MEG literature)?
3. **Success criteria**: What outcome would count as "the model works" for the case studies? Qualitative match or quantitative fit?
4. **Scope for Phase 3**: Does he want to involve Drossi now, or produce predictions first?
5. **AI alignment angle**: Is this a brief exploration or a full section of the paper?

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Moche data doesn't have trial-level data (only summary stats) | Medium | High | Use summary statistics for moment-matching instead of trial-level MLE |
| pymdp doesn't support factorized models well | Low | Medium | Can implement custom factor graph using raw numpy |
| Zhao fMRI data not actually downloadable | Medium | Medium | Fall back to published summary statistics and contrast maps |
| Phase 2 model too complex to fit | Medium | High | Start with 2 factors (identity + outcome), add affect/distance incrementally |
| David's scope expectations exceed feasible work | High | Medium | Clarify scope questions above before committing to all phases |

---

## Publication Strategy

- **Phase 1 alone** could be a short computational paper or supplementary to the existing manuscript
- **Phase 1 + 2** is a full computational neuroethics paper
- **Phase 1 + 2 + 3** is a major paper + potentially a second paper on AI alignment
- Target journals: Neuroscience of Consciousness, Computational Brain & Behavior, Frontiers in Computational Neuroscience, or PNAS (if Phase 3 is strong)
