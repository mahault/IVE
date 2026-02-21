# Data Summary

## Dataset 1: Moche et al. (2024) - Behavioural IVE
**Source**: https://osf.io/ukqs8
**Paper**: PLOS ONE 19(3): e0300863
**Status**: Downloaded (6 SPSS files)

### Files
| File | N | Key variables |
|------|---|--------------|
| Data_set_study_1.sav | 997 (984 valid) | Identifiability (Unid/Id), DV_20Children (SEK), Unit Asking |
| Data_set_study_2a.sav | 1209 (1194 valid) | 3 levels of ID (Non/Id/HighlyId), DV_20Children |
| Data_set_study_2b_affective.sav | 596 (581 valid) | 3 levels of ID + Sympathy + Distress ratings |
| Dataset_study_3_IVE_mental_imagery.sav | 1500 | IVE + imagery manipulation, emotion ratings |
| Dataset_study_4_UA_identifiability_order.sav | ~1500 | Unit asking x identifiability x emotion order |
| Data_set_study_5_UA_singularity.sav | 2000 | Singularity (1 vs 5 start), group size (20 vs 200) |

### Key empirical findings (from our analysis)
- **Study 1**: Unidentified Mean=72.6 SEK, Identified Mean=64.2 SEK, Cohen's d = -0.048
  - IVE does NOT appear in donation amounts
- **Study 2a**: Non-ID=79.0, ID=65.4, Highly-ID=97.5 SEK
  - No clear identifiability gradient in donations
- **Study 2b (with affect)**:
  - Sympathy: Non-ID=4.01, ID=3.96, Highly-ID=4.23 (increases with identifiability)
  - Distress: Non-ID=2.77, ID=3.00, Highly-ID=3.51 (increases with identifiability)
  - Donations: Non-ID=57.2, ID=67.4, Highly-ID=48.5 (no clear pattern)

### Implication for modelling
The IVE manifests primarily as an AFFECTIVE shift, not a donation shift.
This supports modelling IVE as:
- delta_C (preference/valuation shift) -> maps to affect ratings
- delta_gamma (precision shift) -> maps to decision urgency
- Donations depend on the interaction of affect, precision, AND cost

## Dataset 2: Zhao et al. (2024) - fMRI IVE
**Source**: https://www.scidb.cn/s/YjY32q (Science Data Bank)
**Status**: NOT YET DOWNLOADED (requires manual access to Chinese site)
**Phase**: 3

## Dataset 3: Gaesser et al. (2019) - Prosocial fMRI
**Source**: OpenNeuro ds001439 + OSF 9k4n7
**Status**: Downloaded (366 files)

### Downloaded content
- participants.tsv: 18 subjects (ages 18-35)
- Task events (BIDS format): 8 runs of "ieh" task + 2 runs of "tom" localizer per subject
- OSF behavioral: 2 Excel files (Experiment 1 fMRI + Experiment 2 TMS)
- NIfTI fMRI files NOT downloaded (use aws s3 sync for Phase 3)

### Tasks
- **task-ieh**: "Imagine, Estimate, Help" - prosocial willingness-to-help
- **task-tom**: Theory of Mind localizer (for TPJ ROI definition)
