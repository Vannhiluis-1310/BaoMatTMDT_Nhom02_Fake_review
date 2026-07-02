# Phase 6 Plan Check

## Verdict

`VERIFICATION PASSED`

## Coverage

| Requirement | Covered By | Notes |
|-------------|------------|-------|
| ROB-01 | Task 3 | FGSM and PGD perturbations in PCA feature space using Phase 4 DL model as differentiable surrogate. |
| ROB-02 | Task 3 | Clean vs adversarial metrics table for DL and final ensemble on seeded subset. |
| XAI-01 | Task 4 | SHAP global importance on LightGBM/tree model plus PCA component loading map. |
| XAI-02 | Task 5 | LIME tabular explanations for representative fake/real cases. |

## Project Rules Check

- No local notebook execution is planned.
- No local EDA/training/tuning/dataset processing is planned.
- Executable ML source remains inside `notebooks/06_Adversarial_XAI.ipynb`.
- Existing artifacts are consumed; Phase 2-5 are not retrained.
- RAM 12GB is handled through fixed subset sizes.

## Risk Review

| Risk | Handling |
|------|----------|
| Final ensemble is a weighted blend, not a calibrated stacker | Plan includes metadata-driven prediction helper and fallback recording. |
| SHAP over final ensemble may be slow | Plan uses efficient LightGBM TreeExplainer and labels this as tree/global evidence. |
| PCA explanations are abstract | Plan includes PCA component loading map back to raw feature groups. |
| Adversarial examples on tree ensemble are not directly differentiable | Plan uses DL surrogate attacks and transfer evaluation on the ensemble. |

## Notes For Executor

Build one complete notebook rather than splitting Phase 6 into multiple notebooks or helper `.py` files. Keep subset sizes configurable at the top of the notebook.
