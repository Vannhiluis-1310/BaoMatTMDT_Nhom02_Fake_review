---
phase: 10
plan: 01
status: implemented_static
completed: 2026-06-09
requirements-completed:
  - REV-ENS-01
  - REV-ENS-02
  - REV-ENS-03
  - REV-ENS-04
key-files:
  created:
    - notebooks/05_00_Phase5_Run_Order.ipynb
    - notebooks/05_01_LightGBM_Raw.ipynb
    - notebooks/05_02_XGBoost_Raw.ipynb
    - notebooks/05_03_MLP_Raw.ipynb
    - notebooks/05_04_CNN_BiLSTM_Sequence.ipynb
    - notebooks/05_05_Weighted_Blending.ipynb
    - notebooks/05_06_Stacking_Calibration.ipynb
  modified:
    - notebooks/03_PCA_Feature_Selection.ipynb
    - notebooks/04_PSO_Model_Training.ipynb
    - notebooks/05_Hybrid_Ensemble.ipynb
    - notebooks/07_Evaluation_Ablation.ipynb
    - notebooks/08_Final_Report_Kaggle.ipynb
    - .planning/NOTEBOOK_PLAN.md
    - .planning/ROADMAP.md
    - .planning/TASKLIST.md
    - .planning/REQUIREMENTS.md
    - .planning/PROJECT_STRUCTURE.md
    - .planning/STATE.md
---

# Phase 10 Plan 01: Pipeline Redesign and Ensemble Model Zoo Summary

Implemented the notebook-source redesign for the revised ModernBERT raw-feature + sequence-track pipeline.

## What Changed

- Created a Phase 5 notebook family so each major model has its own `.ipynb`.
- Rewrote `05_Hybrid_Ensemble.ipynb` into an orchestrator/summary notebook that reads saved probabilities instead of training every model directly.
- Added `05_05_Weighted_Blending.ipynb` for pair/triple/full weighted blend sweeps.
- Added `05_06_Stacking_Calibration.ipynb` for stackers and bounded calibration.
- Reframed `04_PSO_Model_Training.ipynb` as legacy PSO/PCA ablation evidence, not the revised final DL path.
- Reframed `03_PCA_Feature_Selection.ipynb` as PCA/SVD diagnostic rather than the default final model input.
- Updated Phase 7 and Phase 8 opening narratives and run order to reference the revised Phase 5 family.
- Updated planning docs to reflect the Phase 5 notebook family and the two-track pipeline.

## Phase 5 Notebook Family

| Notebook | Purpose |
|----------|---------|
| `05_00_Phase5_Run_Order.ipynb` | Validate Phase 5 inputs and document execution order. |
| `05_01_LightGBM_Raw.ipynb` | Train/evaluate LightGBM on raw 777 features. |
| `05_02_XGBoost_Raw.ipynb` | Train/evaluate XGBoost on raw 777 features. |
| `05_03_MLP_Raw.ipynb` | Train/evaluate MLP on raw 777 features. |
| `05_04_CNN_BiLSTM_Sequence.ipynb` | Train/evaluate CNN-BiLSTM-Attention on token sequences with behavioral late fusion. |
| `05_05_Weighted_Blending.ipynb` | Run weighted blend sweeps across available model probabilities. |
| `05_06_Stacking_Calibration.ipynb` | Run stacking meta-models and calibration. |
| `05_Hybrid_Ensemble.ipynb` | Aggregate candidates and select balanced/precision-first winners. |

## Verification Performed

- Parsed all new/modified notebooks as JSON.
- Confirmed each new Phase 5 notebook exists and has notebook format version 4.
- Confirmed `05_Hybrid_Ensemble.ipynb` is summary-only and no longer contains direct base-model training classes/imports.
- Confirmed no temporary notebook-writer script remains.
- Confirmed planning docs contain the new Phase 5 notebook family.

## Colab Runtime (completed 2026-06-09/10)

Final track outputs synced. SSOT: `reports/tables/phase7_final_metrics.csv`.

| Mode | τ | Test Macro F1 | Prec. Fake | ROC-AUC |
|------|---|---------------|------------|---------|
| Default | 0.50 | 0.9433 | 0.9699 | 0.9769 |
| Balanced | 0.30 | 0.9463 | 0.9344 | 0.9769 |
| Precision-first | 0.60 | 0.9126 | 0.9816 | 0.9769 |

## Not Run Locally

- Agent did not execute notebook cells locally (Colab-first policy).

## Next Step (optional)

Re-run only if pipeline changes. Otherwise:

1. `05_00_Phase5_Run_Order.ipynb`
2. `05_01_LightGBM_Raw.ipynb`
3. `05_02_XGBoost_Raw.ipynb`
4. `05_03_MLP_Raw.ipynb`
5. `05_04_CNN_BiLSTM_Sequence.ipynb`
6. `05_05_Weighted_Blending.ipynb`
7. `05_06_Stacking_Calibration.ipynb`
8. `05_Hybrid_Ensemble.ipynb`

Then sync artifacts and run `$gsd-verify-work phase 10` or verify Phase 5 outputs directly.
