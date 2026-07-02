---
phase: 9
plan: 01
status: implemented
completed: 2026-06-09
requirements-completed:
  - DIAG-01
  - DIAG-02
  - DIAG-03
  - DIAG-04
key-files:
  created:
    - notebooks/tests/01_DL_PCA_Diagnostic_Test.ipynb
    - artifacts/diagnostics/dl_pca_test/.gitkeep
    - reports/diagnostics/dl_pca_test/.gitkeep
    - reports/diagnostics/dl_pca_test/input_validation.csv
    - reports/diagnostics/dl_pca_test/baseline_snapshot.csv
    - reports/diagnostics/dl_pca_test/mlp_raw_vs_pca_metrics.csv
    - reports/diagnostics/dl_pca_test/architecture_mismatch_metrics.csv
    - reports/diagnostics/dl_pca_test/constrained_dl_ensemble_sweep.csv
    - reports/diagnostics/dl_pca_test/constrained_dl_ensemble_best.csv
    - reports/diagnostics/dl_pca_test/diagnostic_decision_table.csv
    - reports/diagnostics/dl_pca_test/diagnostic_summary.md
    - reports/diagnostics/dl_pca_test/diagnostic_summary_table.csv
    - reports/diagnostics/dl_pca_test/runtime_summary.csv
    - .planning/phases/09-dl-pca-diagnostic-test/09-VERIFICATION.md
---

# Phase 9 Plan 01: DL PCA Diagnostic Test Summary

Implemented a standalone Colab-only diagnostic notebook to test whether weak DL performance is driven by PCA information loss, architecture mismatch on static vectors, or both, without modifying the main Phase 1-8 notebooks or shared artifacts.

## What Changed

- Created `notebooks/tests/01_DL_PCA_Diagnostic_Test.ipynb` as a separate diagnostic notebook under `notebooks/tests/`.
- Added Colab guard, Drive mount logic, project-root fallback paths, and read-only inputs from Phase 2/3/5/7 artifacts.
- Added input validation and baseline snapshot cells for the existing Phase 7 evidence.
- Added controlled `mlp_raw_777` vs `mlp_pca_400` comparison.
- Added `cnn_bilstm_pca_400_diagnostic` to isolate architecture mismatch on the same PCA input.
- Added constrained non-zero DL ensemble sweep with runtime-aware outputs.
- Added diagnostic interpretation and recommendation generation.
- Created isolated diagnostics folders under `artifacts/diagnostics/dl_pca_test/` and `reports/diagnostics/dl_pca_test/`.

## Created Outputs

- `notebooks/tests/01_DL_PCA_Diagnostic_Test.ipynb`
- `reports/diagnostics/dl_pca_test/input_validation.csv`
- `reports/diagnostics/dl_pca_test/baseline_snapshot.csv`
- `reports/diagnostics/dl_pca_test/mlp_raw_vs_pca_metrics.csv`
- `reports/diagnostics/dl_pca_test/architecture_mismatch_metrics.csv`
- `reports/diagnostics/dl_pca_test/constrained_dl_ensemble_sweep.csv`
- `reports/diagnostics/dl_pca_test/constrained_dl_ensemble_best.csv`
- `reports/diagnostics/dl_pca_test/diagnostic_decision_table.csv`
- `reports/diagnostics/dl_pca_test/diagnostic_summary.md`
- `reports/diagnostics/dl_pca_test/diagnostic_summary_table.csv`
- `reports/diagnostics/dl_pca_test/runtime_summary.csv`

## Colab Run Results

- Raw MLP default Macro F1: `0.893034`, Precision Fake: `0.921401`, ROC-AUC: `0.949882`.
- Raw MLP selected-threshold Precision Fake: `0.974830`, just below the original `0.975` target.
- PCA MLP default Macro F1: `0.868496`; raw-vs-PCA delta is `+0.024538`.
- CNN-BiLSTM on PCA default Macro F1: `0.774833`; MLP-PCA-vs-CNN delta is `+0.093663`.
- Best constrained non-zero DL blend: `mlp_raw_dl0.30_xgb0.00_lgbm0.70`.
- Best constrained blend default metrics: Macro F1 `0.879132`, Precision Fake `0.932907`, ROC-AUC `0.937180`.

## Verification Performed

- Parsed the notebook JSON successfully after fixing one malformed cell object.
- Parsed all notebook code cells with Python AST successfully.
- Synced the executed Colab notebook and all diagnostic CSV/MD outputs from the downloaded run.
- Verified `input_validation.csv` contains only `pass` statuses.
- Verified required diagnostic markers are present:
  - `dl_weight >= 0.10`
  - `mlp_raw_777`
  - `mlp_pca_400`
  - `cnn_bilstm_pca_400`
  - `diagnostic_summary.md`
- Verified Phase 1-8 notebook paths only appear as read-only context references, not write targets.

## Not Run

- No notebook cells were executed locally.
- No EDA, training, tuning, or dataset processing was run locally.
- Diagnostic CSV/MD outputs were produced in Colab by the user and synced into the workspace.

## Deviations from Plan

None. The notebook stays isolated from the main source notebooks and the shared Phase 1-8 artifacts.

## Next Step

Decide whether to redesign the DL branch or reframe the project around raw-feature MLP/LightGBM evidence before changing the final report/title.
