# Phase 5 Summary: Hybrid Stacking Ensemble

> **Legacy PCA sweep (2026-06-01).** Final track: `phase5_weighted_blend` — test F1 **0.9433–0.9463**, AUC **0.9769** (`phase7_final_metrics.csv`). Content below = historical candidate `blend_dl01_xgb00_lgbm09`.

## Completed

- XGBoost and LightGBM were trained on Phase 3 PCA features.
- Phase 4 DL probabilities were loaded and validated.
- Candidate sweep evaluated 71 stacker/blend variants.
- Selected final candidate: `blend_dl01_xgb00_lgbm09`.
- Final model artifacts, candidate selection tables, final probabilities and figures were regenerated.

## Selected Ensemble

- DL PSO probability weight: 0.1
- XGBoost probability weight: 0.0
- LightGBM probability weight: 0.9
- Selected threshold: 0.77
- Validation Precision Fake target: met

## Test Metrics

- Default threshold Macro F1: 0.852779 (legacy PCA candidate sweep — superseded by `weighted_blend` 0.9433, `phase7_final_metrics.csv`)
- Default threshold Precision Fake: 0.920154
- Default threshold ROC-AUC: 0.913779
- Optimized threshold Precision Fake: 0.965413
- Optimized threshold Recall Fake: 0.606326

## Verification Result

Phase 5 is artifact-complete and technically verified. It has a metric gap because all final project metric targets are still missed on the test set.
