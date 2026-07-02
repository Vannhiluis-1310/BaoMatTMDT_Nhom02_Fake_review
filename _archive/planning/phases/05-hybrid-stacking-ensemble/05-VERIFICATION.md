# Phase 5 Verification: Hybrid Stacking Ensemble

Status: verified_with_metric_gap

## Artifact Checks

- `artifacts/ensemble/xgboost_model.pkl` exists.
- `artifacts/ensemble/lightgbm_model.pkl` exists.
- `artifacts/ensemble/stacking_meta_model.pkl` exists.
- `artifacts/ensemble/probability_calibrator.pkl` exists.
- `artifacts/ensemble/final_ensemble_model.pkl` exists.
- `artifacts/ensemble/phase5_metadata.json` exists.
- `artifacts/predictions/final_ensemble_train_prob.npy` exists with shape `(29923,)`.
- `artifacts/predictions/final_ensemble_val_prob.npy` exists with shape `(6413,)`.
- `artifacts/predictions/final_ensemble_test_prob.npy` exists with shape `(6413,)`.
- Phase 5 tables and figures were regenerated under `reports/tables/` and `reports/figures/`.

## Candidate Sweep Verification

`phase5_metadata.json` confirms:

- Stacker mode: `candidate_sweep`
- Candidate count: 71
- Stack validation ratio: 0.6
- Feature order: `dl_pso_prob`, `xgboost_prob`, `lightgbm_prob`
- Selected candidate: `blend_dl01_xgb00_lgbm09`
- Selected candidate config:
  - DL PSO probability weight: 0.1
  - XGBoost probability weight: 0.0
  - LightGBM probability weight: 0.9
- Selected threshold: 0.77
- Validation target met: true
- Selection rule: `validation_precision_target_then_macro_f1_roc_auc`

`phase5_candidate_selection.csv` exists and contains 71 candidate rows.

## Base Model Test Metrics

| Model | Macro F1 | Precision Fake | Recall Fake | ROC-AUC | PR-AUC |
|-------|----------|----------------|-------------|---------|--------|
| dl_pso_phase4 | 0.810489 | 0.836586 | 0.702363 | 0.867929 | 0.860599 |
| xgboost | 0.849100 | 0.925652 | 0.716463 | 0.907062 | 0.901153 |
| lightgbm | 0.852960 | 0.920192 | 0.729421 | 0.913226 | 0.907658 |

## Final Ensemble Test Metrics

| Threshold Strategy | Threshold | Macro F1 | Precision Fake | Recall Fake | ROC-AUC | PR-AUC |
|--------------------|-----------|----------|----------------|-------------|---------|--------|
| default_0.5 | 0.50 | 0.852779 | 0.920154 | 0.729040 | 0.913779 | 0.907551 |
| optimized_precision_fake | 0.77 | 0.808712 | 0.965413 | 0.606326 | 0.913779 | 0.907551 |

## Target Check

| Target | Result | Status |
|--------|--------|--------|
| Macro F1 >= 0.89 | 0.852779 default / 0.808712 optimized | Miss |
| Precision Fake >= 0.975 | 0.965413 optimized | Miss |
| ROC-AUC >= 0.93 | 0.913779 | Miss |

## Conclusion

Phase 5 candidate sweep executed correctly and selected the best validation candidate, but the test-set targets are still not met. Validation target transfer is weak: validation Precision Fake target was met, but test Precision Fake stayed at 0.965413.

## Not Run Locally

Per project policy, no notebook cells, EDA, training, tuning, or dataset processing were run locally by the agent.
