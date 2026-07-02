---
status: complete
phase: 05-hybrid-stacking-ensemble
source:
  - .planning/phases/05-hybrid-stacking-ensemble/05-VERIFICATION.md
started: 2026-06-01T10:15:00+07:00
updated: 2026-06-01T10:20:00+07:00
---

## Current Test

[testing complete]

## Tests

### 1. Ensemble Artifacts
expected: XGBoost, LightGBM, stacking metadata, final ensemble model, calibrator/selector and final probabilities are present.
result: pass

### 2. Candidate Sweep
expected: Phase 5 evaluates multiple candidate stackers/blends and records candidate selection.
result: pass

### 3. Validation-Only Selection
expected: Candidate and threshold are selected on validation/calibration data, not test labels.
result: pass

### 4. Final Target Metrics
expected: Final ensemble should approach Macro F1 >= 0.89, Precision Fake >= 0.975 and ROC-AUC >= 0.93.
result: issue
reported: "Final test metrics remain below target: default Macro F1 0.852779, optimized Precision Fake 0.965413, ROC-AUC 0.913779."
severity: major

## Summary

total: 4
passed: 3
issues: 1
pending: 0
skipped: 0

## Gaps

- truth: "Final ensemble should approach Macro F1 >= 0.89, Precision Fake >= 0.975 and ROC-AUC >= 0.93."
  status: failed
  reason: "Candidate sweep selected a validation-positive blend, but test metrics did not reach the project targets."
  severity: major
  test: 4
  artifacts:
    - reports/tables/phase5_final_metrics.csv
    - reports/tables/phase5_candidate_selection.csv
    - artifacts/ensemble/phase5_metadata.json
  missing:
    - Stronger base features/model signal or revised objective expectations/evidence framing.
