---
status: complete
phase: 06-adversarial-robustness-and-xai
source:
  - 06-01-SUMMARY.md
started: 2026-06-01
updated: 2026-06-01
---

# Phase 6 UAT

## Current Test

[testing complete]

## Tests

### 1. Input And Artifact Validation
expected: Phase 6 writes validation tables proving Phase 3-5 inputs are present and consistent.
result: pass
evidence:
  - `reports/tables/phase6_input_validation.csv`
  - `reports/tables/phase6_artifact_validation.csv`
  - `reports/tables/phase6_predictor_smoke_test.csv`

### 2. FGSM/PGD Robustness Metrics
expected: Robustness table contains clean, FGSM and PGD rows for both `dl_pso` and `final_ensemble`, with seed/subset/epsilon context and core metrics.
result: pass
evidence:
  - `reports/tables/phase6_robustness_metrics.csv`
  - `reports/tables/phase6_robustness_metric_drops.csv`
  - `reports/tables/phase6_adversarial_config.csv`
  - `reports/figures/phase6_robustness_metric_drop.png`
notes: `phase6_robustness_metrics.csv` has 12 rows: clean plus FGSM eps 0.01/0.03/0.05 and PGD eps 0.01/0.03 for both models.

### 3. SHAP Global Explanations
expected: SHAP global evidence exists for LightGBM and ranks PCA components.
result: pass
evidence:
  - `artifacts/xai/shap_values_lightgbm.npy`
  - `reports/tables/phase6_shap_global_importance.csv`
  - `reports/figures/phase6_shap_top_components.png`
  - `reports/figures/phase6_shap_summary.png`
notes: SHAP array shape verified as `(500, 400)` and importance table has 400 PCA components.

### 4. PCA Component Interpretation
expected: Important PCA components are mapped back to raw feature groups where possible.
result: pass
evidence:
  - `reports/tables/phase6_pca_component_loading_map.csv`
notes: Loading map has 300 rows and includes `bert_embedding` and `behavioral` raw feature groups.

### 5. LIME Local Explanations
expected: LIME exports representative local fake/real explanations as HTML and CSV tables.
result: pass
evidence:
  - `reports/tables/phase6_lime_case_summary.csv`
  - `reports/tables/phase6_lime_feature_weights.csv`
  - `artifacts/xai/lime_case_*.html`
notes: Six LIME case summaries and six HTML explanation files verified.

### 6. Reproducibility Metadata
expected: Phase 6 metadata records seed, subset sizes, source artifact paths, output paths, fallbacks and limitations.
result: pass
evidence:
  - `artifacts/xai/phase6_metadata.json`
notes: Metadata records seed 42, robustness subset 1000, SHAP subset 500, LIME background 1000 and six LIME cases.

## Summary

total: 6
passed: 6
issues: 0
pending: 0
skipped: 0
blocked: 0

## Gaps

[none]
