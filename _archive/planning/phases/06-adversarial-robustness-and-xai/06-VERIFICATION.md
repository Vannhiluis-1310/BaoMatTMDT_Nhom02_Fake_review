---
phase: 6
status: passed
verified_at: 2026-06-01
requirements:
  ROB-01: passed
  ROB-02: passed
  XAI-01: passed
  XAI-02: passed
---

# Phase 6 Verification

## Verdict

Phase 6 passed artifact verification. The notebook was run on Colab and produced the expected robustness, SHAP, LIME and metadata outputs. No local notebook execution, training, tuning, EDA or dataset processing was performed during verification.

## Requirement Traceability

| Requirement | Status | Evidence |
|-------------|--------|----------|
| ROB-01 | passed | `phase6_adversarial_config.csv` records FGSM and PGD feature-space attacks using `dl_pso` surrogate. |
| ROB-02 | passed | `phase6_robustness_metrics.csv` contains clean vs FGSM/PGD metrics on seeded subset of 1000 test rows. |
| XAI-01 | passed | `shap_values_lightgbm.npy`, `phase6_shap_global_importance.csv`, SHAP figures and PCA loading map exist. |
| XAI-02 | passed | `phase6_lime_case_summary.csv`, `phase6_lime_feature_weights.csv` and six `lime_case_*.html` files exist. |

## Automated Checks

| Check | Result | Detail |
|-------|--------|--------|
| Required artifacts exist | pass | All Phase 6 CSV/JSON/NPY/PNG/HTML outputs are present and non-empty. |
| Robustness rows | pass | 12 rows: clean, FGSM eps 0.01/0.03/0.05 and PGD eps 0.01/0.03 for `dl_pso` and `final_ensemble`. |
| Robustness metrics | pass | Core columns include accuracy, Macro F1, Precision Fake, Recall Fake, F1 Fake, ROC-AUC, PR-AUC and confusion matrix values. |
| SHAP shape | pass | `shap_values_lightgbm.npy` shape is `(500, 400)` float32. |
| SHAP table | pass | `phase6_shap_global_importance.csv` has 400 PCA components with rank and mean absolute SHAP. |
| PCA loading map | pass | `phase6_pca_component_loading_map.csv` has 300 rows and maps to `bert_embedding` and `behavioral` groups. |
| LIME cases | pass | Six case summaries, six HTML files and 72 local feature-weight rows exist. |
| Metadata | pass | `phase6_metadata.json` records seed, subset sizes, output paths, source artifacts, limitations and fallback decisions. |

## Robustness Snapshot

| Condition | Model | Macro F1 | Precision Fake | Recall Fake | ROC-AUC |
|-----------|-------|----------|----------------|-------------|---------|
| clean | dl_pso | 0.7724 | 0.7644 | 0.6822 | 0.8481 |
| clean | final_ensemble | 0.8000 | 0.9603 | 0.5917 | 0.9210 |
| FGSM eps 0.05 | dl_pso | 0.6737 | 0.6196 | 0.6015 | 0.7048 |
| FGSM eps 0.05 | final_ensemble | 0.7923 | 0.9594 | 0.5770 | 0.9042 |
| PGD eps 0.03 | dl_pso | 0.7038 | 0.6658 | 0.6186 | 0.7595 |
| PGD eps 0.03 | final_ensemble | 0.7962 | 0.9598 | 0.5844 | 0.9108 |

## Notes

- Phase 6 correctly presents robustness/XAI as evidence for the current best model. It does not claim the Phase 5 target metrics were reached.
- SHAP/LIME are PCA-component explanations; the final report should explicitly discuss this limitation.
- The LightGBM feature-name warning observed in Colab is non-blocking because prediction completed and outputs were generated.
