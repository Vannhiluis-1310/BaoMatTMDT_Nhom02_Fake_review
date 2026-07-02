# Phase 7 Validation Strategy

## Validation Goal

Confirm that Phase 7 produces final evaluation evidence that is reproducible, honest about target gaps, and directly usable in the final report.

## Required Checks

| Check | Method | Pass Criteria |
|-------|--------|---------------|
| Input validation | Inspect `phase7_input_validation.csv` | Required Phase 3-6 artifacts exist and row counts match labels/probabilities. |
| Final metrics | Inspect `phase7_final_metrics.csv` | Core metrics and confusion values exist for train/val/test and key model variants. |
| Target audit | Inspect `phase7_target_audit.csv` | Macro F1, Precision Fake and ROC-AUC targets have pass/fail and gap values. |
| CV evidence | Inspect `phase7_cv_metrics.csv` and summary | 5-fold or resource-aware CV includes seed, folds, model, rows and metrics. |
| Ablation variants | Inspect `phase7_ablation_results.csv` | Full, Model A, B, C, D and E exist with evidence type labels. |
| Ablation deltas | Inspect `phase7_ablation_delta.csv` | Deltas vs Full Model are computed for Macro F1, Precision Fake and ROC-AUC. |
| Figures | Inspect report figures | Final metrics, target gap, CV and ablation figures exist. |
| Error analysis | Inspect `phase7_error_analysis.csv` | FP/FN/high-confidence errors are exported with threshold context. |
| Metadata | Inspect `phase7_metadata.json` | Seed, split, configs, versions, fallbacks and output paths are saved. |

## Human Review Points

- Confirm that controlled ablations are clearly described as controlled evidence in the final report.
- Confirm that target misses are framed as limitations rather than hidden.
- Confirm whether reduced CV is acceptable if full 5-fold CV is too slow.
