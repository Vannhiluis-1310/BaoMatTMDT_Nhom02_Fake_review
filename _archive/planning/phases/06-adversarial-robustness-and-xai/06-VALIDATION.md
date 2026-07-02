# Phase 6 Validation Strategy

## Validation Goal

Confirm that Phase 6 produces trustworthy robustness and XAI evidence for the current Phase 4/5 models without retraining, leakage, or excessive Colab resource use.

## Required Checks

| Check | Method | Pass Criteria |
|-------|--------|---------------|
| Input availability | Read Phase 3/4/5 metadata and arrays | All required artifacts exist; shapes match split row counts. |
| DL predictor reconstruction | Run on small Colab slice | Produces fake probabilities in `[0, 1]` with expected row count. |
| Final ensemble predictor | Run on small Colab slice | Uses selected Phase 5 candidate logic or records fallback. |
| FGSM/PGD generation | Inspect robustness config and output table | Clean, FGSM and PGD rows exist with epsilon/method metadata. |
| Robustness metrics | Compare clean vs adversarial | Macro F1, Precision Fake, Recall Fake and ROC-AUC are reported for each condition. |
| SHAP outputs | Inspect table/files | Global importance table and SHAP artifact exist. |
| PCA interpretation | Inspect loading map | Top PCA components map to raw feature groups where possible. |
| LIME outputs | Inspect HTML and CSV | Local explanations exist for representative cases. |
| Reproducibility | Inspect metadata | Seed, subset sizes, source artifact paths and fallback decisions are saved. |

## Human Review Points

- Whether the robustness drop is acceptable should be interpreted in the report, not hidden.
- Whether PCA-level SHAP/LIME explanations are satisfying enough for the paper should be discussed as a limitation.
- Phase 6 runs on **legacy PCA ensemble** @τ=0.79 — present robustness/XAI as diagnostic/ablation evidence, not headline for `weighted_blend` final (2026-06-10).
