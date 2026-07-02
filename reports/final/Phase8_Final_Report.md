# Phase 8 Final Report: PSO-Optimized Hybrid Fake Review Detection

**Generated at:** 2026-06-11T09:04:13.464781+00:00  
**Seed:** 42  
**Dataset:** Amazon Labeled Fake Reviews Dataset  
**Runtime constraint:** 12GB RAM Colab workflow

## Abstract

This project builds a fake review detection pipeline around a ModernBERT raw-feature tabular track and a fair sequence-model track. The tabular track uses ModernBERT pooled embeddings plus 9 behavioral features as raw 777-dimensional input for LightGBM, XGBoost and MLP candidates. The sequence track evaluates CNN-BiLSTM-Attention on real token sequences with behavioral late fusion. PCA/SVD is retained as diagnostic and ablation evidence rather than assumed as the final input path. The report presents balanced and precision-first operating modes with Recall Fake reported beside Precision Fake.

## Dataset and Preprocessing

Phase 1 cleans the input dataset and creates train/validation/test splits with seed 42. The original dataset remains at `data/final_labeled_fake_reviews.csv` and should not be rewritten.

## Feature Engineering

- Transformer backbone: `answerdotai/ModernBERT-base`.
- Pooling: `masked_mean_last_hidden_state`.
- Max length: 160.
- BERT/ModernBERT embedding dimension: 768.
- Behavioral features: 9 total, with five basic and four advanced features.
- Fused feature dimension before PCA: 777.
- Leakage policy: train-fitted scaler and train-fitted aggregates are reused for validation/test.

## PCA / SVD Feature Selection

Phase 3 records PCA/SVD diagnostics from 777 dimensions to 400 components when the diagnostic reducer exists. Retained variance is 0.951030. Phase 9 and controlled ablation evidence show PCA can hurt DL performance, so PCA is reported as a RAM/diagnostic tool rather than a universally beneficial final component.

## Model Architecture

The revised architecture has two fair tracks: raw 777 tabular candidates (`LightGBM`, `XGBoost`, `MLP`) and a sequence CNN-BiLSTM-Attention candidate that reads token sequences rather than PCA/static vectors. Phase 5 then evaluates single models, weighted blending, stacking and calibration before selecting `phase5_weighted_blend` with threshold 0.6 for audit.

## Final Test Metrics

- Macro F1 = NA
- Precision Fake = NA
- Recall Fake = NA
- ROC-AUC = NA
- PR-AUC = NA

At the selected precision threshold, Precision Fake = NA, Macro F1 = NA, and Recall Fake = NA. This threshold improves precision but gives up substantial recall.

## Target Audit

All original target metrics remain below target.

- macro_f1 at default_0.5: actual 0.943320, target 0.890000, gap 0.053320, pass True.
- precision_fake at default_0.5: actual 0.969872, target 0.975000, gap -0.005128, pass False.
- roc_auc at default_0.5: actual 0.976900, target 0.930000, gap 0.046900, pass True.
- macro_f1 at phase5_balanced_macro_f1_threshold: actual 0.946343, target 0.890000, gap 0.056343, pass True.
- precision_fake at phase5_balanced_macro_f1_threshold: actual 0.934395, target 0.975000, gap -0.040605, pass False.
- roc_auc at phase5_balanced_macro_f1_threshold: actual 0.976900, target 0.930000, gap 0.046900, pass True.
- macro_f1 at phase5_selected_precision_threshold: actual 0.912628, target 0.890000, gap 0.022628, pass True.
- precision_fake at phase5_selected_precision_threshold: actual 0.981643, target 0.975000, gap 0.006643, pass True.
- roc_auc at phase5_selected_precision_threshold: actual 0.976900, target 0.930000, gap 0.046900, pass True.

## Cross-Validation Evidence

Phase 7 runs `full_5fold_cv` with 5 folds and 36336 rows. Mean Macro F1 is 0.865940, mean Precision Fake is 0.903090, and mean ROC-AUC is 0.923336.

## Ablation Study

- Full Model: removed none; evidence direct_artifact; Macro F1 0.943320, Precision Fake 0.969872, ROC-AUC 0.976900.
- Model A: removed cnn_sequence_branch; evidence direct_artifact_branch_level; Macro F1 0.905938, Precision Fake 0.968579, ROC-AUC 0.953063.
- Model B: removed PCA; evidence controlled_lightgbm; Macro F1 0.905821, Precision Fake 0.962230, ROC-AUC 0.952377.
- Model C: removed advanced_behavioral_features; evidence controlled_lightgbm; Macro F1 0.868396, Precision Fake 0.904425, ROC-AUC 0.924356.
- Model D: removed tabular_blend_branch; evidence direct_artifact; Macro F1 0.934290, Precision Fake 0.946500, ROC-AUC 0.972582.
- Model E: removed PSO_PCA_advanced_behavioral_ensemble_nearest_available_baseline; evidence direct_artifact_nearest_baseline; Macro F1 0.766460, Precision Fake 0.781827, ROC-AUC 0.838937.

Interpretation notes: PCA and legacy PSO/PCA CNN-BiLSTM are treated as diagnostic/ablation evidence. If a single raw-feature model beats ensemble variants, the report should select the single model honestly rather than forcing a hybrid winner.

## Robustness and XAI

Phase 6 evaluates robustness/XAI on selected candidates within RAM limits. SHAP uses 500 samples and LIME exports 6 representative cases when available. These outputs explain the selected evidence-backed model and must be refreshed after the revised Phase 5 run.

## Error Analysis

Phase 7 exports false positives, false negatives, high-confidence mistakes and borderline predictions in `reports/tables/phase7_error_analysis.csv`.

## Limitations

- The original target metrics are not reached.
- PCA is diagnostic/ablation evidence, not a guaranteed score-improving final component.
- Some ablations are controlled/surrogate evidence instead of complete full-hybrid retrains.
- Robustness/XAI scope may differ for tabular and sequence winners under the 12GB RAM constraint.
- SHAP/LIME explanations must be regenerated for the selected revised Phase 5 winner.

## Future Work

- Run the revised Phase 5 notebook family and rerun Phase 7/8 after the new artifacts are synced.
- Explore stronger ensemble calibration or threshold policies to close the fake-precision gap.
- Add cross-dataset validation if time allows.
- Convert the final Markdown report to PDF/DOCX after the metric narrative is approved.

## Reproducibility and Package Checklist

- Run notebooks 01 through 08 in order on Colab.
- Keep seed 42 in all metric tables.
- Submit the final notebook, final report, Phase 8 manifests, Phase 7 tables/figures and saved model artifacts.
