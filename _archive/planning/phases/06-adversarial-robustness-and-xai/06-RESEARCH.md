# Phase 6 Research: Adversarial Robustness and XAI

## Context

Phase 6 must use the artifacts produced by Phase 3, 4 and 5:

- Phase 3 PCA features: `artifacts/pca/features_final_{split}.npy`, shape around 400 dims.
- Phase 4 deep model: `artifacts/models/best_model_dl.pth` and `artifacts/predictions/dl_pso_*_prob.npy`.
- Phase 5 ensemble: `artifacts/ensemble/final_ensemble_model.pkl`, `artifacts/ensemble/phase5_metadata.json`, and final probabilities.

Current Phase 4/5 verification status is `verified_with_gap`: artifacts are usable, but final targets are not met. Phase 6 should report robustness/XAI truthfully for the current best model rather than implying the target metrics were reached.

## Robustness Approach

Tree models and weighted blends are not directly differentiable end-to-end. Use the Phase 4 CNN-BiLSTM-Attention model as a differentiable surrogate over PCA feature vectors:

1. Reconstruct the Phase 4 PyTorch model architecture inside `06_Adversarial_XAI.ipynb`.
2. Load `best_model_dl.pth`.
3. Generate adversarial PCA features on a fixed seeded test subset:
   - FGSM with epsilon grid, e.g. `0.01`, `0.03`, `0.05`.
   - PGD with small step count, e.g. `5` steps, alpha `epsilon / 4`.
4. Clamp perturbed PCA features to train-set min/max per component to avoid wildly invalid inputs.
5. Evaluate both:
   - `dl_pso` on clean/FGSM/PGD features.
   - `final_ensemble` by recomputing base probabilities on perturbed features and applying the selected Phase 5 candidate logic.

This gives a consistent robustness story: adversarial perturbations are generated against the differentiable DL model and transferred to the ensemble.

## Ensemble Prediction Handling

Phase 5 may select either a weighted blend or a sklearn/calibrated stacker. The Phase 6 notebook should implement a small `predict_final_ensemble_proba(X)` helper:

- Load XGBoost/LightGBM wrappers from `artifacts/ensemble/`.
- Recompute `dl_pso`, XGBoost and LightGBM probabilities for any PCA feature matrix.
- Read `phase5_metadata.json`.
- If selected candidate type is `weighted_blend`, combine base probabilities using recorded weights.
- If selected candidate is a fitted sklearn/calibrated model, build stack features and call `predict_proba`.
- If metadata is missing or unsupported, fall back to LightGBM and mark the fallback clearly in metadata.

## SHAP Approach

Use SHAP primarily on LightGBM because:

- The selected Phase 5 candidate heavily relies on LightGBM in current artifacts.
- `shap.TreeExplainer` is efficient on tree models.
- It avoids expensive model-agnostic SHAP over the full stacked ensemble.

Recommended outputs:

- `artifacts/xai/shap_values_lightgbm.npy`
- `reports/tables/phase6_shap_global_importance.csv`
- `reports/figures/phase6_shap_top_components.png`
- Optional SHAP summary plot if resources allow.

To make PCA features interpretable, map top PCA components back to raw feature groups:

- Load `artifacts/pca/pca_or_svd.joblib`.
- Load Phase 2 feature dictionary if available.
- For each important PCA component, record top raw loadings and whether they come from BERT embedding dimensions or behavioral features.
- Save `reports/tables/phase6_pca_component_loading_map.csv`.

## LIME Approach

Use `lime.lime_tabular.LimeTabularExplainer` on PCA features:

- Background: seeded subset from train PCA features, e.g. 1000 rows.
- Cases: small fixed set from test, e.g. TP, TN, FP, FN if available, otherwise highest/lowest confidence fake/real cases.
- Prediction function: `predict_final_ensemble_proba`.
- Save HTML and CSV summaries:
  - `artifacts/xai/lime_case_*.html`
  - `reports/tables/phase6_lime_case_summary.csv`
  - `reports/tables/phase6_lime_feature_weights.csv`

## Resource Guardrails

- Do not run locally.
- Keep all executable pipeline code inside `notebooks/06_Adversarial_XAI.ipynb`.
- Default subset sizes:
  - robustness: 1000 test rows
  - SHAP: 500 test rows
  - LIME background: 1000 train rows
  - LIME cases: 6 examples
- Allow lowering subset sizes if Colab RAM is tight.
- Save all subset IDs and seeds for reproducibility.

## Validation Architecture

Phase 6 is complete when:

1. Clean vs adversarial metrics exist for DL and final ensemble.
2. Robustness metric drops are reported, not hidden.
3. SHAP global importance exists and maps PCA components to raw feature groups.
4. LIME examples exist for representative fake/real cases.
5. Metadata records seed, subset sizes, model artifact versions and fallback decisions.
