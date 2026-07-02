# Phase 7 Research: Evaluation and Ablation Study

> **Updated 2026-06-10.** Final track SSOT: `phase7_final_metrics.csv`. Targets **pass** on `phase5_weighted_blend` (balanced F1 0.9463, precision-first Prec. Fake 0.9816, ROC-AUC 0.9769). Section below through "Legacy snapshot" documents the **superseded PCA pipeline** (~2026-06-01).

## Final Track (current)

- Model: `phase5_weighted_blend` (CNN 50% / XGB 35% / LGBM 15%).
- Test @τ=0.3 balanced: Macro F1 **0.9463**; @τ=0.6 precision-first: Prec. Fake **0.9816**; ROC-AUC **0.9769** (all τ).
- Multi-seed: balanced **0.9485 ± 0.0018** (`phase7_multiseed_summary.csv`).
- Ablation: raw +0.0397 vs PCA ref.; ensemble +0.164 vs DL-PSO; advanced behavioral +0.0008 (controlled).
- Phase 6 robustness: legacy PCA ensemble @τ=0.79 only — disclose in report.

## Context (legacy snapshot — pre Phase 10)

Phase 7 originally evaluated a PCA-track ensemble that missed targets:

- Phase 4 `dl_pso`: Macro F1 0.7793, Precision Fake 0.7819, ROC-AUC 0.8517.
- Legacy final ensemble @0.5: Macro F1 0.8558, Precision Fake 0.9156, ROC-AUC 0.9115.
- Phase 6 robustness subset clean legacy ensemble: Macro F1 0.8000 @τ=0.79.

## Final Evaluation Approach

Use existing Phase 4/5 probability artifacts and labels to recompute metrics consistently:

- Accuracy
- Macro F1
- Precision Fake
- Recall Fake
- F1 Fake
- ROC-AUC
- PR-AUC
- Brier score
- Confusion matrix values
- Threshold strategy

This avoids local or unnecessary Colab retraining and guarantees all metrics use the same label arrays.

## CV Strategy

Full CV of the PSO deep model and final stacking ensemble is resource-heavy because it implies repeated DL/PSO/ensemble training. Use a staged strategy:

1. Primary CV: 5-fold StratifiedKFold LightGBM on Phase 3 PCA features.
2. Optional quick CV: Logistic Regression on PCA features for a linear baseline.
3. Fallback: reduced-row StratifiedKFold if full CV is slow or RAM constrained.

The notebook must record `cv_mode`, folds, subset size, seed and reason. This satisfies EVAL-02 because the requirement explicitly allows reduced CV with explanation when resources are insufficient.

## Ablation Strategy

Use a mixed direct + controlled ablation plan:

| Variant | Evidence Type | Practical Implementation |
|---------|---------------|--------------------------|
| Full Model | direct_artifact | Phase 5 final ensemble predictions and metrics. |
| Model A No PSO | direct_artifact + optional blend swap | Compare `dl_baseline` vs `dl_pso`; if selected blend uses DL weight, recompute selected blend using baseline DL probabilities. |
| Model B No PCA | controlled_lightgbm | Train LightGBM on raw fused 777-dim features and compare to LightGBM on PCA 400-dim features under same config. |
| Model C No advanced behavioral | controlled_lightgbm | Drop the 4 advanced behavioral columns from raw features, fit PCA/SVD if needed, train LightGBM and compare. |
| Model D No ensemble | direct_artifact | Phase 4 `dl_pso` metrics. |
| Model E Baseline | direct_artifact | Phase 4 `dl_baseline` metrics. |

For Model B and C, if raw-feature training is slow or memory-heavy, sample with stratification and record `resource_fallback`. This is better than fabricating an unsupported full hybrid ablation.

## Error Analysis Strategy

Use final ensemble probabilities on the test split:

- False positives at default threshold.
- False negatives at default threshold.
- False positives at optimized precision threshold.
- False negatives at optimized precision threshold.
- Highest-confidence wrong predictions.

If row IDs and cleaned split text can be joined from Phase 1/2 artifacts, export text snippets. If text cannot be safely joined, export row indices, labels, probabilities and confidence only.

## Outputs

Recommended Phase 7 outputs:

- `reports/tables/phase7_input_validation.csv`
- `reports/tables/phase7_final_metrics.csv`
- `reports/tables/phase7_target_audit.csv`
- `reports/tables/phase7_cv_metrics.csv`
- `reports/tables/phase7_ablation_results.csv`
- `reports/tables/phase7_ablation_delta.csv`
- `reports/tables/phase7_error_analysis.csv`
- `reports/tables/phase7_report_highlights.csv`
- `reports/figures/phase7_final_metrics.png`
- `reports/figures/phase7_target_gap.png`
- `reports/figures/phase7_ablation_delta.png`
- `reports/figures/phase7_cv_summary.png`
- `artifacts/evaluation/phase7_metadata.json`

## Risk Handling

- If targets are missed, make this explicit and frame the contribution around robustness, interpretability, and which components helped.
- If ablation Model B/C is not perfectly equivalent to full hybrid, label it as controlled model ablation, not full-model ablation.
- If CV is reduced, record exact fallback reason and fold details.
