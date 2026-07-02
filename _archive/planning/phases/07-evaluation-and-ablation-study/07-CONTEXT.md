# Phase 7: Evaluation and Ablation Study - Context

**Gathered:** 2026-06-01  
**Status:** Ready for planning  
**Source:** Roadmap + Phase 4-6 verified artifacts

<domain>

## Phase Boundary

Phase 7 builds `notebooks/07_Evaluation_Ablation.ipynb` into the final experiment-evidence notebook. It must compute final metrics, target pass/fail audit, CV evidence, ablation tables, plots and error analysis from existing artifacts, with resource-aware retraining only where necessary for controlled ablations.

</domain>

<decisions>

## Implementation Decisions

### Runtime

- Run on Google Colab only.
- Do not run notebook cells, EDA, training, tuning, CV, ablation or dataset processing locally.
- Keep executable evaluation source inside `notebooks/07_Evaluation_Ablation.ipynb`.

### Evidence Policy (updated 2026-06-10)

- SSOT: `phase7_final_metrics.csv` on `phase5_weighted_blend`. Targets pass in appropriate modes (precision-first passes all three).
- Report honestly: mode-dependent targets, overfit train, Phase 6 legacy scope, behavioral ablation +0.0008 only.
- Every metric row must include seed, split, model variant, threshold and data source.
- Target audit compares against Macro F1 ≥0.89, Precision Fake ≥0.975, ROC-AUC ≥0.93.

### Ablation Policy

- Use existing prediction logs wherever possible to avoid expensive retraining.
- Label ablations as either:
  - `direct_artifact`: uses existing trained model/probabilities.
  - `controlled_lightgbm`: retrains a lightweight controlled LightGBM comparison.
  - `resource_fallback`: uses sample/CV fallback because full version exceeds RAM/time.
- Do not fabricate deltas for ablations that cannot be fairly measured. Record limitation instead.

### CV Policy

- Prefer 5-fold CV with LightGBM on PCA features because full PSO + ensemble CV is too expensive under 12GB and timeline constraints.
- If full 5-fold CV is too slow, use StratifiedKFold with fewer rows and record `cv_mode`, subset size and reason.

</decisions>

<canonical_refs>

## Canonical References

### Planning

- `.planning/ROADMAP.md` - Phase 7 goal, tasks and success criteria.
- `.planning/REQUIREMENTS.md` - EVAL-01..EVAL-05.
- `.planning/STATE.md` - Current metric gaps and verified Phase 6 state.

### Inputs

- `artifacts/pca/features_final_{train,val,test}.npy`
- `artifacts/pca/labels_{train,val,test}.npy`
- `artifacts/features/features_raw_{train,val,test}.npy`
- `artifacts/features/feature_metadata.json`
- `artifacts/features/feature_dictionary.csv`
- `artifacts/predictions/*_prob.npy`
- `reports/tables/phase4_model_comparison.csv`
- `reports/tables/phase5_base_model_metrics.csv`
- `reports/tables/phase5_stacking_metrics.csv`
- `reports/tables/phase5_final_metrics.csv`
- `reports/tables/phase5_candidate_selection.csv`
- `reports/tables/phase6_robustness_metrics.csv`
- `artifacts/xai/phase6_metadata.json`

### Notebook

- `notebooks/07_Evaluation_Ablation.ipynb` - target notebook to implement.

</canonical_refs>

<specifics>

## Specific Ideas

- Final metrics should include both default threshold and optimized precision threshold for final ensemble.
- Ablation table should include the requested variants:
  - Full Model: current Phase 5 final ensemble.
  - Model A No PSO: direct deep branch comparison and, if applicable, selected blend with baseline DL swapped in.
  - Model B No PCA: controlled LightGBM raw features vs PCA features.
  - Model C No advanced behavioral: controlled LightGBM with BERT + 5 basic behavioral features.
  - Model D No ensemble: Phase 4 PSO DL model.
  - Model E Baseline: Phase 4 baseline DL.
- Add a target gap figure to make missed targets easy to explain in the final report.
- Error analysis should export top false positives/false negatives by confidence, with review text if row-level text can be joined safely.

</specifics>

<deferred>

## Deferred Ideas

- Full 5-fold CV of PSO deep model and full hybrid ensemble is deferred unless Colab time is clearly available.
- Retuning Phase 4/5 is out of scope for Phase 7; Phase 7 evaluates and explains current artifacts.

</deferred>
