# Phase 9: DL PCA Diagnostic Test - Context

**Gathered:** 2026-06-09  
**Status:** Ready for planning  
**Source:** User concern + Phase 4/5/7 verified artifacts

<domain>

## Phase Boundary

This is a standalone diagnostic test phase. It is not part of the main 8-notebook pipeline and must not modify the main source notebooks.

The goal is to test why the DL branch is weak:

1. Is DL low mainly because PCA removed useful signal?
2. Is DL low mainly because CNN-BiLSTM-Attention is architecturally mismatched with static PCA/tabular vectors?
3. Can any DL branch make a meaningful non-zero contribution to an ensemble under a constrained selection rule?

</domain>

<non_negotiables>

## Hard Constraints

- Do not modify notebooks `01_*.ipynb` through `08_*.ipynb`.
- Do not overwrite existing Phase 1-8 artifacts.
- Do not write into `artifacts/models/`, `artifacts/ensemble/`, `artifacts/predictions/`, `reports/tables/phase*.csv` or `reports/final/`.
- Do not run notebook cells, training, tuning, EDA, PCA, or dataset processing locally.
- The diagnostic source must be a separate Colab notebook, not a `.py` module.
- All diagnostic outputs must go under:
  - `artifacts/diagnostics/dl_pca_test/`
  - `reports/diagnostics/dl_pca_test/`
- Preserve `data/final_labeled_fake_reviews.csv`.
- RAM 12GB remains a hard constraint.

</non_negotiables>

<known_evidence>

## Current Evidence

- Phase 2 fused features:
  - ModernBERT embedding: 768 dimensions.
  - Behavioral features: 9.
  - Raw fused vector: 777 dimensions.
- Phase 3 PCA:
  - Reduced dimension: 400.
  - Retained variance: 0.951030.
- Phase 4 DL branch:
  - CNN-BiLSTM-Attention + PSO remains weaker than tree models.
- Phase 7 final ensemble:
  - Selected candidate: `blend_dl00_xgb05_lgbm05`.
  - DL weight: 0.
  - Legacy default test Macro F1: 0.855820 (superseded; final blend 0.9433–0.9463).
  - Selected threshold Precision Fake: 0.960441.
- Phase 7 controlled no-PCA LightGBM:
  - Macro F1: 0.905821.
  - Precision Fake: 0.962230.
  - ROC-AUC: 0.952377.

</known_evidence>

<test_design_summary>

## Diagnostic Test Idea

Use controlled comparisons rather than another broad retune:

1. **PCA effect on DL**
   - Train the same lightweight MLP on raw 777 features and PCA 400 features.
   - If MLP raw strongly beats MLP PCA, PCA is hurting DL-relevant signal.

2. **Architecture mismatch on PCA vector**
   - Compare MLP PCA vs CNN-BiLSTM-Attention PCA under the same split/seed.
   - If MLP PCA beats CNN-BiLSTM PCA, the sequence architecture is likely mismatched.

3. **Existing tree reference**
   - Use existing LightGBM PCA and no-PCA evidence as non-DL controls.

4. **Constrained ensemble probe**
   - Sweep simple weighted blends with `dl_weight >= 0.10` and `dl_weight >= 0.20`.
   - Check whether any DL contribution can improve or at least preserve Macro F1 / Precision Fake / ROC-AUC.

</test_design_summary>

<deferred>

## Deferred

- Full ModernBERT token-level sequence training is deferred unless the first diagnostic test proves the current static-vector DL branch is not salvageable.
- Any rewrite of Phase 4/5/7 is deferred until this diagnostic phase produces evidence.
- Title/report changes are deferred until the diagnostic result is known.

</deferred>
