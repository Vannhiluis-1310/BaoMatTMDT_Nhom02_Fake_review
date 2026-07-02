# Phase 9 Research Notes: DL PCA Diagnostic Test

**Date:** 2026-06-09  
**Mode:** Local artifact reasoning only

## Question

The current final ensemble assigns DL weight 0. This makes the original title and novelty claim weak. Before changing the title or retraining the whole pipeline, we need a focused diagnostic test:

> Is the DL branch weak because PCA removed information, because CNN-BiLSTM-Attention is mismatched with static vectors, or both?

## Recommended Experimental Controls

| Comparison | What It Isolates | Interpretation |
|---|---|---|
| `MLP_raw_777` vs `MLP_pca_400` | PCA effect on a DL-compatible tabular architecture | If raw wins by >= 2-3 Macro F1 points, PCA likely removes discriminative signal. |
| `MLP_pca_400` vs `CNNBiLSTM_pca_400` | Architecture mismatch on the same PCA features | If MLP wins, CNN/BiLSTM is likely not appropriate for static PCA vectors. |
| `LightGBM_raw_777` vs `LightGBM_pca_400` | Existing tree-model PCA sensitivity | Already suggests raw features can be stronger than PCA features. |
| constrained blend with `dl_weight >= 0.10` | Whether DL can contribute non-zero to final ensemble | If metrics collapse, DL branch is not ready for title-level claim. |

## Why MLP Is The Right Diagnostic Baseline

MLP is not necessarily the final recommended model. It is a clean control because it treats raw/PCA feature vectors as tabular static vectors. That makes it better for testing whether PCA itself hurts the DL signal, without introducing the sequence assumptions of CNN/BiLSTM.

## Why Not Immediately Train Token-Level CNN/BiLSTM

Token-level ModernBERT output would be a more semantically valid sequence input, but it is a larger experiment:

- It requires transformer forward passes or token-level hidden-state caching.
- It may exceed time/RAM if run across the full dataset.
- It does not directly isolate whether current PCA/static-vector design is the problem.

Therefore, token-level DL should be a follow-up only after the diagnostic notebook establishes the failure mode.

## Evidence Thresholds

Suggested interpretation thresholds:

- PCA hurts DL: `MLP_raw_777_macro_f1 - MLP_pca_400_macro_f1 >= 0.02`.
- Architecture mismatch: `MLP_pca_400_macro_f1 - CNNBiLSTM_pca_400_macro_f1 >= 0.02`.
- DL usable in final title: at least one constrained ensemble with `dl_weight >= 0.10` is within 0.01 Macro F1 and 0.01 ROC-AUC of the unconstrained best, or improves a target metric without unacceptable recall loss.
- DL not title-ready: constrained DL ensemble is clearly worse, and standalone DL/MLP branches do not justify inclusion.

## Expected Outputs

- Diagnostic notebook.
- Diagnostic metrics table.
- Diagnostic ensemble sweep table.
- Runtime table.
- Interpretation summary.
- Recommendation for next action:
  - keep title and retrain with non-zero DL,
  - change title/framing,
  - redesign DL branch.
