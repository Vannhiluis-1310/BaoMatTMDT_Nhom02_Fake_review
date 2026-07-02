# DL / PCA Diagnostic Summary

- Raw MLP default Macro F1: 0.893034
- PCA MLP default Macro F1: 0.868496
- CNN-BiLSTM on PCA default Macro F1: 0.774833
- Best constrained non-zero DL blend: mlp_raw_dl0.30_xgb0.00_lgbm0.70
- Best constrained blend test Macro F1: 0.847788
- Best constrained blend test Precision Fake: 0.966165
- Best constrained blend test ROC-AUC: 0.937180

## Decision

1. Keep title and rerun with non-zero DL contribution
2. Redesign the DL branch before keeping the title
3. Change title/framing to a tree-based ensemble with DL as ablation

This diagnostic notebook is meant to decide which of those three directions is justified by evidence.
