# Phase 4 Summary: PSO-Tuned CNN-BiLSTM-Attention

> **Ablation track (PCA 400-d).** Artifact SSOT: `phase7_final_metrics.csv` — dl_baseline test F1 **0.7665**, dl_pso **0.7793** (+0.0128). Numbers below are an earlier Colab snapshot.

## Completed

- Stronger PSO retune was run on Colab.
- 90 PSO trials completed successfully.
- Best params, DL checkpoint, metadata, training history and probability artifacts were regenerated.
- Phase 4 figures and metrics tables were regenerated.

## Test Metrics

- Test Macro F1: 0.810489
- Test Precision Fake: 0.836586
- Test Recall Fake: 0.702363
- Test ROC-AUC: 0.867929
- Test PR-AUC: 0.860599

## Verification Result

Phase 4 is artifact-complete and technically verified. It has a metric gap because the standalone DL signal remains below the final project targets.
