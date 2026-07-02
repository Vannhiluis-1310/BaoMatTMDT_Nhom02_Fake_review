# Phase 4 Verification: PSO-Tuned CNN-BiLSTM-Attention

Status: verified_with_metric_gap

## Artifact Checks

- `artifacts/models/baseline_model_dl.pth` exists.
- `artifacts/models/best_model_dl.pth` exists.
- `artifacts/models/best_params.json` exists.
- `artifacts/models/phase4_metadata.json` exists.
- `artifacts/predictions/dl_pso_train_prob.npy` exists with shape `(29923,)`.
- `artifacts/predictions/dl_pso_val_prob.npy` exists with shape `(6413,)`.
- `artifacts/predictions/dl_pso_test_prob.npy` exists with shape `(6413,)`.
- Phase 4 tables and figures were regenerated under `reports/tables/` and `reports/figures/`.

## Retune Config Verification

`phase4_metadata.json` confirms:

- `pso_subset_ratio = 0.2`
- `pso_particles = 10`
- `pso_iterations = 8`
- `pso_trial_epochs = 5`
- `final_max_epochs = 30`
- `early_stopping_patience = 6`
- `max_batch_size = 64`
- Composite objective weights:
  - Macro F1: 0.50
  - ROC-AUC: 0.30
  - Precision Fake: 0.20

`phase4_pso_trial_history.csv` contains 90 successful PSO trials, all `status=ok`.

## Best PSO Trial

- Trial: 82
- Validation objective score: 0.795292
- Validation Macro F1: 0.758734
- Validation Precision Fake: 0.844170
- Validation ROC-AUC: 0.823638
- Best params include:
  - learning_rate: 0.000394896
  - dropout: 0.144660
  - cnn_filters: 96
  - kernel_size: 7
  - lstm_hidden: 64
  - attention_dim: 96
  - focal_gamma: 1.608726
  - batch_size: 32

## Final Phase 4 Metrics

| Split | Model | Macro F1 | Precision Fake | Recall Fake | ROC-AUC | PR-AUC |
|-------|-------|----------|----------------|-------------|---------|--------|
| val | dl_pso | 0.798606 | 0.828082 | 0.681021 | 0.860606 | 0.850172 |
| test | dl_pso | 0.810489 | 0.836586 | 0.702363 | 0.867929 | 0.860599 |

## Conclusion

Phase 4 retune executed correctly and improved the DL model compared with the previous Phase 4 run, but the standalone DL model remains below the final project target. This is acceptable for Phase 4 as an input to ensemble, but it is still a metric gap for the overall target.

## Not Run Locally

Per project policy, no notebook cells, EDA, training, tuning, or dataset processing were run locally by the agent.
