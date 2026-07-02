# Notebook Plan

## Execution Rule

- All executable project source must live in `.ipynb` files under `notebooks/`.
- Run notebooks on Google Colab unless the project owner explicitly approves local execution.
- Each notebook must read the previous phase output from Drive/artifacts and write its own outputs clearly.
- Do not duplicate heavy computation across notebooks if an artifact already exists and is still compatible with the revised pipeline.
- Phase 9 diagnostic remains isolated under `notebooks/tests/` and is not part of the submission notebook chain.

## Revised Pipeline Contract

The current project direction is no longer "PCA-first CNN-BiLSTM over static vectors". The revised pipeline uses two fair model tracks:

1. **Tabular Track**: ModernBERT pooled embeddings + 9 behavioral features -> raw fused 777 features -> LightGBM, XGBoost, MLP, and weighted/stacked ensembles.
2. **Sequence Track**: raw text -> ModernBERT tokenizer -> token sequence -> CNN-BiLSTM-Attention, with behavioral features added through late fusion.

PCA/SVD is retained as diagnostic and ablation evidence, not as the default final path.

## Notebook Map

| File | Main Content | Main Output | Note |
|------|--------------|-------------|------|
| `01_EDA_Preprocessing.ipynb` | Load data, deep EDA, missing/duplicate handling, fixed stratified split | `data/processed/` clean data and split files | Phase 1; no rerun needed unless split/schema changes |
| `02_Feature_Engineering.ipynb` | ModernBERT pooled embeddings, 9 behavioral features, raw fused 777 features, tokenizer artifacts for sequence track | `artifacts/features/features_raw_777_*`, behavioral metadata, tokenization metadata | Phase 2; ModernBERT must be explicit |
| `03_PCA_Feature_Selection.ipynb` | PCA/SVD and optional Autoencoder diagnostic, memory/time report | `artifacts/pca/` diagnostic features and reports | Phase 3; not default final model input |
| `04_PSO_Model_Training.ipynb` | Legacy PSO CNN-BiLSTM over PCA retained for ablation evidence only | `artifacts/models/best_model_dl.pth`, legacy DL probabilities | Phase 4; not the revised final DL path |
| `05_Hybrid_Ensemble.ipynb` | Phase 5 orchestrator/summary: reads per-model notebook outputs, runs final ensemble selection and exports the final Phase 5 report tables | `artifacts/ensemble/model_zoo_*`, `reports/tables/phase5_ensemble_sweep.csv` | Phase 5 main submission notebook; single models are allowed to win |
| `06_Adversarial_XAI.ipynb` | Robustness and XAI for selected candidates; SHAP/LIME for tabular winner and DL explanation subset | Robustness tables + explanation files under `artifacts/xai/` | Phase 6; subset allowed under RAM 12GB |
| `07_Evaluation_Ablation.ipynb` | Final evaluation, threshold sweep, recall-aware precision-first selection, ablation including PCA diagnostic | Final metrics, ablation charts, winner tables under `reports/` | Phase 7; every metric tied to seed/split/model variant |
| `08_Final_Report_Kaggle.ipynb` | Final presentation notebook, inference path, balanced winner and precision-first winner narrative | Clean submission notebook and final report artifacts | Phase 8; revise after new Phase 5/7 outputs |

## Dependency Chain

```text
01_EDA_Preprocessing
  -> 02_Feature_Engineering
       -> raw fused 777 features
       -> sequence tokenization metadata
  -> 03_PCA_Feature_Selection
       -> diagnostic-only PCA/SVD/Autoencoder evidence
  -> 04_PSO_Model_Training
       -> legacy PSO/PCA CNN-BiLSTM ablation evidence
  -> 05_Hybrid_Ensemble
       -> LightGBM raw / XGBoost raw / MLP raw / Sequence DL
       -> soft voting / weighted blending / stacking / calibration
  -> 06_Adversarial_XAI
  -> 07_Evaluation_Ablation
  -> 08_Final_Report_Kaggle
```

## Phase 5 Notebook Family

Phase 5 should be split into independent model notebooks for easier Colab execution, debugging and reruns. The main `05_Hybrid_Ensemble.ipynb` remains the Phase 5 orchestrator/summary notebook.

| Notebook | Responsibility | Reads | Writes |
|----------|----------------|-------|--------|
| `05_00_Phase5_Run_Order.ipynb` | Validate Phase 5 inputs and explain execution order | Phase 2/4 artifact metadata | run-order checklist and input validation table |
| `05_01_LightGBM_Raw.ipynb` | Train/evaluate LightGBM on raw 777 features | raw 777 train/val/test | LightGBM model, probabilities, metrics |
| `05_02_XGBoost_Raw.ipynb` | Train/evaluate XGBoost on raw 777 features | raw 777 train/val/test | XGBoost model, probabilities, metrics |
| `05_03_MLP_Raw.ipynb` | Train/evaluate MLP on raw 777 features or standardize Phase 4 MLP output | raw 777 train/val/test | MLP model/probabilities/metrics |
| `05_04_CNN_BiLSTM_Sequence.ipynb` | Train/evaluate CNN-BiLSTM-Attention on token sequences with behavioral late fusion | tokenized text + behavioral features | sequence DL model/probabilities/metrics |
| `05_05_Weighted_Blending.ipynb` | Run pair/triple/full weighted blends over available model probabilities | per-model validation/test probabilities | weighted blend sweep table |
| `05_06_Stacking_Calibration.ipynb` | Run stacking meta-models and probability calibration | per-model validation/test probabilities | stacking/calibration artifacts and metrics |
| `05_Hybrid_Ensemble.ipynb` | Aggregate all Phase 5 outputs and choose selected candidates | all Phase 5 model/sweep outputs | final Phase 5 leaderboard and selected candidates |

If a strict "8 notebooks only" submission is required later, `05_Hybrid_Ensemble.ipynb` should summarize the Phase 5 family outputs, while the per-model notebooks remain supplementary reproducibility artifacts.

## Artifact Contract

| Phase | Must Read | Must Write |
|-------|-----------|------------|
| 1 | `data/final_labeled_fake_reviews.csv` | cleaned dataset, train/val/test split files |
| 2 | Phase 1 split files | ModernBERT pooled embeddings, 9 behavioral features, raw fused 777 features, feature metadata |
| 3 | Phase 2 raw fused features | PCA/SVD/Autoencoder diagnostic artifacts, explained variance and memory reports |
| 4 | Phase 3 PCA/SVD diagnostic artifacts when legacy ablation is needed | legacy PSO/PCA DL checkpoint and probabilities |
| 5 | Phase 2 raw features; Phase 4 probabilities if reused | per-model artifacts, model zoo table, ensemble sweep table, calibrated probabilities, threshold report |
| 6 | Selected Phase 5 candidates and probabilities | adversarial metrics, SHAP/LIME outputs |
| 7 | All model probabilities, logs, and diagnostic artifacts | final metrics table, ablation table, winner selection table, plots |
| 8 | Reports, figures, final artifacts | clean presentation notebook and submission-ready outputs |

## Model Selection Contract

Two final operating modes must be reported:

| Mode | Selection Rule |
|------|----------------|
| Balanced | Maximize validation Macro F1, then check Precision Fake, Recall Fake, ROC-AUC and PR-AUC |
| Precision-first | Require `Precision Fake >= 97.5%`, then maximize Recall Fake, Macro F1 and ROC-AUC |

Recall Fake must always be reported when Precision Fake is optimized.
