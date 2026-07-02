# Roadmap: ModernBERT Raw-Feature + Sequence-Track Fake Review Detection

**Timeline:** revised continuation after Phase 9 diagnostic  
**Runtime policy:** Run on Google Colab. Do not run notebook cells, EDA, training, tuning, or dataset processing locally unless the project owner explicitly approves.  
**Source policy:** All executable pipeline source remains inside the 8 phase notebooks under `notebooks/`. Phase 9/10 are planning and diagnostic/control phases, not extra submission notebooks.

## Current Direction

Phase 9 showed that the earlier PCA-first DL path is weak:

- MLP on raw 777 features is stronger than MLP on PCA 400.
- CNN-BiLSTM on static PCA vectors is architecturally mismatched.
- LightGBM raw no-PCA is currently a very strong balanced candidate.

The revised work is therefore:

1. Keep Phase 1 split and Phase 2 raw fused feature concept.
2. Make ModernBERT explicit in Phase 2.
3. Treat PCA/SVD as diagnostic/ablation, not the default final path.
4. Optimize DL fairly through:
   - MLP over raw 777 tabular features.
   - CNN-BiLSTM-Attention over real token sequences, with behavioral late fusion.
5. Run a broad model zoo and ensemble sweep before choosing the final model.

## Notebook Plan

| Phase | Notebook | Main Output |
|-------|----------|-------------|
| 1 | `01_EDA_Preprocessing.ipynb` | `data/processed/` clean data and fixed train/val/test split |
| 2 | `02_Feature_Engineering.ipynb` | ModernBERT pooled embeddings, 9 behavioral features, raw fused 777 features |
| 3 | `03_PCA_Feature_Selection.ipynb` | PCA/SVD/Autoencoder diagnostic artifacts and memory report |
| 4 | `04_PSO_Model_Training.ipynb` | Legacy PSO/PCA CNN-BiLSTM ablation probabilities |
| 5 | `05_*` notebook family + `05_Hybrid_Ensemble.ipynb` | Per-model notebooks for LightGBM/XGBoost/MLP/sequence-DL plus ensemble sweep/orchestrator |
| 6 | `06_Adversarial_XAI.ipynb` | Robustness + SHAP/LIME for selected candidates |
| 7 | `07_Evaluation_Ablation.ipynb` | Final metrics, threshold sweep, ablation and winner tables |
| 8 | `08_Final_Report_Kaggle.ipynb` | Clean final notebook/report with balanced and precision-first winners |
| 9 | `tests/01_DL_PCA_Diagnostic_Test.ipynb` | Completed standalone diagnostic evidence |
| 10 | Planning/update phase | Updates notebooks 02-08 according to the revised pipeline |

## Revised Phase Summary

| # | Phase | Goal | Key Change |
|---|-------|------|------------|
| 1 | Data Preparation and EDA | Keep clean data and fixed split stable | No rerun unless schema/split is invalid |
| 2 | Feature Engineering | Produce ModernBERT raw 777 features and sequence-token metadata | ModernBERT is the text backbone for pooled embeddings; sequence track uses tokenizer/input ids |
| 3 | PCA / Feature Diagnostics | Keep PCA/SVD/Autoencoder as controlled diagnostic evidence | PCA is no longer default final input |
| 4 | Legacy DL Ablation | Retain PSO/PCA CNN-BiLSTM only as ablation evidence | This branch must not be framed as the final DL architecture |
| 5 | Model Zoo and Ensemble Search | Compare LGBM, XGB, MLP, sequence DL, and many ensemble types | Each model gets its own notebook; single model may win; ensemble is not forced |
| 6 | Robustness and XAI | Explain and stress-test selected winners | Use subset when RAM is tight |
| 7 | Evaluation and Ablation | Report metrics, recall-aware thresholds and ablations | Balanced and precision-first winners must be selected on validation |
| 8 | Delivery and Report | Present revised method and results cleanly | Old PCA-first narrative must be revised |
| 9 | DL PCA Diagnostic Test | Completed evidence used to justify redesign | Isolated from submission chain |
| 10 | Pipeline Redesign Execution Plan | Update notebooks and artifact contracts for the revised pipeline | Next GSD phase to plan/execute |

## Revised Model Matrix

| Track | Input | Candidates | Purpose |
|-------|-------|------------|---------|
| Tabular | ModernBERT pooled 768 + 9 behavioral = raw 777 | LightGBM, XGBoost, MLP | Main high-performing track |
| Sequence | ModernBERT tokenizer `input_ids`/`attention_mask` | CNN-BiLSTM-Attention, optional ModernBERT hidden-state variant | Fair DL sequence comparison |
| Diagnostic | PCA/SVD/Autoencoder variants | MLP PCA, LightGBM PCA, CNN-BiLSTM PCA legacy | Ablation and explanation only |
| Ensemble | Validation probabilities from candidates | soft voting, weighted blending, stacking, calibration | Determine if combining models beats the best single |

## Ensemble Search Contract

Phase 5 must evaluate:

- Single models in separate notebooks: LightGBM raw, XGBoost raw, MLP raw, CNN-BiLSTM sequence.
- Pair blends: LGBM+XGB, LGBM+MLP, XGB+MLP, LGBM+SequenceDL, MLP+SequenceDL.
- Triple/full blends where validation probability files exist.
- Weighted blending with grid or PSO weights where weights sum to 1.
- Stacking meta-models: Logistic/Ridge, RandomForest, and/or LightGBM meta if RAM allows.
- Calibration: Platt/sigmoid and isotonic when validation sample size is sufficient.

Every row must include `macro_f1`, `precision_fake`, `recall_fake`, `f1_fake`, `roc_auc`, `pr_auc`, `threshold`, confusion matrix values, seed, split, feature source, and runtime seconds.

## Critical Path

1. Keep Phase 1 split fixed.
2. Update Phase 2 to ensure ModernBERT raw 777 artifacts and sequence metadata are complete.
3. Keep Phase 3 diagnostic-only unless a specific experiment needs PCA/SVD.
4. Keep Phase 4 as legacy PSO/PCA ablation evidence only.
5. Update Phase 5 as a notebook family: one notebook per model, including MLP raw and sequence CNN-BiLSTM, then blending/stacking/orchestrator notebooks.
6. Update Phase 7 to choose balanced and precision-first winners based on validation, then report test once.
7. Update Phase 8 narrative and final diagrams after new results exist.

## Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| CNN-BiLSTM sequence branch is slower than tabular models | Colab runtime pressure | Use max_length 128/160, batch caching, early stopping and subset-first validation |
| Sequence DL underperforms LightGBM raw | Title novelty pressure | Report it honestly as fair sequence-track evidence; use ensemble only if validation proves value |
| Precision-first threshold hurts Recall Fake | Misleading high precision | Always report Recall Fake and F1 Fake beside Precision Fake |
| Too many ensemble combinations | Timeline bloat | Start with deterministic grid and prune combinations with weak validation probabilities |
| PCA narrative conflict | Report inconsistency | Reframe PCA as diagnostic/ablation, not mandatory final component |
| Local execution accidentally triggered | Violates project rule | Keep all heavy code in notebooks and run only in Colab with user action |
