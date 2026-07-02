# Tasklist

## Working Rules

- Do not run notebook cells, EDA, training, tuning, or dataset processing locally unless the project owner explicitly approves it.
- All executable pipeline source must stay inside the 8 phase notebooks under `notebooks/`.
- Do not create separate `.py` modules for the main ML pipeline.
- Preserve `data/final_labeled_fake_reviews.csv`; do not move or rewrite it without permission.
- Treat RAM 12GB as a hard design constraint.
- Every experiment must record seed, split, model variant, input source, threshold, metrics and artifact path.
- Precision Fake optimization must always report Recall Fake and F1 Fake.

## Revised Execution Plan

### Step 0: Lock Current Evidence

- [x] Phase 9 diagnostic confirms PCA hurts MLP raw and CNN-BiLSTM on static PCA vectors.
- [x] Current strong candidate identified: LightGBM raw no-PCA.
- [x] Current strong DL candidate identified: MLP raw 777.
- [ ] Mark old PCA-first CNN-BiLSTM path as legacy/ablation in notebooks and report.

### Step 1: Update Phase 2 Feature Contract

- [ ] Ensure `02_Feature_Engineering.ipynb` explicitly uses `answerdotai/ModernBERT-base` or documents fallback if unavailable.
- [ ] Save ModernBERT pooled embeddings for train/val/test.
- [ ] Save 9 behavioral features for train/val/test with leakage-safe metadata.
- [ ] Save raw fused 777 feature matrices for train/val/test.
- [ ] Save tokenizer config and sequence metadata needed by CNN-BiLSTM.
- [ ] Confirm artifact names are stable for Phase 4 and Phase 5.

### Step 2: Reframe Phase 3 as Diagnostic

- [ ] Update `03_PCA_Feature_Selection.ipynb` text to say PCA/SVD is diagnostic/ablation, not final default.
- [ ] Keep PCA/SVD outputs for controlled comparison.
- [ ] Optionally add Autoencoder diagnostic if Colab time allows.
- [ ] Export memory/time comparison: raw 777 vs PCA/SVD vs optional Autoencoder.

### Step 3: Reframe Phase 4 DL

- [x] Mark `04_PSO_Model_Training.ipynb` as legacy PSO/PCA ablation evidence.
- [x] Move revised MLP raw 777 candidate into `05_03_MLP_Raw.ipynb`.
- [x] Move revised CNN-BiLSTM sequence candidate into `05_04_CNN_BiLSTM_Sequence.ipynb`.
- [ ] Save validation/test probabilities for each revised DL candidate after Colab execution.
- [x] Keep legacy CNN-BiLSTM-PCA result only as ablation evidence.

### Step 4: Expand Phase 5 Model Zoo and Ensemble Search

- [x] Create Phase 5 notebook family while keeping `05_Hybrid_Ensemble.ipynb` as orchestrator/summary.
- [x] `05_00_Phase5_Run_Order.ipynb`: validate inputs and document run order.
- [x] `05_01_LightGBM_Raw.ipynb`: train/evaluate LightGBM raw 777.
- [x] `05_02_XGBoost_Raw.ipynb`: train/evaluate XGBoost raw 777.
- [x] `05_03_MLP_Raw.ipynb`: train/evaluate MLP raw 777.
- [x] `05_04_CNN_BiLSTM_Sequence.ipynb`: train/evaluate CNN-BiLSTM over real token sequences with behavioral late fusion.
- [x] `05_05_Weighted_Blending.ipynb`: run pair/triple/full weighted blends where probabilities exist.
- [x] `05_06_Stacking_Calibration.ipynb`: test stacking meta-models, calibration and threshold sweep.
- [x] `05_Hybrid_Ensemble.ipynb`: aggregate all Phase 5 outputs, produce final leaderboard and selected candidates.
- [ ] Export full sweep table with recall, not only precision.

### Step 5: Update Phase 6 Robustness/XAI Scope

- [ ] Select candidates from Phase 5: balanced winner, precision-first winner, and best DL branch.
- [ ] Run SHAP for tabular winner on subset.
- [ ] Run LIME or representative examples for text/sequence branch.
- [ ] Run robustness only where method is technically valid and resource-safe.

### Step 6: Update Phase 7 Evaluation/Ablation

- [x] Select winners on validation, then evaluate test once (`phase5_weighted_blend`, 2026-06-10).
- [x] Report Balanced winner: τ=0.30, test Macro F1 0.9463.
- [x] Report Precision-first winner: τ=0.60, Prec. Fake 0.9816.
- [x] Include single-model, ensemble, PCA diagnostic and sequence-vs-tabular comparisons (`phase7_ablation_results.csv`).
- [x] Export final metrics (`phase7_final_metrics.csv`, multi-seed summary).
- [x] Ablation: PCA not beneficial on fused vector (+0.0397 raw); behavioral +0.0008 only.

### Step 7: Update Phase 8 Final Report

- [x] Two-track narrative in `docs/`, `thesis/`, `Phase8_Final_Report.md`.
- [x] ModernBERT + behavioral + sequence documented.
- [x] Dual operating modes (balanced / precision-first) with SSOT numbers.
- [ ] Discuss why CNN-BiLSTM is evaluated on sequence input for fairness.
- [ ] Keep limitations honest if LightGBM raw beats sequence DL.

## GSD Command Plan

Recommended order:

1. Plan the redesign phase:
   - `$gsd-plan-phase 10`
2. Execute the redesign phase:
   - `$gsd-execute-phase 10`
3. Verify notebook source updates:
   - `$gsd-verify-work phase 10`
4. Then run Colab notebooks manually in this order:
   - `02_Feature_Engineering.ipynb`
   - `03_PCA_Feature_Selection.ipynb` only if diagnostic artifacts need refresh
   - `04_PSO_Model_Training.ipynb`
   - `05_Hybrid_Ensemble.ipynb`
   - `07_Evaluation_Ablation.ipynb`
   - `08_Final_Report_Kaggle.ipynb`
5. Verify phases after Colab outputs are synced:
   - `$gsd-verify-work phase 4`
   - `$gsd-verify-work phase 5`
   - `$gsd-verify-work phase 7`
   - `$gsd-verify-work phase 8`

## Definition of Done

- [ ] Phase 2 produces raw 777 ModernBERT+behavioral features and sequence metadata.
- [x] Phase 4 is clearly labeled legacy/ablation, while MLP raw and CNN-BiLSTM sequence live in Phase 5 notebooks.
- [ ] Phase 5 has one notebook per major model plus separate ensemble notebooks and can choose single winner if it wins.
- [ ] Phase 7 reports Balanced and Precision-first winners with recall-aware metrics.
- [ ] Phase 8 final report matches the revised pipeline and no longer overclaims PCA-first DL.
