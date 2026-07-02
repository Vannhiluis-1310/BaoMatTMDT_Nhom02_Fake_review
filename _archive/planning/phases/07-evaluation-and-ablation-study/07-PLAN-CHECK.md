# Phase 7 Plan Check

**Verdict:** VERIFICATION PASSED

## Requirement Coverage

| Requirement | Covered By | Notes |
|-------------|------------|-------|
| EVAL-01 | Task 2 | Final metrics include Macro F1, Precision Fake, Recall Fake, F1 Fake, ROC-AUC, PR-AUC and confusion matrix values. |
| EVAL-02 | Task 3 | Resource-aware CV with explicit fallback policy. |
| EVAL-03 | Task 4 | Full, A, B, C, D and E ablation variants are planned. |
| EVAL-04 | Tasks 2-5 | Metrics tables and figures are exported for report use. |
| EVAL-05 | Task 6 | Metadata records seed, split, configs, versions and fallback decisions. |

## Project Constraint Check

| Constraint | Status |
|------------|--------|
| Colab-only execution | pass |
| No local training/evaluation | pass |
| Executable source stays inside `notebooks/07_Evaluation_Ablation.ipynb` | pass |
| RAM 12GB respected | pass, via CV/ablation sampling fallback |
| No claims that target was reached if missed | pass |

## Risk Review

| Risk | Handling |
|------|----------|
| Full hybrid 5-fold CV too expensive | Use LightGBM PCA CV and record as resource-aware CV. |
| Model B/C not exact full-hybrid ablations | Label as `controlled_lightgbm`, not direct full-hybrid deltas. |
| Phase 5 target metrics still missed | Target audit makes gaps explicit. |
| Selected ensemble may have low/no DL weight | Model A includes branch-level PSO evidence and records blend limitation. |
| Error analysis cannot join review text | Export row indices/probs/labels and record limitation. |

## Execution Shape

One plan only. Phase 7 modifies a single notebook, so multiple executor agents would risk JSON conflicts.
