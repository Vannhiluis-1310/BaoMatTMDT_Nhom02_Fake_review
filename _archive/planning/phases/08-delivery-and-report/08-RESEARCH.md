# Phase 8 Research Notes

**Date:** 2026-06-01
**Mode:** Local artifact review only

## Findings

Phase 8 should be a delivery/report consolidation phase. The verified Phase 7 artifacts already contain the final metrics, target audit, CV evidence, ablation results, error analysis and report highlights needed for the final notebook and report.

The existing `notebooks/08_Final_Report_Kaggle.ipynb` is currently a stub. It needs to become a clean Colab notebook that reads project artifacts and writes final report/package files.

An older `reports/final/Phase9_Final_Report.md` exists, but it contains stale phase naming and stale experiment facts. It should not be treated as a source of truth. At most, it can inform structure or wording after all numbers are corrected against Phase 7 artifacts.

## Source of Truth

| Topic | Preferred Source | Notes |
|-------|------------------|-------|
| Final metrics | `reports/tables/phase7_final_metrics.csv` | Use current Phase 7 numbers. |
| Target gaps | `reports/tables/phase7_target_audit.csv` | Must be visible in report. |
| Selected final configuration | `artifacts/evaluation/phase7_metadata.json` | Current selected candidate and limitations. |
| CV evidence | `phase7_cv_metrics.csv`, `phase7_cv_summary.csv` | Full 5-fold CV evidence from Phase 7. |
| Ablation | `phase7_ablation_results.csv`, `phase7_ablation_delta.csv` | Include evidence type and caveats. |
| Robustness/XAI | Phase 6 tables and metadata | Can be summarized, not recomputed. |
| PCA/RAM story | Phase 3 metadata and Phase 7 Model B caveat | Avoid overclaiming PCA as metric-superior. |

## Report Shape

The final report should include:

1. Abstract / executive summary.
2. Problem statement and dataset.
3. Pipeline overview.
4. Feature engineering:
   - transformer embeddings
   - 9 behavioral features
   - leakage controls
5. PCA/SVD feature selection and RAM constraint handling.
6. PSO-tuned CNN-BiLSTM-Attention model.
7. XGBoost/LightGBM and stacking/blending ensemble.
8. Robustness and XAI.
9. Final metrics and target audit.
10. 5-fold CV evidence.
11. Ablation study with Model A-E.
12. Error analysis.
13. Limitations and future work.
14. Reproducibility and run order.

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Stale final report numbers are reused | Notebook must load current Phase 7 CSV/JSON and generate report from them. |
| Report implies targets were reached | Include explicit target audit and target gap statements. |
| PCA is overclaimed | Include Model B caveat and frame PCA as RAM/pipeline stabilizer. |
| Notebook becomes too heavy | Use precomputed artifacts only; no retraining or recomputing SHAP/LIME/CV. |
| Artifact folder still contains old files | Generate manifest and mark stale/optional files instead of deleting automatically. |
| Requirements for Phase 6 are stale in `REQUIREMENTS.md` | Phase 8 report can reference verified Phase 6 artifacts; requirement status cleanup can be handled separately if desired. |

## Recommendation

Use one implementation plan. Phase 8 edits a single final notebook and final report outputs, so parallel plans would create unnecessary conflicts.
