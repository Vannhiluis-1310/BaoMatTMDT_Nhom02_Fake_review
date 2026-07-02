# Phase 8: Delivery and Report - Context

**Gathered:** 2026-06-01
**Status:** Ready for planning
**Source:** Roadmap + Phase 7 verified artifacts

<domain>

## Phase Boundary

Phase 8 builds the final delivery layer for the fake review detection project. It should turn `notebooks/08_Final_Report_Kaggle.ipynb` into a clean Colab-facing report notebook and produce final report/package artifacts from already generated Phase 1-7 outputs.

This phase is not a training, tuning, feature extraction, PCA, CV or ablation phase. It should read existing small CSV/JSON/table/figure artifacts and summarize them for submission.

</domain>

<decisions>

## Implementation Decisions

### Runtime

- Run on Google Colab only when the user chooses to execute the notebook.
- Do not run notebook cells, EDA, feature extraction, PCA, training, tuning, CV, ablation or dataset processing locally.
- Keep executable delivery/report source inside `notebooks/08_Final_Report_Kaggle.ipynb`.
- Phase 8 local verification is limited to static file/notebook checks and small artifact metadata reads.

### Evidence Policy (updated 2026-06-10)

- SSOT: `reports/tables/phase7_final_metrics.csv`, `phase7_multiseed_summary.csv`, `phase7_target_audit.csv`.
- **Final track** `phase5_weighted_blend`: balanced Macro F1 **0.9463**, precision-first Prec. Fake **0.9816**, ROC-AUC **0.9769** — all original targets pass in the appropriate mode.
- Disclose: LGBM raw train F1=1.0; blend train ≈0.976; Phase 6 on legacy PCA ensemble @τ=0.79.
- Legacy PCA metrics (0.8558) may appear as ablation/history — never as headline.
- Controlled ablation: raw +0.0397 vs PCA; advanced behavioral +0.0008 (not +0.0023).

### Report Policy

- Treat `reports/final/Phase9_Final_Report.md` as stale reference material only.
- The final report should be regenerated as `reports/final/Phase8_Final_Report.md` using current Phase 7 tables/metadata.
- Remove or avoid stale claims such as old selected blends, old threshold values and "target achieved" language.
- Include novelty, architecture, feature engineering, PCA/RAM story, PSO, ensemble, robustness, XAI, ablation, limitations and future work.

### Delivery Policy

- Produce a clear artifact inventory and submission manifest.
- Keep final outputs under `reports/final/`.
- Do not delete old artifacts during execution unless the user explicitly asks.
- Do not move or rewrite `data/final_labeled_fake_reviews.csv`.

</decisions>

<canonical_refs>

## Canonical References

### Planning

- `.planning/ROADMAP.md` - Phase 8 goal, tasks and success criteria.
- `.planning/REQUIREMENTS.md` - DEL-01..DEL-04.
- `.planning/STATE.md` - verified metric gaps and current workspace state.
- `.planning/phases/07-evaluation-and-ablation-study/07-VERIFICATION.md` - latest verified Phase 7 status.

### Inputs

- `notebooks/08_Final_Report_Kaggle.ipynb` - target notebook.
- `reports/tables/phase7_final_metrics.csv` - final metrics source.
- `reports/tables/phase7_target_audit.csv` - target pass/fail and gap source.
- `reports/tables/phase7_cv_metrics.csv` and `phase7_cv_summary.csv` - CV evidence.
- `reports/tables/phase7_ablation_results.csv` and `phase7_ablation_delta.csv` - ablation evidence.
- `reports/tables/phase7_error_analysis.csv` - error analysis.
- `reports/tables/phase7_report_highlights.csv` - report-ready highlights.
- `artifacts/evaluation/phase7_metadata.json` - selected final configuration and limitations.
- `reports/tables/phase6_robustness_metrics.csv` and `artifacts/xai/phase6_metadata.json` - robustness/XAI pointers.
- `artifacts/pca/phase3_metadata.json` - PCA dimensions, variance and RAM notes.
- `artifacts/features/feature_metadata.json` and `feature_dictionary.csv` - feature engineering summary.

</canonical_refs>

<specifics>

## Specific Ideas

- Notebook 08 should be organized as a final narrative notebook:
  - Title and project contract.
  - How to run on Colab.
  - Artifact validation and inventory.
  - Dataset/features/PCA summary.
  - Architecture and training summary.
  - Final metrics and target gaps.
  - Ablation study.
  - Robustness and XAI.
  - Error analysis.
  - Final limitations and future work.
  - Submission checklist.
- Use precomputed figures from `reports/figures/` when available instead of recomputing heavy experiments.
- Save final delivery artifacts:
  - `reports/final/Phase8_Final_Report.md`
  - `reports/final/phase8_artifact_inventory.csv`
  - `reports/final/phase8_run_order_checklist.csv`
  - `reports/final/phase8_submission_package_manifest.csv`
  - `reports/final/phase8_report_summary.csv`

</specifics>

<deferred>

## Deferred Ideas

- PDF export is optional and deferred unless explicitly requested.
- ModernBERT rerun and Phase 2-5 retraining are pending todos, not Phase 8 delivery work.
- Production API, dashboard and external dataset validation are out of scope for Phase 8.

</deferred>
