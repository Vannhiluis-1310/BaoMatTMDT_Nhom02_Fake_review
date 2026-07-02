# State

## Project Reference

See: `.planning/PROJECT.md` (updated 2026-06-10)

**Core value:** Chứng minh mô hình hybrid cải thiện phát hiện fake review một cách có kiểm chứng bằng metrics, ablation và giải thích mô hình, trong giới hạn RAM 12GB và bộ notebook Colab theo phase.

## Current Focus

**Final track complete (2026-06-10).** Headline metrics audited in Phase 7/8. Phase 6 robustness/XAI still on legacy PCA ensemble — disclosed, not rerun on `weighted_blend`.

## Final Track Results (SSOT)

**Source:** `reports/tables/phase7_final_metrics.csv`, `phase7_multiseed_summary.csv`, `phase7_target_audit.csv` (seed 42, test n=6.413, generated 2026-06-10).

| Mode | τ | Macro F1 | Prec. Fake | Rec. Fake | ROC-AUC |
|------|---|----------|------------|-----------|---------|
| Default | 0.50 | **0.9433** | 0.9699 | 0.8956 | 0.9769 |
| Balanced | 0.30 | **0.9463** | 0.9344 | 0.9390 | 0.9769 |
| Precision-first | 0.60 | 0.9126 | **0.9816** | 0.8152 | 0.9769 |

**Multi-seed (42, 123, 456):** balanced Macro F1 **0.9485 ± 0.0018**; precision-first Prec. Fake **0.9763 ± 0.0029**.

**Target audit:** All three original targets pass under the appropriate mode (`phase7_target_audit.csv`).

**Model:** `phase5_weighted_blend` — CNN-BiLSTM 50% / XGBoost 35% / LightGBM 15%.

**Overfit (disclosed):** LGBM raw train Macro F1 = 1.0 (`phase5_lgbm_raw_metrics.csv`); blend train ≈0.976; val–test gap ≈0.0005.

**Ablation highlights** (`phase7_ablation_results.csv`): raw LGBM +0.0397 vs PCA ref.; ensemble +0.164 vs DL-PSO; advanced behavioral +0.0008 (controlled).

## Legacy PCA Track (historical, ablation only)

| Metric @τ=0.5 | Value | Notes |
|---------------|-------|-------|
| Legacy blend Macro F1 | 0.8558 | Missed F1 target; superseded |
| DL-PSO test | 0.7793 | `phase7_final_metrics.csv` |
| Phase 6 clean legacy ensemble F1 | 0.8000 | @τ=0.79, subset n=1000 |

Phase 4–8 UAT/VERIFICATION files dated ~2026-06-01 document this legacy state. Do not use as headline.

## Phase Status

| Phase | Status | Notes |
|-------|--------|-------|
| 1. Data Preparation and EDA | Complete | Artifacts under `artifacts/` |
| 2. Feature Engineering | Complete | ModernBERT 768 + 9 behavioral → 777-d |
| 3. PCA Feature Selection | Complete (ablation) | Diagnostic/ablation, not final input |
| 4. PSO-Tuned CNN-BiLSTM | Complete (ablation) | DL-PSO test 0.7793; +0.0128 vs baseline 0.7665 |
| 5. Hybrid Ensemble | **Complete (final)** | `weighted_blend`; Colab outputs synced 2026-06-10 |
| 6. Adversarial Robustness and XAI | Complete (legacy scope) | PCA ensemble @τ=0.79; not final blend |
| 7. Evaluation and Ablation | **Complete** | Phase 7 audit 2026-06-10; targets pass on final track |
| 8. Delivery and Report | **Complete** | `reports/final/Phase8_Final_Report.md` synced |
| 9. DL PCA Diagnostic Test | Complete | Raw MLP 0.8930; PCA hurts DL |
| 10. Pipeline Redesign | **Complete (runtime)** | Phase 5 notebook family run on Colab 2026-06-09/10 |

## Known Workspace State

- Dataset: `data/final_labeled_fake_reviews.csv` (42,749 after clean).
- Git repository was not initialized by agent setup.
- Phase 5–8 Colab outputs synced; SSOT = `phase7_final_metrics.csv`.
- Phase 6 `phase6_metadata.json` generated 2026-06-10T12:54:11 (legacy model).
- Phase 7 `phase7_metadata.json` generated 2026-06-10T09:54:50.
- Workspace cleanup 2026-06-11: non-submission files moved to `_archive/` (see `_archive/README.md`).

## Phase 10 Benchmark (2026-06-18)

**Notebook:** `notebooks/10_Baseline_Algorithm_Benchmark.ipynb` — Option C Tier-1 (6 models @ τ=0.5).

| Model | Macro F1 | Nguồn |
|-------|----------|-------|
| Logistic Regression | 0.8734 | train sklearn |
| Linear SVM | 0.8741 | train sklearn |
| Random Forest | 0.8906 | train sklearn |
| XGBoost raw | 0.9059 | Phase 7 artifact |
| CNN-BiLSTM | 0.9343 | Phase 7 artifact |
| **Weighted blend** | **0.9433** | Phase 7 artifact |

SSOT: `reports/tables/baseline_benchmark_metrics.csv`. Literature Tier-2: `baseline_benchmark_literature_tier2.csv`.

## Next Actions

1. Optional: rerun Phase 6 on `weighted_blend` (raw/sequence feature space).
2. Optional: formal statistical tests on multi-seed results.
3. Optional: cross-dataset validation (Yelp).
4. Thesis submission: use `thesis/` + `docs/08` + `phase7_*` + `baseline_benchmark_*` CSVs as evidence.

## Verification

- Final SSOT: `reports/tables/phase7_final_metrics.csv`
- Submission manifest: `reports/final/phase8_submission_package_manifest.csv`
- Historical planning/UAT (legacy PCA snapshot): `_archive/planning/phases/`

## Open Questions

- Rerun Phase 6 on final pipeline before claiming robustness for `weighted_blend`.
- Fine-tune ModernBERT (LoRA) if timeline allows.