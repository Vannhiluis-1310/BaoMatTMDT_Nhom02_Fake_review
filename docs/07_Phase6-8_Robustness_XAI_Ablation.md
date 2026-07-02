# Phase 6–8: Robustness, XAI, Ablation, Final Report

**Trạng thái:** ✅ Hoàn tất (2026-06-10). Phase 7/8 dùng pipeline `weighted_blend` final; Phase 6 notebook vẫn **legacy PCA ensemble** @τ=0.79.

**Kết quả Phase 5 final:** weighted_blend test Macro F1 **0.9433** @0.5, **0.9463** @0.3, Precision Fake **0.9816** @0.6, ROC-AUC **0.9769**. Multi-seed: balanced **0.9485 ± 0.0018**, precision-first Prec. Fake **0.9763 ± 0.0029**.

---

## 1. Phase 6: Adversarial Robustness & XAI

**Artifact:** `artifacts/xai/phase6_metadata.json` (`generated_at_utc` 2026-06-10T12:54:11)

> **Footnote:** Phase 6 load `final_ensemble_model.pkl` (blend XGB 50% / LGBM 50%, τ=0.79, PCA 400-d) — **không phải** `phase5_weighted_blend`. Số robustness/XAI minh họa legacy track; headline báo cáo lấy từ Phase 5/7.

### 1.1. Thiết kế

- FGSM/PGD trên **PCA feature space** (subset test n=1.000, stratified)
- Surrogate attack model: `dl_pso`
- SHAP global importance trên LightGBM (PCA components, n=500)
- LIME local cases (n=6)

### 1.2. Robustness (`phase6_robustness_metrics.csv`)

| Condition | Model | τ | Macro F1 | Prec. Fake |
|-----------|-------|---|----------|------------|
| clean | final_ensemble (legacy) | 0.79 | 0.8000 | 0.9603 |
| clean | dl_pso | 0.50 | 0.7724 | 0.7644 |
| FGSM ε=0.03 | final_ensemble (legacy) | 0.79 | 0.7949 | 0.9597 |
| FGSM ε=0.03 | dl_pso | 0.50 | 0.7068 | 0.6702 |
| FGSM ε=0.05 | final_ensemble (legacy) | 0.79 | 0.7923 | 0.9593 |
| FGSM ε=0.05 | dl_pso | 0.50 | 0.6737 | 0.6196 |

**Metric drops (FGSM ε=0.03, `phase6_robustness_metric_drops.csv`):**

| Model | Δ Macro F1 | Δ Prec. Fake |
|-------|------------|--------------|
| final_ensemble (legacy) | −0.0052 | −0.0006 |
| dl_pso | −0.0656 | −0.0942 |

Legacy ensemble ổn định hơn DL-PSO đơn lẻ trước nhiễu FGSM; cần rerun Phase 6 trên `weighted_blend` để khớp final track.

### 1.3. XAI (`phase6_shap_global_importance.csv`)

Top SHAP (PCA space): `pca_000` (1.2295), `pca_005` (0.2738), `pca_147` (0.2704). LIME: 6 case HTML trong `artifacts/xai/lime_case_*.html`.

---

## 2. Phase 7: Evaluation & Ablation

**Artifact:** `artifacts/evaluation/phase7_metadata.json` (`generated_at_utc` 2026-06-10T09:54:50)

### 2.1. Target audit (`phase7_target_audit.csv`)

| Chế độ | τ | Metric | Actual | Target | Pass |
|--------|---|--------|--------|--------|------|
| default | 0.50 | macro_f1 | 0.9433 | 0.89 | ✅ |
| default | 0.50 | precision_fake | 0.9699 | 0.975 | ❌ |
| default | 0.50 | roc_auc | 0.9769 | 0.93 | ✅ |
| balanced | 0.30 | macro_f1 | **0.9463** | 0.89 | ✅ |
| balanced | 0.30 | precision_fake | 0.9344 | 0.975 | ❌ |
| balanced | 0.30 | roc_auc | 0.9769 | 0.93 | ✅ |
| precision-first | 0.60 | macro_f1 | 0.9126 | 0.89 | ✅ |
| precision-first | 0.60 | precision_fake | **0.9816** | 0.975 | ✅ |
| precision-first | 0.60 | roc_auc | 0.9769 | 0.93 | ✅ |

**Headline:** precision-first @τ=0.6 pass cả 3 target.

### 2.2. Ablation (`phase7_ablation_results.csv`)

| Variant | Test Macro F1 @0.5 | Ghi chú |
|---------|-------------------|---------|
| Full weighted_blend | **0.9433** | Final |
| Raw LGBM (no PCA) | 0.9058 | Δ +0.0397 vs PCA LGBM |
| No advanced behavioral | 0.8670 | Δ +0.0008 vs PCA+9 feat. ref. (`phase7_ablation_controlled_reference.csv`) |
| DL-PSO single | 0.7793 | Không ensemble |
| DL baseline (no PSO) | 0.7665 | Branch-level |
| Legacy PCA+PSO blend | 0.8558 | Ablation 01/06 |

### 2.3. 5-fold CV surrogate (`phase7_cv_summary.csv`)

LightGBM PCA: Macro F1 mean **0.8659** ±0.0036, ROC-AUC **0.9233** ±0.0041 (5 folds, 36.336 rows).

---

## 3. Phase 8: Final Report

**Deliverables (đã có):**

```
reports/final/Phase8_Final_Report.md
reports/final/phase8_artifact_inventory.csv
reports/final/phase8_run_order_checklist.csv
reports/final/phase8_submission_package_manifest.csv
```

Phase 8 đọc thêm: `phase7_final_metrics.csv` (SSOT headline), `literature_sota_comparison.csv`, `phase7_target_audit.csv`, `phase7_multiseed_summary.csv`. `phase5_leaderboard.csv` chỉ dùng cho base-model comparison (không thay SSOT blend).

---

## 4. Checklist pipeline 6 → 7 → 8

- [x] Phase 5 leaderboard `2026-06-09`
- [x] `phase5_weighted_blend_test_prob.npy` tồn tại
- [x] Literature CSV + `00_Literature_Review_SOTA.md`
- [x] `phase5_selected_candidates_test_audit.csv` (2026-06-10)
- [x] Chạy `06_Adversarial_XAI.ipynb` (2026-06-10, legacy model)
- [x] Chạy `07_Evaluation_Ablation.ipynb` (2026-06-10)
- [x] Chạy `08_Final_Report_Kaggle.ipynb` (2026-06-10)