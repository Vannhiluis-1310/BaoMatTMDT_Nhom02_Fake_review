# Tổng kết và Hướng dẫn — Fake Review Detection (2026-06-10)

## 1. Cấu trúc tài liệu

| File | Nội dung |
|------|----------|
| `docs/00_Literature_Review_SOTA.md` | 20 papers, bảng SOTA 3-tier, 8 research gaps |
| `docs/01_Tong_Quan_Du_An.md` | Tổng quan + abstract cập nhật |
| `docs/02`–`05` | Phase methodology |
| `docs/06_Phase5_Hybrid_Ensemble.md` | Kết quả pipeline 09/06 |
| `docs/07_Phase6-8_*.md` | Robustness, ablation, delivery |
| `docs/08_Ket_Luan_va_Huong_Dan.md` | File này |

## 2. Số liệu chính (pipeline 2026-06-10, `phase7_final_metrics.csv`)

### Dataset

- Input: 50,000 → **42,749** sau clean
- Split: 29,923 / 6,413 / 6,413 (70/15/15, seed 42)
- Class sau clean: ~59% real / ~41% fake

### Final model: weighted_blend

| Chế độ | τ | Test Macro F1 | Prec. Fake | ROC-AUC | Target |
|--------|---|---------------|------------|---------|--------|
| Default | 0.50 | **0.9433** | 0.9699 | 0.9769 | F1 ≥0.89 ✓ |
| Balanced | 0.30 | **0.9463** | 0.9344 | 0.9769 | Headline SOTA |
| Precision-first | 0.60 | 0.9126 | **0.9816** | 0.9769 | Prec ≥0.975 ✓ |

**Multi-seed (42, 123, 456):** balanced Macro F1 **0.9485 ± 0.0018**; precision-first Prec. Fake **0.9763 ± 0.0029** (`phase7_multiseed_summary.csv`).

### Target audit (pipeline mới)

| Metric | Target | Actual (best) | Pass |
|--------|--------|---------------|------|
| macro_f1 | ≥0.890 | **0.9463** @τ=0.3 | ✓ |
| precision_fake | ≥0.975 | **0.9816** @τ=0.6 | ✓ |
| roc_auc | ≥0.930 | **0.9769** | ✓ |

### Legacy ablation (PCA+PSO, 01/06)

- Macro F1 0.8558 @0.5 — giữ làm evidence PCA track thua raw.

## 3. Hướng dẫn chạy Colab

### 3.1. Thứ tự đầy đủ

```
01_EDA_Preprocessing.ipynb
02_Feature_Engineering.ipynb
03_PCA_Feature_Selection.ipynb          # diagnostic / ablation
04_PSO_Model_Training.ipynb            # ablation track
05_00_Phase5_Run_Order.ipynb
05_01_LightGBM_Raw.ipynb
05_02_XGBoost_Raw.ipynb
05_03_MLP_Raw.ipynb
05_04_CNN_BiLSTM_Sequence.ipynb
05_05_Weighted_Blending.ipynb
05_06_Stacking_Calibration.ipynb
05_Hybrid_Ensemble.ipynb
06_Adversarial_XAI.ipynb               # ✅ đã chạy 2026-06-10 (legacy model)
07_Evaluation_Ablation.ipynb           # ✅ đã chạy 2026-06-10
10_Baseline_Algorithm_Benchmark.ipynb  # Tier-1 fair benchmark (>=5 thuật toán)
09_Bieu_Do_So_Sanh_Mo_Hinh.ipynb       # biểu đồ từ baseline_benchmark_metrics.csv
08_Final_Report_Kaggle.ipynb           # ✅ đã chạy 2026-06-10
```

## 7. Pipeline Phase 6 → 7 → 8 — hoàn tất (2026-06-10)

**Đã chạy trên Colab:**

- [x] `06_Adversarial_XAI.ipynb` — FGSM/PGD + SHAP/LIME (legacy PCA ensemble @τ=0.79)
- [x] `07_Evaluation_Ablation.ipynb` — ablation, CV, target audit (`phase7_metadata.json` 2026-06-10)
- [x] `08_Final_Report_Kaggle.ipynb` — `Phase8_Final_Report.md` + manifest

**Kiểm định:** `phase7_target_audit.csv` — precision-first @τ=0.6 pass cả 3 target (macro_f1, precision_fake, roc_auc).

**Việc tùy chọn còn lại:** Rerun Phase 6 trên `phase5_weighted_blend` để robustness/XAI khớp final track; refresh `05_Hybrid` nếu cần metadata mới.

### 3.2. Artifacts bắt buộc

```
artifacts/ensemble/phase5_metadata.json
artifacts/predictions/phase5_weighted_blend_*_prob.npy
reports/tables/phase7_final_metrics.csv
reports/tables/phase7_multiseed_summary.csv
reports/tables/phase5_leaderboard.csv
reports/tables/literature_sota_comparison.csv
reports/tables/literature_references_20.csv
```

## 4. Điểm mạnh

1. **Vượt SOTA Tier A** trên Amazon text classification (xem `00_Literature_Review_SOTA.md`)
2. **Dual-mode deployment:** balanced F1 + precision-first e-commerce
3. **Reproducible:** seed 42, train-only fit, metadata JSON
4. **8 research gaps** từ 20 papers đã kiểm chứng

## 5. Hạn chế (trung thực)

1. LGBM **raw** train Macro F1 = 1.0 (`phase5_lgbm_raw_metrics.csv`; test 0.9051); blend train Macro F1 ≈0.976 (`phase7_final_metrics.csv`; test 0.9433–0.9463). Val–test gap ≈0.0005.
2. Phase 6 robustness/XAI trên **legacy PCA ensemble** — chưa rerun trên `weighted_blend`
3. Multi-seed đã thực hiện; seed 123 precision-first đơn lẻ (0.9728) hơi dưới 0.975
4. Graph/multimodal SOTA khác setup — không so sánh trực tiếp

## 6. Tài liệu tham khảo

- `docs/00_Literature_Review_SOTA.md`
- `reports/tables/literature_references_20.csv`
- `reports/tables/phase7_final_metrics.csv
reports/tables/phase7_multiseed_summary.csv
reports/tables/phase5_leaderboard.csv`