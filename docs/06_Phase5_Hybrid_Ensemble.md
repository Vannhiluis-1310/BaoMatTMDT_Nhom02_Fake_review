# Phase 5: Hybrid Ensemble (Pipeline 2026-06-10)

**Artifact Phase 5:** `phase5_weighted_blend_metrics.csv` (2026-06-10T09:37). **SSOT test audit:** `phase7_final_metrics.csv` (2026-06-10T09:43).

## 1. Kiến trúc pipeline chính

### 1.1. Hai track song song

| Track | Notebook | Input | Vai trò |
|-------|----------|-------|---------|
| Tabular | `05_01`–`05_03` | Raw 777 (ModernBERT 768 + 9 behavioral) | XGB, LGBM, MLP |
| Sequence | `05_04` | Token sequence + behavioral late fusion | CNN-BiLSTM-Attention |
| Ensemble | `05_05`, `05_06`, `05_Hybrid` | Saved probabilities | Weighted blend, stacking, threshold sweep |

**Lưu ý:** PCA+PSO (Phase 3–4) là **ablation track**, không phải đường chính báo cáo SOTA.

### 1.2. Weighted blend đã chọn (val grid)

```
CNN-BiLSTM sequence : 0.50
XGBoost raw         : 0.35
LightGBM raw        : 0.15
MLP                 : 0.00
```

Nguồn: `artifacts/ensemble/phase5_weighted_blend_metadata.json`

## 2. Kết quả test (SSOT: `phase7_final_metrics.csv`; base models: `phase5_leaderboard.csv` 2026-06-09)

| Model | τ | Macro F1 | Prec. Fake | Rec. Fake | ROC-AUC |
|-------|---|----------|------------|-----------|---------|
| **weighted_blend** | 0.50 | **0.9433** | 0.9699 | 0.8956 | 0.9769 |
| **weighted_blend** | 0.30 | **0.9463** | 0.9344 | 0.9390 | 0.9769 |
| **weighted_blend** | 0.60 | 0.9126 | **0.9816** | 0.8152 | 0.9769 |
| cnn_bilstm_sequence | 0.50 | 0.9324 | 0.9405 | 0.8982 | 0.9637 |
| xgb_raw | 0.50 | 0.9059 | 0.9686 | 0.8106 | 0.9531 |
| lgbm_raw | 0.50 | 0.9051 | 0.9677 | 0.8095 | 0.9548 |
| stacking_calibrated | 0.50 | 0.9105 | 0.9728 | 0.8175 | 0.9731 |

*Test @τ=0.3 và @0.6: tính từ `phase5_weighted_blend_test_prob.npy` (val-select, test audit).*

## 3. Chọn mô hình trên validation

| Chế độ | Model | τ (val) | Val Macro F1 | Val Prec. Fake | Target 0.975 |
|--------|-------|---------|--------------|----------------|--------------|
| Balanced | weighted_blend | 0.30 | 0.9468 | 0.9361 | Chưa (ưu tiên F1) |
| Precision-first | weighted_blend | 0.60 | 0.9098 | **0.9784** | **Đạt** |

Policy: `selection_policy` trong `phase5_metadata.json` — test chỉ audit sau khi chọn trên val.

## 4. So sánh với pipeline cũ (01/06)

| Pipeline | Test Macro F1 @0.5 | Ghi chú |
|----------|-------------------|---------|
| Legacy PCA+PSO+blend | 0.8558 | Ablation / negative result PCA |
| **Mới raw+sequence+blend** | **0.9433** | **+0.0875** |

## 5. Insights

- **Sequence track mạnh nhất:** CNN-BiLSTM 0.9324 — blend default +0.0109.
- **Blend ổn định ROC:** 0.9769 > mọi single model.
- **Dual reporting:** Macro F1 0.9463 (balanced) + Precision Fake 0.9816 (e-commerce); multi-seed mean 0.9485±0.0018 / 0.9763±0.0029.
- **Overfit train:** LGBM raw train Macro F1 = 1.0 (`phase5_lgbm_raw_metrics.csv`); blend train ≈0.976 (`phase7_final_metrics.csv`). Disclose trong limitations; weight LGBM chỉ 15%.

## 6. Handoff Phase 6 / 7 / 8

**Prerequisites trước khi chạy 06–08:**

- [x] `phase5_leaderboard.csv` (09/06)
- [x] `phase5_metadata.json` (09/06)
- [x] `phase5_weighted_blend_*_prob.npy` (train/val/test)
- [x] `phase5_selected_candidates_test_audit.csv` (2026-06-10)
- [x] Phase 6–8 đã chạy trên Colab (2026-06-10)

**Thứ tự Colab:**

```
05_00 → 05_01..05_06 → 05_Hybrid → 06_Adversarial_XAI → 07_Evaluation_Ablation → 08_Final_Report_Kaggle
```