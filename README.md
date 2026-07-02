# Fake Review Detection Research Project

**Tên đề tài (2026):** Phát hiện đánh giá giả trên Amazon bằng pipeline hai nhánh ModernBERT–đặc trưng hành vi–ensemble học sâu và GBDT với chính sách ngưỡng kép

**English:** Dual-Track ModernBERT, Behavioral Features, and Threshold-Selected Ensemble for Fake Review Detection on Amazon Labeled Fake Reviews

Dataset: 42,749 mẫu (sau clean) | Seed 42 | Split 70/15/15 | Colab 12GB RAM

## Kết quả chính (test, `phase5_weighted_blend`, seed 42)

| Chế độ | τ | Macro F1 | Precision Fake | ROC-AUC |
|--------|---|----------|----------------|---------|
| Default | 0.50 | 0.9433 | 0.9699 | 0.9769 |
| Balanced | 0.30 | **0.9463** | 0.9344 | 0.9769 |
| Precision-first | 0.60 | 0.9126 | **0.9816** | 0.9769 |

**Multi-seed (42, 123, 456):** balanced Macro F1 **0.9485 ± 0.0018**; precision-first Prec. Fake **0.9763 ± 0.0029** (`phase7_multiseed_summary.csv`).

Nguồn SSOT: `reports/tables/phase7_final_metrics.csv`, `phase7_multiseed_summary.csv`, `phase7_target_audit.csv` (2026-06-10). Overfit train: LGBM raw train F1=1.0; blend train ≈0.976 (`phase5_lgbm_raw_metrics.csv`).

## Start Here

1. Read `docs/08_Ket_Luan_va_Huong_Dan.md` — tổng kết và số liệu.
2. Read `docs/00_Literature_Review_SOTA.md` — SOTA và 20 papers.
3. Read `docs/01_Tong_Quan_Du_An.md` — tổng quan đầy đủ.
4. Run phase notebooks in `notebooks/` on Colab.

## Archive

Non-submission scratch files (AI agents, old planning UAT, PSO per-trial CSVs) live in `_archive/` — see `_archive/README.md`.

## Runtime Rule

Colab-first. Do not run training/tuning locally unless explicitly approved.

## Pipeline

```
01 → 02 → 03 (ablation) → 04 (ablation) → 05_* → 05_Hybrid → 06 → 07 → 08
```

Final track: raw 777-d + CNN-BiLSTM sequence + weighted blend (CNN 50% / XGB 35% / LGBM 15%).  
Ablation track: PCA 777→400 + PSO-tuned DL (legacy 0.8558 @0.5).