---
created: 2026-06-01T10:44:15.3093684+07:00
title: Đổi BERT sang ModernBERT
status: completed
resolved: 2026-06-10
area: feature-engineering
files:
  - notebooks/02_Feature_Engineering.ipynb
  - notebooks/05_* (Phase 5 family)
---

## Resolution (2026-06-10)

ModernBERT pipeline đã chạy trên Colab. Final track `phase5_weighted_blend` đạt target trên test (`phase7_final_metrics.csv`):

- Balanced Macro F1 **0.9463** @τ=0.3 (≥0.89 ✓)
- Precision-first Prec. Fake **0.9816** @τ=0.6 (≥0.975 ✓)
- ROC-AUC **0.9769** (≥0.93 ✓)

## Original Problem (historical)

Phase 4/5 legacy PCA pipeline chưa đạt target (Macro F1 ~0.85). Đổi sang ModernBERT + raw 777-d + sequence track là một trong các bước redesign (Phase 10).

## Original Solution

Đổi `BERT_MODEL_NAME` → `answerdotai/ModernBERT-base` trong Phase 2, rerun pipeline. Đã thực hiện.