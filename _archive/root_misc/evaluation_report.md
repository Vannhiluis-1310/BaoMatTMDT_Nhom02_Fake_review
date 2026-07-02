# Evaluation Report: Dual-Track ModernBERT, Behavioral Features, and Threshold-Selected Ensemble for Fake Review Detection

## Điểm mạnh (Strengths)
1. **Kiến trúc đột phá (Dual-Track Hybrid Model)**: Việc kết hợp nhánh Tabular (ModernBERT 768-d + 9 Behavioral Features) và nhánh Sequence (CNN-BiLSTM) mang tính toàn diện.
2. **Hiệu suất ấn tượng (Vượt SOTA Tier A)**: Macro F1 **0.9463** (balanced), Precision Fake **0.9816** (precision-first), ROC-AUC **0.9769**. Multi-seed: balanced **0.9485 ± 0.0018**, prec.-first **0.9763 ± 0.0029**. FPR precision-first **1,06%** (40/3.789).
3. **Luận chứng khoa học chặt chẽ (Ablation Study)**: Định lượng PCA vs Raw, behavioral, ensemble; negative result PCA được báo cáo trung thực.
4. **Reproducibility**: 8 Phase, seed 42, fit train-only, metadata Phase 5/7/8.

## Điểm yếu (Weaknesses)
1. **Overfitting train** (`phase5_lgbm_raw_metrics.csv`, `phase7_final_metrics.csv`): LightGBM **raw** train Macro F1 = 1.0 (test 0.9051); weighted_blend train Macro F1 ≈ 0.976 (test 0.9433–0.9463). Val–test gap ≈ 0.0005.
2. **Phase 6 lệch pha**: FGSM/PGD + SHAP/LIME trên legacy PCA ensemble @τ=0,79 — chưa `weighted_blend`.
3. **Chưa cross-dataset**: Multi-seed đã có; seed 123 prec.-first đơn lẻ 0,9728 hơi dưới 0,975.

## Chấm điểm từng phần
- **Methodology**: 9.5/10
- **Architecture**: 9.0/10
- **Results & SOTA**: 9.5/10
- **Code/Notebook**: 8.5/10

## Đề xuất
1. Rerun Phase 6 trên `weighted_blend`
2. Regularization LightGBM/XGBoost
3. Kiểm định thống kê formal (bootstrap/paired test) trên multi-seed đã có; cross-dataset