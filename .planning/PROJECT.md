# ModernBERT Raw-Feature + Sequence-Track Fake Review Detection

## What This Is

Dự án nghiên cứu xây dựng bộ 8 notebook Colab theo phase để phát hiện fake review trên Amazon Labeled Fake Reviews Dataset khoảng 50.000 mẫu. Pipeline hiện tại được thiết kế lại sau Phase 9 diagnostic: dùng ModernBERT pooled embeddings + 9 behavioral features thành raw fused 777 features cho LightGBM/XGBoost/MLP, đồng thời đánh giá CNN-BiLSTM-Attention trên sequence input thật để so sánh công bằng.

Đối tượng sử dụng là người làm báo cáo/nghiên cứu bảo mật thương mại điện tử, cần kết quả có thể tái lập, chạy được trong giới hạn RAM 12GB, và chứng minh rõ novelty bằng ablation study.

## Core Value

Chứng minh mô hình hybrid cải thiện phát hiện fake review một cách có kiểm chứng bằng metrics, ablation và giải thích mô hình, trong giới hạn RAM 12GB và bộ notebook Colab theo phase.

## Requirements

### Validated

- [x] Final track `phase5_weighted_blend` audited Phase 7 (2026-06-10): Macro F1 0.9463 @τ=0.3, Prec. Fake 0.9816 @τ=0.6, ROC-AUC 0.9769.
- [x] Multi-seed stability (42, 123, 456): balanced 0.9485±0.0018.
- [x] Ablation controlled: raw +0.0397 vs PCA; ensemble +0.164 vs DL-PSO.
- [x] Phase 6–8 notebooks run; reports synced (`Phase8_Final_Report.md`).

### Active

- [ ] Dataset được EDA, làm sạch missing/duplicated và chia train/validation/test stratified.
- [ ] Toàn bộ source code thực thi nằm trong các file `.ipynb` chạy trên Google Colab, chia theo 8 phase.
- [ ] BERT embeddings được trích xuất theo batch, có cache ra artifact để tránh chạy lại tốn RAM/thời gian.
- [ ] 9 behavioral features được tạo rõ ràng, gồm 5 feature cơ bản và 4 feature nâng cao: review velocity, burst pattern, reviewer embedding, time gap.
- [ ] PCA hoặc TruncatedSVD được giữ như diagnostic/ablation để đo trade-off RAM/time/metric, không mặc định là final input.
- [ ] DL được đánh giá theo hai hướng công bằng: MLP trên raw 777 tabular features và CNN-BiLSTM-Attention trên token sequence thật với behavioral late fusion.
- [ ] PSO hoặc grid/validation tuning được dùng có kiểm soát cho hyperparameters/ensemble weights nếu thật sự tăng validation metrics.
- [ ] XGBoost/LightGBM/MLP/sequence DL được đưa vào model zoo; ensemble được chọn chỉ khi thắng single model trên validation.
- [ ] FGSM/PGD robustness và SHAP/LIME được chạy trên subset kiểm soát RAM.
- [ ] Đánh giá đầy đủ Macro F1, Precision Fake, Recall Fake, ROC-AUC, PR-AUC, confusion matrix, threshold analysis.
- [ ] Ablation study chứng minh đóng góp của PSO, PCA, behavioral nâng cao và ensemble.
- [ ] Notebook và báo cáo cuối có bảng kết quả, biểu đồ, mô tả novelty, giới hạn và hướng phát triển.

### Out of Scope

- Production web app/API realtime - không cần cho mục tiêu nghiên cứu 18-22 ngày.
- Tự động crawl Amazon hoặc tự gán nhãn dữ liệu mới - dataset đã được xác định.
- Chạy training/EDA local khi chưa có sự đồng ý của chủ dự án - ràng buộc đã nêu.
- Tách source thành package `.py` riêng - yêu cầu toàn bộ source pipeline nằm trong các notebook `.ipynb`.
- Fine-tune nhiều backbone transformer lớn như RoBERTa-large/DeBERTa-large - dễ vượt RAM 12GB và lệch timeline.
- Ép CNN-BiLSTM đọc PCA/static vector làm final path - Phase 9 cho thấy đây là mismatch kiến trúc.

## Context

- Dataset: `data/final_labeled_fake_reviews.csv`, tương ứng Amazon Labeled Fake Reviews khoảng 50.000 mẫu.
- Mục tiêu metrics: Macro F1 >= 89-92%, Precision Fake >= 97.5%, ROC-AUC >= 93%.
- Trọng tâm khoa học: cân bằng Precision-Recall, robustness, giải thích mô hình, và ablation study.
- Ràng buộc quan trọng: RAM 12GB; PCA/SVD được đánh giá như công cụ giảm RAM nhưng không còn mặc định là final model path.
- Công cụ dự kiến: Python, HuggingFace Transformers, PyTorch/Keras, pyswarm, scikit-learn, XGBoost/LightGBM, SHAP, LIME, VADER/TextBlob.
- Ưu tiên thực hiện: khóa Phase 2 raw 777 + sequence metadata, sau đó chạy Phase 4/5 model zoo và ensemble sweep để chọn winner theo validation.

## Constraints

- **Runtime**: Google Colab là môi trường chạy chính - không chạy notebook/training local nếu chưa được phép.
- **Source layout**: Toàn bộ source code thực thi nằm trong 8 notebook `.ipynb` theo phase - không tạo module `.py` cho pipeline chính.
- **Memory**: RAM 12GB - BERT extraction, behavioral aggregation, PCA và XAI phải chạy batch/subset.
- **Timeline**: 18-22 ngày - phase phải có output kiểm chứng được, tránh mở rộng backbone/model ngoài kế hoạch.
- **Data**: Dataset khoảng 50.000 mẫu - split stratified và seed cố định để tái lập.
- **Optimization**: PSO chạy CPU và bắt đầu bằng subset 20% - tránh grid quá rộng.
- **Evidence**: Ablation là phần bắt buộc - full model không đủ, phải chứng minh từng cải tiến.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Bộ 8 notebook Colab theo phase | Dễ kiểm soát output từng phase, giảm rủi ro rerun toàn pipeline và phù hợp nộp/publish | Pending |
| Dùng PCA/SVD sau feature fusion | Giảm noise và giữ RAM dưới 12GB | Superseded as final default; keep as diagnostic/ablation |
| PSO thử trên 20% data trước | Giảm chi phí tối ưu và phát hiện cấu hình lỗi sớm | Keep only where validation benefit is clear |
| Chạy XAI/adversarial trên subset | SHAP/LIME/PGD tốn RAM, vẫn đủ bằng chứng nếu subset có seed rõ | Pending |
| Đánh giá theo Macro F1, Precision Fake và ROC-AUC | Tránh tối ưu một metric gây lệch Precision-Recall | Pending |
| ModernBERT raw 777 is the main tabular feature track | Phase 9 shows raw features outperform PCA for DL; LightGBM raw is a strong balanced candidate | Active |
| CNN-BiLSTM must use sequence input for fairness | CNN/BiLSTM need token order/sequence signal, not a static PCA vector | Active |
| Ensemble is optional, not forced | A single model can be the final winner if it beats all ensembles on validation | Active |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition**:
1. Requirements invalidated? Move to Out of Scope with reason.
2. Requirements validated? Move to Validated with phase reference.
3. New requirements emerged? Add to Active.
4. Decisions to log? Add to Key Decisions.
5. "What This Is" still accurate? Update if drifted.

**After each milestone**:
1. Full review of all sections.
2. Core Value check - still the right priority?
3. Audit Out of Scope - reasons still valid?
4. Update Context with current state.

---
*Last updated: 2026-06-10 after Phase 7/8 final-track audit (SSOT: `phase7_final_metrics.csv`)*
