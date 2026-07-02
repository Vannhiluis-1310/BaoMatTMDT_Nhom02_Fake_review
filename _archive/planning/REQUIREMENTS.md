# Requirements: PSO-Optimized Hybrid Fake Review Detection

**Defined:** 2026-05-31  
**Core Value:** Chứng minh mô hình hybrid cải thiện phát hiện fake review một cách có kiểm chứng bằng metrics, ablation và giải thích mô hình, trong giới hạn RAM 12GB và bộ notebook Colab theo phase.

## v1 Requirements

### Data

- [ ] **DATA-01**: Notebook tải hoặc đọc được Amazon Labeled Fake Reviews Dataset từ đường dẫn Colab/Drive được cấu hình.
- [ ] **DATA-02**: Notebook tạo EDA gồm phân phối nhãn, độ dài review, missing values, duplicate rows và phân tích mất cân bằng lớp.
- [ ] **DATA-03**: Notebook xử lý missing/duplicated theo rule được ghi rõ và lưu dataset sạch.
- [ ] **DATA-04**: Notebook tạo train/validation/test split stratified với seed cố định và lưu metadata split.

### Features

- [ ] **FEAT-01**: Notebook trích xuất BERT embeddings 768 chiều theo batch và cache artifact.
- [ ] **FEAT-02**: Notebook tạo 5 behavioral features cơ bản có mô tả công thức rõ ràng.
- [ ] **FEAT-03**: Notebook tạo 4 behavioral features nâng cao: review velocity, burst pattern, reviewer embedding, time gap.
- [ ] **FEAT-04**: Notebook chuẩn hóa/scale behavioral features và nối với BERT embeddings thành feature vector thống nhất.
- [ ] **FEAT-05**: Notebook kiểm tra leakage cho feature theo reviewer/time trước khi train.

### Feature Selection

- [ ] **PCA-01**: Notebook áp dụng PCA hoặc TruncatedSVD trên train set, không fit trên validation/test.
- [ ] **PCA-02**: Notebook chọn số chiều giữ 95-98% variance, mục tiêu khoảng 300-400 chiều.
- [ ] **PCA-03**: Notebook lưu PCA/SVD object, explained variance plot và transformed train/validation/test features.
- [ ] **PCA-04**: Notebook đo RAM/time trước và sau PCA để chứng minh giá trị giảm chiều.

### Deep Model

- [x] **DL-01**: Notebook xây CNN-BiLSTM-Attention nhận vector feature sau PCA.
- [x] **DL-02**: Notebook dùng focal loss hoặc class weighting để xử lý mất cân bằng lớp.
- [x] **DL-03**: Notebook train với batch size <= 64, early stopping và checkpoint best validation Macro F1.
- [x] **DL-04**: Notebook log training curves, validation metrics và inference probabilities.

### PSO Optimization

- [x] **PSO-01**: Notebook định nghĩa search space PSO cho hyperparameters và/hoặc feature weights.
- [x] **PSO-02**: Notebook chạy PSO trial trên subset 20% data trước khi chạy cấu hình chính.
- [x] **PSO-03**: Notebook ghi lại best parameters, objective history và so sánh với cấu hình không PSO.
- [x] **PSO-04**: Notebook có guardrail max iterations/particles để không vượt RAM/time.

### Ensemble

- [x] **ENS-01**: Notebook train XGBoost và LightGBM trên feature sau PCA.
- [x] **ENS-02**: Notebook xây stacking ensemble từ deep model, XGBoost và LightGBM.
- [x] **ENS-03**: Notebook áp dụng probability calibration và đánh giá calibration curve/Brier score.
- [x] **ENS-04**: Notebook chọn threshold tối ưu cho Precision Fake mà vẫn báo cáo Recall Fake.

### Robustness and XAI

- [ ] **ROB-01**: Notebook tạo FGSM hoặc PGD adversarial perturbation phù hợp với input feature/model.
- [ ] **ROB-02**: Notebook đánh giá robustness trên subset có seed và so sánh clean vs adversarial metrics.
- [ ] **XAI-01**: Notebook chạy SHAP trên subset để phân tích global feature importance.
- [ ] **XAI-02**: Notebook chạy LIME cho một số case fake/real đại diện và lưu hình/bảng giải thích.

### Evaluation

- [x] **EVAL-01**: Notebook báo cáo Macro F1, Precision Fake, Recall Fake, F1 Fake, ROC-AUC, PR-AUC và confusion matrix.
- [x] **EVAL-02**: Notebook chạy 5-fold CV hoặc CV rút gọn có giải thích nếu tài nguyên không đủ.
- [x] **EVAL-03**: Notebook có ablation cho Full Model, Model A, B, C, D, E theo bảng đã định nghĩa.
- [x] **EVAL-04**: Notebook xuất bảng tổng hợp metrics và biểu đồ ablation dùng được trong báo cáo.
- [x] **EVAL-05**: Notebook ghi rõ seed, version thư viện, split và cấu hình để tái lập.

### Delivery

- [x] **DEL-01**: Notebook lưu model artifacts: PCA/SVD, scaler, deep model checkpoint, ensemble pickle/joblib, calibration object.
- [x] **DEL-02**: Notebook có mục "How to run on Colab" với thứ tự cell rõ ràng.
- [x] **DEL-03**: Báo cáo cuối trình bày novelty, architecture, ablation, robustness, XAI, limitations và future work.
- [x] **DEL-04**: Folder artifact/report có cấu trúc rõ ràng, không chứa file tạm không cần thiết.

## v2 Requirements

### Extended Research

- **V2-01**: So sánh thêm nhiều transformer backbone khác như RoBERTa hoặc DistilBERT nếu còn thời gian.
- **V2-02**: Thử thêm dataset fake review khác để kiểm tra cross-dataset generalization.
- **V2-03**: Tạo dashboard/demo inference nhỏ sau khi báo cáo nghiên cứu hoàn tất.
- **V2-04**: Tự động tạo LaTeX tables từ kết quả experiment.

## v3 Revised Pipeline Requirements

These requirements supersede conflicting v1 assumptions after Phase 9 diagnostic evidence.

### Revised Features

- [ ] **REV-FEAT-01**: Phase 2 explicitly uses ModernBERT pooled embeddings for the tabular raw feature track, or records a justified fallback if ModernBERT cannot run in Colab.
- [ ] **REV-FEAT-02**: Phase 2 writes raw fused 777 features: 768 text embedding + 9 behavioral features.
- [ ] **REV-FEAT-03**: Phase 2 writes enough tokenizer/sequence metadata for CNN-BiLSTM to consume real token sequences.

### Revised Feature Selection

- [ ] **REV-PCA-01**: Phase 3 treats PCA/SVD as diagnostic and ablation evidence, not as the default final model path.
- [ ] **REV-PCA-02**: PCA/SVD reports must include memory/time benefit and metric impact compared with raw 777 features.

### Revised Deep Learning

- [ ] **REV-DL-01**: MLP raw 777 is the primary tabular DL branch.
- [ ] **REV-DL-02**: CNN-BiLSTM-Attention must consume real token sequences or ModernBERT hidden states, not PCA/static feature vectors.
- [ ] **REV-DL-03**: Behavioral features for sequence DL must be added by a late-fusion branch.
- [ ] **REV-DL-04**: DL threshold sweep must report Precision Fake, Recall Fake, F1 Fake and Macro F1.

### Revised Ensemble and Evaluation

- [ ] **REV-ENS-01**: Phase 5 evaluates single models and allows a single model to win if it beats ensembles.
- [ ] **REV-ENS-02**: Phase 5 evaluates weighted blends among LightGBM, XGBoost, MLP and sequence DL where probability files exist.
- [ ] **REV-ENS-03**: Phase 5 evaluates at least one stacking meta-model and one calibration strategy.
- [ ] **REV-ENS-04**: Phase 5 is split into one notebook per major model plus separate blending/stacking notebooks, with `05_Hybrid_Ensemble.ipynb` as the orchestrator/summary.
- [x] **REV-EVAL-01**: Phase 7 selects a Balanced winner by validation Macro F1 (τ=0.30, test F1 0.9463).
- [x] **REV-EVAL-02**: Phase 7 selects a Precision-first winner by Precision Fake >= 97.5% (τ=0.60, test 0.9816).
- [ ] **REV-EVAL-03**: Phase 8 report must show ModernBERT placement and the two-track pipeline diagram.

## Out of Scope

| Feature | Reason |
|---------|--------|
| Local training mặc định | Chủ dự án yêu cầu không tự tiện chạy local nếu chưa đồng ý |
| Production API/web app | Không cần cho mục tiêu nghiên cứu 18-22 ngày |
| Web scraping Amazon | Dataset đã cố định, scraping tăng rủi ro pháp lý và lệch scope |
| Package Python riêng | Yêu cầu source thực thi nằm trong các notebook `.ipynb` |
| Tuning toàn bộ transformer lớn | Dễ vượt RAM 12GB và làm vỡ timeline |
| Treating PCA as automatically beneficial | Phase 9 evidence shows PCA can hurt DL and should be evaluated, not assumed |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| DATA-01 | Phase 1 | Pending |
| DATA-02 | Phase 1 | Pending |
| DATA-03 | Phase 1 | Pending |
| DATA-04 | Phase 1 | Pending |
| FEAT-01 | Phase 2 | Pending |
| FEAT-02 | Phase 2 | Pending |
| FEAT-03 | Phase 2 | Pending |
| FEAT-04 | Phase 2 | Pending |
| FEAT-05 | Phase 2 | Pending |
| PCA-01 | Phase 3 | Pending |
| PCA-02 | Phase 3 | Pending |
| PCA-03 | Phase 3 | Pending |
| PCA-04 | Phase 3 | Pending |
| DL-01 | Phase 4 | Complete with metric gap |
| DL-02 | Phase 4 | Complete with metric gap |
| DL-03 | Phase 4 | Complete with metric gap |
| DL-04 | Phase 4 | Complete with metric gap |
| PSO-01 | Phase 4 | Complete with metric gap |
| PSO-02 | Phase 4 | Complete with metric gap |
| PSO-03 | Phase 4 | Complete with metric gap |
| PSO-04 | Phase 4 | Complete with metric gap |
| ENS-01 | Phase 5 | Complete with metric gap |
| ENS-02 | Phase 5 | Complete with metric gap |
| ENS-03 | Phase 5 | Complete with metric gap |
| ENS-04 | Phase 5 | Complete with metric gap |
| ROB-01 | Phase 6 | Pending |
| ROB-02 | Phase 6 | Pending |
| XAI-01 | Phase 6 | Pending |
| XAI-02 | Phase 6 | Pending |
| EVAL-01 | Phase 7 | Complete with metric gap |
| EVAL-02 | Phase 7 | Complete |
| EVAL-03 | Phase 7 | Complete |
| EVAL-04 | Phase 7 | Complete |
| EVAL-05 | Phase 7 | Complete |
| DEL-01 | Phase 8 | Complete |
| DEL-02 | Phase 8 | Complete |
| DEL-03 | Phase 8 | Complete |
| DEL-04 | Phase 8 | Complete |

**Coverage:**
- v1 requirements: 36 total
- Mapped to phases: 36
- Unmapped: 0
- v3 revised requirements: 17 total
- Mapped primarily to Phases 2, 3, 4, 5, 7, 8 and orchestration Phase 10

---
*Requirements defined: 2026-05-31*  
*Last updated: 2026-06-09 after Phase 9 diagnostic and pipeline redesign*
