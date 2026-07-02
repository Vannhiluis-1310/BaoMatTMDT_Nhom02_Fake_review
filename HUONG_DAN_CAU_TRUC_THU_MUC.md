# Hướng dẫn cấu trúc thư mục `Fake_reviews`

Tài liệu giải thích **mục đích** và **nội dung** từng file/thư mục trong dự án phát hiện đánh giá giả (Fake Review Detection) trên Amazon Labeled Fake Reviews Dataset.

**Đường dẫn gốc:** `C:\Users\vanhi\Desktop\HCMUTE_TMDT\HKII_Nam_3\Bao_Mat_TMDT\Fake_reviews`  
**Cập nhật:** 2026-06-25 | Pipeline final track: `phase5_weighted_blend` (Macro F1 **0,9463** @ τ=0,30)

---

## 1. Tổng quan dự án

Dự án xây dựng pipeline ML hai nhánh để phân loại review thật/giả:

| Nhánh | Đặc trưng | Mô hình tiêu biểu |
|-------|-----------|-------------------|
| **Final track** (luồng chính) | ModernBERT 768-d + 9 behavioral → **777-d** | XGBoost, CNN-BiLSTM, **weighted blend** |
| **Ablation / legacy** | PCA/SVD 777→**400-d** | DL baseline, DL-PSO, legacy ensemble |

**Luồng dữ liệu end-to-end:**

```
data/ (CSV gốc)
  → notebooks/ Phase 1–8 (Jupyter trên Colab)
  → artifacts/ (model, feature, xác suất, XAI)
  → reports/ (bảng CSV, biểu đồ PNG, gói nộp Phase 8)
  → thesis/ + docs/ (luận văn và tài liệu kỹ thuật)
```

**Nguồn số liệu chính thức (SSOT):** `reports/tables/phase7_final_metrics.csv`

---

## 2. Cây thư mục cấp một

```
Fake_reviews/          (~596 file)
├── data/              (10)   Dữ liệu gốc và đã xử lý
├── notebooks/         (21)   Code Jupyter Phase 1–10
├── artifacts/         (157)  Output model, feature, prediction
├── reports/           (158)  Bảng số, hình, gói nộp cuối
├── thesis/            (19)   Luận văn Chương 1–6
├── docs/              (9)    Tài liệu kỹ thuật theo phase
├── scripts/           (7)    Script phụ trợ (không phải pipeline chính)
├── logs/              (1)    Log chạy thí nghiệm
├── .planning/         (5)    Ghi chú lập kế hoạch nội bộ
├── _archive/          (203)  Bản cũ — không dùng khi nộp bài
├── README.md
├── .gitignore
├── extract.py
├── extract_nb.txt
├── extract_nb_utf8.txt
└── HUONG_DAN_CAU_TRUC_THU_MUC.md   ← file này
```

**Không dùng khi báo cáo / nộp bài:** `_archive/`, `.planning/` (trừ tham khảo lịch sử).

---

## 3. File gốc (root)

| File | Mục đích | Nội dung |
|------|----------|----------|
| `README.md` | Điểm vào dự án | Tên đề tài, kết quả headline, hướng dẫn đọc docs, thứ tự pipeline, quy tắc chạy Colab |
| `.gitignore` | Cấu hình Git | Bỏ qua cache Python, checkpoint notebook, artifact lớn (`.npy`, `.pth`, `.joblib`…), data processed |
| `extract.py` | Tiện ích trích xuất | Script Python đọc cell từ `01_EDA_Preprocessing.ipynb` → ghi ra file text |
| `extract_nb.txt` | Output trích xuất | Nội dung cell notebook (encoding cũ) |
| `extract_nb_utf8.txt` | Output trích xuất UTF-8 | Nội dung cell notebook dạng text thuần, dùng để đọc/annotate nhanh |
| `HUONG_DAN_CAU_TRUC_THU_MUC.md` | Tài liệu này | Giải thích toàn bộ cấu trúc file trong `Fake_reviews/` |

---

## 4. `data/` — Dữ liệu

| File / thư mục | Mục đích | Nội dung |
|----------------|----------|----------|
| `README.md` | Hướng dẫn thư mục data | Quy tắc không ghi đè CSV gốc |
| `final_labeled_fake_reviews.csv` | **Dataset gốc** | ~50.000 review Amazon đã gán nhãn fake/real (~15 MB). **Không ghi đè** sau Phase 1 |
| `interim/.gitkeep` | Giữ thư mục trống | File trung gian từ cleaning/EDA (nếu có) |
| `processed/.gitkeep` | Giữ thư mục trống | Placeholder cho output processed |
| `processed/clean_reviews.csv` | Sau làm sạch | **42.749** mẫu sau loại missing/duplicate |
| `processed/train.csv` | Tập train | **29.923** mẫu (70%), stratified split |
| `processed/val.csv` | Tập validation | **6.413** mẫu (15%) |
| `processed/test.csv` | Tập test | **6.413** mẫu (15%) |
| `processed/split_metadata.json` | Metadata split | Seed=42, tỷ lệ 70/15/15, stratified |
| `processed/schema_metadata.json` | Schema cột | Mô tả cấu trúc cột sau xử lý |

**Notebook sinh ra:** `notebooks/01_EDA_Preprocessing.ipynb`

---

## 5. `notebooks/` — Source code chính (Jupyter)

Toàn bộ pipeline thực thi nằm trong notebook, chạy trên **Google Colab** (RAM 12 GB).

### 5.1. Pipeline chính (Phase 1–8)

| File | Phase | Mục đích | Nội dung chính |
|------|-------|----------|----------------|
| `01_EDA_Preprocessing.ipynb` | 1 | EDA & tiền xử lý | Khám phá dữ liệu, làm sạch, chia train/val/test stratified |
| `02_Feature_Engineering.ipynb` | 2 | Trích xuất đặc trưng | ModernBERT embedding 768-d + 9 behavioral → fused **777-d** |
| `03_PCA_Feature_Selection.ipynb` | 3 | PCA/SVD diagnostic | Giảm chiều 777→400-d cho **ablation track** (không phải final input) |
| `04_PSO_Model_Training.ipynb` | 4 | DL + PSO | DL baseline và DL-PSO trên feature PCA 400-d |
| `05_00_Phase5_Run_Order.ipynb` | 5 | Điều phối Phase 5 | Checklist và thứ tự chạy các notebook con Phase 5 |
| `05_01_LightGBM_Raw.ipynb` | 5 | LightGBM | GBDT trên raw 777-d |
| `05_02_XGBoost_Raw.ipynb` | 5 | XGBoost | GBDT trên raw 777-d |
| `05_03_MLP_Raw.ipynb` | 5 | MLP tabular | Mạng nơ-ron fully-connected trên 777-d |
| `05_04a_CNN_BiLSTM_Sequence_seed42.ipynb` | 5 | CNN-BiLSTM (seed 42) | Deep learning trên **token sequence** + behavioral late fusion |
| `05_04b_CNN_BiLSTM_Sequence_seed123.ipynb` | 5 | CNN-BiLSTM (seed 123) | Cùng kiến trúc, seed 123 — đánh giá ổn định |
| `05_04c_CNN_BiLSTM_Sequence_seed456.ipynb` | 5 | CNN-BiLSTM (seed 456) | Cùng kiến trúc, seed 456 — đánh giá ổn định |
| `05_04_CNN_BiLSTM_Sequence.ipynb` | 5 | CNN-BiLSTM (gốc) | Phiên bản gốc trước khi tách multi-seed |
| `05_05_Weighted_Blending.ipynb` | 5 | Weighted blend | Grid search trọng số kết hợp model trên validation |
| `05_06_Stacking_Calibration.ipynb` | 5 | Stacking | Meta-learner stacking + probability calibration |
| `05_Hybrid_Ensemble.ipynb` | 5 | Tổng hợp ensemble | Orchestrator: leaderboard, chọn model cuối, metadata |
| `06_Adversarial_XAI.ipynb` | 6 | Robustness & XAI | FGSM/PGD adversarial attack + SHAP/LIME giải thích |
| `07_Evaluation_Ablation.ipynb` | 7 | Đánh giá & ablation | Metrics đầy đủ, ablation Models A–E, CV, target audit |
| `08_Final_Report_Kaggle.ipynb` | 8 | Báo cáo cuối | Manifest nộp bài, inventory artifact, `Phase8_Final_Report.md` |

### 5.2. Notebook bổ sung

| File | Mục đích | Nội dung |
|------|----------|----------|
| `09_Bieu_Do_So_Sanh_Mo_Hinh.ipynb` | Trực quan hóa | Vẽ biểu đồ so sánh model từ `baseline_benchmark_metrics.csv` |
| `10_Baseline_Algorithm_Benchmark.ipynb` | Benchmark Tier-1 | So sánh công bằng ≥5 thuật toán classical (LR, SVM, RF…) |

### 5.3. Notebook kiểm thử

| File | Mục đích | Nội dung |
|------|----------|----------|
| `tests/01_DL_PCA_Diagnostic_Test.ipynb` | Diagnostic Phase 9 | Kiểm tra mismatch DL + PCA vector — **không bắt buộc nộp** |

**Thứ tự chạy đầy đủ:** xem `reports/final/phase8_run_order_checklist.csv` hoặc `docs/08_Ket_Luan_va_Huong_Dan.md`.

---

## 6. `artifacts/` — Kết quả chạy pipeline

### 6.1. `artifacts/features/` (Phase 2)

| File | Nội dung |
|------|----------|
| `features_raw_{train,val,test}.npy` | Vector fused **777-d** (ModernBERT 768 + behavioral 9) |
| `bert_{train,val,test}.npy` | Embedding ModernBERT **768-d** riêng |
| `behavioral_{train,val,test}.csv` | **9** đặc trưng hành vi (velocity, burst, time gap…) |
| `labels_{train,val,test}.npy` | Nhãn nhị phân 0=real, 1=fake |
| `feature_metadata.json` | Backbone `answerdotai/ModernBERT-base`, fused_dim=777 |
| `feature_dictionary.csv` | Tên và mô tả từng chiều feature |
| `behavioral_scaler.joblib` | StandardScaler fit **chỉ trên train** |
| `row_ids_{train,val,test}.csv` | Ánh xạ index → ID gốc trong CSV |

### 6.2. `artifacts/pca/` (Phase 3 — ablation)

| File | Nội dung |
|------|----------|
| `pca_or_svd.joblib` | Reducer PCA/SVD fit chỉ train |
| `pca_scaler.joblib` | Scaler trước PCA |
| `features_final_{train,val,test}.npy` | Vector **400-d** sau giảm chiều |
| `phase3_metadata.json` | Retained variance ≈ 95,10% |
| `labels_{train,val,test}.npy` | Nhãn (copy từ Phase 2) |

### 6.3. `artifacts/models/` (Phase 4–5)

| File | Mô hình | Ghi chú |
|------|---------|---------|
| `baseline_model_dl.pth` | DL baseline (PCA, không PSO) | Macro F1 test ≈ 0,7665 |
| `best_model_dl.pth` | DL-PSO (PCA) | Macro F1 test ≈ 0,7793 |
| `best_params.json` | Hyperparameter tối ưu PSO | |
| `phase4_metadata.json` | Metadata training Phase 4 | |
| `phase5_cnn_bilstm_sequence.pth` | CNN-BiLSTM-Attention | Macro F1 test ≈ 0,9343 |
| `phase5_cnn_bilstm_sequence_metadata.json` | Config & history CNN-BiLSTM | |
| `phase5_mlp_raw.pth` | MLP tabular 777-d | Macro F1 test ≈ 0,8990 |
| `phase5_mlp_raw_metadata.json` | Config MLP | |
| `seed_123/`, `seed_456/` | Checkpoint multi-seed | Bản sao model theo seed |

### 6.4. `artifacts/ensemble/` (Phase 5)

| File | Vai trò |
|------|---------|
| `phase5_weighted_blend_metadata.json` | **★ Cấu hình headline:** seed 42 → CNN 50% + XGB 50%; τ balanced 0,30 / precision-first 0,60 |
| `phase5_metadata.json` | Metadata tổng Phase 5 |
| `phase5_xgb_raw.joblib` + `phase5_xgb_raw_metadata.json` | XGBoost raw 777-d đã train |
| `phase5_lgbm_raw.joblib` + `phase5_lgbm_raw_metadata.json` | LightGBM raw 777-d |
| `phase5_stacking_calibrated_metadata.json` | Stacking + calibration |
| `phase5_stack_*.joblib` | Thành phần meta-learner (logistic, RF, extra trees…) |
| `phase5_cnn_bilstm_behavior_scaler.joblib` | Scaler behavioral cho CNN-BiLSTM |
| `phase5_mlp_raw_scaler.joblib` | Scaler cho MLP |
| `final_ensemble_model.pkl` | Legacy PCA ensemble (ablation) |
| `xgboost_model.pkl`, `lightgbm_model.pkl`, `stacking_meta_model.pkl` | Legacy Phase 4 |
| `probability_calibrator.pkl` | Calibrator xác suất legacy |
| `seed_123/`, `seed_456/` | Metadata blend theo seed |

### 6.5. `artifacts/predictions/` — Xác suất P(fake)

Mỗi model có 3 file: `*_train_prob.npy`, `*_val_prob.npy`, `*_test_prob.npy`.

| Prefix | Mô hình | Macro F1 test @0,50 |
|--------|---------|---------------------|
| `phase5_weighted_blend_*` | **Weighted blend (đề xuất)** | **0,9433** |
| `phase5_cnn_bilstm_sequence_*` | CNN-BiLSTM sequence | 0,9343 |
| `phase5_stacking_calibrated_*` | Stacking calibrated | 0,9105 |
| `phase5_xgb_raw_*` | XGBoost raw | 0,9059 |
| `phase5_lgbm_raw_*` | LightGBM raw | 0,9051 |
| `phase5_mlp_raw_*` | MLP raw | 0,8990 |
| `dl_pso_*` | DL-PSO (PCA, ablation) | 0,7793 |
| `dl_baseline_*` | DL baseline (PCA) | 0,7665 |
| `final_ensemble_*` | Legacy PCA blend | ~0,80–0,86 |
| `xgboost_*`, `lightgbm_*`, `stacking_*` | Legacy GBDT Phase 4 | Khác `phase5_*` |
| `sklearn_*` | Baseline sklearn (notebook 10) | LR, LinearSVC, RF |
| `predictions/seed_123/`, `seed_456/` | Xác suất multi-seed | Theo từng seed |

### 6.6. `artifacts/xai/` (Phase 6)

| File | Track | Nội dung |
|------|-------|----------|
| `phase6_final_lime_case_0{1..6}_*.html` | **Headline** | 6 case LIME: TP, TN, FP, FN, high-confidence fake/real |
| `phase6_final_metadata.json` | Headline | Metadata XAI final track (`weighted_blend`) |
| `phase6_final_shap_values_xgb.npy` | Headline | SHAP values subset XGBoost |
| `lime_case_0{1..6}_*.html` | Legacy | LIME trên legacy PCA ensemble |
| `phase6_metadata.json` | Dual-track | Metadata cả final + legacy |
| `shap_values_lightgbm.npy` | Legacy | SHAP values LightGBM legacy |

### 6.7. Khác trong `artifacts/`

| Thư mục / file | Vai trò |
|----------------|---------|
| `evaluation/phase7_metadata.json` | Config Phase 7: ablation map, CV, targets |
| `evaluation/baseline_benchmark_metadata.json` | Metadata benchmark sklearn |
| `diagnostics/dl_pca_test/` | Kết quả diagnostic DL trên PCA |
| `figures/` | Hình phụ (hình chính ở `reports/figures/`) |

---

## 7. `reports/` — Bảng số và biểu đồ

### 7.1. `reports/tables/` — 98 file CSV

#### SSOT — file quan trọng nhất

| File | Nội dung |
|------|----------|
| `phase7_final_metrics.csv` | **★ SSOT:** So sánh tất cả model, mọi split và threshold |
| `phase7_target_audit.csv` | Pass/fail target F1≥0,89 / Prec≥0,975 / AUC≥0,93 |
| `phase7_ablation_results.csv` | Ablation Models A–E @ τ=0,50 |
| `phase7_ablation_delta.csv` | Δ metric so với full model |
| `phase7_multiseed_summary.csv` | Mean ± std 3 seed (42, 123, 456) |
| `phase7_multiseed_metrics.csv` | Metric chi tiết theo seed |
| `phase5_leaderboard.csv` | Xếp hạng candidate Phase 5 |
| `phase5_weighted_blending_sweep.csv` | Grid trọng số blend |
| `phase6_final_shap_global_importance.csv` | SHAP top feature headline |
| `phase6_final_lime_case_summary.csv` | Tóm tắt 6 case LIME |

#### Phase 1 — EDA & cleaning (12 file)

| File | Nội dung |
|------|----------|
| `phase1_eda_summary.csv` | Tổng quan EDA |
| `phase1_advanced_eda_summary.csv` | EDA nâng cao |
| `phase1_cleaning_report.csv` | Báo cáo làm sạch (50k→42.749) |
| `phase1_samples_by_label.csv` | Phân bố nhãn |
| `phase1_length_by_label.csv` | Độ dài review theo nhãn |
| `phase1_sentiment_by_label.csv` | Sentiment theo nhãn |
| `phase1_rating_label_stats.csv` | Thống kê rating |
| `phase1_temporal_stats.csv` | Thống kê thời gian |
| `phase1_top_terms_by_label.csv` | Từ khóa nổi bật theo nhãn |
| `phase1_category_fake_rate.csv` | Tỷ lệ fake theo category |
| `phase1_product_fake_rate.csv` | Tỷ lệ fake theo sản phẩm |
| `phase1_user_burst_stats.csv` | Thống kê burst pattern người dùng |

#### Phase 2 — Feature engineering (5 file)

| File | Nội dung |
|------|----------|
| `phase2_bert_embedding_report.csv` | Báo cáo trích xuất ModernBERT |
| `phase2_feature_quality_report.csv` | Chất lượng 777-d fused features |
| `phase2_input_validation.csv` | Kiểm tra input Phase 2 |
| `phase2_leakage_check_report.csv` | Kiểm tra data leakage |
| `phase2_memory_report.csv` | Sử dụng RAM khi extract |

#### Phase 3 — PCA (5 file)

| File | Nội dung |
|------|----------|
| `phase3_pca_selection_report.csv` | Báo cáo chọn số component |
| `phase3_component_variance.csv` | Explained variance từng component |
| `phase3_feature_quality_report.csv` | Chất lượng feature 400-d |
| `phase3_input_validation.csv` | Kiểm tra input |
| `phase3_memory_report.csv` | RAM trước/sau PCA |

#### Phase 4 — PSO & DL (7 file)

| File | Nội dung |
|------|----------|
| `phase4_baseline_metrics.csv` | Metric DL baseline |
| `phase4_baseline_training_history.csv` | Lịch sử train baseline |
| `phase4_pso_final_metrics.csv` | Metric DL-PSO cuối |
| `phase4_pso_training_history.csv` | Lịch sử train PSO |
| `phase4_pso_trial_history.csv` | Tổng hợp các trial PSO |
| `phase4_model_comparison.csv` | So sánh baseline vs PSO |
| `phase4_input_validation.csv` | Kiểm tra input |

#### Phase 5 — Model zoo & ensemble (29 file)

| File | Nội dung |
|------|----------|
| `phase5_lgbm_raw_metrics.csv` | Metric LightGBM (train F1=1,0) |
| `phase5_xgb_raw_metrics.csv` | Metric XGBoost |
| `phase5_mlp_raw_metrics.csv` + `phase5_mlp_raw_history.csv` | Metric & history MLP |
| `phase5_cnn_bilstm_sequence_metrics.csv` + `..._history.csv` | Metric & history CNN-BiLSTM |
| `phase5_weighted_blend_metrics.csv` | Metric weighted blend |
| `phase5_weighted_blending_best.csv` | Cấu hình blend tốt nhất |
| `phase5_stacking_calibrated_metrics.csv` | Metric stacking |
| `phase5_stacking_calibration_metrics.csv` | Metric calibration |
| `phase5_stacking_metrics.csv` | Metric stacking thô |
| `phase5_calibration_curve.csv` + `phase5_calibration_report.csv` | Đường cong calibration |
| `phase5_threshold_report.csv` | Phân tích ngưỡng τ |
| `phase5_all_candidate_threshold_sweep.csv` | Sweep τ mọi candidate |
| `phase5_candidate_selection.csv` | Lý do chọn candidate |
| `phase5_selected_candidates.csv` | Danh sách model được chọn |
| `phase5_selected_threshold.csv` | Ngưỡng được chọn |
| `phase5_run_order.csv` | Thứ tự chạy Phase 5 |
| `phase5_final_metrics.csv` | Metric tổng Phase 5 |
| `phase5_model_comparison.csv` + `phase5_base_model_metrics.csv` | So sánh model |
| `phase5_input_validation.csv` | Kiểm tra input |
| `phase5_stack_validation_split.csv` | Split cho stacking |

#### Phase 6 — Robustness & XAI (20 file)

| File | Nội dung |
|------|----------|
| `phase6_robustness_metrics.csv` | Metric FGSM/PGD (legacy) |
| `phase6_robustness_metric_drops.csv` | Mức giảm metric khi tấn công |
| `phase6_adversarial_config.csv` | Cấu hình adversarial |
| `phase6_shap_global_importance.csv` | SHAP legacy |
| `phase6_final_shap_global_importance.csv` | SHAP headline |
| `phase6_final_shap_block_importance.csv` | SHAP theo block feature |
| `phase6_*_lime_*` | Background indices, feature weights, case summary |
| `phase6_pca_component_loading_map.csv` | Loading map PCA cho XAI |
| `phase6_predictor_smoke_test.csv` | Smoke test predictor |
| `phase6_input_validation.csv` | Kiểm tra input |

#### Phase 7 — Đánh giá cuối (12 file)

| File | Nội dung |
|------|----------|
| `phase7_cv_metrics.csv` + `phase7_cv_summary.csv` | 5-fold CV LightGBM PCA |
| `phase7_error_analysis.csv` | Phân tích lỗi dự đoán |
| `phase7_ablation_controlled_reference.csv` | Reference controlled ablation |
| `phase7_report_highlights.csv` | Điểm nổi bật báo cáo |
| `phase7_input_validation.csv` | Kiểm tra input |

#### Benchmark & SOTA

| File | Nội dung |
|------|----------|
| `baseline_benchmark_metrics.csv` | Metric ≥5 thuật toán classical |
| `baseline_benchmark_model_config.csv` | Cấu hình từng model benchmark |
| `baseline_benchmark_literature_tier2.csv` | So sánh với SOTA Tier-2 |
| `literature_references_20.csv` | 20 paper tham khảo |
| `literature_sota_comparison.csv` | So sánh SOTA 3-tier |
| `viz_comparison_models.csv` | Data cho biểu đồ so sánh |
| `viz_radar_metrics.csv` | Data radar chart |

### 7.2. `reports/figures/` — 40 file PNG

| Nhóm file | Nội dung |
|-----------|----------|
| `phase1_*.png` (12) | EDA: phân bố nhãn, rating, sentiment, độ dài, thời gian, từ khóa |
| `phase3_*.png` (2) | Explained variance, RAM trước/sau PCA |
| `phase4_*.png` (4) | Training curves, so sánh model, PSO history |
| `phase5_*.png` (4) | So sánh base model, calibration, threshold |
| `phase6_*.png` (5) | SHAP summary, robustness drop |
| `phase7_*.png` (4) | Ablation delta, CV, final metrics, target gap |
| `viz_comparison_*.png` (9) | Biểu đồ so sánh model (notebook 09) |

### 7.3. `reports/final/` — Gói nộp Phase 8

| File | Nội dung |
|------|----------|
| `Phase8_Final_Report.md` | Báo cáo Markdown tổng hợp cuối cùng |
| `phase8_report_summary.csv` | Facts compact trích từ Phase 7 |
| `phase8_submission_package_manifest.csv` | **Danh mục file bắt buộc nộp** |
| `phase8_artifact_inventory.csv` | Kiểm tra exists + kích thước artifact |
| `phase8_run_order_checklist.csv` | Thứ tự notebook 01→08 |

### 7.4. `reports/diagnostics/` — 14 file

Bảng và hình diagnostic bổ sung (DL-PCA test, validation phụ).

---

## 8. `thesis/` — Luận văn

### 8.1. Chương chính (bản nộp)

| File | Nội dung |
|------|----------|
| `Chapter1_Introduction.md` | Mở đầu, câu hỏi nghiên cứu RQ1–RQ6, đóng góp |
| `Chapter2_Theory.md` | Cơ sở lý thuyết, SOTA 20 papers, research gaps G1–G8 |
| `Chapter3_Methodology.md` | Pipeline dual-track, Phase 1–8, thiết kế thí nghiệm |
| `Chapter4_Results.md` | Kết quả triển khai, Bảng 4.1–4.15, rubric |
| `Chapter5_Discussion.md` | Thảo luận kết quả, trả lời RQ |
| `Chapter6_Conclusion.md` | Kết luận, hạn chế, hướng phát triển |
| `References.md` | Tài liệu tham khảo định dạng APA |
| `Self_Assessment_Rubric.md` | Rubric tự đánh giá D0–D8 (94,5/100) |
| `Appendix_Artifacts_and_Folder_Structure.md` | Phụ lục cấu trúc artifact (bản kỹ thuật chi tiết) |

### 8.2. Bản nháp / mở rộng

| File | Nội dung |
|------|----------|
| `new_Chapter2.md` … `new_Chapter5.md` | Bản mở rộng từng chương (draft) |
| `expandChapter.md` | Hướng dẫn mở rộng nội dung chương |
| `Thesis_Full 37fd93dfd386805781b7caf6a458fcda.md` | Bản gộp full thesis (export) |

### 8.3. Script hỗ trợ thesis

| File | Nội dung |
|------|----------|
| `_restructure_expand_full.py` | Script tái cấu trúc và mở rộng thesis |
| `_fix_expand_numbering.py` | Sửa đánh số heading sau expand |
| `_verify_expand.py` | Kiểm tra chất lượng sau expand |
| `_verify_expand_report.json` | Báo cáo kết quả verify |

---

## 9. `docs/` — Tài liệu kỹ thuật

| File | Nội dung |
|------|----------|
| `00_Literature_Review_SOTA.md` | 20 papers, bảng SOTA 3-tier, 8 research gaps |
| `01_Tong_Quan_Du_An.md` | Tổng quan dự án, abstract, mục tiêu |
| `02_Phase1_Data_Preparation.md` | Hướng dẫn Phase 1: EDA, clean, split |
| `03_Phase2_Feature_Engineering.md` | Phase 2: ModernBERT + behavioral 777-d |
| `04_Phase3_PCA_Feature_Selection.md` | Phase 3: PCA diagnostic/ablation |
| `05_Phase4_PSO_Model_Training.md` | Phase 4: DL + PSO trên PCA |
| `06_Phase5_Hybrid_Ensemble.md` | Phase 5: model zoo, blend, stacking |
| `07_Phase6-8_Robustness_XAI_Ablation.md` | Phase 6–8: adversarial, XAI, đánh giá |
| `08_Ket_Luan_va_Huong_Dan.md` | **Tổng kết + hướng dẫn chạy Colab** — đọc đầu tiên |

> Khi demo code trước hội đồng, ưu tiên mở `notebooks/` thay vì `docs/`.

---

## 10. `scripts/` — Script phụ trợ

> **Lưu ý:** Pipeline chính nằm trong notebook. Các script này hỗ trợ annotate, benchmark và kiểm tra — không thay thế notebook.

| File | Mục đích | Nội dung |
|------|----------|----------|
| `run_baseline_benchmark.py` | Benchmark sklearn | Train LR, LinearSVC, RF trên 777-d; ghi `baseline_benchmark_*.csv` |
| `annotate_notebooks_vn.py` | Annotate notebook | Thêm comment tiếng Việt trước import/def trong cell code |
| `annotate_detailed_vn.py` | Annotate chi tiết | Comment chi tiết hơn cho từng dòng logic |
| `fix_comments_vn.py` | Sửa comment | Chỉnh sửa comment tiếng Việt đã annotate |
| `verify_comments.py` | Kiểm tra comment | Verify coverage comment trong notebook |
| `verify_coverage.py` | Kiểm tra coverage | Báo cáo % cell đã có comment |
| `__pycache__/*.pyc` | Cache Python | File biên dịch tự sinh — có thể bỏ qua |

---

## 11. `logs/` — Log thí nghiệm

| File | Nội dung |
|------|----------|
| `.gitkeep` | Giữ thư mục trống trong Git |

Thư mục dành cho log PSO history, run summary (hiện chưa có file log cụ thể).

---

## 12. `.planning/` — Lập kế hoạch nội bộ

| File | Nội dung |
|------|----------|
| `PROJECT.md` | Mô tả dự án, requirements, constraints, key decisions |
| `STATE.md` | Trạng thái hiện tại các phase |
| `research/SUMMARY.md` | Tóm tắt nghiên cứu ban đầu |
| `research/STACK.md` | Stack công nghệ đã chọn |
| `research/PITFALLS.md` | Các bẫy / rủi ro đã ghi nhận |

---

## 13. `_archive/` — Lưu trữ (203 file, không nộp bài)

Thư mục chứa file cũ được chuyển ra khỏi gói nộp chính. Xem `_archive/README.md` để biết lý do.

| Thư mục con | Nội dung | Lý do archive |
|-------------|----------|---------------|
| `.agents/` | Log AI agent: handoff, briefing, draft chương, scripts parse | Scratch workspace, không phải deliverable |
| `planning/` | UAT/VERIFICATION legacy Phase 01–10, ROADMAP, TASKLIST cũ | Đã thay bằng `.planning/PROJECT.md` |
| `root_misc/` | `AGENTS.md`, `ORIGINAL_REQUEST.md`, PDF nhóm cũ | File gốc đã di chuyển |
| `reports/tables/pso_trials/` | ~90 file `phase4_pso_trial_XXX_history.csv` | Chi tiết từng trial; tổng hợp tại `phase4_pso_trial_history.csv` |
| `reports/tables/seed_123/`, `seed_456/` | Bảng Phase 5 chi tiết theo seed | Tổng hợp tại `phase7_multiseed_summary.csv` |

**Khôi phục:** di chuyển ngược từ `_archive/` về vị trí gốc tương ứng.

---

## 14. Bảng tra nhanh

| Câu hỏi | Mở file / thư mục |
|---------|-------------------|
| Bắt đầu từ đâu? | `README.md` → `docs/08_Ket_Luan_va_Huong_Dan.md` |
| Dataset và split? | `data/processed/*.csv`, notebook `01` |
| Feature 777-d gồm gì? | `artifacts/features/feature_metadata.json`, notebook `02` |
| Model cuối cùng? | `artifacts/ensemble/phase5_weighted_blend_metadata.json` |
| Số 0,9463 từ đâu? | `reports/tables/phase7_final_metrics.csv` (τ=0,30) |
| So sánh ≥5 model? | Cùng file, filter `split=test`, `threshold=0.5` |
| Ablation? | `reports/tables/phase7_ablation_results.csv` |
| SHAP top feature? | `reports/tables/phase6_final_shap_global_importance.csv` |
| Gói nộp bài? | `reports/final/phase8_submission_package_manifest.csv` |
| Luận văn đầy đủ? | `thesis/Chapter1` → `Chapter6` + `References.md` |
| Cấu trúc artifact chi tiết? | `thesis/Appendix_Artifacts_and_Folder_Structure.md` |

---

## 15. Dual-track — Không nhầm artifact

| Track | Feature | Model headline | Macro F1 test |
|-------|---------|----------------|---------------|
| **Final** | Raw **777-d** | `phase5_weighted_blend` | **0,9463** @ τ=0,30 |
| **Ablation** | PCA **400-d** | `dl_baseline`, `dl_pso` | 0,77–0,86 |

Hai track **không hợp nhất** ở inference. Mọi số SOTA và XAI headline đến từ **final track**.

---

## 16. Tóm tắt một dòng

> **`data/`** cung cấp dữ liệu → **`notebooks/`** chạy pipeline → sinh **`artifacts/`** → **`reports/tables/phase7_*`** tổng hợp metric → **`thesis/`** và **`reports/final/`** đóng gói nộp bài.

---

*Tài liệu bổ sung cho `thesis/Appendix_Artifacts_and_Folder_Structure.md`. Mọi con số headline lấy từ `reports/tables/phase7_final_metrics.csv`.*