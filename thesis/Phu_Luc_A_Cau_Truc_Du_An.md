# PHỤ LỤC A — CẤU TRÚC DỰ ÁN VÀ HƯỚNG DẪN TÁI LẬP

**Đề tài:** Dual-Track ModernBERT and Behavioral Fusion with Threshold-Selected CNN–GBDT Ensemble for Amazon Fake Review Detection
**Repository:** `Fake_reviews/`
**Đường dẫn Colab:** `/content/drive/MyDrive/BaoMatCuoiKy/Fake_reviews`
**Cập nhật:** 2026-07-02 | Pipeline final: `phase5_weighted_blend` (Macro F1 **0,9463** @ τ = 0,30)

> Phụ lục này bổ sung Chương 3 (phương pháp) và Chương 4 (triển khai). Mọi con số headline lấy từ `reports/tables/phase7_final_metrics.csv` (SSOT).

---

## A.1. Cấu trúc tổng thể thư mục dự án

Dự án được tổ chức theo **ba lớp** tách bạch: đầu vào & mã nguồn, kết quả chạy pipeline, và gói nộp bài.

### A.1.1. Ba lớp chức năng

| Lớp | Thư mục chính | Vai trò |
|-----|---------------|---------|
| **Lớp 1 — Đầu vào & mã nguồn** | `data/`, `notebooks/` | Dataset gốc; notebook Jupyter thực thi Phase 1–10 |
| **Lớp 2 — Kết quả chạy** | `artifacts/`, `reports/` | Model đã train, feature, xác suất, bảng số, biểu đồ |
| **Lớp 3 — Nộp bài & luận văn** | `thesis/`, `reports/final/`, `docs/` | Văn bản học thuật; manifest Phase 8; tài liệu kỹ thuật |

### A.1.2. Luồng dữ liệu end-to-end

```
data/final_labeled_fake_reviews.csv          (50.000 mẫu gốc)
        │
        ▼ Phase 1 — 01_EDA_Preprocessing.ipynb
data/processed/  (clean 42.749 + split 70/15/15, seed 42)
        │
        ▼ Phase 2 — 02_Feature_Engineering.ipynb
artifacts/features/  (ModernBERT 768-d + behavioral 9-d → fused 777-d)
        │
        ├──────────────────────────────┬─────────────────────────────┐
        ▼ Phase 5 (final track)        ▼ Phase 3–4 (ablation track)    │
artifacts/ensemble/                    artifacts/pca/ + models/       │
artifacts/predictions/                 artifacts/predictions/ (PCA)    │
        │                              │                             │
        ▼ Phase 6 — XAI / robustness   │                             │
artifacts/xai/                         │                             │
        │                              │                             │
        ▼ Phase 7 — đánh giá & ablation◄─────────────────────────────┘
reports/tables/phase7_*
        │
        ▼ Phase 8 — đóng gói
reports/final/  →  thesis/ (trích số vào luận văn)
```

### A.1.3. Hai luồng thí nghiệm (dual-track)

| Track | Không gian đặc trưng | Mô hình tiêu biểu | Vai trò trong luận văn |
|-------|----------------------|-------------------|------------------------|
| **Final (luồng chính)** | Raw **777-d** | `phase5_weighted_blend`, CNN-BiLSTM, XGBoost raw | Kết quả SOTA, XAI headline, ngưỡng kép |
| **Ablation / legacy** | PCA **400-d** | `dl_baseline`, `dl_pso`, `final_ensemble` | Negative result; robustness appendix |

Hai track **không hợp nhất** ở inference. Chỉ số báo cáo chính thức đến từ **final track**.

### A.1.4. Thư mục không dùng khi nộp bài

| Thư mục | Lý do loại trừ |
|---------|----------------|
| `_archive/` | Bản cũ, trial PSO chi tiết, planning legacy |
| `.planning/` | Ghi chú lập kế hoạch nội bộ |
| `notebooks/tests/` | Diagnostic, không bắt buộc |
| `scripts/` | Tiện ích annotate/benchmark; pipeline chính nằm trong notebook |

---

## A.2. Cây thư mục chi tiết

### A.2.1. Cây thư mục cấp một

```
Fake_reviews/
├── data/                    # Dữ liệu gốc và đã xử lý
├── notebooks/               # Source code Jupyter Phase 1–10
├── artifacts/               # Output model, feature, prediction, XAI
├── reports/                 # Bảng CSV, hình PNG, gói final Phase 8
├── thesis/                  # Luận văn Chương 1–6 + phụ lục
├── docs/                    # Tài liệu kỹ thuật theo phase
├── scripts/                 # Script phụ trợ (không phải pipeline chính)
├── logs/                    # Log chạy thí nghiệm
├── README.md
└── HUONG_DAN_CAU_TRUC_THU_MUC.md
```

### A.2.2. `data/` — Dữ liệu

```
data/
├── final_labeled_fake_reviews.csv    # ~50.000 review Amazon (KHÔNG ghi đè)
├── interim/                          # File trung gian cleaning
└── processed/
    ├── clean_reviews.csv             # 42.749 mẫu sau làm sạch
    ├── train.csv                     # 29.923 (70%)
    ├── val.csv                       # 6.413 (15%)
    ├── test.csv                      # 6.413 (15%)
    ├── split_metadata.json           # seed=42, stratified
    └── schema_metadata.json          # mô tả cột
```

### A.2.3. `notebooks/` — Pipeline chính

```
notebooks/
├── 01_EDA_Preprocessing.ipynb
├── 02_Feature_Engineering.ipynb
├── 03_PCA_Feature_Selection.ipynb
├── 04_PSO_Model_Training.ipynb
├── 05_00_Phase5_Run_Order.ipynb
├── 05_01_LightGBM_Raw.ipynb
├── 05_02_XGBoost_Raw.ipynb
├── 05_03_MLP_Raw.ipynb
├── 05_04a_CNN_BiLSTM_Sequence_seed42.ipynb
├── 05_04b_CNN_BiLSTM_Sequence_seed123.ipynb
├── 05_04c_CNN_BiLSTM_Sequence_seed456.ipynb
├── 05_05_Weighted_Blending.ipynb
├── 05_06_Stacking_Calibration.ipynb
├── 05_Hybrid_Ensemble.ipynb
├── 06_Adversarial_XAI.ipynb
├── 07_Evaluation_Ablation.ipynb
├── 08_Final_Report_Kaggle.ipynb
├── 09_Bieu_Do_So_Sanh_Mo_Hinh.ipynb
├── 10_Baseline_Algorithm_Benchmark.ipynb
└── tests/
    └── 01_DL_PCA_Diagnostic_Test.ipynb
```

### A.2.4. `artifacts/` — Kết quả model

```
artifacts/
├── features/          # Phase 2: 777-d, behavioral, BERT embedding
├── pca/               # Phase 3: PCA 400-d (ablation)
├── models/            # Phase 4–5: checkpoint DL (.pth)
├── ensemble/          # Phase 5: GBDT, blend metadata, scaler
├── predictions/       # Xác suất P(fake) train/val/test
├── xai/               # Phase 6: SHAP, LIME HTML
├── evaluation/        # Metadata Phase 7, benchmark
├── diagnostics/       # DL-PCA diagnostic
└── figures/           # Hình phụ (hình chính ở reports/figures/)
```

### A.2.5. `reports/` — Bảng số và gói nộp

```
reports/
├── tables/
│   ├── phase1_*   (12 file)   # EDA, cleaning
│   ├── phase2_*   (5 file)    # Feature engineering
│   ├── phase3_*   (5 file)    # PCA
│   ├── phase4_*   (7 file)    # PSO / DL
│   ├── phase5_*   (29 file)   # Leaderboard, blend, SOTA
│   ├── phase6_*   (20 file)   # Robustness, SHAP, LIME
│   └── phase7_*   (12 file)   # ★ SSOT đánh giá cuối
├── figures/                   # PNG biểu đồ theo phase
├── final/                     # Gói Phase 8
│   ├── Phase8_Final_Report.md
│   ├── phase8_report_summary.csv
│   ├── phase8_submission_package_manifest.csv
│   ├── phase8_artifact_inventory.csv
│   └── phase8_run_order_checklist.csv
└── diagnostics/
```

### A.2.6. `thesis/` và `docs/`

```
thesis/
├── Chapter1_Introduction.md … Chapter6_Conclusion.md
├── References.md
├── Self_Assessment_Rubric.md
└── Phu_Luc_A_Cau_Truc_Du_An.md    ← phụ lục này

docs/
├── 00_Literature_Review_SOTA.md
├── 01_Tong_Quan_Du_An.md … 07_Phase6-8_*.md
└── 08_Ket_Luan_va_Huong_Dan.md    ← hướng dẫn chạy tóm tắt
```

---

## A.3. Mô tả các artifact theo phase

Bảng dưới liệt kê **artifact bắt buộc** theo từng phase. Kiểm tra tồn tại: `reports/final/phase8_artifact_inventory.csv`.

### Phase 1 — EDA và tiền xử lý

| Artifact | Đường dẫn | Mô tả |
|----------|-----------|-------|
| Dataset gốc | `data/final_labeled_fake_reviews.csv` | ~50.000 review; không rewrite sau Phase 1 |
| Corpus sạch | `data/processed/clean_reviews.csv` | 42.749 mẫu (loại dup + missing) |
| Split | `data/processed/{train,val,test}.csv` | 70/15/15 stratified, seed 42 |
| Metadata split | `data/processed/split_metadata.json` | Ghi seed, tỷ lệ, số dòng |
| Báo cáo EDA | `reports/tables/phase1_*.csv` | 12 bảng: nhãn, rating, sentiment, temporal… |
| Hình EDA | `reports/figures/phase1_*.png` | Biểu đồ phân bố, fake rate |

**Notebook:** `01_EDA_Preprocessing.ipynb`

---

### Phase 2 — Trích xuất đặc trưng

| Artifact | Đường dẫn | Mô tả |
|----------|-----------|-------|
| Vector fused 777-d | `artifacts/features/features_raw_{train,val,test}.npy` | ModernBERT 768 + behavioral 9 |
| BERT embedding | `artifacts/features/bert_{train,val,test}.npy` | Embedding 768-d riêng |
| Behavioral | `artifacts/features/behavioral_{train,val,test}.csv` | 9 đặc trưng hành vi |
| Nhãn | `artifacts/features/labels_{train,val,test}.npy` | 0=real, 1=fake |
| Scaler | `artifacts/features/behavioral_scaler.joblib` | Fit **chỉ train** |
| Metadata | `artifacts/features/feature_metadata.json` | Backbone `answerdotai/ModernBERT-base` |
| Từ điển feature | `artifacts/features/feature_dictionary.csv` | Tên và mô tả từng chiều |
| Báo cáo | `reports/tables/phase2_*.csv` | Chất lượng feature, leakage check |

**Notebook:** `02_Feature_Engineering.ipynb`

---

### Phase 3 — PCA / giảm chiều (ablation track)

| Artifact | Đường dẫn | Mô tả |
|----------|-----------|-------|
| Reducer | `artifacts/pca/pca_or_svd.joblib` | PCA/SVD fit train-only |
| Scaler PCA | `artifacts/pca/pca_scaler.joblib` | Chuẩn hóa trước PCA |
| Feature 400-d | `artifacts/pca/features_final_{train,val,test}.npy` | Vector sau giảm chiều |
| Metadata | `artifacts/pca/phase3_metadata.json` | Retained variance ≈ 95,10% |
| Báo cáo | `reports/tables/phase3_*.csv` | Variance, RAM |

**Notebook:** `03_PCA_Feature_Selection.ipynb`
**Lưu ý:** Không phải input của pipeline final; dùng cho ablation và legacy DL.

---

### Phase 4 — DL baseline + PSO (ablation track)

| Artifact | Đường dẫn | Mô tả |
|----------|-----------|-------|
| DL baseline | `artifacts/models/baseline_model_dl.pth` | Macro F1 test ≈ 0,7665 |
| DL-PSO | `artifacts/models/best_model_dl.pth` | Macro F1 test ≈ 0,7793 |
| Hyperparameter | `artifacts/models/best_params.json` | Tham số PSO tối ưu |
| Metadata | `artifacts/models/phase4_metadata.json` | Config training |
| Xác suất | `artifacts/predictions/dl_{baseline,pso}_*_prob.npy` | P(fake) train/val/test |
| Báo cáo | `reports/tables/phase4_*.csv` | So sánh baseline vs PSO |

**Notebook:** `04_PSO_Model_Training.ipynb`

---

### Phase 5 — Model zoo và ensemble (final track)

| Artifact | Đường dẫn | Mô tả |
|----------|-----------|-------|
| **★ Blend headline** | `artifacts/ensemble/phase5_weighted_blend_metadata.json` | CNN 50% + XGB 50%; τ balanced 0,30 / precision-first 0,60 |
| XGBoost raw | `artifacts/ensemble/phase5_xgb_raw.joblib` | Macro F1 test 0,9059 @ τ=0,50 |
| LightGBM raw | `artifacts/ensemble/phase5_lgbm_raw.joblib` | Macro F1 test 0,9051 |
| CNN-BiLSTM | `artifacts/models/phase5_cnn_bilstm_sequence.pth` | Macro F1 test 0,9343 |
| MLP raw | `artifacts/models/phase5_mlp_raw.pth` | Macro F1 test 0,8990 |
| Stacking | `artifacts/ensemble/phase5_stack_*.joblib` | Meta-learner + calibration |
| Xác suất blend | `artifacts/predictions/phase5_weighted_blend_*_prob.npy` | Nguồn metric Phase 7 |
| Multi-seed | `artifacts/ensemble/seed_{123,456}/`, `predictions/seed_*/` | Ổn định 3 seed |
| Legacy ensemble | `artifacts/ensemble/final_ensemble_model.pkl` | PCA blend; adversarial appendix |
| Báo cáo | `reports/tables/phase5_*.csv` | Leaderboard, sweep blend, threshold |
| SOTA | `reports/tables/literature_references_20.csv` | 20 paper đã xác minh |
| SOTA so sánh | `reports/tables/literature_sota_comparison.csv` | Ma trận 3-tier |

**Notebooks:** `05_00` → `05_06` → `05_Hybrid_Ensemble.ipynb`

#### Bảng metric headline (test set, seed 42)

| Mô hình | τ | Macro F1 | Prec. Fake | ROC-AUC |
|---------|---|----------|------------|---------|
| weighted_blend (balanced) | 0,30 | **0,9463** | 0,9344 | 0,9769 |
| weighted_blend (default) | 0,50 | 0,9433 | 0,9699 | 0,9769 |
| weighted_blend (precision-first) | 0,60 | 0,9126 | **0,9816** | 0,9769 |
| CNN-BiLSTM sequence | 0,50 | 0,9343 | 0,9465 | 0,9726 |
| XGBoost raw 777-d | 0,50 | 0,9059 | 0,9686 | 0,9531 |

*Nguồn: `reports/tables/phase7_final_metrics.csv`*

---

### Phase 6 — Robustness và XAI

| Artifact | Đường dẫn | Mô tả |
|----------|-----------|-------|
| SHAP headline | `reports/tables/phase6_final_shap_global_importance.csv` | Top feature trên XGBoost final |
| LIME cases | `artifacts/xai/phase6_final_lime_case_0{1..6}_*.html` | 6 case: TP, TN, FP, FN, high-conf |
| Tóm tắt LIME | `reports/tables/phase6_final_lime_case_summary.csv` | Mô tả 6 case |
| Robustness legacy | `reports/tables/phase6_robustness_metric_drops.csv` | FGSM/PGD trên legacy ensemble |
| Metadata | `artifacts/xai/phase6_final_metadata.json` | Config XAI final track |

**Notebook:** `06_Adversarial_XAI.ipynb`
**Lưu ý:** Robustness adversarial chạy trên legacy PCA ensemble; XAI headline trên final track.

---

### Phase 7 — Đánh giá, ablation, audit

| Artifact | Đường dẫn | Mô tả |
|----------|-----------|-------|
| **★ SSOT metrics** | `reports/tables/phase7_final_metrics.csv` | Tất cả model, split, threshold |
| Target audit | `reports/tables/phase7_target_audit.csv` | Pass/fail F1≥0,89 / Prec≥0,975 / AUC≥0,93 |
| Ablation | `reports/tables/phase7_ablation_results.csv` | Models A–E @ τ=0,50 |
| Delta ablation | `reports/tables/phase7_ablation_delta.csv` | Δ so với full model |
| Multi-seed | `reports/tables/phase7_multiseed_summary.csv` | Mean ± std (42, 123, 456) |
| CV | `reports/tables/phase7_cv_summary.csv` | 5-fold CV surrogate |
| Error analysis | `reports/tables/phase7_error_analysis.csv` | Phân tích lỗi dự đoán |
| Metadata | `artifacts/evaluation/phase7_metadata.json` | Config ablation, targets |

**Notebook:** `07_Evaluation_Ablation.ipynb`

#### Ablation Models A–E (test, τ = 0,50)

| Model | Mô tả | Macro F1 |
|-------|-------|----------|
| Full | `phase5_weighted_blend` | 0,9433 |
| A | Chỉ XGBoost raw | 0,9059 |
| B | LightGBM raw (no PCA) | 0,9058 |
| C | LGBM thiếu advanced behavioral | 0,8684 |
| D | Chỉ CNN-BiLSTM sequence | 0,9343 |
| E | DL baseline gần nhất | 0,7665 |

---

### Phase 8 — Đóng gói và manifest

| Artifact | Đường dẫn | Mô tả |
|----------|-----------|-------|
| Báo cáo cuối | `reports/final/Phase8_Final_Report.md` | Tổng hợp Phase 1–7 |
| Facts compact | `reports/final/phase8_report_summary.csv` | Số liệu rút gọn cho báo cáo |
| Manifest nộp | `reports/final/phase8_submission_package_manifest.csv` | Danh mục file bắt buộc |
| Inventory | `reports/final/phase8_artifact_inventory.csv` | Kiểm tra exists + kích thước |
| Thứ tự chạy | `reports/final/phase8_run_order_checklist.csv` | Notebook 01→08 |

**Notebook:** `08_Final_Report_Kaggle.ipynb`

---

### Phase 9–10 — Bổ sung (không bắt buộc nộp)

| Phase | Notebook | Output chính |
|-------|----------|--------------|
| 9 | `09_Bieu_Do_So_Sanh_Mo_Hinh.ipynb` | `reports/figures/benchmark_tier1_macro_f1.png` |
| 10 | `10_Baseline_Algorithm_Benchmark.ipynb` | `reports/tables/baseline_benchmark_metrics.csv` |

---

## A.4. Hướng dẫn tái lập thí nghiệm

### A.4.1. Yêu cầu môi trường

| Thành phần | Phiên bản / cấu hình |
|------------|----------------------|
| Nền tảng | **Google Colab** (khuyến nghị); RAM cap **12 GB** |
| GPU | T4 16 GB (freeze ModernBERT; `max_length=160`) |
| Python | 3.12 |
| PyTorch | 2.11 |
| XGBoost / LightGBM | 3.2 / 4.6 |
| SHAP | 0.52 |
| Seed hệ thống | **42** (split + train chính) |
| Dataset | `data/final_labeled_fake_reviews.csv` (Kaggle Amazon Labeled Fake Reviews) |

> **Quy tắc:** Không chạy training/tuning trên máy local trừ khi được phê duyệt. Pipeline thiết kế Colab-first.

### A.4.2. Thứ tự chạy notebook (bắt buộc)

Nguồn chính thức: `reports/final/phase8_run_order_checklist.csv`.

| STT | Notebook | Phase | Đầu vào | Đầu ra chính |
|-----|----------|-------|---------|--------------|
| 1 | `01_EDA_Preprocessing.ipynb` | 1 | `data/final_labeled_fake_reviews.csv` | `data/processed/` |
| 2 | `02_Feature_Engineering.ipynb` | 2 | `data/processed/` | `artifacts/features/` |
| 3 | `03_PCA_Feature_Selection.ipynb` | 3 | `artifacts/features/` | `artifacts/pca/` |
| 4 | `04_PSO_Model_Training.ipynb` | 4 | `artifacts/pca/` | `artifacts/models/`, `predictions/dl_*` |
| 5 | `05_00_Phase5_Run_Order.ipynb` | 5 | Checklist | — |
| 5a | `05_01_LightGBM_Raw.ipynb` | 5 | `artifacts/features/` | `ensemble/phase5_lgbm_raw.*` |
| 5b | `05_02_XGBoost_Raw.ipynb` | 5 | `artifacts/features/` | `ensemble/phase5_xgb_raw.*` |
| 5c | `05_03_MLP_Raw.ipynb` | 5 | `artifacts/features/` | `models/phase5_mlp_raw.*` |
| 5d | `05_04a_CNN_BiLSTM_Sequence_seed42.ipynb` | 5 | `artifacts/features/` | `models/phase5_cnn_bilstm_sequence.*` |
| 5e | `05_05_Weighted_Blending.ipynb` | 5 | `artifacts/predictions/` | `ensemble/phase5_weighted_blend_metadata.json` |
| 5f | `05_06_Stacking_Calibration.ipynb` | 5 | predictions | `ensemble/phase5_stack_*` |
| 5g | `05_Hybrid_Ensemble.ipynb` | 5 | Phase 5 artifacts | `reports/tables/phase5_*` |
| 6 | `06_Adversarial_XAI.ipynb` | 6 | `artifacts/ensemble/` | `artifacts/xai/`, `phase6_*` |
| 7 | `07_Evaluation_Ablation.ipynb` | 7 | Phase 2–6 | `reports/tables/phase7_*` |
| 8 | `08_Final_Report_Kaggle.ipynb` | 8 | Phase 1–7 | `reports/final/` |

**Tùy chọn sau Phase 8:**

```
10_Baseline_Algorithm_Benchmark.ipynb  →  baseline_benchmark_metrics.csv
09_Bieu_Do_So_Sanh_Mo_Hinh.ipynb       →  biểu đồ so sánh
05_04b/c_*_seed{123,456}.ipynb         →  multi-seed evaluation
```

### A.4.3. Quy tắc phương pháp luận (bắt buộc tuân thủ)

| Quy tắc | Mô tả | Áp dụng |
|---------|-------|---------|
| **Seed cố định** | `seed = 42` cho split và train chính | Toàn pipeline |
| **Stratified split** | 70% train / 15% val / 15% test; giữ tỷ lệ nhãn | Phase 1 |
| **Fit train-only** | Scaler, PCA, aggregate chỉ học trên train | Phase 2–5 |
| **Chọn τ trên val** | Ngưỡng và trọng số blend chọn từ validation | Phase 5 |
| **Audit test một lần** | Không chỉnh pipeline sau khi đọc test | Phase 7 |
| **Không rewrite raw** | `final_labeled_fake_reviews.csv` bất biến | Phase 1+ |

### A.4.4. Kiểm tra tái lập nhanh (smoke test)

Sau khi chạy xong Phase 7, xác nhận các file sau **tồn tại** và metric khớp SSOT:

```text
reports/tables/phase7_final_metrics.csv
reports/tables/phase7_target_audit.csv
reports/tables/phase7_ablation_results.csv
artifacts/ensemble/phase5_weighted_blend_metadata.json
artifacts/predictions/phase5_weighted_blend_test_prob.npy
```

**Metric kỳ vọng (seed 42, test set):**

| Kiểm tra | Giá trị kỳ vọng | File |
|----------|-----------------|------|
| Balanced Macro F1 | ≈ 0,9463 @ τ=0,30 | `phase7_final_metrics.csv` |
| Precision-first Prec. Fake | ≈ 0,9816 @ τ=0,60 | `phase7_final_metrics.csv` |
| ROC-AUC | ≈ 0,9769 | `phase7_final_metrics.csv` |
| Target audit | 3/3 PASS | `phase7_target_audit.csv` |
| Multi-seed balanced F1 | 0,9485 ± 0,0018 | `phase7_multiseed_summary.csv` |

Sai lệch > ±0,005 so với SSOT → kiểm tra seed, đường dẫn artifact, hoặc thứ tự notebook.

### A.4.5. Gói nộp tối thiểu

Theo `phase8_submission_package_manifest.csv`:

| Hạng mục | Đường dẫn |
|----------|-----------|
| Notebook cuối | `notebooks/08_Final_Report_Kaggle.ipynb` |
| Báo cáo Markdown | `reports/final/Phase8_Final_Report.md` |
| Metrics SSOT | `reports/tables/phase7_final_metrics.csv` |
| Target audit | `reports/tables/phase7_target_audit.csv` |
| Ablation | `reports/tables/phase7_ablation_results.csv` |
| Model artifacts | `artifacts/models/`, `artifacts/ensemble/` |
| Inventory + checklist | `reports/final/phase8_*.csv` |

### A.4.6. Lộ trình demo trước hội đồng (gợi ý 15–20 phút)

```
01_EDA_Preprocessing        →  dataset + split
02_Feature_Engineering      →  777-d fusion
05_02_XGBoost_Raw           →  nhánh tabular
05_04a_CNN_BiLSTM (seed42)  →  nhánh sequence
05_05_Weighted_Blending     →  grid blend + τ
07_Evaluation_Ablation      →  ablation + target audit
(tuỳ chọn) 08_Final_Report   →  manifest Phase 8
```

### A.4.7. Bảng tra nhanh

| Câu hỏi | Mở file / notebook |
|---------|-------------------|
| Bắt đầu từ đâu? | `README.md` → `docs/08_Ket_Luan_va_Huong_Dan.md` |
| Dataset và split? | `data/processed/*.csv`, notebook `01` |
| Feature 777-d? | `artifacts/features/feature_metadata.json`, notebook `02` |
| Model cuối cùng? | `artifacts/ensemble/phase5_weighted_blend_metadata.json` |
| Số 0,9463 từ đâu? | `reports/tables/phase7_final_metrics.csv` (τ=0,30) |
| Ablation? | `reports/tables/phase7_ablation_results.csv` |
| SHAP top feature? | `reports/tables/phase6_final_shap_global_importance.csv` |
| Gói nộp đầy đủ? | `reports/final/phase8_submission_package_manifest.csv` |

---

*Phụ lục A — kết thúc. Liên hệ Chương 3 (§3.12 kiến trúc pipeline) và Chương 4 (§4.15 gói artifact Phase 8).*
