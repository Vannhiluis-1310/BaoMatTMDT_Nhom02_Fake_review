# PHỤ LỤC: CẤU TRÚC THƯ MỤC VÀ ARTIFACT DỰ ÁN

Tài liệu này mô tả cấu trúc repository `Fake_reviews`, luồng dữ liệu giữa các phase, và ánh xạ file artifact → mô hình. Dùng khi tái lập thí nghiệm, nộp bài, hoặc trình bày trực tiếp notebook trước hội đồng. Cập nhật theo artifact Phase 6–8 (generated **2026-06-11**).

**Đường dẫn gốc:** `Fake_reviews/` (Google Colab: `/content/drive/MyDrive/BaoMatCuoiKy/Fake_reviews`).

---

## A.1. Tổng quan ba lớp

| Lớp | Thư mục | Vai trò |
|-----|---------|---------|
| **Đầu vào & code** | `data/`, `notebooks/` | Dữ liệu gốc; notebook Jupyter chạy tuần tự Phase 1–8 |
| **Kết quả chạy** | `artifacts/`, `reports/` | Model đã train, feature, xác suất, bảng số, hình |
| **Nộp bài & luận văn** | `thesis/`, `reports/final/` | Văn bản học thuật; manifest Phase 8 |

**Luồng end-to-end:**

```
data/ (CSV gốc)
  → Phase 1: data/processed/ (split 70/15/15)
  → Phase 2: artifacts/features/ (777-d)
       ├→ Phase 5 final track → artifacts/ensemble/ + artifacts/predictions/
       └→ Phase 3–4 ablation (PCA 400-d) → artifacts/pca/ + artifacts/models/
  → Phase 6: artifacts/xai/
  → Phase 7: reports/tables/phase7_*
  → Phase 8: reports/final/
  → thesis/ (trích số từ reports/tables/)
```

**Nguyên tắc đọc số:** Luận văn và báo cáo lấy số từ `reports/tables/phase7_final_metrics.csv` (SSOT metrics). File `*.npy` trong `artifacts/predictions/` là nguồn xác suất gốc; Phase 7 aggregate thành CSV.

---

## A.2. Cây thư mục cấp một

```
Fake_reviews/
├── data/                 # Dữ liệu gốc và đã xử lý
├── notebooks/            # Code Jupyter Phase 1–8 (+ tests/)
├── artifacts/            # Output model, feature, prediction, XAI
├── reports/              # Bảng CSV, hình PNG, gói final Phase 8
├── thesis/               # Luận văn Chương 1–6 + phụ lục này
├── docs/                 # Tài liệu kỹ thuật theo phase (tham khảo)
├── logs/                 # Log chạy
├── _archive/             # Bản cũ — không dùng khi báo cáo
├── .planning/            # Ghi chú lập kế hoạch nội bộ
├── README.md
└── .gitignore
```

**Không dùng khi báo cáo:** `_archive/`, `.planning/` (trừ khi tham khảo lịch sử dự án).

---

## A.3. `data/` — Dữ liệu

| Đường dẫn | Mô tả |
|-----------|--------|
| `data/final_labeled_fake_reviews.csv` | Dataset gốc Amazon (~50.000 dòng). **Không ghi đè** sau Phase 1 |
| `data/processed/clean_reviews.csv` | Sau làm sạch: **42.749** mẫu |
| `data/processed/train.csv` | Train **29.923** (70%) |
| `data/processed/val.csv` | Validation **6.413** (15%) |
| `data/processed/test.csv` | Test **6.413** (15%) |
| `data/processed/split_metadata.json` | Stratified split, seed = 42 |
| `data/interim/` | File trung gian (nếu có) |

**Notebook:** `notebooks/01_EDA_Preprocessing.ipynb`

---

## A.4. `notebooks/` — Thứ tự chạy (Phase 1–8)

Nguồn chính thức: `reports/final/phase8_run_order_checklist.csv`.

| STT | Notebook | Phase | Mục đích |
|-----|----------|-------|----------|
| 1 | `01_EDA_Preprocessing.ipynb` | 1 | EDA, làm sạch, split |
| 2 | `02_Feature_Engineering.ipynb` | 2 | ModernBERT + 9 behavioral → 777-d |
| 3 | `03_PCA_Feature_Selection.ipynb` | 3 | PCA/SVD diagnostic 777→400 |
| 4 | `04_PSO_Model_Training.ipynb` | 4 | DL baseline + DL-PSO trên PCA |
| 5 | `05_01_LightGBM_Raw.ipynb` | 5 | LightGBM raw 777-d |
| 5 | `05_02_XGBoost_Raw.ipynb` | 5 | XGBoost raw 777-d |
| 5 | `05_03_MLP_Raw.ipynb` | 5 | MLP raw 777-d |
| 5 | `05_04a/b/c_CNN_BiLSTM_Sequence_seed{42,123,456}.ipynb` | 5 | Sequence DL (multi-seed) |
| 5 | `05_05_Weighted_Blending.ipynb` | 5 | Grid trọng số blend trên val |
| 5 | `05_06_Stacking_Calibration.ipynb` | 5 | Stacking + calibration |
| 5 | `05_Hybrid_Ensemble.ipynb` | 5 | Tổng hợp ensemble Phase 5 |
| 6 | `06_Adversarial_XAI.ipynb` | 6 | FGSM/PGD + SHAP/LIME |
| 7 | `07_Evaluation_Ablation.ipynb` | 7 | Metrics, audit, ablation, CV |
| 8 | `08_Final_Report_Kaggle.ipynb` | 8 | Manifest, inventory, báo cáo cuối |

**Phụ:** `notebooks/tests/01_DL_PCA_Diagnostic_Test.ipynb` — diagnostic, không bắt buộc nộp.

**Gợi ý demo trước hội đồng:** `01` → `02` → `05_02` + `05_04a` → `05_05` → `07` → (tuỳ chọn) `08`.

---

## A.5. `artifacts/` — Bản đồ artifact theo phase

### A.5.1. `artifacts/features/` (Phase 2)

| File | Ý nghĩa |
|------|---------|
| `features_raw_{train,val,test}.npy` | Vector fused **777-d** (headline input final track) |
| `bert_{train,val,test}.npy` | Embedding ModernBERT **768-d** |
| `behavioral_{train,val,test}.csv` | **9** đặc trưng hành vi (5 basic + 4 advanced) |
| `labels_{train,val,test}.npy` | Nhãn nhị phân |
| `feature_metadata.json` | Backbone `answerdotai/ModernBERT-base`, fused_dim=777 |
| `feature_dictionary.csv` | Tên từng chiều feature |
| `behavioral_scaler.joblib` | Scaler fit train-only |
| `row_ids_*.csv` | Ánh xạ dòng → ID gốc |

### A.5.2. `artifacts/pca/` (Phase 3 — ablation track)

| File | Ý nghĩa |
|------|---------|
| `pca_or_svd.joblib`, `pca_scaler.joblib` | Reducer fit **chỉ train** |
| `features_final_{train,val,test}.npy` | Vector **400-d** sau PCA |
| `phase3_metadata.json` | Retained variance ≈ 95,10% |

### A.5.3. `artifacts/models/` (Phase 4–5 — deep learning)

| File | Model | Macro F1 test @ τ=0,50 |
|------|-------|------------------------|
| `baseline_model_dl.pth` | DL baseline (PCA, không PSO) | 0,7665 |
| `best_model_dl.pth` | DL-PSO | 0,7793 |
| `phase5_cnn_bilstm_sequence.pth` | CNN-BiLSTM-Attention (sequence) | 0,9343 |
| `phase5_mlp_raw.pth` | MLP tabular 777-d | 0,8990 |
| `best_params.json` | Hyperparameter PSO |
| `phase4_metadata.json`, `phase5_cnn_bilstm_sequence_metadata.json`, `phase5_mlp_raw_metadata.json` | Metadata train |
| `seed_123/`, `seed_456/` | Checkpoint multi-seed |

### A.5.4. `artifacts/ensemble/` (Phase 5 — model đã lưu)

| File | Model / vai trò |
|------|-----------------|
| `phase5_weighted_blend_metadata.json` | **★★ Cấu hình headline:** seed 42 → CNN 50% + XGB 50%; τ balanced 0,30 / precision-first 0,60 |
| `phase5_metadata.json` | Metadata tổng Phase 5 |
| `phase5_xgb_raw.joblib` + `phase5_xgb_raw_metadata.json` | XGBoost raw 777-d |
| `phase5_lgbm_raw.joblib` + `phase5_lgbm_raw_metadata.json` | LightGBM raw (train F1=1,0; seed 42 không vào blend) |
| `phase5_stacking_calibrated_metadata.json` | Stacking + calibration |
| `phase5_stack_*.joblib` | Thành phần meta-learner |
| `final_ensemble_model.pkl` | Legacy PCA ensemble (τ=0,79; adversarial appendix) |
| `xgboost_model.pkl`, `lightgbm_model.pkl`, `stacking_meta_model.pkl` | Legacy Phase 4 |
| `seed_123/`, `seed_456/` | Metadata blend theo seed (456: CNN 60% + LGBM 35% + XGB 5%) |

### A.5.5. `artifacts/predictions/` (xác suất P(fake))

Mỗi model: `*_train_prob.npy`, `*_val_prob.npy`, `*_test_prob.npy`.

#### Final track (pipeline chính — so sánh ≥5 model)

| Prefix | Mô hình | Macro F1 test @0,50 | Notebook train |
|--------|---------|---------------------|----------------|
| `phase5_weighted_blend_*` | **Weighted blend (đề xuất)** | **0,9433** | `05_05`, `05_Hybrid` |
| `phase5_cnn_bilstm_sequence_*` | CNN-BiLSTM sequence | 0,9343 | `05_04a` |
| `phase5_stacking_calibrated_*` | Stacking calibrated | 0,9105 | `05_06` |
| `phase5_xgb_raw_*` | XGBoost raw 777-d | 0,9059 | `05_02` |
| `phase5_lgbm_raw_*` | LightGBM raw 777-d | 0,9051 | `05_01` |
| `phase5_mlp_raw_*` | MLP raw 777-d | 0,8990 | `05_03` |

#### Ablation / legacy track

| Prefix | Mô hình | Macro F1 test @0,50 |
|--------|---------|---------------------|
| `dl_pso_*` | DL-PSO (PCA) | 0,7793 |
| `dl_baseline_*` | DL baseline (PCA) | 0,7665 |
| `final_ensemble_*` | Legacy PCA blend | ~0,80–0,86 |
| `xgboost_*`, `lightgbm_*`, `stacking_*` | Legacy GBDT/stacking Phase 4 | Khác `phase5_*` |

#### Multi-seed

| Thư mục | Nội dung |
|---------|----------|
| `predictions/seed_123/` | Xác suất + metadata seed 123 |
| `predictions/seed_456/` | Xác suất + metadata seed 456 |

### A.5.6. `artifacts/xai/` (Phase 6)

| File / nhóm | Track | Ghi chú |
|-------------|-------|---------|
| `phase6_final_shap_global_importance.csv` (trong `reports/tables/`) | **Headline** `final_raw_777` | Top-4 behavioral |
| `phase6_final_lime_case_0{1..6}_*.html` | LIME 6 case @ τ=0,30 | Case chọn từ `weighted_blend` |
| `phase6_final_metadata.json` | Metadata XAI final | |
| `phase6_metadata.json` | Dual-track: final + legacy appendix | |
| `phase6_shap_*`, `lime_case_*` (không prefix `final`) | Legacy PCA appendix | Không headline |

### A.5.7. Khác trong `artifacts/`

| Thư mục / file | Vai trò |
|----------------|---------|
| `evaluation/phase7_metadata.json` | Config Phase 7: ablation_evidence_map, CV, targets |
| `diagnostics/dl_pca_test/` | Diagnostic DL trên PCA |
| `figures/` | Hình phụ (hình chính ở `reports/figures/`) |

---

## A.6. `reports/` — Bảng số và hình

```
reports/
├── tables/
│   ├── phase1_*  (12)  EDA, cleaning
│   ├── phase2_*  (5)   feature stats
│   ├── phase3_*  (5)   PCA
│   ├── phase4_*  (7)   PSO / DL
│   ├── phase5_*  (29)  leaderboard, blend sweep, SOTA
│   ├── phase6_*  (20)  robustness, SHAP, LIME
│   └── phase7_*  (12)  ★ SSOT đánh giá cuối
├── figures/            PNG biểu đồ theo phase
├── final/              Gói Phase 8
└── diagnostics/
```

### A.6.1. File SSOT (single source of truth)

| File | Nội dung | Liên kết luận văn |
|------|----------|-------------------|
| `phase7_final_metrics.csv` | So sánh **tất cả model**, train/val/test, mọi τ | Bảng 4.1 (§4.7) |
| `phase7_target_audit.csv` | Pass/fail target 0,89 / 0,975 / 0,93 | §4.8 |
| `phase7_ablation_results.csv` | Models A–E @ τ=0,50 | Bảng 4.4 (§4.10) |
| `phase7_ablation_delta.csv` | Δ so với full model | §4.10, §5.2 |
| `phase7_multiseed_summary.csv` | Mean ± std 3 seed | §4.11 |
| `phase7_multiseed_metrics.csv` | τ theo seed trên val | §4.11 |
| `phase7_cv_summary.csv` | 5-fold CV LGBM PCA | §4.10 |
| `phase5_leaderboard.csv` | Xếp hạng candidate Phase 5 | §4.7 |
| `phase5_weighted_blending_sweep.csv` | Grid trọng số blend | Bảng 4.1b (§4.7.1) |
| `phase6_final_shap_global_importance.csv` | SHAP headline | §4.9.2 |
| `phase6_final_lime_case_summary.csv` | Tóm tắt 6 case LIME | §4.9.2 |
| `phase6_robustness_metric_drops.csv` | FGSM/PGD legacy | §4.9.1 |

### A.6.2. `reports/final/` — Gói Phase 8 (2026-06-11)

| File | Vai trò |
|------|---------|
| `Phase8_Final_Report.md` | Báo cáo Markdown tổng hợp |
| `phase8_report_summary.csv` | Facts compact (trích Phase 7) |
| `phase8_submission_package_manifest.csv` | Danh mục file bắt buộc nộp |
| `phase8_artifact_inventory.csv` | Kiểm tra exists + kích thước artifact |
| `phase8_run_order_checklist.csv` | Thứ tự notebook 01→08 |

Chi tiết gói nộp: **Bảng 4.15** (Chương 4, §4.15).

**Lưu ý inventory:** Liệt kê `phase6_robustness_metrics.csv` (legacy); XAI headline `phase6_final_*` nằm tại `artifacts/xai/` và `reports/tables/phase6_final_*` — luận văn trích dẫn trực tiếp (§4.9.2).

---

## A.7. `thesis/` — Tài liệu luận văn

| File | Nội dung |
|------|----------|
| `Chapter1_Introduction.md` | Mở đầu, RQ, đóng góp |
| `Chapter2_Theory.md` | Lý thuyết, SOTA 20 papers |
| `Chapter3_Methodology.md` | Pipeline, dual-track, Phase 7–8 |
| `Chapter4_Results.md` | Triển khai + kết quả, Bảng 4.1–4.15, rubric §4.14 |
| `Chapter5_Discussion.md` | Thảo luận, RQ1–RQ6 |
| `Chapter6_Conclusion.md` | Kết luận, triển khai |
| `References.md` | Tài liệu tham khảo APA |
| `Self_Assessment_Rubric.md` | Rubric D0–D8 (94,5/100) |
| `Appendix_Artifacts_and_Folder_Structure.md` | **Phụ lục này** |

---

## A.8. `docs/` — Tài liệu kỹ thuật (tham khảo)

| File | Nội dung |
|------|----------|
| `00_Literature_Review_SOTA.md` | 20 papers, gap G1–G8 |
| `01_Tong_Quan_Du_An.md` … `08_Ket_Luan_va_Huong_Dan.md` | Hướng dẫn từng phase |

Song song với notebook; khi demo code ưu tiên mở `notebooks/`.

---

## A.9. Dual-track — Không nhầm artifact

| Track | Feature | Model tiêu biểu | Số headline |
|-------|---------|-----------------|-------------|
| **Final (luồng chính)** | Raw **777-d** | `phase5_weighted_blend` | Macro F1 **0,9463** @ τ=0,30 |
| **Ablation / legacy** | PCA **400-d** | `dl_baseline`, `dl_pso`, `final_ensemble` | Thấp hơn; dùng negative result |

Hai track **không hợp nhất** ở inference. Mọi số SOTA và XAI headline đến từ final track trừ khi ghi chú `legacy_appendix_only`.

---

## A.10. Bảng tra nhanh — Câu hỏi → File / Notebook

| Câu hỏi | Mở |
|---------|-----|
| Dataset và split? | `data/processed/*.csv`, `01_EDA_Preprocessing.ipynb` |
| Feature 777-d gồm gì? | `artifacts/features/feature_metadata.json`, `02_Feature_Engineering.ipynb` |
| Train XGB / CNN ở đâu? | `05_02`, `05_04a` + `ensemble/phase5_xgb_raw.joblib`, `models/phase5_cnn_bilstm_sequence.pth` |
| Trọng số blend seed 42? | `ensemble/phase5_weighted_blend_metadata.json` |
| Số 0,9463 / 0,9816 từ đâu? | `reports/tables/phase7_final_metrics.csv` |
| So sánh ≥5 model? | Cùng file, filter `split=test`, `threshold=0.5` |
| Ablation Models A–E? | `reports/tables/phase7_ablation_results.csv` |
| Multi-seed ổn định? | `phase7_multiseed_summary.csv` |
| SHAP top feature? | `phase6_final_shap_global_importance.csv` |
| Gói nộp Phase 8? | `reports/final/phase8_submission_package_manifest.csv` |

---

## A.11. Bộ năm model gợi ý khi báo cáo (cùng test, τ=0,50)

| STT | Mô hình | Macro F1 | Artifact prediction |
|-----|---------|----------|---------------------|
| 1 | DL baseline (PCA) | 0,7665 | `dl_baseline_test_prob.npy` |
| 2 | XGBoost raw 777-d | 0,9059 | `phase5_xgb_raw_test_prob.npy` |
| 3 | CNN-BiLSTM sequence | 0,9343 | `phase5_cnn_bilstm_sequence_test_prob.npy` |
| 4 | Stacking calibrated | 0,9105 | `phase5_stacking_calibrated_test_prob.npy` |
| 5 | **Weighted blend (đề xuất)** | **0,9433** | `phase5_weighted_blend_test_prob.npy` |

*Có thể thay (1) bằng DL-PSO (0,7793) hoặc thêm LightGBM raw (0,9051) — tổng cộng repo có **≥7** họ mô hình trên cùng split.*

---

## A.12. Tóm tắt một dòng

> **`data/`** → **`notebooks/`** sinh **`artifacts/`** → **`reports/tables/phase7_*`** tổng hợp metric → **`thesis/`** và **`reports/final/`** đóng gói nộp bài.

---

*Phụ lục này bổ sung Chương 3 (§3.12) và Chương 4 (§4.15). Không thay thế SSOT số liệu: mọi con số headline vẫn lấy từ `phase7_final_metrics.csv`.*