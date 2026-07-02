# Bảng hàm thuật toán — Cell, Dòng & Giải thích

Tài liệu tra cứu **hàm thuật toán tự viết** trong pipeline Fake Review Detection: vị trí định nghĩa, nơi gọi, và ý nghĩa từng hàm.

**Quy ước đếm:**
- **Cell** = thứ tự cell trong notebook (gồm markdown), đúng như Jupyter/Colab hiển thị.
- **Dòng** = số dòng trong source của cell **code** đó.

**Bảng tra nhanh số liệu:** `reports/tables/phase7_final_metrics.csv`

---

## Luồng gọi tổng thể

```
Phase 2: load_bert_model → extract_or_load_bert_embeddings
         → add_basic_behavioral_features → fit_reviewer_behavior_map → add_advanced_behavioral_features
Phase 3: fit_selection_reducer → fit_final_reducer → transform (PCA/SVD 777→400)
Phase 4: build_model → fit_model → evaluate_model
         → pso_objective (callback) → fallback_pso → DL-PSO checkpoint
Phase 5: LGBM/XGB .fit() | MLP/CNN predict() | threshold_sweep → weighted blend → write_metrics
Phase 6: fgsm/pgd_attack_batch → evaluate_condition | SHAP TreeExplainer | LIME explain_instance
Phase 7: evaluate_test_probability → evaluate_probabilities (ablation, CV, multi-seed)
```

---

## Phase 2 — ModernBERT & Behavioral Features

**Notebook:** `notebooks/02_Feature_Engineering.ipynb`  
**Mục tiêu phase:** Tạo vector **777-d** = ModernBERT 768-d + 9 behavioral features.

### `load_bert_model(model_name)`

| | Cell | Dòng |
|---|------|------|
| **Định nghĩa** | 10 | 32 |
| **Gọi** | 11 | 2 |

**Giải thích:** Tải `AutoTokenizer` và `AutoModel` (ModernBERT) từ HuggingFace, chuyển lên GPU nếu có, đặt `eval()` để inference. Trả về `(tokenizer, model, device)` dùng cho toàn phase 2.

---

### `valid_cached_embedding(path, expected_rows)`

| | Cell | Dòng |
|---|------|------|
| **Định nghĩa** | 10 | 54 |
| **Gọi** | 10 | 75–76 |

**Giải thích:** Kiểm tra file `.npy` cache embedding đã tồn tại và đúng số dòng chưa. Tránh chạy lại BERT tốn RAM/thời gian khi rerun notebook.

---

### `mean_pool_last_hidden_state(...)`

| | Cell | Dòng |
|---|------|------|
| **Định nghĩa** | 10 | 18 |
| **Gọi** | 10 | 129 |

**Giải thích:** Mean pooling có attention mask trên `last_hidden_state` của BERT → vector **768-d** đại diện cho mỗi review.

---

### `extract_or_load_bert_embeddings(split_name, frame, ...)`

| | Cell | Dòng |
|---|------|------|
| **Định nghĩa** | 10 | 72 |
| **Gọi train** | 11 | 16 |
| **Gọi val** | 11 | 22 |
| **Gọi test** | 11 | 28 |

**Giải thích:** Với mỗi split (train/val/test): nếu cache hợp lệ thì load `.npy`, không thì tokenize batch, forward ModernBERT, pool → lưu `bert_{split}.npy`. Đây là bước tốn RAM nhất của pipeline.

---

### `add_basic_behavioral_features(frame)`

| | Cell | Dòng |
|---|------|------|
| **Định nghĩa** | 14 | 20 |
| **Gọi** | 14 | 71 |

**Giải thích:** Tính **5 đặc trưng hành vi cơ bản** từ metadata review: độ dài text, sentiment (VADER/TextBlob), rating, v.v. Gọi qua dict comprehension cho từng split.

---

### `fit_reviewer_behavior_map(train_frame, train_basic_features)`

| | Cell | Dòng |
|---|------|------|
| **Định nghĩa** | 15 | 142 |
| **Gọi** | 15 | 209 |

**Giải thích:** Fit **reviewer embedding / behavior score** chỉ trên **train** (không dùng nhãn fake/real → tránh leakage). Map này dùng cho feature nâng cao ở val/test.

---

### `add_advanced_behavioral_features(split_name, frame)`

| | Cell | Dòng |
|---|------|------|
| **Định nghĩa** | 16 | 29 |
| **Gọi** | 16 | 78 |

**Giải thích:** Tính **4 đặc trưng hành vi nâng cao**: review velocity, burst pattern, time gap, reviewer behavior score. Chỉ dùng lịch sử **train** khi tính cho val/test.

---

### `behavioral_scaler.fit_transform()` / `.transform()` (sklearn)

| | Cell | Dòng |
|---|------|------|
| **fit train** | 19 | 4 |
| **transform val/test** | 19 | 6, 8 |

**Giải thích:** Chuẩn hóa 9 behavioral features — scaler fit **chỉ train**, transform val/test. Sau đó ghép với BERT → `features_raw_{split}.npy` (777-d).

---

## Phase 3 — PCA / SVD (Ablation track)

**Notebook:** `notebooks/03_PCA_Feature_Selection.ipynb`  
**Mục tiêu phase:** Giảm chiều 777→**400-d** để so sánh ablation (không phải input final track).

### `fit_selection_reducer()` / `fit_final_reducer()`

| Hàm | Định nghĩa | Gọi |
|-----|------------|-----|
| `fit_selection_reducer` | Cell 12 L10 | Cell 13 L19 |
| `fit_final_reducer` | Cell 12 L52 | Cell 13 L49 |
| `choose_component_count` | Cell 12 L78 | Cell 13 L46 |

**Giải thích:**
- `fit_selection_reducer`: thử PCA/SVD/IncrementalPCA, chọn backend phù hợp RAM.
- `choose_component_count`: chọn số component giữ ~95% variance.
- `fit_final_reducer`: fit reducer cuối trên train, lưu `pca_or_svd.joblib`.

### `scaler.fit_transform()` / `model.fit()` / `transform()` (sklearn)

| Thao tác | Cell | Dòng |
|----------|------|------|
| Scale trước PCA | 11 | 13–17 |
| Fit reducer | 12 | 18, 30, 44, 72 |
| Transform split | 15 | 31 |

**Giải thích:** Pipeline chuẩn sklearn: scale train → fit PCA/SVD trên train → transform cả 3 split → `features_final_{split}.npy`.

---

## Phase 4 — PSO + CNN-BiLSTM (DL trên PCA)

**Notebook:** `notebooks/04_PSO_Model_Training.ipynb`  
**Mục tiêu phase:** Train DL baseline và DL-PSO trên feature **400-d** (ablation).

### `build_model(config, input_dim)`

| | Cell | Dòng |
|---|------|------|
| **Định nghĩa** | 13 | 175 |
| **Gọi** | 14 | 168 |

**Giải thích:** Khởi tạo `CNNBiLSTMAttention` với hyperparameter từ config (filters, kernel, LSTM hidden, attention dim, dropout). Kiến trúc: Conv1d → BiLSTM → Attention → FC.

---

### `fit_model(model, loaders, ...)`

| | Cell | Dòng |
|---|------|------|
| **Định nghĩa** | 14 | 144 |
| **Gọi baseline** | 16 | 34 |
| **Gọi PSO trial** | 19 | 45 |
| **Gọi PSO final** | 22 | 19 |

**Giải thích:** Vòng lặp train: `train_one_epoch` → evaluate trên val → early stopping → lưu checkpoint tốt nhất. Dùng cho cả DL baseline và từng trial PSO.

---

### `train_one_epoch()` / `predict_probabilities()` / `evaluate_model()`

| Hàm | Định nghĩa | Gọi chính |
|-----|------------|-----------|
| `train_one_epoch` | Cell 14 L2 | Cell 14 L188 |
| `predict_probabilities` | Cell 14 L52 | Cell 14 L120 |
| `evaluate_model` | Cell 14 L106 | Cell 16 L101, Cell 22 L86 |

**Giải thích:**
- `train_one_epoch`: 1 epoch forward + backward + optimizer step.
- `predict_probabilities`: inference toàn loader → xác suất P(fake).
- `evaluate_model`: gọi predict + `evaluate_predictions` → metric dict.

---

### `evaluate_predictions(y_true, prob_fake, ...)`

| | Cell | Dòng |
|---|------|------|
| **Định nghĩa** | 11 | 32 |
| **Gọi** | 14 | 122 |

**Giải thích:** Hàm metric **dùng chung** cả pipeline: tính Macro F1, Precision/Recall Fake, ROC-AUC, PR-AUC, confusion matrix tại ngưỡng τ. Được copy sang notebook Phase 5.

---

### `decode_particle(position)` → `pso_objective(position)` → `fallback_pso(...)`

| Hàm | Định nghĩa | Gọi |
|-----|------------|-----|
| `decode_particle` | Cell 18 L146 | Cell 19 L10, Cell 20 L162 |
| `pso_objective` | Cell 19 L6 | Cell 20 L82, L104 |
| `compute_pso_objective_score` | Cell 18 L176 | Cell 19 L73 |
| `fallback_pso` | Cell 20 L2 | Cell 20 L102 |
| `normalize_pso_result` | Cell 20 L49 | Cell 20 L95–96, L117–118 |

**Giải thích (chuỗi PSO):**
1. `decode_particle`: vector hạt PSO [filters, kernel, lr, dropout, ...] → dict config.
2. `pso_objective`: **hàm mục tiêu** — train DL nhanh trên subset 20% → trả score càng thấp càng tốt.
3. `compute_pso_objective_score`: weighted sum Macro F1 + Precision Fake + ROC-AUC.
4. `fallback_pso`: wrapper PSO (pyswarm hoặc grid fallback) gọi `pso_objective` nhiều lần.
5. `normalize_pso_result`: chuẩn hóa output PSO → `best_params.json`.

**Kết quả:** `best_model_dl.pth` (DL-PSO), Macro F1 test ≈ 0,7793.

---

## Phase 5 — Model Zoo & Ensemble (Final track)

**Input:** `features_raw_{split}.npy` (**777-d**)

### LightGBM — `05_01_LightGBM_Raw.ipynb`

| Thao tác | Cell | Dòng | Giải thích |
|----------|------|------|------------|
| `LGBMClassifier(**CONFIG)` | 7 | 42 | Khởi tạo GBDT với hyperparameter cố định |
| `model.fit(X_train, y_train)` | 7 | 44 | Train trên raw 777-d, eval_set=val |
| `model.predict_proba()` | 7 | 52 | Sinh xác suất 3 split → `.npy` |
| `write_metrics()` định nghĩa | 5 | 340 | |
| `write_metrics()` gọi | 7 | 54 | Ghi CSV metric + lưu artifact |

---

### XGBoost — `05_02_XGBoost_Raw.ipynb`

| Thao tác | Cell | Dòng | Giải thích |
|----------|------|------|------------|
| `XGBClassifier(**CONFIG)` | 7 | 44 | Tương tự LGBM |
| `model.fit()` | 7 | 46 | Train + early stopping trên val |
| `predict_proba()` | 7 | 54 | Test Macro F1 ≈ 0,9059 |
| `write_metrics()` gọi | 7 | 56 | |

---

### MLP — `05_03_MLP_Raw.ipynb`

| Hàm | Định nghĩa | Gọi | Giải thích |
|-----|------------|-----|------------|
| `forward()` (class) | Cell 7 L22 | qua `model(x)` | FC layers trên 777-d |
| `predict(split)` | Cell 7 L36 | L97 (val/epoch), L132 (prob_map) | Inference → P(fake) |
| `write_metrics()` | Cell 5 L340 | Cell 7 L132 | |

---

### CNN-BiLSTM Sequence — `05_04a/b/c_*_seed*.ipynb`

| Hàm | Định nghĩa | Gọi (seed42) | Giải thích |
|-----|------------|--------------|------------|
| `forward()` Attention | Cell 8 L45 | nội bộ | Attention trên BiLSTM output |
| `forward()` Model | Cell 8 L72 | nội bộ | Token sequence + behavioral late fusion |
| `predict(split)` | Cell 9 L25 | L66, L101 | Inference sequence model |
| `write_metrics()` | Cell 5 L340 | Cell 9 L103 | Test F1 ≈ 0,9343 (seed 42) |

**Giải thích:** Khác Phase 4 — model đọc **token sequence thật** (input_ids), không phải vector PCA tĩnh. Behavioral features fusion ở cuối mạng.

---

### Weighted Blending — `05_05_Weighted_Blending.ipynb`

| Hàm / logic | Định nghĩa | Gọi | Giải thích |
|-------------|------------|-----|------------|
| `threshold_sweep()` | Cell 6 L50 | L56; Cell 7 L54 | Quét τ từ 0,01→0,99, tìm balanced (0,30) và precision-first (0,60) |
| Blend trọng số | — | Cell 7 L44–50 | `blended = Σ weight_i × prob_i` trên grid CNN/XGB/LGBM |
| `write_metrics()` | Cell 5 L340 | Cell 7 L87 | Lưu `phase5_weighted_blend_*_prob.npy` |

**Kết quả headline:** seed 42 → CNN 50% + XGB 50%; Macro F1 **0,9463** @ τ=0,30.

---

### Stacking + Calibration — `05_06_Stacking_Calibration.ipynb`

| Hàm | Định nghĩa | Gọi | Giải thích |
|-----|------------|-----|------------|
| `stack_matrix(split)` | Cell 7 L16 | L21 | Ghép xác suất base model thành meta-features |
| `model.fit()` stacker | — | L37 | Logistic / RF / ExtraTrees meta-learner |
| `calibrator.fit()` | — | L48 | `CalibratedClassifierCV` hiệu chỉnh xác suất |
| `write_metrics()` | Cell 5 L270 | L77 | Test F1 ≈ 0,9105 |

---

### Tổng hợp — `05_Hybrid_Ensemble.ipynb`

| Hàm | Gọi | Giải thích |
|-----|-----|------------|
| `threshold_sweep()` | Cell 8 L14 | Sweep mọi candidate trên val |
| `evaluate_predictions()` | Cell 8 L8, L118 | Leaderboard + chọn model cuối |

---

## Phase 6 — Adversarial & XAI

**Notebook:** `notebooks/06_Adversarial_XAI.ipynb`

### Dự đoán (wrapper)

| Hàm | Định nghĩa | Gọi | Giải thích |
|-----|------------|-----|------------|
| `predict_dl_proba(X)` | Cell 17 L28 | Cell 20 L90 | DL-PSO inference batch |
| `predict_final_ensemble_fake_proba(X)` | Cell 17 L82 | Cell 20 L92; Cell 26 L90 | Legacy PCA ensemble |
| `predict_xgb_raw_proba(model, X)` | Cell 8 L132 | Cell 12 L125 | XGB raw cho LIME final track |

---

### Adversarial attacks

| Hàm | Định nghĩa | Gọi | Giải thích |
|-----|------------|-----|------------|
| `fgsm_attack_batch()` | Cell 20 L8 | L74 | Fast Gradient Sign Method — nhiễu ±ε theo gradient loss |
| `pgd_attack_batch()` | Cell 20 L30 | L78 | Projected Gradient Descent — FGSM lặp nhiều bước |
| `clamp_features()` | Cell 20 L2 | nội bộ | Giới hạn feature nhiễu trong miền hợp lệ |
| `evaluate_condition()` | Cell 20 L88 | L106, L112, L118 | So metric clean vs FGSM vs PGD |

**Giải thích:** Đo **robustness** — mô hình giảm bao nhiêu F1/Precision khi bị tấn công adversarial trên subset PCA features.

---

### Giải thích mô hình (XAI)

| Hàm / API | Định nghĩa | Gọi | Giải thích |
|-----------|------------|-----|------------|
| `shap.TreeExplainer()` | — | Cell 10 L11; Cell 23 L13 | SHAP global importance cho XGB/LGBM |
| `select_final_lime_cases()` | Cell 12 L2 | L74 | Chọn 6 case: TP, TN, FP, FN, high-conf fake/real |
| `explain_instance()` (LIME) | — | Cell 12 L121; Cell 26 L129 | Giải thích local từng review → HTML |
| `select_lime_cases()` | Cell 26 L20 | Cell 27 L2 | LIME legacy ensemble |
| `build_raw_feature_names()` | Cell 8 L2 | L99 | Tên 777 chiều cho LIME/SHAP |
| `build_raw_feature_metadata()` | Cell 24 L2 | L72 | Metadata feature cho báo cáo XAI |

---

## Phase 7 — Evaluation & Ablation

**Notebook:** `notebooks/07_Evaluation_Ablation.ipynb`  
**Output SSOT:** `reports/tables/phase7_final_metrics.csv`

### `evaluate_probabilities(...)`

| | Cell | Dòng |
|---|------|------|
| **Định nghĩa** | 11 | 22 |
| **Gọi** | 15 L120, L152; 18 L4; 26 L46, L113 | |

**Giải thích:** Hàm metric **trung tâm Phase 7** — nhận xác suất từ artifact `.npy`, tính đầy đủ metric + metadata ablation → ghi vào `phase7_final_metrics.csv`.

---

### `evaluate_test_probability(model_variant, threshold, ...)`

| | Cell | Dòng |
|---|------|------|
| **Định nghĩa** | 18 | 2 |
| **Gọi** | 19 L9, L28, L44, L226, L243 | |

**Giải thích:** Wrapper gọi `evaluate_probabilities` trên split **test** cho từng model variant. Dùng trong ablation Models A–E.

---

### Ablation helpers

| Hàm | Định nghĩa | Gọi | Giải thích |
|-----|------------|-----|------------|
| `fit_lightgbm_predict()` | Cell 15 L54 | L110 | Train LGBM 1 fold CV |
| `lightgbm_config()` | Cell 15 L18 | Cell 18 L106 | Hyperparameter LGBM cho ablation |
| `build_basic_behavioral_features()` | Cell 18 L152 | Cell 19 L141 | Ablation: chỉ 5 feature cơ bản |
| `train_only_reduce_features()` | Cell 18 L208 | Cell 19 L143 | PCA fit chỉ train cho ablation |
| `error_case_table()` | Cell 22 L50 | L116, L118 | Top 25 FP/FN trên test |
| `load_multiseed_blend_probs(seed)` | Cell 26 L18 | L94 | Load prob seed 123/456 |
| `select_multiseed_thresholds(y_val, prob_val)` | Cell 26 L34 | L101 | Chọn τ balanced/precision-first theo seed |

---

### `selected_weighted_blend(split)` — chỉ định nghĩa

| | Cell | Dòng |
|---|------|------|
| **Định nghĩa** | 18 | 26 |
| **Gọi** | — | *(không có lời gọi trực tiếp)* |

**Giải thích:** Hàm dự phòng tái tạo blend legacy PCA (dl_pso + xgboost + lightgbm + stacking). Pipeline final dùng artifact `phase5_weighted_blend_*_prob.npy` đã lưu sẵn từ Phase 5.

---

## Benchmark & Diagnostic

### `10_Baseline_Algorithm_Benchmark.ipynb`

| Hàm | Cell | Giải thích |
|-----|------|------------|
| `compute_metrics()` định nghĩa L40, gọi L21 | 4–5 | Metric cho LR, LinearSVC, RF trên 777-d |
| `est.fit()` | 5 L13 | Train sklearn baseline |

### `09_Bieu_Do_So_Sanh_Mo_Hinh.ipynb`

| Hàm | Cell | Giải thích |
|-----|------|------------|
| `loc_metric_test()` L4, gọi L58 | 6 | Lọc metric split=test |
| `ghep_metric_tu_nhieu_file()` L28, gọi L176 | 6–7 | Gộp baseline + phase7 → vẽ biểu đồ |

### `tests/01_DL_PCA_Diagnostic_Test.ipynb`

| Hàm | Cell | Giải thích |
|-----|------|------------|
| `train_model()` L136 | 6 | Train MLP/CNN so sánh raw vs PCA |
| `metric_bundle()` L132 | 4 | Gói metric nhanh |
| `choose_threshold_by_precision()` L168 | 4 | Chọn τ đạt Precision ≥ 0,975 |

**Giải thích:** Diagnostic Phase 9 — chứng minh CNN-BiLSTM **không nên** đọc vector PCA tĩnh (mismatch kiến trúc).

---

## Hàm dùng chung (xuất hiện nhiều notebook Phase 5)

| Hàm | Định nghĩa gốc | Mục đích |
|-----|----------------|----------|
| `safe_roc_auc()` | `04` Cell 11 L8 | ROC-AUC an toàn khi chỉ 1 class |
| `safe_pr_auc()` | `04` Cell 11 L20 | PR-AUC an toàn |
| `evaluate_predictions()` | `04` Cell 11 L32 | Metric dict chuẩn |
| `write_metrics()` | `05_00` Cell 5 L270 | Ghi CSV + validate artifact |
| `threshold_sweep()` | `05_05` Cell 6 L50 | Quét ngưỡng τ trên val |

---

## Ghi chú khi trình bày trước hội đồng

| Câu hỏi thường gặp | Mở notebook | Cell |
|--------------------|---------------|------|
| BERT embedding ở đâu? | `02_Feature_Engineering` | 10–11 |
| PSO tối ưu gì? | `04_PSO_Model_Training` | 19–20 |
| Model cuối blend thế nào? | `05_05_Weighted_Blending` | 7 |
| Số 0,9463 từ đâu? | `07_Evaluation_Ablation` | 11, 19 |
| SHAP/LIME case nào? | `06_Adversarial_XAI` | 12, 26 |

---

*Cập nhật: 2026-06-25. Bảng cell/dòng đầy đủ (130 hàm): `BANG_HAM_THUAT_TOAN_CELL_DONG.md`*