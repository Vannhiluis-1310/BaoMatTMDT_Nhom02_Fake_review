# Phase 3: PCA Feature Selection - Chi tiết

## 1. Lý do chọn phương án (từ .planning/phases/03-pca-feature-selection/)

### 1.1. Vấn đề cần giải quyết

- 777-d raw features (BERT 768 + behavioral 9) quá lớn cho RAM 12GB
- Cần giảm chiều nhưng giữ ≥ 95% variance
- PCA vs Mutual Information vs Variance Threshold vs AutoEncoder

### 1.2. Quyết định thiết kế

**Tại sao chọn PCA thay vì Mutual Information?**
- Mutual Information: Supervised (dùng label), rủi ro leakage, chậm
- PCA: Unsupervised, giảm chiều mạnh, loại nhiễu, không cần label

**Tại sao PCA thay vì Variance Threshold?**
- Variance Threshold chỉ loại features variance thấp, không giảm chiều
- PCA giảm 777 → 400 (giảm ~49% kích thước)

**Tại sao PCA thay vì AutoEncoder?**
- AutoEncoder cần train thêm network, tốn RAM → vi phạm 12GB constraint
- PCA không cần train, chỉ fit transform

**Fit policy nghiêm ngặt:**
- PCA chỉ fit trên train set
- Val/test transform bằng PCA đã fit → tránh data leakage (Hotelling, 1933; Jolliffe, 2002)

## 2. Luồng code (notebooks/03_PCA_Feature_Selection.ipynb)

### 2.1. PCA Pipeline

```
Input: 777-d raw features
StandardScaler fit train-only
PCA fit train-only, n_components=400
Output: 400-d PCA features
Variance retained: 0.9619 (96.19%)
```

### 2.2. Memory Report (phase3_memory_report.csv)

```
Before PCA: 777-d feature matrix
After PCA: 400-d feature matrix
Memory reduction: ~49%
RAM usage: Within 12GB constraint
```

## 3. Kết quả và phân tích (reports/tables/)

### 3.1. Số liệu chính xác

- **Input dimension:** 777
- **Output dimension:** 400
- **Variance retained:** 0.9619 (96.19%)
- **Reduction ratio:** 400/777 ≈ 0.5148 (51.48% dimensions kept)

### 3.2. Insight kinh doanh

**Nỗi đau từ high-dimensional features:**
- 777-d feature matrix tiêu tốn RAM lớn
- Model training chậm, dễ OOM trên Colab 12GB
- Nhiều chiều có variance thấp (nhiễu) làm giảm signal-to-noise ratio

**Ý nghĩa PCA:**
- Giảm 777 → 400 dimensions (giảm 49% kích thước)
- Giữ 96.19% variance → mất mát thông tin tối thiểu
- Loại nhiễu tự động (regularization ngầm)

**Lưu ý quan trọng (từ Phase8_Final_Report):**
- Model B (no-PCA LightGBM surrogate): Macro F1 = 0.905821, Precision Fake = 0.962230, ROC-AUC = 0.952377
- PCA gây mất mát thông tin hữu ích
- PCA được giữ vì **RAM/pipeline stabilizer**, không phải vì cải thiện metrics

### 3.3. Lý do chuyển sang Phase 4

- 400-d PCA features là input cho DL model
- Cần train CNN-BiLSTM-Attention baseline
- PSO optimization 12 hyperparameters

## 4. Ràng buộc và quyết định cho phase sau

**Ràng buộc cho Phase 4:**
- Input: 400-d PCA features
- PSO: 10 particles × 8 iterations × 5 trial epochs
- Objective: 0.5 × Macro F1 + 0.3 × ROC-AUC + 0.2 × Precision Fake
- Subset 20% cho PSO trials

**Quyết định cho Phase 5:**
- DL PSO-tuned vs XGBoost vs LightGBM comparison
- Stacking ensemble selection (71 candidates)
- Threshold optimization (0.79)
