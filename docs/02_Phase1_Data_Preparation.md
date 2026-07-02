# Phase 1: Data Preparation and EDA - Chi tiết

## 1. Lý do chọn phương án (từ .planning/phases/01-data-preparation-and-eda/)

### 1.1. Vấn đề cần giải quyết

- Dataset Amazon Labeled Fake Reviews có 50,000 rows × 13 cols
- Tồn tại missing values và duplicate records cần làm sạch
- Cần stratified split 70/15/15 để đảm bảo phân phối class cân bằng
- Yêu cầu reproducibility với seed 42

### 1.2. Quyết định thiết kế

**Tại sao chọn cleaning thay vì imputation?**
- Missing values trong text/review fields không thể imputation hợp lý
- Duplicate records là lỗi dữ liệu, không phải missing data
- Kết quả: 50,000 → 42,749 rows (loại 7,251 duplicate/missing)

**Tại sao stratified split?**
- Class ratio ban đầu ≈ 1.02 (cân bằng)
- Stratified đảm bảo train/val/test có cùng phân phối
- Tránh bias khi class distribution thay đổi giữa splits

**Tại sao seed 42?**
- Reproducibility requirement từ REQUIREMENTS.md
- Giá trị phổ biến trong ML community
- Cho phép so sánh kết quả giữa các phase

## 2. Luồng code (notebooks/01_EDA_Preprocessing.ipynb)

### 2.1. EDA Summary (phase1_eda_summary.csv)

```
Dataset: 50,000 rows, 13 columns
Class ratio (fake/real): ≈ 1.02
Columns: review_id, reviewer_id, product_id, review_text, rating, verified_purchase, review_date, ...
```

### 2.2. Cleaning Report (phase1_cleaning_report.csv)

```
Input: 50,000 rows
Removed duplicates: [số lượng]
Removed missing: [số lượng]
Output: 42,749 rows (giảm 14.502%)
```

### 2.3. Stratified Split

```
Train: 29,923 rows (70%)
Val: 6,413 rows (15%)
Test: 6,413 rows (15%)
Seed: 42
Strategy: StratifiedShuffleSplit
```

**Fit policy:**
- Tất cả transformers (scaler, PCA) chỉ fit trên train set
- Val/test chỉ transform → tránh data leakage

## 3. Kết quả và phân tích (reports/tables/)

### 3.1. Số liệu chính xác (không làm tròn)

- **Input rows:** 50000
- **Output rows:** 42749
- **Removed rows:** 7251 (14.502%)
- **Class ratio sau cleaning:** ≈ 1.02 (vẫn cân bằng)
- **Train/Val/Test split:** 29923 / 6413 / 6413

### 3.2. Insight kinh doanh

**Nỗi đau từ duplicate/missing:**
- 7,251 reviews bị loại bỏ → có thể là spammer copy-paste review hoặc lỗi crawl
- Tỷ lệ 14.502% cho thấy chất lượng dữ liệu đầu vào chưa cao
- Nếu không cleaning → model học nhiễu từ duplicate entries

**Ý nghĩa cho e-commerce:**
- 42,749 reviews sạch đủ lớn để train model robust
- Class balance (1.02) là điều kiện lý tưởng cho binary classification
- Stratified split đảm bảo model không bị bias về phân phối class

### 3.3. Lý do chuyển sang Phase 2

- Dữ liệu đã sạch, có split rõ ràng
- Cần trích xuất features từ text (BERT) + behavioral features
- Phase 2 sẽ fuse 768-d BERT + 9 behavioral features → 777-d raw features

## 4. Ràng buộc và quyết định cho phase sau

**Ràng buộc cho Phase 2:**
- BERT embeddings phải extract trên train-only fit policy
- 9 behavioral features cần StandardScaler fit train-only
- Feature fusion (concatenate) phải đảm bảo không có leakage

**Quyết định cho Phase 3:**
- 777-d raw features cần giảm chiều → PCA 400 components
- PCA fit train-only, transform val/test
- Giữ 96.19% variance để không mất thông tin
