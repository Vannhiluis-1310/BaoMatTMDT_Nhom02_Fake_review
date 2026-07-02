# Phase 2: Feature Engineering - Chi tiết

**Output pipeline chính:** `features_raw_{train,val,test}.npy` (777-d) → tabular models `05_01–05_03` + token sequences → `05_04` late fusion.

## 1. Lý do chọn phương án (từ .planning/phases/02-feature-engineering/)

### 1.1. Vấn đề cần giải quyết

- Text-only analysis không đủ để phát hiện fake review tinh vi
- Cần behavioral features bổ sung tín hiệu hành vi
- BERT embeddings cần extract đúng cách (masked_mean_last_hidden_state)
- 9 behavioral features cần thiết kế dựa trên nghiên cứu spam detection

### 1.2. Quyết định thiết kế

**Tại sao chọn bert-base-uncased (sau nâng cấp ModernBERT)?**
- BERT-base là backbone kinh điển, đã chứng minh hiệu quả trong nhiều nghiên cứu fake review
- ModernBERT-base (Warner et al., 2025) được chọn sau vì:
  - Context length 8192 tokens (BERT-base chỉ 512)
  - Flash Attention + unpadding → inference nhanh 2.65-3x
  - RoPE, GeGLU, alternating local-global attention

**Tại sao 9 behavioral features (5 basic + 4 advanced)?**
- Basic features: length, rating deviation, sentiment, verified_purchase
- Advanced features: velocity, burst, reviewer behavior score, time gap
- Dựa trên nghiên cứu spam detection và reviewer behavior analysis

**Tại sao fit policy train-only?**
- StandardScaler fit train, transform val/test → tránh leakage
- Behavioral feature aggregates (mean rating) tính trên train only

## 2. Luồng code (notebooks/02_Feature_Engineering.ipynb)

### 2.1. BERT Embeddings

```
Model: bert-base-uncased (sau: ModernBERT-base)
Pooling: masked_mean_last_hidden_state
Max length: 128 (sau nâng cấp: 160)
Output dim: 768-d
```

### 2.2. 9 Behavioral Features

**Basic (5 features):**
- `basic_char_len_log`: log1p(số ký tự)
- `basic_word_count_log`: log1p(số từ)
- `basic_rating_deviation`: |rating - mean_train|
- `basic_sentiment_compound`: VADER compound score
- `basic_verified_purchase`: Binary flag (0/1)

**Advanced (4 features):**
- `adv_review_velocity_30d`: Số reviews trong 30 ngày trước
- `adv_product_burst_7d`: Số reviews cho sản phẩm trong 7 ngày
- `adv_reviewer_behavior_score`: Unsupervised anomaly score
- `adv_time_gap_hours_log`: log1p(giờ từ review trước)

### 2.3. Feature Fusion

```
BERT (768-d) + Behavioral (9-d) → Concatenate → 777-d raw features
StandardScaler fit train-only
```

## 3. Kết quả và phân tích (reports/tables/)

### 3.1. Số liệu chính xác

- **BERT embedding dim:** 768
- **Behavioral features:** 9
- **Fused dimension:** 777
- **Variance retained sau PCA (Phase 3):** 0.9619 (96.19%)

### 3.2. Insight kinh doanh

**Nỗi đau từ text-only analysis:**
- Fake reviews viết bởi AI/crowdsourcing có văn phong tự nhiên
- BERT embeddings phát hiện ngữ nghĩa phi tự nhiên nhưng không bắt được:
  - Velocity attack (nhiều reviews trong thời gian ngắn)
  - Burst attack (flood reviews cho một sản phẩm)
  - Reviewer behavior anomaly (tài khoản spammer)

**Ý nghĩa behavioral features:**
- `adv_review_velocity_30d` và `adv_product_burst_7d` phát hiện campaign spam
- `adv_reviewer_behavior_score` phát hiện anomaly ở mức tài khoản
- `adv_time_gap_hours_log` phát hiện tốc độ đánh giá bất thường

### 3.3. Lý do chuyển sang Phase 3

- 777-d raw features quá lớn cho RAM 12GB
- Cần giảm chiều → PCA 400 components
- PCA cũng loại nhiễu (regularization ngầm)

## 4. Ràng buộc và quyết định cho phase sau

**Ràng buộc cho Phase 3:**
- PCA fit train-only, transform val/test
- Giữ ≥ 95% variance
- Giảm 777 → ~400 dimensions để fit RAM 12GB

**Quyết định cho Phase 4:**
- 400-d PCA features là input cho CNN-BiLSTM-Attention
- PSO optimization 12 hyperparameters
- Baseline DL vs PSO-tuned DL comparison
