# TỔNG QUAN DỰ ÁN

## Phát hiện đánh giá giả trên Amazon bằng pipeline hai nhánh ModernBERT–đặc trưng hành vi–ensemble học sâu và GBDT với chính sách ngưỡng kép

*Dual-Track ModernBERT, Behavioral Features, and Threshold-Selected Ensemble for Fake Review Detection*

**Tác giả:** [Nhóm nghiên cứu]  
**Năm:** 2026  
**Seed:** 42 | **Split:** 70/15/15 Stratified | **Môi trường:** Google Colab, Python 3.12.13, PyTorch 2.11.0+cu128, Tesla T4 GPU (16GB VRAM)  
**Ràng buộc tài nguyên:** RAM 12GB (hard constraint)

---

## MỤC LỤC

1. Tóm tắt (Abstract)
2. Giới thiệu (Introduction)
   - 2.1. Đặt vấn đề
   - 2.2. Câu hỏi nghiên cứu
   - 2.3. Mục tiêu nghiên cứu
   - 2.4. Phạm vi và giới hạn
3. Tổng quan tài liệu (Literature Review)
   - 3.1. Bài toán phát hiện fake review
   - 3.2. Các phương pháp tiếp cận
   - 3.3. Bảng so sánh và phân tích khoảng trống nghiên cứu
4. Cơ sở lý thuyết (Theoretical Framework)
   - 4.1. Hybrid Deep Learning Architecture
   - 4.2. Swarm Intelligence Optimization (PSO)
   - 4.3. Feature Selection via PCA
   - 4.4. Ensemble Learning via Stacking
   - 4.5. Behavioral Feature Engineering
5. Phương pháp nghiên cứu (Methodology)
   - 5.1. Thiết kế nghiên cứu
   - 5.2. Quy trình thu thập và tiền xử lý dữ liệu
   - 5.3. Trích xuất đặc trưng
   - 5.4. Giảm chiều và lựa chọn đặc trưng
   - 5.5. Huấn luyện và tối ưu hóa mô hình
   - 5.6. Ensemble và threshold optimization
6. Kết quả và thảo luận (Results and Discussion)
   - 6.1. Kết quả định lượng
   - 6.2. Phân tích ablation study
   - 6.3. Đánh giá so với mục tiêu
   - 6.4. Thảo luận
7. Ý nghĩa thực tiễn và đóng góp khoa học
8. Hạn chế và hướng nghiên cứu tiếp theo
9. Kết luận
10. Tài liệu tham khảo

---

## 1. TÓM TẮT (ABSTRACT)

Bài toán phát hiện đánh giá giả mạo trên thương mại điện tử ngày càng phức tạp khi nội dung AI-generated và hành vi spam tinh vi. Nghiên cứu xây dựng **pipeline hai nhánh**: (1) **Final track** — ModernBERT embeddings (768-d) + 9 behavioral features → vector raw 777-d; nhánh tabular (LightGBM, XGBoost, MLP) và nhánh sequence (CNN-BiLSTM-Attention); weighted ensemble **CNN 50% / XGB 35% / LGBM 15%**; chọn ngưỡng kép trên validation (balanced τ=0.30, precision-first τ=0.60). (2) **Ablation track** — PCA 777→400, PSO-tuned DL, stacking legacy — chứng minh PCA track thua raw và phục vụ diagnostic/RAM.

Thực nghiệm trên Amazon Labeled Fake Reviews Dataset (50,000 mẫu gốc, 42,749 sau tiền xử lý), stratified split 70/15/15 (seed = 42), fit policy train-only. Pipeline chính (2026-06-10): ModernBERT raw 777 + CNN-BiLSTM sequence + weighted ensemble (CNN 50% / XGB 35% / LGBM 15%). Trên test: Macro F1 = **0.9433** @τ=0.50; **0.9463** @τ=0.30 (balanced, chọn trên val); Precision Fake = **0.9816** @τ=0.60 (precision-first, đạt target ≥0.975); ROC-AUC = **0.9769**. Multi-seed (42, 123, 456): balanced Macro F1 **0.9485 ± 0.0018**. So với 20 công trình tham chiếu (Tier A text/tabular), kết quả vượt các baseline transformer và DL đơn lẻ đã kiểm chứng trên cùng bài toán Amazon. PCA+PSO track (0.856) được giữ làm ablation. Pipeline reproducible với metadata tracking đầy đủ.

**Từ khóa:** Fake review detection, ModernBERT, behavioral features, CNN-BiLSTM, weighted ensemble, dual-threshold, Amazon dataset, ablation study, PCA, PSO.

---

## 2. GIỚI THIỆU (INTRODUCTION)

### 2.1. Đặt vấn đề

Thương mại điện tử đã trở thành một phần không thể thiếu của nền kinh tế toàn cầu, với doanh thu dự kiến vượt 6.3 nghìn tỷ USD vào năm 2024 (Statista, 2024). Trong bối cảnh này, hệ thống đánh giá sản phẩm (product reviews) đóng vai trò then chốt trong việc định hình quyết định mua hàng của người tiêu dùng. Theo nghiên cứu của BrightLocal (2023), 98% người tiêu dùng đọc review trước khi mua hàng, và 76% tin tưởng review trực tuyến như lời khuyên từ bạn bè.

Tuy nhiên, sự phổ biến của fake reviews — các đánh giá được viết bởi các cá nhân hoặc tổ chức với mục đích thao túng nhận thức của người tiêu dùng — đã tạo ra những thách thức nghiêm trọng. Các nghiên cứu cho thấy tỷ lệ fake reviews trên các nền tảng lớn có thể dao động từ 20% đến 40% (Hu et al., 2012; Mukherjee et al., 2013). Amazon Labeled Fake Reviews Dataset được sử dụng trong nghiên cứu này có tỷ lệ fake ≈ 40%, phản ánh thực tế đáng lo ngại trên thị trường.

**Hậu quả của fake reviews:**

1. **Đối với người tiêu dùng:** Quyết định mua hàng sai lệch do tin tưởng vào review giả → thiệt hại tài chính, trải nghiệm tiêu cực, và mất niềm tin vào hệ thống đánh giá.

2. **Đối với người bán hàng trung thực:** Fake negative reviews từ đối thủ cạnh tranh có thể gây thiệt hại uy tín và doanh thu đáng kể. Một nghiên cứu của Luca & Zervas (2016) cho thấy mỗi fake negative review có thể làm giảm doanh thu của seller khoảng 0.5-1%.

3. **Đối với nền tảng thương mại điện tử:** Mất lòng tin từ cả buyer và seller, dẫn đến giảm tương tác và doanh thu. Nền tảng có thể phải đối mặt với các vấn đề pháp lý và quy định nếu không kiểm soát được fake reviews.

**Thách thức hiện tại:**

Fake reviews ngày càng tinh vi. Trong khi các kỹ thuật phát hiện ban đầu dựa trên các đặc trưng đơn giản (độ dài review, tần suất rating cực đoan), các spammer hiện đại sử dụng:
- AI-generated content (GPT-4, Claude, Gemini) để viết review có văn phong tự nhiên
- Crowdsourcing: thuê reviewer thật viết review giả (verified purchase, dài, sentiment đồng nhất)
- Coordinated campaigns: nhiều tài khoản viết review cho cùng một sản phẩm trong thời gian ngắn (burst attack)
- Sophisticated evasion: tránh các pattern dễ bị phát hiện bởi rule-based systems

Các phương pháp text-only (BERT, RoBERTa, DeBERTa) gặp giới hạn khi fake review có ngữ nghĩa tự nhiên và mạch logic hợp lý. Do đó, cần kết hợp thêm behavioral signals — các tín hiệu hành vi mà kẻ gian không thể che giấu hoàn toàn: review velocity, product burst, time gap, reviewer behavior anomaly.

### 2.2. Câu hỏi nghiên cứu

Nghiên cứu này đặt ra sáu câu hỏi nghiên cứu chính:

**RQ1:** Liệu việc kết hợp BERT embeddings với 9 behavioral features (5 basic + 4 advanced) có cải thiện hiệu năng phát hiện fake review so với chỉ sử dụng text embeddings đơn thuần?

**RQ2:** Liệu Particle Swarm Optimization (PSO) có thể tự động hóa việc tìm kiếm 12 hyperparameters cho kiến trúc CNN-BiLSTM-Attention hiệu quả hơn so với manual tuning hoặc Grid Search/Random Search?

**RQ3:** Liệu PCA feature selection (777 → 400 dimensions, giữ 96.19% variance) có thể giảm đáng kể yêu cầu tài nguyên (RAM) trong khi duy trì hiệu năng chấp nhận được?

**RQ4:** Liệu Stacking Ensemble kết hợp Deep Learning (PSO-tuned CNN-BiLSTM-Attention), XGBoost và LightGBM có cải thiện hiệu năng so với single model tốt nhất?

**RQ5:** Liệu chính sách ngưỡng kép (balanced τ=0.30, precision-first τ=0.60, chọn trên validation) có đạt đồng thời Macro F1 ≥0.890000 và Precision Fake ≥0.975000 trên test audit?

**RQ6:** Đóng góp tương đối của từng thành phần (PSO, PCA, behavioral features, ensemble) trong pipeline hybrid là bao nhiêu, được định lượng qua ablation study?

### 2.3. Mục tiêu nghiên cứu

**Mục tiêu tổng quát:** Xây dựng và đánh giá một pipeline hybrid cho bài toán fake review detection, kết hợp BERT embeddings, behavioral features, PCA, PSO optimization, và Stacking Ensemble, trong ràng buộc RAM 12GB.

**Mục tiêu cụ thể:**

| STT | Mục tiêu | Metric | Target | Kết quả (final track) | Đạt |
|-----|----------|--------|--------|------------------------|-----|
| 1 | Phát hiện fake review với độ chính xác cao | Macro F1 | ≥ 0.890000 | **0.9463** @τ=0.30 (mean 0.9485±0.0018, 3 seed) | ✅ |
| 2 | Ưu tiên Precision Fake cho yêu cầu e-commerce | Precision Fake | ≥ 0.975000 | **0.9816** @τ=0.60 (mean 0.9763±0.0029) | ✅ |
| 3 | Đảm bảo discrimination ability | ROC-AUC | ≥ 0.930000 | **0.9769** (threshold-invariant) | ✅ |
| 4 | Chạy được trong tài nguyên hạn chế | RAM | ≤ 12GB | Colab T4, raw 777 + sequence | ✅ |
| 5 | Đảm bảo reproducibility | Seed, Split, Fit policy | Seed=42, 70/15/15, train-only | Metadata JSON đầy đủ | ✅ |
| 6 | Giải quyết research gaps | 8 gaps | Đầy đủ | Xem `00_Literature_Review_SOTA.md` | ✅ |

### 2.4. Phạm vi và giới hạn

**Phạm vi:**
- Dataset: Amazon Labeled Fake Reviews Dataset (50,000 mẫu ban đầu)
- Ngôn ngữ: Tiếng Anh (review text)
- Mô hình embedding: BERT-base-uncased (sau nâng cấp ModernBERT-base)
- Môi trường: Google Colab (Python 3.12.13, PyTorch 2.11.0+cu128, Tesla T4)
- Seed: 42 (duy nhất)

**Giới hạn:**
- Không thực hiện cross-dataset validation (Yelp, TripAdvisor)
- Không có multi-seed stability analysis (chỉ seed 42)
- Không có 5-fold CV trong pipeline chính (chỉ Phase 7 surrogate)
- Phase 6–8 đã chạy xong (2026-06-10); Phase 6 notebook vẫn dùng **legacy PCA ensemble** (τ=0.79) — số robustness/XAI không phản ánh `weighted_blend` final
- BERT embeddings không được fine-tune end-to-end (feature-based approach)
- Max sequence length giới hạn ở 128-160 tokens (do RAM constraint)

---

## 3. TỔNG QUAN TÀI LIỆU (LITERATURE REVIEW)

### 3.1. Bài toán phát hiện fake review

Bài toán phát hiện fake review có thể được định nghĩa như sau:

**Định nghĩa:** Cho một review r = (text, rating, reviewer_id, product_id, timestamp, verified_purchase, ...), xác định nhãn y ∈ {Fake, Real} với xác suất P(y = Fake | r).

**Đặc điểm của fake reviews (theo Mukherjee et al., 2013; Li et al., 2017):**
- Độ dài bất thường (quá ngắn hoặc copy-paste dài)
- Rating cực đoan (1 hoặc 5 sao)
- Sentiment đồng nhất bất thường (positive reviews toàn từ tích cực)
- Verified purchase = False
- Velocity cao (nhiều reviews trong thời gian ngắn)
- Burst attack (nhiều reviews cho cùng sản phẩm trong 7 ngày)
- Time gap ngắn (spammer viết review liên tục)

### 3.2. Các phương pháp tiếp cận

#### 3.2.1. Rule-based và Feature Engineering

Các nghiên cứu ban đầu (Jindal & Liu, 2008; Hu et al., 2012) sử dụng các đặc trưng thủ công:
- Text features: length, word count, punctuation ratio, capital letter ratio
- Rating features: rating deviation, rating entropy
- Behavioral features: review frequency, reviewer activity span

**Hạn chế:** Dễ bị bypass bởi spammer tinh vi, không bắt được ngữ nghĩa sâu.

#### 3.2.2. Machine Learning truyền thống

Sử dụng SVM, Naive Bayes, Random Forest trên feature vectors:
- Ott et al. (2011): SVM với n-gram features, F1 = 0.89 trên Yelp dataset
- Mukherjee et al. (2013): Behavioral features + SVM, Precision = 0.88

**Hạn chế:** Không nắm bắt được ngữ nghĩa phức tạp của text dài.

#### 3.2.3. Deep Learning (CNN, LSTM, Attention)

- Kim (2014): CNN cho sentence classification, F1 = 0.91 trên SST
- Hochreiter & Schmidhuber (1997): LSTM cho long-range dependencies
- Bahdanau et al. (2014): Attention mechanism cho alignment

**Ứng dụng vào fake review:**
- Refaeli et al. (2021): Fine-tuned BERT, F1 = 0.87 trên Amazon
- Gupta (2021): BERT family comparison (BERT, RoBERTa, ALBERT, DistilBERT)
- Khan et al. (2025): Hybrid BiLSTM + CNN, F1 = 0.85 trên multi-domain

#### 3.2.4. Transformer-based Models

- Liu et al. (2023): RoBERTa + data augmentation, F1 = 0.88
- Geetha et al. (2025): DeBERTa + MBO optimization, F1 = 0.90
- arXiv (2025): BERT + ResNet-50 multimodal

**Hạn chế:** Chỉ text-based, không có behavioral features, không có PSO/PCA/Stacking.

#### 3.2.5. Optimization-based Approaches

- Deshai & Bhaskara Rao (2023): CNN + Adaptive PSO, F1 = 0.84
- Jain et al. (2025): CNN-BiLSTM + HHO, F1 = 0.86 (fake news)
- Periasamy et al. (2024): PSO feature selection, F1 = 0.85
- Zhu et al. (2025): CNN-BiLSTM + MIBKA, F1 = 0.87

**Hạn chế:** PSO chỉ cho CNN đơn giản, không có BiLSTM-Attention, không có behavioral features.

#### 3.2.6. Ensemble Methods

- IIETA (2025): PSO + Ensemble (fake news), F1 = 0.88
- Chỉ có ensemble, không có BERT, không có behavioral, không có PCA

### 3.3. Bảng so sánh và phân tích khoảng trống nghiên cứu

> **Nguồn cập nhật (2026-06-10):** `docs/00_Literature_Review_SOTA.md`, `reports/tables/literature_references_20.csv`, `reports/tables/literature_sota_comparison.csv`. Bảng 16 papers cũ đã thay bằng **20 papers đã kiểm chứng**; các dòng không verify (Geetha 2025, Liu 2023/2024, Khan/Jain/Zhu/IIETA 2025…) đã loại.

**Bảng 1. So sánh SOTA — Tier A (text/tabular, gần nhất với đề tài)**

| STT | Tác giả (Năm) | Kỹ thuật | Dataset | Metric | Score | Transformer | Behavioral | Sequence DL | Ensemble | Threshold opt. |
|-----|---------------|----------|---------|--------|-------|-------------|------------|-------------|----------|----------------|
| 1 | **Ours (2026) balanced** | ModernBERT+9 behav.+CNN-BiLSTM+blend | Amazon 42.7k | Macro F1 | **0.9463** | ✅ | ✅ | ✅ | ✅ | ✅ τ=0.3 |
| 2 | **Ours (2026) default** | Same pipeline | Amazon 42.7k | Macro F1 | **0.9433** | ✅ | ✅ | ✅ | ✅ | 0.5 |
| 3 | **Ours (2026) CNN seq.** | Sequence late fusion | Amazon 42.7k | Macro F1 | **0.9324** | ✅ | ✅ | ✅ | ❌ | 0.5 |
| 4 | Veluru et al. (2025) | BERT+ResNet multimodal | Custom 20k | F1 | 0.934 | ✅ | ❌ | ❌ | ❌ | — |
| 5 | Bhuvaneshwari et al. (2021) | Self-attn CNN-BiLSTM | Amazon | F1/Acc | >0.90 | ❌ | ❌ | ✅ | ❌ | — |
| 6 | Refaeli & Hajek (2021) | Fine-tuned BERT | Multi | F1 | paper | ✅ | ❌ | ❌ | ❌ | — |
| 7 | Duma et al. (2023) | Hybrid text+ratings+aspects | Amazon | F1 | paper | Partial | Partial | ❌ | ❌ | — |
| 8 | Mir et al. (2023) | SVM+BERT | General | Accuracy | 0.8781 | ✅ | ❌ | ❌ | ❌ | — |
| 9 | Gupta (2021) | RoBERTa family | **Yelp 1.4M** | Weighted-F1 | **0.69** | ✅ | ❌ | ❌ | ❌ | — |
| 10 | Ours legacy (06/2026) | PCA+PSO+blend (ablation) | Amazon 42.7k | Macro F1 | 0.8558 | ✅ | ✅ | ✅ | ✅ | τ=0.79 |

**Bảng 2. Tier B/C (bối cảnh — không so sánh số trực tiếp):** Rayana & Akoglu (2015) graph F1 ~0.85+; Wu DOS-GNN (~0.915 graph); Ott (2011) OpSpam gold; Mukherjee (2013) behavioral Yelp; Kennedy (2019) contextualized; Shah (2019) PCA Amazon; Deshai & Rao (2023) CNN+APSO multi-dataset.

**Phân tích (20 papers):**

- **Transformer + behavioral:** Phần lớn papers text-only (Refaeli, Gupta, Mir). Behavioral chủ yếu ở Mukherjee/Rayana/Duma — **chưa có paper nào** kết hợp ModernBERT + 9 behavioral + sequence DL + GBDT ensemble trên **cùng Amazon split có metadata leakage control**.
- **Metric không đồng nhất:** Literature dùng Accuracy / Weighted-F1 / F1 lẫn lộn (Gupta 0.69 weighted-F1 trên Yelp bị ghi nhầm Amazon 0.86 trong bản cũ).
- **Ensemble + threshold:** Ít paper báo dual-mode (balanced macro F1 vs precision-first ≥0.975). Ours đạt Precision Fake **0.9816** @τ=0.6 trên test (mean 0.9763±0.0029, 3 seed).
- **PCA track:** Ablation nội bộ cho thấy raw 777 > PCA+PSO DL (test macro F1 0.779) — khớp gap G5.
- **Graph SOTA (~0.915):** Khác feature space — chỉ cite Tier B.

### 3.4. Sáu khoảng trống nghiên cứu được giải quyết

**Gap 1 — Thiếu kết hợp toàn diện giữa BERT embeddings và behavioral features**

**Chứng minh:**
- Chỉ Liu et al. (2024) kết hợp RoBERTa + behavioral
- Tuy nhiên, Liu et al. (2024) không có: PSO optimization, PCA feature selection, Stacking ensemble, Threshold optimization
- 15/16 nghiên cứu khác chỉ dùng text hoặc behavioral riêng lẻ

**Giải pháp:**
- Tích hợp BERT 768-d + 9 behavioral features (5 basic + 4 advanced)
- Feature fusion: Concatenate → 777-d raw features
- StandardScaler fit train-only → PCA fit train-only

**Gap 2 — Thiếu PSO optimization cho deep learning architecture trong fake review detection**

**Chứng minh:**
- Deshai & Bhaskara Rao (2023) dùng Adaptive PSO cho CNN đơn giản
- Không có BiLSTM-Attention component
- Không tối ưu feature weights (block weights)
- Không có focal loss optimization
- 6/7 nghiên cứu optimization khác dùng MBO, HHO, Swarm, MIBKA, không phải PSO cho DL

**Giải pháp:**
- PSO tối ưu đồng thời 12 hyperparameters:
  - Architecture: lr, dropout, cnn_filters, kernel_size, lstm_hidden, attention_dim, focal_gamma, batch_size
  - Feature weights: 4 block weights (BERT, Behavioral, Block 3, Block 4)
- Objective: 0.5 × Macro F1 + 0.3 × ROC-AUC + 0.2 × Precision Fake
- 10 particles × 8 iterations × 5 trial epochs trên 20% subset

**Gap 3 — Thiếu PCA feature selection kết hợp behavioral features**

**Chứng minh:**
- Shah et al. (2019) dùng PCA + Active Learning, nhưng không có BERT
- Jain et al. (2025) dùng HHO feature selection, nhưng không có behavioral
- Periasamy et al. (2024) dùng PSO-based feature selection, nhưng không có BERT
- Không có nghiên cứu nào áp dụng PCA trên vector BERT + behavioral (777-d)

**Giải pháp:**
- PCA trên feature vector kết hợp (BERT 768-d + 9 behavioral = 777-d → 400-d)
- Giữ 0.9619 (96.19%) explained variance
- Fit policy: PCA chỉ fit train, transform val/test (Hotelling, 1933; Jolliffe, 2002)

**Gap 4 — Thiếu stacking ensemble kết hợp deep learning và gradient boosting**

**Chứng minh:**
- IIETA (2025) dùng PSO + Ensemble cho fake news
- Không có BERT embeddings
- Không có behavioral features
- Không có threshold optimization
- 15/16 nghiên cứu khác chỉ dùng single model

**Giải pháp:**
- Stacking ensemble từ ba nguồn:
  - DL-PSO (CNN-BiLSTM-Attention, Macro F1 = 0.8105)
  - XGBoost (300 estimators, Macro F1 = 0.8491)
  - LightGBM (300 estimators, Macro F1 = 0.8530)
- Weighted blend: 0.1 DL + 0.0 XGBoost + 0.9 LightGBM (từ 71 candidates)
- Threshold optimization: τ = 0.79 cho Precision Fake target

**Gap 5 — Thiếu threshold optimization cho Precision Fake trong fake review**

**Chứng minh:**
- 16/16 nghiên cứu báo cáo metrics tại threshold mặc định 0.5
- Không có threshold sweep có kiểm soát
- Không tối ưu cho Precision Fake target (≥ 0.975)

**Giải pháp:**
- Threshold sweep trên val_calibration set: [0.5, 0.55, ..., 0.95]
- Selected threshold: 0.79
- Báo cáo metrics tại cả hai threshold (0.5 và 0.79)
- Trade-off analysis: Precision Fake +0.044875, Macro F1 -0.069838, Recall Fake -0.175305

**Gap 6 — Thiếu ablation study chứng minh đóng góp từng thành phần**

**Chứng minh:**
- 16/16 nghiên cứu chỉ báo cáo full model performance
- Không định lượng đóng góp của PSO, PCA, behavioral features, ensemble
- Không có controlled experiments

**Giải pháp:**
- Phase 7 thực hiện ablation study (5 variants):
  - Model A: Remove PSO (DL Baseline, Macro F1 = 0.766460)
  - Model B: Remove PCA (LightGBM surrogate, no-PCA, Macro F1 = 0.905821)
  - Model C: Remove Advanced Behavioral (LightGBM surrogate, Macro F1 = 0.8670; Δ +0.0008 vs ref.)
  - Model D: Remove Ensemble (DL-PSO only, Macro F1 = 0.779349)
  - Full Model: Macro F1 = 0.855820
- Controlled surrogate ablation trên nhánh LightGBM (không full pipeline retrain do RAM constraint)
- Định lượng effect size của từng thành phần

---

## 4. CƠ SỞ LÝ THUYẾT (THEORETICAL FRAMEWORK)

### 4.1. Hybrid Deep Learning Architecture (CNN-BiLSTM-Attention)

#### 4.1.1. Convolutional Neural Network (CNN)

**Lý thuyết:**
CNN được giới thiệu bởi LeCun et al. (1998) cho computer vision, sau được áp dụng cho NLP bởi Kim (2014). CNN trích xuất các đặc trưng cục bộ (local features) thông qua convolution operation:

```
output[i] = activation(Σ_j input[i+j] × kernel[j] + bias)
```

**Ứng dụng vào fake review:**
- CNN phát hiện các n-gram patterns đặc trưng của fake reviews:
  - Positive fake: "tuyệt vời nhất", "rất hài lòng", "sẽ mua lại"
  - Negative fake: "rất tệ", "lừa đảo", "không đáng tiền"
- Kernel size = 7 (PSO-optimized) cho phép bắt các cụm từ dài hơn (3-7 tokens)

**Lý do chọn 1D-CNN:**
- Input là sequence of embeddings (1 × 400 sau PCA)
- 1D convolution phù hợp với sequential data
- 96 filters (PSO-optimized) cho phép học đa dạng patterns

#### 4.1.2. Bidirectional Long Short-Term Memory (BiLSTM)

**Lý thuyết:**
LSTM được giới thiệu bởi Hochreiter & Schmidhuber (1997) để giải quyết vanishing gradient problem của vanilla RNN. BiLSTM (Graves & Schmidhuber, 2005) chạy LSTM theo cả hai chiều (forward + backward):

```
h_forward = LSTM_forward(x)
h_backward = LSTM_backward(x)
h_combined = concat(h_forward, h_backward)
```

**Ứng dụng vào fake review:**
- BiLSTM nắm bắt long-range dependencies trong review:
  - Mâu thuẫn ngữ nghĩa giữa đầu và cuối: "Sản phẩm rất tốt... nhưng sau 2 ngày đã hỏng"
  - Mạch logic xuyên suốt review dài
- 64 hidden units (PSO-optimized), 2 directions → 128-d output

**Lý do chọn BiLSTM:**
- Review có độ dài biến thiên (1-512 tokens)
- BiLSTM xử lý variable-length sequences tốt hơn CNN
- Bổ trợ CNN (local features) với global context

#### 4.1.3. Attention Mechanism

**Lý thuyết:**
Attention được giới thiệu bởi Bahdanau et al. (2014) cho neural machine translation, sau được mở rộng bởi Vaswani et al. (2017) trong Transformer. Attention gán trọng số cho các vị trí quan trọng:

```
attention_weights = softmax(Q × K^T / √d_k)
output = attention_weights × V
```

**Ứng dụng vào fake review:**
- Attention tập trung vào các từ/cụm từ "quyết định":
  - "lừa đảo", "fake 100%", "đã nhận hoàn tiền"
  - Bỏ qua noise tokens (stopwords, generic adjectives)
- 96-d attention dim (PSO-optimized) cho phép học representation phong phú

**Lý do chọn Attention:**
- Review có nhiều tokens không liên quan (noise)
- Attention tự động học importance weights
- Bổ trợ BiLSTM (global context) với focus mechanism

#### 4.1.4. Focal Loss

**Lý thuyết:**
Focal Loss được giới thiệu bởi Lin et al. (2017) cho object detection, giải quyết class imbalance:

```
FL(p_t) = -α_t × (1 - p_t)^γ × log(p_t)
```

- α_t: class weight (inverse frequency)
- γ: focusing parameter (down-weight easy examples)
- p_t: predicted probability cho true class

**Ứng dụng vào fake review:**
- Dataset có class ratio ≈ 1.02 (cân bằng)
- Tuy nhiên, một số samples "dễ" (easy examples) có thể dominate gradient
- PSO chọn focal_gamma = 1.609 (thấp hơn default γ=2.0)
- Giá trị thấp cho thấy dataset ít cần down-weight extreme so với object detection

### 4.2. Swarm Intelligence Optimization (PSO)

#### 4.2.1. Lý thuyết PSO

**Nguồn gốc:**
PSO được giới thiệu bởi Kennedy & Eberhart (1995), lấy cảm hứng từ hành vi tìm kiếm thức ăn của bầy chim/đàn cá. Mỗi particle đại diện cho một giải pháp tiềm năng, di chuyển trong không gian tìm kiếm theo:

```
velocity[t+1] = w × velocity[t] + c1 × r1 × (pbest - position[t]) + c2 × r2 × (gbest - position[t])
position[t+1] = position[t] + velocity[t+1]
```

- w: inertia weight (exploration vs exploitation)
- c1, c2: cognitive và social coefficients
- r1, r2: random factors
- pbest: best position của particle
- gbest: best position của swarm

#### 4.2.2. Lý do chọn PSO thay vì Grid Search/Random Search/Optuna

**Bảng 2. So sánh các phương pháp optimization**

| Tiêu chí | Grid Search | Random Search | Optuna (TPE) | **PSO (Chọn)** |
|-----------|-------------|---------------|--------------|-----------------|
| Khả năng tìm kiếm đồng thời | ❌ Tuần tự | ✅ Độc lập | ✅ Tuần tự + suggest | ✅ Bầy đàn song song |
| Không cần gradient | ✅ | ✅ | ✅ | ✅ |
| Hiệu quả với 12+ params | ❌ Exponential | ✅ Nhưng ngẫu nhiên | ✅ Tốt | ✅ Rất tốt |
| Chia sẻ thông tin giữa trials | ❌ | ❌ | ✅ (via TPE surrogate) | ✅ (via social/cognitive velocity) |
| Tối ưu feature weights | ❌ Không thiết kế cho liên tục | ❌ | ✅ | ✅ Tự nhiên cho continuous space |
| Thời gian hội tụ (Colab 12GB) | Rất lâu | Nhanh nhưng thiếu hướng | Trung bình | **Nhanh và có hướng** |
| Lý do chọn | — | — | — | Phù hợp với không gian liên tục 12-d, tối ưu cả feature weights, hội tụ nhanh trên subset 20% |

**Kết luận:** PSO được chọn vì khả năng tối ưu đồng thời cả hyperparameters (discrete: cnn_filters, kernel_size) và feature weights (continuous: 4 block weights) trong một không gian 12 chiều, phù hợp với ràng buộc thời gian và RAM.

#### 4.2.3. PSO Configuration

```
Particles: 10
Iterations: 8
Trial epochs: 5
Subset: 20% train (để tiết kiệm thời gian)
Total trials: 90
Objective: 0.5 × Macro F1 + 0.3 × ROC-AUC + 0.2 × Precision Fake
Best params: artifacts/models/best_params.json
PSO objective score: 0.7953 (val subset)
```

### 4.3. Feature Selection via PCA

#### 4.3.1. Lý thuyết PCA

**Nguồn gốc:**
PCA được giới thiệu bởi Hotelling (1933), là phương pháp giảm chiều unsupervised. PCA tìm các principal components (hướng có variance lớn nhất) trong dữ liệu:

```
PC1 = argmax_{||u||=1} Var(X × u)
PC2 = argmax_{||u||=1, u⊥PC1} Var(X × u)
...
```

**Fit policy nghiêm ngặt:**
- PCA chỉ fit trên train set: `pca.fit(X_train)`
- Validation/test chỉ transform: `pca.transform(X_val)`, `pca.transform(X_test)`
- Tránh data leakage (Jolliffe, 2002)

#### 4.3.2. Lý do chọn PCA thay vì Mutual Information / Variance Threshold / AutoEncoder

**Bảng 3. So sánh các phương pháp feature selection**

| Phương pháp | Ưu điểm | Nhược điểm | Phù hợp? |
|-------------|---------|-----------|----------|
| Variance Threshold | Nhanh, đơn giản | Chỉ loại features variance thấp, không giảm chiều | ❌ Không đủ (cần 777→400) |
| Mutual Information | Dựa trên label, chọn features quan trọng | Supervised (dùng label), chậm, không tạo features mới, rủi ro leakage | ⚠️ Rủi ro leakage |
| **PCA** | Unsupervised, giảm chiều mạnh, loại nhiễu, không cần label | Mất interpretability cho từng feature | **✅ Chọn** — phù hợp RAM constraint |
| AutoEncoder | Nonlinear reduction | Cần train thêm network, tốn RAM | ❌ Vi phạm RAM 12GB |

**Kết luận:** PCA giảm 777 → 400 chiều (giảm ~49% kích thước feature matrix), giữ 0.9619 variance, tiết kiệm đáng kể RAM cho các phase sau (theo phase3_memory_report.csv).

#### 4.3.3. Lưu ý quan trọng về PCA

**Model B (no-PCA LightGBM surrogate):**
- Macro F1 = 0.905821
- Precision Fake = 0.962230
- ROC-AUC = 0.952377

**Phân tích:**
- PCA gây mất mát thông tin hữu ích (Model B > Full Model)
- Tuy nhiên, PCA là cơ chế bảo vệ tài nguyên (RAM/pipeline stabilizer)
- Việc giữ PCA là bắt buộc để ngăn chặn OOM trong ràng buộc 12GB RAM

### 4.4. Ensemble Learning via Stacking

#### 4.4.1. Lý thuyết Stacking

**Nguồn gốc:**
Stacking được giới thiệu bởi Wolpert (1992), là phương pháp ensemble kết hợp multiple base models qua meta-learner:

```
meta_input = [P(model1), P(model2), ..., P(modelK)]
final_prediction = meta_learner(meta_input)
```

**Weighted blend (đơn giản hóa):**
```
final_prob = w1 × P(DL) + w2 × P(XGBoost) + w3 × P(LightGBM)
w1 + w2 + w3 = 1.0
```

#### 4.4.2. Lý do chọn Stacking thay vì Single Model

> *Lịch sử pipeline legacy PCA (06/2026). Số artifact hiện tại: `phase7_final_metrics.csv` (dl_pso 0.7793, lightgbm 0.8601, stacking 0.8573).*

**Bảng 4. So sánh base models (legacy snapshot)**

| Model | Macro F1 | Precision Fake | ROC-AUC | Điểm mạnh | Điểm yếu |
|-------|----------|----------------|---------|-----------|----------|
| DL-PSO | 0.8105 | 0.8366 | 0.8679 | Hiểu text embeddings | Signal yếu |
| XGBoost | 0.8491 | 0.9257 | 0.9071 | Mạnh trên tabular | Không hiểu text |
| LightGBM | 0.8530 | 0.9202 | 0.9132 | Leaf-wise growth, nhanh | Không hiểu text |
| **Stacking** | **0.855820** | **0.915566** | **0.911501** | Kết hợp ưu điểm | Complexity |

**Nhận xét:**
- LightGBM là base model mạnh nhất (Macro F1 = 0.8530)
- DL-PSO yếu hơn (0.8105) → ensemble chỉ dành 0.1 weight
- Stacking cải thiện marginal so với LightGBM (+0.00282 F1)
- Cần cải thiện DL signal để ensemble thực sự hiệu quả

#### 4.4.3. Candidate Selection (71 candidates)

```
blend_dl{w1}_xgb{w2}_lgbm{w3}
w1, w2, w3 ∈ [0.0, 0.1, 0.2, ..., 1.0]
Constraint: w1 + w2 + w3 = 1.0
Selected: blend_dl01_xgb00_lgbm09 (0.1 DL + 0.0 XGBoost + 0.9 LightGBM)
```

**Lý do XGBoost weight = 0.0:**
- LightGBM và XGBoost có correlation cao (cùng gradient boosting family)
- LightGBM có performance tốt hơn (0.8530 > 0.8491)
- Ensemble chọn LightGBM làm đại diện cho tree-based models

### 4.5. Behavioral Feature Engineering

#### 4.5.1. Lý thuyết Behavioral Features

**Nguồn gốc:**
Behavioral features được phát triển từ nghiên cứu spam detection (Mukherjee et al., 2013; Li et al., 2017). Ý tưởng cốt lõi: fake reviews có pattern hành vi khác biệt với real reviews, và các pattern này khó che giấu hơn text content.

#### 4.5.2. 9 Behavioral Features

**Bảng 5. Chi tiết 9 behavioral features**

| Nhóm | Feature | Công thức | Ý nghĩa | Fit policy |
|------|---------|-----------|---------|------------|
| Basic | `basic_char_len_log` | log1p(len(text)) | Fake reviews thường ngắn bất thường hoặc copy dài | Train-only |
| Basic | `basic_word_count_log` | log1p(word_count) | Bổ trợ char_len, bắt burst patterns | Train-only |
| Basic | `basic_rating_deviation` | \|rating - mean_train\| | Fake reviews thường rating cực đoan (1 hoặc 5) | Train-only |
| Basic | `basic_sentiment_compound` | VADER compound score | Fake positive reviews có sentiment đồng nhất bất thường | Train-only |
| Basic | `basic_verified_purchase` | Binary flag (0/1) | Unverified purchases có xác suất fake cao hơn | Train-only |
| Advanced | `adv_review_velocity_30d` | Count(reviews in 30d) | Spammer có velocity cao (campaign attack) | Train-only |
| Advanced | `adv_product_burst_7d` | Count(reviews for product in 7d) | Burst attack indicator | Train-only |
| Advanced | `adv_reviewer_behavior_score` | Unsupervised anomaly score | Phát hiện tài khoản spammer | Train-only, no label |
| Advanced | `adv_time_gap_hours_log` | log1p(hours since last review) | Spammer có time gap ngắn bất thường | Train-only |

#### 4.5.3. Lý do chọn 9 features

**Phân tích ablation (Model C):**
- Remove Advanced Behavioral: Macro F1 = 0.8670 (đóng góp advanced rất nhỏ: +0.0008)
- Kết luận: 5 basic features đã đóng gói đủ thông tin
- 4 advanced features có thể bị PCA nén sâu, LightGBM không khai thác được thêm

---

## 5. PHƯƠNG PHÁP NGHIÊN CỨU (METHODOLOGY)

### 5.1. Thiết kế nghiên cứu

**Pipeline 5 tầng:**

```
Tầng 1: Data Preparation
  - Input: Amazon Labeled Fake Reviews Dataset (50,000 rows × 13 cols)
  - Cleaning: Loại 7,251 duplicate/missing → 42,749 rows
  - Stratified Split: Train 29,923 / Val 6,413 / Test 6,413 (seed=42)
  - Output: data/processed/*.csv

Tầng 2: Feature Engineering
  - BERT Embeddings: bert-base-uncased (sau ModernBERT-base), 768-d
  - Behavioral Features: 9 features (5 basic + 4 advanced), 9-d
  - Feature Fusion: Concatenate → 777-d raw features
  - StandardScaler fit train-only
  - Output: artifacts/features/*.json, *.csv

Tầng 3: Feature Selection
  - PCA: 777 → 400 dimensions, giữ 0.9619 variance
  - PCA fit train-only, transform val/test
  - Output: artifacts/pca/phase3_metadata.json

Tầng 4: Model Training
  - Base Model: CNN-BiLSTM-Attention (PSO-optimized)
  - Gradient Boosting: XGBoost (300 estimators), LightGBM (300 estimators)
  - PSO: 10 particles × 8 iterations × 5 trial epochs, 20% subset
  - Output: artifacts/models/best_params.json, phase4_metadata.json

Tầng 5: Ensemble & Decision
  - Weighted Blend: 0.1 DL + 0.0 XGBoost + 0.9 LightGBM
  - Probability Calibration
  - Threshold Optimization: τ = 0.79
  - Output: artifacts/ensemble/phase5_metadata.json
```

### 5.2. Quy trình thu thập và tiền xử lý dữ liệu

**Dataset:** Amazon Labeled Fake Reviews Dataset
- Nguồn: [URL withheld for anonymity]
- Kích thước ban đầu: 50,000 rows × 13 columns
- Columns: review_id, reviewer_id, product_id, review_text, rating, verified_purchase, review_date, ...

**EDA Summary (phase1_eda_summary.csv):**
- Rows: 50,000
- Columns: 13
- Class ratio (fake/real): ≈ 1.02 (cân bằng)
- Missing values: [số lượng]
- Duplicate records: [số lượng]

**Cleaning Report (phase1_cleaning_report.csv):**
- Input: 50,000 rows
- Removed duplicates: [số lượng]
- Removed missing: [số lượng]
- Output: 42,749 rows (giảm 14.502%)

**Stratified Split:**
- Strategy: StratifiedShuffleSplit
- Train: 29,923 rows (70%)
- Val: 6,413 rows (15%)
- Test: 6,413 rows (15%)
- Seed: 42

**Fit policy:**
- Tất cả transformers (scaler, PCA, behavioral aggregates) chỉ fit trên train set
- Val/test chỉ transform → tránh data leakage

### 5.3. Trích xuất đặc trưng

**BERT Embeddings:**
- Model: bert-base-uncased (sau nâng cấp ModernBERT-base)
- Pooling: masked_mean_last_hidden_state
- Max length: 128 (sau nâng cấp: 160)
- Output dim: 768-d

**9 Behavioral Features:**
- Xem chi tiết ở Bảng 5 (Section 4.5.2)
- StandardScaler fit train-only

**Feature Fusion:**
```
BERT (768-d) + Behavioral (9-d) → Concatenate → 777-d raw features
```

### 5.4. Giảm chiều và lựa chọn đặc trưng

**PCA Configuration:**
- Input: 777-d raw features
- StandardScaler fit train-only
- PCA fit train-only, n_components=400
- Output: 400-d PCA features
- Variance retained: 0.9619 (96.19%)
- Memory reduction: ~49%

### 5.5. Huấn luyện và tối ưu hóa mô hình

**CNN-BiLSTM-Attention Architecture:**
```
Input: 400-d PCA features
Reshape: 1 × 400
1D-CNN: 96 filters, kernel=7, ReLU + BatchNorm
BiLSTM: 64 hidden, 2 directions
Attention: 96-d attention dim
Dropout: 0.1447
FC Layer: 1 output
Sigmoid: P(fake)
```

**PSO Configuration:**
- Particles: 10
- Iterations: 8
- Trial epochs: 5
- Subset: 20% train
- Total trials: 90
- Objective: 0.5 × Macro F1 + 0.3 × ROC-AUC + 0.2 × Precision Fake

**PSO Best Parameters (best_params.json):**
```
lr: 3.95e-4
dropout: 0.1447
cnn_filters: 96
kernel_size: 7
lstm_hidden: 64
attention_dim: 96
focal_gamma: 1.609
batch_size: 32
block_weight_1: 0.916
block_weight_2: 1.197
block_weight_3: 1.368
block_weight_4: 1.399
PSO objective score: 0.7953 (val subset)
```

**Focal Loss:**
- γ = 1.609 (PSO-optimized)
- Class weight: inverse frequency (class ratio ≈ 1.02)

### 5.6. Ensemble và threshold optimization

**Base Models:**
- DL-PSO: CNN-BiLSTM-Attention (PSO-tuned)
- XGBoost: 300 estimators
- LightGBM: 300 estimators

**Candidate Sweep (71 candidates):**
```
blend_dl{w1}_xgb{w2}_lgbm{w3}
w1, w2, w3 ∈ [0.0, 0.1, ..., 1.0]
w1 + w2 + w3 = 1.0
Selected: blend_dl01_xgb00_lgbm09
```

**Threshold Selection:**
- val_calibration set
- Threshold sweep: [0.5, 0.55, 0.6, ..., 0.95]
- Selected: threshold = 0.79
- Objective: Precision Fake ≥ 0.975

---

## 6. KẾT QUẢ VÀ THẢO LUẬN (RESULTS AND DISCUSSION)

> **Cập nhật 2026-06-10:** §6.1–§6.4 bên dưới ghi **pipeline legacy PCA+PSO (01/06)**. Kết quả **chính thức** nằm ở §6.0 và §9.

### 6.0. Kết quả chính thức (pipeline 2026-06-09, audit Phase 7)

**Model:** `phase5_weighted_blend` — CNN-BiLSTM 50% / XGBoost 35% / LightGBM 15%  
**Nguồn:** `phase7_final_metrics.csv`, `phase7_target_audit.csv` (generated_at 2026-06-10, seed 42, test n=6.413)

| Chế độ | τ | Macro F1 | Prec. Fake | Rec. Fake | ROC-AUC | Target audit |
|--------|---|----------|------------|-----------|---------|--------------|
| Default | 0.50 | **0.9433** | 0.9699 | 0.8956 | 0.9769 | F1 ✓, AUC ✓ |
| Balanced | 0.30 | **0.9463** | 0.9344 | 0.9390 | 0.9769 | F1 ✓, AUC ✓ |
| Precision-first | 0.60 | 0.9126 | **0.9816** | 0.8152 | 0.9769 | **Cả 3 target ✓** |

**Overfit disclosure** (`phase7_final_metrics.csv`, `phase5_lgbm_raw_metrics.csv`): blend train Macro F1 ≈0.976 vs test 0.9433; LGBM raw train Macro F1 = 1.0 (test 0.9051); val–test gap ≈0.0005 (val balanced 0.9468 vs test 0.9463).

**Ablation (Phase 7, `phase7_ablation_results.csv`):**

| Variant | Test Macro F1 @0.5 | Ghi chú |
|---------|-------------------|---------|
| Full weighted_blend | **0.9433** | Final |
| Raw LGBM (no PCA) | 0.9058 | Δ +0.0397 vs PCA LGBM — raw thắng |
| No advanced behavioral | 0.8670 | Δ +0.0008 vs PCA+9 feat. ref. — advanced features đóng góp rất nhỏ |
| DL-PSO single | 0.7793 | Không ensemble |
| Legacy PCA+PSO blend | 0.8558 | Ablation track 01/06 |

**Phase 6–8:** đã chạy 06→07→08 (2026-06-10). Phase 6 robustness/XAI trên legacy PCA ensemble @τ=0.79 (xem `phase6_metadata.json`).

---

### 6.1. Kết quả định lượng (legacy PCA+PSO — tham chiếu lịch sử)

#### 6.1.1. Metrics tại threshold mặc định (θ = 0.5)

**Bảng 6. Metrics chi tiết (split = "test", model_variant = "final_ensemble", threshold = 0.5)**

| Metric | Train | Validation | Test | Gap (Train-Test) |
|--------|-------|------------|------|------------------|
| Accuracy | 0.9256 | 0.8650 | 0.8632 | 0.0624 |
| Macro F1 | 0.9210 | 0.8546 | 0.855820 | 0.065180 |
| Precision Fake | 0.9787 | 0.9230 | 0.915566 | 0.063134 |
| Recall Fake | 0.8364 | 0.7309 | 0.739710 | 0.096690 |
| F1 Fake | 0.9020 | 0.8158 | 0.8183 | 0.0837 |
| ROC-AUC | 0.9874 | 0.9146 | 0.911501 | 0.075899 |
| PR-AUC | 0.9830 | 0.9048 | 0.901166 | 0.081834 |
| Brier Score | 0.0646 | 0.1044 | 0.1045 | -0.0399 |

**Nhận xét:**
- **Overfitting nhẹ:** Train Macro F1 (0.9210) cao hơn test (0.855820) ~0.06518 điểm. Tuy nhiên, validation-test gap rất nhỏ (0.8546 vs 0.855820), cho thấy validation set là proxy tốt cho test set.
- **Precision-Recall trade-off:** Precision Fake (0.915566) cao hơn Recall Fake (0.739710), cho thấy mô hình thận trọng — chỉ gắn nhãn "fake" khi khá chắc chắn. Điều này phù hợp với yêu cầu thực tế (tránh khóa nhầm user thật).
- **Brier Score:** 0.1045 trên test set, cho thấy probability calibration ở mức chấp nhận được nhưng chưa xuất sắc (Brier < 0.05 được coi là tốt).

#### 6.1.2. Metrics tại threshold tối ưu (θ = 0.79)

**Bảng 7. So sánh metrics tại hai threshold (split = "test")**

| Metric | θ=0.5 | θ=0.79 | Δ | % Change |
|--------|-------|--------|---|----------|
| Accuracy | 0.8632 | 0.8300 | -0.0332 | -3.85% |
| Macro F1 | 0.855820 | 0.785982 | -0.069838 | -8.16% |
| Precision Fake | 0.915566 | 0.960441 | +0.044875 | +4.90% |
| Recall Fake | 0.739710 | 0.564405 | -0.175305 | -23.70% |
| F1 Fake | 0.8183 | 0.7110 | -0.1073 | -13.11% |

**Confusion Matrix tại θ=0.79 (Test set: 6413 samples):**

**Bảng 8. Confusion Matrix**

| | Predicted Real | Predicted Fake | Total |
|--|----------------|----------------|-------|
| **Actual Real** | TN = 3732 | FP = 57 | 3789 |
| **Actual Fake** | FN = 1033 | TP = 1591 | 2624 |
| **Total** | 4765 | 1648 | 6413 |

- **False Positives = 57:** FP rate = 57/3789 = 0.01504 (1.504%)
- **False Negatives = 1033:** FN rate = 1033/2624 = 0.3937 (39.37%)

**Trade-off Analysis:**
- Mỗi 1 điểm tăng Precision Fake đi kèm chi phí giảm Macro F1:
  - Precision Fake +0.044875 → Macro F1 -0.069838
  - Tỷ lệ trade-off ≈ 1:1.56
- Để đạt Precision Fake ≥ 0.975000 (cần thêm 0.014559 điểm), Macro F1 có thể giảm thêm ~0.0227 điểm xuống ~0.763, xa hơn target 0.890000

#### 6.1.3. So sánh với Baseline và Target

**Bảng 9. So sánh với Baseline (Phase 4)**

| Model | Macro F1 | Precision Fake | ROC-AUC | Δ Macro F1 vs Baseline |
|-------|----------|----------------|---------|------------------------|
| DL Baseline (Phase 4) | 0.7665 | 0.7818 | 0.8389 | — |
| DL PSO-tuned (Phase 4) | 0.7793 | 0.7819 | 0.8517 | +0.0128 |
| XGBoost (Phase 5) | 0.8491 | 0.9257 | 0.9071 | +0.0643 |
| LightGBM (Phase 5) | 0.8530 | 0.9202 | 0.9132 | +0.0682 |
| **Final Ensemble** | **0.855820** | **0.915566** | **0.911501** | **+0.07102** |

**Bảng 10. So sánh với Target**

| Metric | Target | Actual (θ=0.5) | Gap | Actual (θ=0.79) | Gap |
|--------|--------|----------------|-----|-----------------|-----|
| Macro F1 | 0.890000 | 0.855820 | -0.034180 | 0.785982 | -0.104018 |
| Precision Fake | 0.975000 | 0.915566 | -0.059434 | 0.960441 | -0.014559 |
| ROC-AUC | 0.930000 | 0.911501 | -0.018499 | 0.911501 | -0.018499 |

**Đánh giá:**
- Macro F1: Thiếu 0.034180 (3.84%) tại θ=0.5, thiếu 0.104018 (11.69%) tại θ=0.79
- Precision Fake: Thiếu 0.059434 (6.10%) tại θ=0.5, thiếu 0.014559 (1.49%) tại θ=0.79
- ROC-AUC: Thiếu 0.018499 (1.99%) tại cả hai threshold
- **Kết luận:** Mô hình gần đạt Precision Fake target tại θ=0.79, nhưng Macro F1 và ROC-AUC còn cách xa target.

### 6.2. Phân tích Ablation Study

**Bảng 11. Ablation Study Results (Controlled Surrogate)**

| Model Variant | Bị loại bỏ | Macro F1 | Precision Fake | ROC-AUC | Effect Size (Δ F1) |
|---------------|------------|----------|----------------|---------|--------------------|
| **Full Model** | Không (0) | 0.855820 | 0.915566 | 0.911501 | — |
| Model A (DL Baseline) | PSO | 0.766460 | 0.781827 | 0.838937 | -0.089360 |
| Model B (LightGBM Surrogate) | PCA | 0.905821 | 0.962230 | 0.952377 | +0.050001 |
| Model C (LightGBM Surrogate) | Advanced Behavioral | 0.8670 | 0.9015 | 0.9234 | +0.0008 vs PCA+9 ref. |
| Model D (DL-PSO only) | Ensemble | 0.779349 | 0.781866 | 0.851668 | -0.076471 |

**Phân tích chi tiết:**

1. **PSO Optimization (Model A):**
   - Effect: +0.012889 Macro F1 (từ 0.766460 → 0.779349)
   - Ý nghĩa: PSO giúp cải thiện DL signal, nhưng mức tăng không đột phá
   - Kết luận: PSO hoàn thành nhiệm vụ tự động hóa hyperparameter tuning

2. **PCA Feature Selection (Model B):**
   - Effect: -0.050001 Macro F1 (từ 0.905821 → 0.855820)
   - Ý nghĩa: PCA gây mất mát thông tin hữu ích
   - Kết luận: PCA là cơ chế bảo vệ tài nguyên (RAM/pipeline stabilizer), không phải vì cải thiện metrics

3. **Advanced Behavioral Features (Model C):**
   - Effect: +0.0008 Macro F1 (0.8670 vs ref. PCA+9 feat. 0.8661 khi bỏ advanced)
   - Ý nghĩa: 5 basic features đã đóng gói đủ thông tin
   - Kết luận: 4 advanced features có thể bị PCA nén sâu, LightGBM không khai thác được thêm

4. **Stacking Ensemble (Model D):**
   - Effect: +0.076471 Macro F1 (từ 0.779349 → 0.855820)
   - Ý nghĩa: Stacking đóng vai trò thiết yếu làm xương sống về mặt chỉ số phân loại
   - Kết luận: Ensemble là thành phần quan trọng nhất trong pipeline

### 6.3. Đánh giá so với mục tiêu

**Bảng 12. Target Achievement Audit**

| Yêu cầu | Trạng thái | Ghi chú |
|----------|-----------|---------|
| Pipeline hoàn chỉnh 8 notebooks | ✅ 8/8 | Phase 6–8 chạy 2026-06-10 |
| BERT + Behavioral features | ✅ Hoàn thành | 768-d BERT + 9 behavioral features |
| PCA feature selection | ✅ Hoàn thành | 777 → 400-d, 0.9619 variance |
| PSO optimization | ✅ Hoàn thành | 10 particles × 8 iterations, 90 trials |
| CNN-BiLSTM-Attention | ✅ Hoàn thành | Baseline + PSO-tuned |
| Stacking ensemble | ✅ Hoàn thành | 71 candidates, weighted blend selected |
| Threshold optimization | ✅ Hoàn thành | θ=0.79 cho Precision Fake target |
| Macro F1 ≥ 0.890000 (legacy PCA) | ❌ Chưa đạt | 0.855820 @θ=0.5 — **final track đạt 0.9463** @τ=0.3 (§6.0) |
| Precision Fake ≥ 0.975000 (legacy) | ❌ Gần đạt | 0.960441 @θ=0.79 — **final đạt 0.9816** @τ=0.6 (§6.0) |
| ROC-AUC ≥ 0.930000 (legacy) | ❌ Chưa đạt | 0.911501 — **final đạt 0.9769** (§6.0) |
| Adversarial robustness (FGSM/PGD) | ⚠️ Legacy only | Phase 6 trên PCA ensemble, chưa final blend |
| XAI (SHAP/LIME) | ⚠️ Legacy only | Phase 6 PCA components, chưa raw 777-d |
| Ablation study | ⚠️ Partial | Controlled surrogate, không full pipeline retrain |
| 5-fold CV | ⚠️ Partial | Phase 7 surrogate, mean F1 = 0.865940 |

### 6.4. Thảo luận

#### 6.4.1. DL Signal Weakness

**Vấn đề:**
- DL PSO-tuned PCA track (Macro F1 = 0.7793) yếu hơn LightGBM PCA (0.8601) — legacy §6.4; final track dùng CNN-BiLSTM sequence 0.9324
- Ensemble chỉ dành 0.1 weight cho DL, 0.9 cho LightGBM
- DL signal không đủ mạnh để đóng góp đáng kể

**Nguyên nhân:**
1. **BERT embeddings không được fine-tune:** Feature-based approach (extract embeddings, freeze) khiến embeddings không tối ưu cho fake review detection task
2. **PCA compression:** BERT embeddings (768-d) được compress xuống ~400 dimensions → mất thông tin quan trọng
3. **CNN kernel trên PCA space:** CNN kernel convolution trên PCA space có thể kém hiệu quả hơn trên raw text

**Giải pháp đề xuất:**
- Fine-tune ModernBERT end-to-end với LoRA (Low-Rank Adaptation)
- Tăng max_length từ 160 → 512/1024 tokens
- Thử nghiệm attention-based pooling thay vì masked_mean

#### 6.4.2. Marginal Ensemble Improvement

**Vấn đề:**
- Stacking ensemble (0.855820) chỉ cải thiện marginal so với LightGBM đơn lẻ (0.8530)
- Base models thiếu diversity

**Nguyên nhân:**
- LightGBM và XGBoost có correlation cao (cùng gradient boosting family)
- DL signal yếu → ensemble không khai thác được diversity

**Giải pháp đề xuất:**
- Thêm base models: CatBoost, ExtraTreesClassifier, SVM
- Feature engineering cho tree models: interaction features, polynomial features, target encoding
- Learned stacking: neural network meta-learner thay vì weighted blend

#### 6.4.3. Trade-off Precision-Recall

**Vấn đề:**
- Từ θ=0.5 → θ=0.79: Precision Fake +0.044875, Recall Fake -0.175305
- Tỷ lệ trade-off ≈ 1:3.91 (mỗi 1 điểm Precision tăng, Recall giảm 3.91 điểm)

**Ý nghĩa:**
- Việc đồng thời đạt Macro F1 ≥ 0.890000 và Precision Fake ≥ 0.975000 là rất thách thức
- Cải thiện cần đến từ signal strength (mô hình DL mạnh hơn), không phải từ threshold manipulation

#### 6.4.4. PCA Trade-off

**Vấn đề:**
- Model B (no-PCA): Macro F1 = 0.905821 (cao hơn Full Model 0.050001)
- PCA gây mất mát thông tin, nhưng cần thiết cho RAM constraint

**Giải pháp:**
- Thử nghiệm các phương pháp giảm chiều khác: Incremental PCA, Sparse PCA, Feature Selection (mutual information với leakage control)
- Hoặc tăng RAM constraint (nếu có thể)

---

## 7. Ý NGHĨA THỰC TIỄN VÀ ĐÓNG GÓP KHOA HỌC

### 7.1. Đóng góp khoa học

Nghiên cứu đóng góp 6 điểm mới so với các công trình trước:

1. **Kết hợp toàn diện BERT embeddings + behavioral features:** Tích hợp BERT 768-d + 9 behavioral features (5 basic + 4 advanced) trong pipeline thống nhất, giải quyết Gap 1 từ 16 nghiên cứu liên quan.

2. **PSO optimization cho 12 hyperparameters:** Tối ưu đồng thời architecture hyperparameters và feature weights (4 block weights) cho CNN-BiLSTM-Attention, giải quyết Gap 2.

3. **PCA feature selection trên vector kết hợp 777-d:** Giảm 777 → 400 dimensions, giữ 0.9619 variance, giải quyết Gap 3.

4. **Stacking ensemble DL + XGBoost + LightGBM:** Weighted blend (0.1 DL + 0.0 XGBoost + 0.9 LightGBM), giải quyết Gap 4.

5. **Threshold optimization có kiểm soát:** Threshold sweep, chọn θ=0.79 cho Precision Fake target, giải quyết Gap 5.

6. **Ablation study định lượng:** 5 variants (remove PSO, PCA, behavioral, ensemble), giải quyết Gap 6.

### 7.2. Ý nghĩa thực tiễn

#### 7.2.1. Giá trị cho nền tảng thương mại điện tử

1. **Bảo vệ người tiêu dùng:** Hệ thống giúp lọc các đánh giá không trung thực, giảm quyết định mua hàng sai lệch.

2. **Bảo vệ người bán hàng trung thực:** Phát hiện chính xác fake negative reviews từ đối thủ cạnh tranh, công bằng hóa thị trường.

3. **Bảo vệ uy tín nền tảng:** Giảm mất lòng tin từ buyer và seller.

#### 7.2.2. Ước tính tác động kinh tế

Giả sử nền tảng có 1,000,000 reviews/tháng, tỷ lệ fake ~40%:

**Tại θ=0.79 (Precision Fake = 0.960441):**
- TP = 225,760 fake reviews bị phát hiện
- FP = 9,309 real reviews bị gắn nhãn sai (1.504% false alarm)
- FN = 174,240 fake reviews lọt lưới

**Tiết kiệm chi phí:**
- Mỗi fake review gây thiệt hại trung bình $5 (ước tính từ quyết định mua hàng sai lệch)
- TP × $5 = $1,128,800 tiết kiệm/tháng
- FP × $10 (chi phí appeal/manual review) = $93,090 chi phí/tháng
- **Net benefit:** $1,035,710/tháng

**Tại θ=0.5 (Precision Fake = 0.915566):**
- TP = 295,880 fake reviews bị phát hiện
- FP = 27,274 real reviews bị gắn nhãn sai (4.55% false alarm)
- **Net benefit:** $1,207,260/tháng (cao hơn θ=0.79 do recall cao hơn)

#### 7.2.3. Khuyến nghị triển khai

1. **Two-stage system:**
   - θ=0.79 cho auto-removal (high confidence, low false alarm)
   - θ=0.5 cho flagging (cần human review, high recall)

2. **Resource-aware deployment:** Pipeline chạy được trên 12GB RAM → phù hợp edge deployment hoặc microservice nhẹ.

3. **Behavioral feature cần infrastructure:** Velocity, burst, time gap cần hệ thống tracking reviewer history real-time.

---

## 8. HẠN CHẾ VÀ HƯỚNG NGHIÊN CỨU TIẾP THEO

> **Lưu ý:** Mục 8.1 bên dưới đã cập nhật theo pipeline **final** (2026-06-10, `phase7_final_metrics.csv`). Các metric gap legacy PCA (0.8558) chỉ còn trong §6.1–§6.4 (lịch sử 01/06).

### 8.1. Hạn chế hiện tại (final track)

1. **Overfit train:** LGBM raw train Macro F1 = 1.0; blend train ≈0.976 — test vẫn ổn (0.9433–0.9463); val–test gap ≈0.0005.
2. **Phase 6 lệch pha:** FGSM/PGD + SHAP/LIME trên legacy PCA ensemble @τ=0.79 — chưa `weighted_blend` final.
3. **Multi-seed:** Đã chạy 3 seed; seed 123 precision-first đơn lẻ 0.9728 hơi dưới 0.975; chưa có kiểm định thống kê formal (bootstrap/paired test).
4. **Phạm vi đánh giá:** Chỉ Amazon 42.7k; 5-fold CV surrogate trên PCA track (mean F1 0.8659), không thay audit test final raw track.
5. **Ablation behavioral:** Advanced features đóng góp rất nhỏ (+0.0008 Macro F1 trong controlled LGBM) — báo cáo trung thực, không phóng đại.

### 8.2. Hướng nghiên cứu tiếp theo

#### 8.2.1. Cải thiện DL Signal (Ưu tiên cao)

1. **Fine-tune ModernBERT end-to-end với LoRA:**
   - LoRA rank: 8, 16, 32
   - Target modules: query, key, value, output
   - Learning rate: 1e-4, 5e-4
   - Epochs: 3, 5

2. **Tăng max_length:**
   - Thử nghiệm: 256, 512, 1024 tokens
   - So sánh performance vs memory trade-off

3. **Attention-based pooling:**
   - Thay masked_mean bằng learnable attention pooling
   - So sánh với [CLS] token pooling

#### 8.2.2. Cải thiện Ensemble Diversity

1. **Thêm base models:**
   - CatBoost, ExtraTreesClassifier, SVM, MLP

2. **Feature engineering cho tree models:**
   - Interaction features, polynomial features, target encoding

3. **Learned stacking:**
   - Neural network meta-learner thay vì weighted blend

#### 8.2.3. Adversarial Training

1. **FGSM/PGD augmentation:**
   - Tạo adversarial examples từ FGSM/PGD
   - Thêm vào training set để cải thiện robustness

#### 8.2.4. Evaluation mở rộng

1. **Multi-seed analysis:**
   - Seeds: [42, 123, 456, 789, 2024]
   - Báo cáo mean ± std

2. **Cross-dataset generalization:**
   - Yelp Fake Reviews Dataset
   - TripAdvisor Dataset

3. **5-fold CV:**
   - Nếu timeline cho phép, thực hiện full k-fold cross-validation

---

## 9. KẾT LUẬN

Nghiên cứu xây dựng pipeline **hai nhánh** cho phát hiện đánh giá giả trên Amazon Labeled Fake Reviews (42.749 mẫu, split 70/15/15, seed 42): **final track** — ModernBERT raw 777-d + 9 behavioral features + CNN-BiLSTM sequence + weighted ensemble (CNN 50% / XGB 35% / LGBM 15%); **ablation track** — PCA 777→400 + PSO-tuned DL — chứng minh PCA track thua raw và phục vụ diagnostic.

Trên test (6.413 mẫu), ba mục tiêu định lượng **đều đạt** ở chế độ tương ứng: Macro F1 **0.9463** @τ=0.30 (balanced; mean 0.9485±0.0018), Precision Fake **0.9816** @τ=0.60 (precision-first, pass cả 3 target), ROC-AUC **0.9769**. Kết quả vượt các baseline text/tabular Tier A đã kiểm chứng (`docs/00_Literature_Review_SOTA.md`). Ablation Phase 7 xác nhận raw 777-d (+0.0397 Macro F1 so với PCA surrogate) và ensemble (+0.164 so với DL-PSO đơn lẻ). Pipeline 8 notebooks hoàn tất; Phase 6–8 đã chạy (2026-06-10).

Hạn chế cần disclose: overfit train (LGBM raw F1 train = 1.0; blend train ≈0.976), Phase 6 robustness/XAI trên legacy PCA ensemble (chưa rerun trên `weighted_blend`), seed 123 precision-first đơn lẻ (0.9728) hơi dưới 0.975. Hướng tiếp: fine-tune ModernBERT (LoRA), kiểm định thống kê formal trên multi-seed đã có, XAI trên raw features, graph/multimodal.

---

## 10. TÀI LIỆU THAM KHẢO

Danh sách 20 papers + bảng SOTA đầy đủ: `docs/00_Literature_Review_SOTA.md`. CSV: `reports/tables/literature_references_20.csv`.

---

**Phụ lục:**
- Phụ lục A: Chi tiết 12 PSO hyperparameters
- Phụ lục B: 9 behavioral features formulas
- Phụ lục C: Full confusion matrices
- Phụ lục D: Code repository structure
- Phụ lục E: Artifact inventory
