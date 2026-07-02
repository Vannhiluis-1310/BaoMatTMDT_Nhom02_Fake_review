# CHƯƠNG 4: TRIỂN KHAI THỰC NGHIỆM VÀ KẾT QUẢ

Chương này bổ sung Chương 3: **triển khai từng phase** (§4.1–4.6) và **kết quả định lượng** (§4.7–4.15). Logic luồng → Ch.3 §3.1; protocol metric *tại sao* → §3.2; khung tự chấm D0–D8 → §3.13 (điểm số tại §4.14).

*Quy ước trình bày:* Mỗi bảng số liệu kèm khối **Diễn giải** (đọc số, so sánh, liên kết EDA/RQ) và **Kết luận** (thông điệp thiết kế hoặc triển khai rút ra từ bảng) — tránh liệt kê thụ động.

Corpus: Amazon Labeled Fake Reviews, **42.749** mẫu sau làm sạch, test **n = 6.413**. Số headline seed 42: `phase7_final_metrics.csv`, `phase7_target_audit.csv` (generated 2026-06-11). Multi-seed → §4.11; XAI/robustness → §4.9 (`phase6_metadata.json`). Trừ khi ghi chú legacy appendix. So sánh literature: **nêu rõ metric, dataset, split** (Gupta et al., 2024; Ren & Ji, 2024).

---

## 4.1. Chuẩn bị dữ liệu (Phase 1)

**Nguồn:** Amazon Labeled Fake Reviews — 50.000 mẫu gốc, nhãn Fake/Real, metadata: rating, reviewer, product, timestamp, verified purchase.

**Triển khai** (`01_EDA_Preprocessing.ipynb`): loại mẫu thiếu trường bắt buộc (57 dòng), deduplicate text+label (7.194 dòng) → **42.749** mẫu (`phase1_cleaning_report.csv`). Không imputation text — mẫu thiếu ngữ nghĩa bị loại (Ren & Ji, 2024).

**Phân chia:** Stratified 70/15/15, seed = 42 — train **29.923**, validation **6.413**, test **6.413** (`train/val/test.csv`, `split_metadata.json`). Tỷ lệ lớp ~41% Fake / 59% Real; stratified giữ phân phối trên mọi tập.

**Fit policy:** Mọi thống kê sau Phase 1 chỉ đọc split đã cố định; scaler/PCA/aggregate behavioral **fit train-only** (Ch.3 §3.2.3; triển khai §4.3).

Sau tiền xử lý, tập gồm **42.749** đánh giá tiếng Anh. Thiết lập đảm bảo đủ dữ liệu huấn luyện ModernBERT và GBDT; validation dành chọn blend + τ; test audit một lần.

So với các corpus lịch sử: Ott et al. (2011) chỉ 400 gold deceptive; Mukherjee et al. (2013) và Rayana & Akoglu (2015) dùng Yelp graph; Gupta (2021) 1,4 triệu Yelp nhưng metric weighted-F1 0,69. Amazon Labeled 42k của đề tài nằm ở quy mô trung bình–lớn cho text classification, tương đương vùng Hajek et al. (2020) và Vidanagama et al. (2020) từng sử dụng, nhưng với protocol leakage control rõ ràng hơn.

Môi trường: Google Colab, Tesla T4, Python 3.12.13, PyTorch 2.11.0+cu128, seed = 42 cố định.

### 4.1.1. Phân tích khám phá dữ liệu (EDA — Phase 1)

Phase 1 thực hiện EDA có cấu trúc theo checklist EDA-01..08 (`phase1_advanced_eda_summary.csv`, generated 2026-05-31). Rubric tự chấm coi đây là chiều **D0** (xem `Self_Assessment_Rubric.md` §D0).

### Bảng 4.0a. Tóm tắt làm sạch và cân bằng lớp

| Chỉ số | Trước clean | Sau clean |
|--------|-------------|-----------|
| Số dòng | 50.000 | **42.749** |
| Fake (label=1) | 24.719 (49,4%) | **17.494 (40,9%)** |
| Real (label=0) | 25.281 (50,6%) | **25.255 (59,1%)** |
| Imbalance ratio | 1,023 | ~1,44 (real/fake count) |
| Dòng bỏ | — | 7.251 (dup text+label 7.194; missing 57) |

*Lưu ý:* Tỷ lệ Fake giảm sau dedup vì nhóm trùng lặp lệch Fake; split stratified 70/15/15 giữ tỷ lệ ~41/59 trên train/val/test.

**Diễn giải:** Mất **14,5%** dòng (7.251/50.000) chủ yếu do trùng lặp text+label — tín hiệu cho chiến dịch spam tái sử dụng cùng mẫu văn bản. Sau dedup, imbalance ratio tăng từ 1,02 (gần cân bằng) lên ~1,44 (real/fake count), nhưng vẫn ở mức **mất cân bằng nhẹ** so với corpus thực tế Amazon (thường real chiếm đa số). Stratified split giữ ~41% Fake trên mọi tập, tránh val/test quá ít mẫu fake khiến Macro F1 không ổn định.

### Bảng 4.0b. Phân bố độ dài văn bản theo nhãn (EDA-01)

| Nhãn | n | Char mean | Char median | Word mean | Word median |
|------|---|-----------|-------------|-----------|-------------|
| Fake | 17.494 | 190,7 | **43** | 35,8 | **7** |
| Real | 25.255 | 176,1 | **125** | 33,5 | **24** |

Fake có phân bố **lệch phải** (median char thấp hơn nhiều, mean cao hơn nhẹ) — gợi ý một phần fake là đánh giá cực ngắn hoặc spam dài, khác Ott et al. (2011) (deceptive thường dài hơn hotel). Kết quả **ủng hộ** đặc trưng `char_len`, `word_count` trong vector behavioral 9-d.

### Bảng 4.0c. Rating theo nhãn (EDA-07)

| Nhãn | Mean rating | Median | % rating 5 sao |
|------|-------------|--------|----------------|
| Fake | **4,06** | 5 | 63,8% (11.161/17.494) |
| Real | **3,84** | 5 | 56,0% (14.143/25.255) |

**Diễn giải:** Fake có mean rating **cao hơn** real (4,06 vs 3,84) dù median cùng là 5 sao — gợi ý fake thiên về đánh giá tích cực cực đoan (astroturfing). Tỷ lệ 5 sao ở fake (63,8%) vượt real (56,0%) ~8 điểm %. Kết quả justify `basic_rating_deviation` và các cờ rating cực đoan trong vector 9-d; đồng thời cảnh báo **không thể** phân loại chỉ bằng “rating = 5” vì hơn một nửa real cũng 5 sao.

### Bảng 4.0d. Sentiment VADER (EDA-03)

| Nhãn | Compound mean | Positive mean | Neutral mean | Negative mean |
|------|---------------|---------------|--------------|---------------|
| Fake | 0,456 | 0,301 | 0,648 | 0,049 |
| Real | 0,445 | 0,216 | 0,732 | 0,052 |

Hai lớp gần nhau về compound — **không đủ** để phân tách đơn lẻ; cần kết hợp embedding + behavioral (nhất quán Mukherjee et al., 2013: rating/behavior quan trọng hơn lexicon thuần).

### Bảng 4.0e. Tỷ lệ Fake theo mức rating (toàn corpus)

| Rating | n reviews | n Fake | Fake rate |
|--------|-----------|--------|-----------|
| 1 | 6.520 | 2.435 | 37,3% |
| 2 | 2.649 | 881 | 33,3% |
| 3 | 3.447 | 1.118 | 32,4% |
| 4 | 4.829 | 1.899 | 39,3% |
| 5 | 25.304 | 11.161 | **44,1%** |

Rating 5 chiếm 59% volume và có fake rate cao nhất — motivate đặc trưng `rating`, `extreme_rating_flag`.

**Diễn giải:** Fake rate **tăng dần** theo mức rating (32,4% ở 3 sao → 44,1% ở 5 sao), trong khi rating 1–2 vẫn có ~33–37% fake — spam không chỉ là “kéo 5 sao” mà còn có nhánh negative fake. Rating 5 chiếm **59%** volume toàn corpus, nên quy tắc “chỉ xét 5 sao” sẽ bỏ sót phần lớn real; mô hình cần kết hợp độ lệch rating (`rating_deviation`) thay vì ngưỡng cứng.

### Bảng 4.0f. Fake rate theo ngày trong tuần (EDA-05)

| Ngày | n | Fake rate |
|------|---|-----------|
| Thứ Hai | 6.403 | 41,5% |
| Thứ Ba | 6.620 | 41,5% |
| Thứ Tư | 6.481 | 40,0% |
| Thứ Năm | 6.221 | 40,9% |
| Thứ Sáu | 5.980 | 41,1% |
| Thứ Bảy | 5.539 | 40,8% |
| Chủ Nhật | 5.505 | 40,6% |

Biến thiên nhẹ (~1,5 điểm %); timestamp parse rate 100%, khoảng thời gian 2003-09-17 → 2025-06-15.

### Bảng 4.0g. Fake rate theo giờ (top/bottom 3)

| Giờ (UTC) | n | Fake rate | Ghi chú |
|-----------|---|-----------|---------|
| 09h | 524 | **44,5%** | Cao nhất |
| 07h | 661 | 43,0% | |
| 12h | 1.206 | 42,8% | |
| 15h | 2.162 | 38,7% | Thấp |
| 10h | 560 | 38,8% | |
| 20h | 2.521 | 39,0% | |

**Diễn giải:** Biến thiên theo giờ (~6 điểm % giữa 09h và 15h) lớn hơn theo ngày (~1,5 điểm %), cho thấy **thời điểm đăng** có tín hiệu phân biệt nhẹ — phù hợp campaign spam tập trung buổi sáng. Tuy nhiên, fake rate vẫn dao động 38,7–44,5% trên mọi khung giờ; không có “giờ vàng” để rule-based filter. Đặc trưng `adv_time_gap_hours_log` và velocity bổ sung chiều thời gian ở mức user/reviewer, không chỉ giờ đăng toàn corpus.

### Bảng 4.0h. User burst (EDA-06)

| Chỉ số | Giá trị |
|--------|---------|
| Số user | 42.204 |
| User có ≥2 review/ngày (max_daily) | 34 |
| User có burst_fake_count > 0 | 17.122 |
| Mean reviews/user | 1,01 |

Phần lớn user chỉ 1 review — burst hiếm nhưng **có tín hiệu** cho nhóm fake tập trung (Rayana & Akoglu, 2015).

### Bảng 4.0i. Phân bố fake theo sản phẩm (EDA-08)

| Chỉ số | Giá trị |
|--------|---------|
| Số ASIN | 23.344 |
| Fake rate trung bình theo ASIN | 41,5% |
| Fake rate median theo ASIN | 21,5% |
| ASIN có fake rate > 50% | 8.548 (36,6%) |

Category EDA **skipped** (không có cột category trong corpus). Wordcloud **skipped** (mặc định); thay bằng `phase1_top_terms_by_label.csv` — top unigram Fake: *br, hair, product, great, like*; Real: *hair, product, great, like, love* (overlap cao → cần BERT).

**Kết luận EDA (§4.1.1):** Tám khía cạnh EDA-01..08 (trừ category/wordcloud) cho ba thông điệp thiết kế: (1) **Độ dài và verified purchase** là tín hiệu behavioral mạnh nhất — median char fake chỉ 43 vs 125 real; (2) **Rating và sentiment đơn lẻ không đủ** — cần fusion embedding + metadata; (3) **Spam có dấu hiệu tập trung** theo sản phẩm (36,6% ASIN fake rate > 50%) và burst user (17.122 user có burst_fake_count > 0), justify khối advanced behavioral. EDA trực tiếp dẫn đến vector 9-d (§4.3) và lý do dual-view BERT+behavioral (Hình 3.3). Chiều D0 tự chấm đạt nấc **Tốt** (6/8 khía cạnh có bảng/hình; §4.14).

**Hình minh họa:** `reports/figures/phase1_length_by_label_boxplot.png`, `phase1_sentiment_by_label.png`, `phase1_fake_rate_by_rating.png`, `phase1_fake_rate_by_hour.png`, `phase1_top_terms_by_label.png`.

---

## 4.2. Trích xuất ModernBERT (Phase 2 — nhánh ngôn ngữ)

Notebook `02_Feature_Engineering.ipynb` dùng `answerdotai/ModernBERT-base` ở chế độ **freeze** (Warner et al., 2024) — inference không cập nhật trọng số, phù hợp RAM ≤ 12GB (Ch.3 §3.12).

| Tham số | Giá trị | Artifact |
|---------|---------|----------|
| Pooling | masked mean last hidden state | `bert_{train,val,test}.npy` |
| Chiều đầu ra tabular | **768-d** / review | `feature_metadata.json` |
| Token sequence | max_length = **160** | `token_ids_{train,val,test}.npy` |
| Fit policy | Encoder cố định; không học từ val/test | `feature_metadata.json` |

Một encoder phục vụ **hai consumer** (Hình 3.3): vector 768-d cho early fusion 777-d; ma trận token cho nhánh sequence §4.4 — tránh hai bản mã hóa không nhất quán.

**Diễn giải:** Freeze encoder là quyết định **do ràng buộc RAM** (Ch.3 §3.12), không phải vì ModernBERT không cải thiện được — trade-off: mất khả năng domain adaptation nhưng giữ reproducibility và chi phí inference thấp. `max_length = 160` cân bằng coverage (EDA median word fake = 7, real = 24) với bộ nhớ batch trên T4.

**Kết luận:** Phase 2 tạo **hub đặc trưng** dùng chung cho mọi nhánh downstream; mọi số liệu Phase 5+ đều phụ thuộc artifact `bert_*.npy` và `token_ids_*.npy` từ bước này.

---

## 4.3. Đặc trưng hành vi và fusion 777-d (Phase 2)

Từ metadata mỗi review, Phase 2 trích **9 đặc trưng hành vi** (5 basic + 4 advanced), thiết kế theo Mukherjee et al. (2013) và Duma et al. (2023). StandardScaler **fit train-only**; aggregate rating (cho `basic_rating_deviation`, velocity, burst) chỉ học từ train.

### Bảng 4.2. Chi tiết 9 đặc trưng hành vi

| Nhóm | Feature | Công thức | Ý nghĩa |
|------|---------|-----------|---------|
| Basic | `basic_char_len_log` | log1p(len(text)) | Fake thường cực ngắn hoặc copy dài (EDA Bảng 4.0b) |
| Basic | `basic_word_count_log` | log1p(word_count) | Bổ trợ char_len; SHAP top-1 §4.9 |
| Basic | `basic_rating_deviation` | \|rating − mean_train\| | Rating cực đoan (Bảng 4.0c–e) |
| Basic | `basic_sentiment_compound` | VADER compound | Hai lớp gần nhau — cần kết hợp embedding |
| Basic | `basic_verified_purchase` | Binary 0/1 | SHAP top-2 §4.9 |
| Advanced | `adv_review_velocity_30d` | Count reviews 30 ngày trước | Campaign spam |
| Advanced | `adv_product_burst_7d` | Count reviews sản phẩm 7 ngày | Burst attack |
| Advanced | `adv_reviewer_behavior_score` | Unsupervised anomaly score | Tài khoản bất thường |
| Advanced | `adv_time_gap_hours_log` | log1p(giờ từ review trước) | Tốc độ đánh giá bất thường |

**Early fusion:** $\mathbf{x} = [\mathbf{e}_{768}; \mathbf{f}_9] \in \mathbb{R}^{777}$ → `features_raw_{train,val,test}.npy`, `behavioral_{train,val,test}.csv`. Đầu vào nhánh tabular §4.5 và PCA ablation §4.6.

**Diễn giải:** Khối **basic** (5 feature) bám trực tiếp EDA: độ dài (`char_len`, `word_count`), rating (`rating_deviation`), sentiment, verified. Khối **advanced** (4 feature) bổ sung tín hiệu thời gian và anomaly ở mức reviewer/product — phản ánh G1 (Gupta et al., 2024): text-only transformer thiếu metadata engineered. Cột “Ý nghĩa” trong bảng là **giả thuyết thiết kế**; mức đóng góp thực tế được kiểm chứng ở Bảng 4.4 (advanced chỉ +0,0023) và SHAP §4.9.2 (basic top-4).

**Kết luận:** Fusion 777-d là điểm neo của final track — mọi nhánh tabular, XAI headline và ablation controlled đều xoay quanh vector này. Không giảm chiều trước GBDT (negative result PCA, §4.6) là hệ quả trực tiếp từ thiết kế này.

---

## 4.4. Nhánh sequence: CNN-BiLSTM-Attention (Phase 5)

Notebook `05_04_CNN_BiLSTM_Sequence`: token sequence từ ModernBERT → 1D-CNN → BiLSTM → attention → FC; **late fusion** concat vector 9-d behavioral sau attention (không trộn vào token). Loss: **Focal Loss** (Lin et al., 2017), `max_length = 160` (`phase5_cnn_bilstm_sequence_metadata.json`).

**Kết quả test** @ τ = 0,50 (`phase7_final_metrics.csv`):

| Metric | Giá trị |
|--------|---------|
| Macro F1 | **0,9324** |
| Precision Fake | 0,9405 |
| Recall Fake | 0,8982 |
| ROC-AUC | 0,9637 |

**Diễn giải:** Macro F1 **0,9324** đặt nhánh sequence ở vị trí thứ hai trong leaderboard (Bảng 4.1), sau blend nhưng trước mọi mô hình tabular đơn lẻ. Recall fake 0,8982 cao hơn XGB/LGBM (~0,81) — CNN-BiLSTM bắt được mẫu fake “khó” về ngữ nghĩa mà tree split kém. Precision 0,9405 thấp hơn XGB (0,9686), phản ánh trade-off sequence vs tree trong blend.

**Kết luận:** Nhánh sequence **không** là headline đơn lẻ nhưng là **thành phần bắt buộc** của ensemble: ablation Model D (−0,0090 vs full, Bảng 4.4) và trọng số CNN 50% seed 42 (Bảng 4.1c) chứng minh đóng góp diversity. Late fusion F9 sau attention giữ inductive bias token (Ch.3 §3.1) mà vẫn khai thác behavioral đã chứng minh qua EDA.

---

## 4.5. Nhánh tabular: GBDT trên raw 777-d (Phase 5)

Notebooks `05_01` LightGBM, `05_02` XGBoost, `05_03` MLP — cùng input `features_raw_*.npy` (777-d). Mỗi model xuất $p_k$ trên val/test (`phase5_*_raw_*_prob.npy`).

**Kết quả test** @ τ = 0,50 (Bảng 4.1):

| Model | Macro F1 | Prec. Fake | Rec. Fake | ROC-AUC |
|-------|----------|------------|-----------|---------|
| XGBoost raw | 0,9059 | **0,9686** | 0,8106 | 0,9531 |
| LightGBM raw | 0,9051 | 0,9677 | 0,8095 | 0,9548 |

**Diễn giải:** Hai họ GBDT cho kết quả **gần như đồng đẳng** trên test (Macro F1 0,9059 vs 0,9051; ROC-AUC 0,9531 vs 0,9548) — khác biệt nhỏ hơn σ multi-seed (0,0018). Cả hai đều **precision-first by nature**: precision fake ~0,97 nhưng recall ~0,81 → bỏ sót ~19% fake khi dùng đơn lẻ. Stacking calibrated (0,9105, Bảng 4.1) không vượt XGB đơn, củng cố lựa chọn weighted blend thay meta-learner.

**Kết luận:** GBDT raw 777-d là **xương sống tabular** của pipeline: XGB được chọn làm 50% blend và là surrogate XAI headline (§4.9.2). LGBM overfit train (F1 = 1,0) khiến grid val loại trọng số dương seed 42 — minh bạch disclose, không ẩn overfit (D8, §4.13.2).

---

## 4.6. Ablation track: PCA và PSO (Phase 3–4)

**PCA** (`03_PCA_Feature_Selection.ipynb`): 777→**400** chiều, giữ **95,10%** phương sai, fit train-only (`features_pca_*.npy`). **PSO** (`04_PSO_Model_Training.ipynb`): 10 particles × 8 iterations, subset train 20%, tối ưu 12 hyperparameter DL trên PCA 400-d.

| Model (PCA track) | Test Macro F1 @ τ=0,50 | Ghi chú |
|-------------------|------------------------|---------|
| DL baseline | **0,7665** | Precision Fake 0,7818 |
| DL-PSO | **0,7793** | +0,0128 vs baseline |
| Legacy PCA+PSO blend | 0,8558 | `phase5_final_metrics.csv` |

**Diễn giải:** PCA giữ 95,10% phương sai nhưng **mất tín hiệu phân loại** quan trọng — Δ +0,0397 Macro F1 khi bỏ PCA (Model B). PSO trên DL đơn chỉ cải thiện +0,0128 (0,7665→0,7793), nhỏ hơn một thứ tự so với redesign pipeline (+0,0907 vs legacy blend). Legacy PCA+PSO blend (0,8558) vẫn thua raw final track ~0,09 điểm, chứng minh bottleneck nằm ở **biểu diễn** chứ không chỉ hyperparameter.

**Kết luận:** Ablation track đóng vai trò **negative result có giá trị** (G5, Bảng 4.12a): PCA trên fused 777-d **không** nên đưa vào SOTA path. Track này được giữ cho FGSM/PGD appendix (§4.9.1) và so sánh lịch sử thiết kế, **tách biệt** inference với final track (Ch.3 §3.1).

---

## 4.7. Weighted ensemble — final track (Phase 5)

Bảng 4.1 tổng hợp hiệu năng test của final track (`phase7_final_metrics.csv`, model_variant = `final_ensemble` = `phase5_weighted_blend`).

### Bảng 4.1. Hiệu năng test theo mô hình và chế độ ngưỡng (n = 6.413)

| Mô hình / chế độ | τ | Macro F1 | Prec. Fake | Rec. Fake | ROC-AUC |
|------------------|---|----------|------------|-----------|---------|
| **weighted_blend (default)** | 0,50 | **0,9433** | 0,9699 | 0,8956 | 0,9769 |
| **weighted_blend (balanced)** | 0,30 | **0,9463** | 0,9344 | 0,9390 | 0,9769 |
| **weighted_blend (precision-first)** | 0,60 | 0,9126 | **0,9816** | 0,8152 | 0,9769 |
| CNN-BiLSTM sequence | 0,50 | 0,9324 | 0,9405 | 0,8982 | 0,9637 |
| XGBoost raw 777 | 0,50 | 0,9059 | 0,9686 | 0,8106 | 0,9531 |
| LightGBM raw 777 | 0,50 | 0,9051 | 0,9677 | 0,8095 | 0,9548 |
| Stacking calibrated | 0,50 | 0,9105 | 0,9728 | 0,8175 | 0,9731 |
| Legacy PCA+PSO blend | 0,50 | 0,8558 | — | — | — |

**Nhận xét so sánh:**

1. **Weighted blend (0,9463)** vượt Veluru et al. (2025) F1 0,934 trên multimodal 20k — nhưng Veluru dùng ảnh+text (Tier khác, không claim trực tiếp).

2. **Vượt Gupta (2021)** weighted-F1 0,69 trên Yelp 1,4M — dataset và metric khác hoàn toàn.

3. **Vượt Mir et al. (2023)** accuracy 87,81% — metric Accuracy thường optimistic hơn Macro F1 trên dữ liệu mất cân bằng nhẹ.

4. **Đứng trên Vidanagama et al. (2020)** khi họ báo 97,3% Accuracy Amazon — Accuracy không tương đương Macro F1 0,9463; cần thận trọng khi trích dẫn.

5. **Vượt legacy nội bộ** 0,8558 (+0,0907 Macro F1) — chứng minh redesign pipeline 09/06 có giá trị.

ROC-AUC **0,9769** bất biến theo τ (thuộc tính của metric threshold-independent), vượt target 0,93 và vượt vùng Hajek et al. (2020) trên protocol tương đương.

**Kết luận (Bảng 4.1):** Headline **balanced @ τ = 0,30** (Macro F1 **0,9463**) là số báo cáo chính; **precision-first @ τ = 0,60** (Prec. Fake **0,9816**) là chế độ triển khai auto-flag. Không có single τ đạt đồng thời cả hai — dual-threshold (§4.8) là giải pháp có chủ đích, không phải hậu xử lý. Weighted blend vượt mọi base model và legacy redesign (+0,0907 vs PCA+PSO), đáp ứng M1–M3 (Ch.1).

### Diễn giải sau Bảng 4.1

Bảng 4.1 cho thấy **thứ bậc hiệu năng rõ ràng** trên cùng test n = 6.413 (`phase7_final_metrics.csv`). Ở τ = 0,50, `weighted_blend` (0,9433) đứng đầu, tiếp theo CNN-BiLSTM sequence (0,9324), stacking calibrated (0,9105), XGBoost/LightGBM raw (0,9059 / 0,9051), và legacy PCA+PSO (0,8558, `phase5_final_metrics.csv`). Khoảng cách blend–sequence (+0,0109) nhỏ hơn blend–legacy (+0,0875), củng cố rằng redesign Phase 5 mang lại cải thiện lớn hơn fine-tuning ensemble trong cùng kiến trúc.

Điểm đáng chú ý: XGB và LGBM đơn lẻ có **precision fake cao** (0,9686 / 0,9677) nhưng recall thấp (~0,81) — phù hợp vai trò “bảo thủ” trong blend precision-first. Ngược lại, balanced @ τ = 0,30 hạ ngưỡng để recall fake đạt 0,9390 (FN = 160), đổi lại precision fake giảm còn 0,9344 — minh họa trade-off cổ điển trên dữ liệu mất cân bằng nhẹ (Ott et al., 2011). ROC-AUC 0,9769 đồng nhất mọi dòng vì đo khả năng ranking, không phụ thuộc τ.

Trọng số blend **không** được đặt trước theo trực giác “chia đều ba nhánh” (ví dụ CNN / LGBM / XGB). Chi tiết grid, so sánh các tỷ lệ ứng viên và trọng số chốt theo từng seed huấn luyện tại **§4.7.1** và Bảng 4.1b–c.

### 4.7.1. Grid chọn trọng số weighted blend (chỉ trên validation)

Sau khi mỗi base model xuất xác suất fake $p_k$ trên train/val/test (`phase5_*_raw_*_prob.npy`, `phase5_cnn_bilstm_sequence_*_prob.npy`), notebook `05_05` / `05_Hybrid_Ensemble` thực hiện **grid search** trên **validation** (n = 6.413): duyệt tổ hợp trọng số bước **0,05** sao cho $\sum_k w_k = 1$, với $k \in \{\text{CNN},\ \text{LGBM},\ \text{XGB},\ \text{MLP}\}$ theo từng subset model được khai báo trong sweep. Tiêu chí chọn: **tối đa hóa Macro F1 validation** @ τ = 0,50 (tie-break: ROC-AUC, precision fake, recall fake). Cấu hình thắng được ghi vào `phase5_weighted_blend_metadata.json` (seed 42) hoặc `artifacts/ensemble/seed_{123,456}/phase5_weighted_blend_metadata.json`, rồi áp dụng cố định lên test — **không** tinh chỉnh lại trọng số trên test.

**Lưu ý phương pháp:** Artifact chính thức (`phase5_weighted_blend_metadata.json`) là nguồn trọng số headline seed 42. Tỷ lệ “CNN 50% / XGB 35% / LGBM 15%” **chỉ là ứng viên trong grid** (val Macro F1 0,9375), không phải cấu hình chốt — Bảng 4.1b chứng minh nó **thua** CNN 50% + XGB 50% (0,9450). Gói Phase 8 (2026-06-11, §4.15) đồng bộ manifest với metadata Phase 5.

#### Bảng 4.1b. Tỷ lệ ứng viên tiêu biểu trên validation (seed 42, blend CNN + LGBM + XGB, τ = 0,50)

| CNN | LGBM | XGB | Val Macro F1 | So với thắng (Δ) | Nguồn |
|-----|------|-----|--------------|------------------|-------|
| **0,50** | **0,00** | **0,50** | **0,9450** | **0** (thắng) | `phase5_weighted_blending_sweep.csv` |
| 0,60 | 0,35 | 0,05 | 0,9426 | −0,0024 | cùng file |
| 0,50 | 0,15 | 0,35 | 0,9415 | −0,0035 | cùng file |
| 0,50 | 0,35 | 0,15 | 0,9375 | −0,0075 | cùng file |
| 0,50 | 0,50 | 0,00 | 0,9327 | −0,0123 | cùng file (chỉ CNN+LGBM) |

*Nguồn số:* `phase5_weighted_blending_sweep.csv`, generated 2026-06-10; các dòng `phase5_weighted_blend_phase5_cnn_bilstm_sequence+phase5_lgbm_raw+phase5_xgb_raw_*`.

**Đọc bảng:** Tỷ lệ “ba model đều có mặt” như **50 / 35 / 15** đạt val Macro F1 **0,9375** — thấp hơn **0,0075** so với **50 / 0 / 50**. Grid vì vậy gán trọng số LGBM = **0** ở điểm tối ưu: thêm LGBM vào convex combination **không** cải thiện Macro F1 validation dù LGBM đơn lẻ vẫn khá (test 0,9051, Bảng 4.1). Điều này nhất quán với overfit train (LGBM raw train Macro F1 = 1,0, `phase5_lgbm_raw_metrics.csv`) và tương quan cao giữa LGBM và XGB trên cùng vector 777-d — diversity hữu ích cho ensemble (Breiman, 1996) nhưng **trọng số dương trên val** chỉ được giữ khi đo được lợi ích trên hold-out (Zhang et al., 2020).

#### Bảng 4.1c. Trọng số blend chốt theo seed huấn luyện (grid max Val Macro F1 @ τ = 0,50)

| Seed | CNN | LGBM | XGB | Val Macro F1 (blend) | Đường dẫn metadata |
|------|-----|------|-----|----------------------|-------------------|
| **42** (headline) | **0,50** | **0,00** | **0,50** | 0,9450 | `artifacts/ensemble/phase5_weighted_blend_metadata.json` |
| 123 | 0,50 | 0,00 | 0,50 | 0,9471 | `artifacts/ensemble/seed_123/phase5_weighted_blend_metadata.json` |
| 456 | 0,60 | 0,35 | 0,05 | 0,9403 | `artifacts/ensemble/seed_456/phase5_weighted_blend_metadata.json` |

*Nguồn số val:* `phase5_weighted_blending_sweep.csv` (seed 42), `seed_123/phase5_weighted_blending_sweep.csv`, `seed_456/phase5_weighted_blending_sweep.csv`.

#### Giải thích chi tiết

**1. Tại sao headline seed 42 chỉ còn hai thành phần (CNN + XGB)?**  
Blend là $p_{\text{blend}} = w_{\text{CNN}} p_{\text{CNN}} + w_{\text{LGBM}} p_{\text{LGBM}} + w_{\text{XGB}} p_{\text{XGB}}$ (các trọng số khác 0 được chuẩn hóa). Trên validation seed 42, mọi điểm grid có $w_{\text{LGBM}} > 0$ trong tam giác CNN–LGBM–XGB đều có Macro F1 **≤ 0,9442**, trong khi $w_{\text{LGBM}} = 0$, $w_{\text{CNN}} = w_{\text{XGB}} = 0{,}5$ đạt **0,9450**. XGB đóng góp precision fake cao (0,9686 test); CNN đóng góp recall và ranking tốt hơn (sequence test 0,9343, `phase5_cnn_bilstm_sequence_metrics.csv`). Kết hợp 50/50 tối ưu hóa Macro F1 val mà không kéo thêm nhiễu từ LGBM đã overfit train.

**2. Tại sao seed 456 lại có LGBM (35%)?**  
Multi-seed (§4.11) **không** dùng chung một bộ trọng số: với **cùng split dữ liệu Phase 1** (seed 42), mỗi seed huấn luyện 42 / 123 / 456 **train lại** CNN, XGB, LGBM → phân phối $p_k$ trên validation **thay đổi** → bề mặt Macro F1 theo $(w_k)$ thay đổi. Ở seed 456, blend CNN 50% + XGB 50% chỉ đạt val Macro F1 **0,9397** (`seed_456/phase5_weighted_blending_sweep.csv`), thấp hơn **0,9403** của cấu hình CNN 60% + LGBM 35% + XGB 5%. Trên seed đó, LGBM mang thêm diversity có lợi trên val; grid **tự** đưa LGBM vào — không phải quy tắc “seed lớn thì phải có LGBM”.

**3. Seed 123 giống seed 42 (50/50), khác seed 456 — có mâu thuẫn không?**  
Không. Seed 123 và 42 độc lập chọn cùng **dạng** tối ưu (CNN+XGB) vì trên validation của từng seed, điểm 50/50 thắng grid. Metric test vẫn dao động nhẹ (Bảng 4.5a) do khởi tạo khác; multi-seed báo **mean ± std** (Bảng 4.5b), không ép một tỷ lệ duy nhất cho mọi seed.

**4. Quan hệ với multi-seed và headline:**  
Headline luận văn (Bảng 4.1, τ = 0,30 / 0,60) lấy **seed 42** làm canonical (`phase5_weighted_blend_*_prob.npy`, trọng số Bảng 4.1c). Kiểm định §4.11 chứng minh **metric** ổn định khi train seed đổi; **trọng số** có thể đổi (456 có LGBM) mà vẫn hợp lệ — đúng protocol “chọn cấu hình trên val từng lần chạy”, không phải lỗi pipeline.

**5. SSOT trọng số và metric blend:**  
Trọng số: `phase5_weighted_blend_metadata.json` (+ bản `seed_*`). Metric test blend @ τ = 0,50: `phase5_weighted_blend_metrics.csv`. Metric @ τ balanced/precision-first: ngưỡng từ `phase5_selected_candidates.csv`, audit test qua `phase5_weighted_blend_test_prob.npy` (đồng bộ số Bảng 4.1).

---

## 4.8. Target audit và dual-threshold (Phase 5–7)

Ngưỡng chọn **chỉ trên validation**; test audit một lần. Validation Macro F1 tại τ = 0,30: 0,9468; tại τ = 0,60 precision-first: val Precision Fake 0,9784 — đạt ngưỡng 0,975 trước khi audit test.

### Bảng 4.3. Target audit (`phase7_target_audit.csv`)

| Chế độ | τ | Metric | Actual | Target | Pass |
|--------|---|--------|--------|--------|------|
| default | 0,50 | macro_f1 | 0,9433 | 0,89 | ✓ |
| default | 0,50 | precision_fake | 0,9699 | 0,975 | ✗ |
| default | 0,50 | roc_auc | 0,9769 | 0,93 | ✓ |
| balanced | 0,30 | macro_f1 | **0,9463** | 0,89 | ✓ |
| balanced | 0,30 | roc_auc | 0,9769 | 0,93 | ✓ |
| precision-first | 0,60 | precision_fake | **0,9816** | 0,975 | ✓ |
| precision-first | 0,60 | macro_f1 | 0,9126 | 0,89 | ✓ |
| precision-first | 0,60 | roc_auc | 0,9769 | 0,93 | ✓ |

**Kết luận audit:** Chế độ **precision-first @ τ = 0,60** pass cả ba target — đáp ứng Gap G7 (Gupta et al., 2024): hiếm paper báo đồng thời macro F1 cao và precision-first cho e-commerce trên cùng pipeline.

Khuyến nghị triển khai: τ = 0,60 cho auto-flag; τ = 0,30 cho moderation cân bằng — thay thế τ = 0,79 của legacy PCA track (Macro F1 chỉ 0,7860 trên test legacy @ τ = 0,79, `phase5_final_metrics.csv`).

### Diễn giải sau Bảng 4.3 (target audit)

Target audit (`phase7_target_audit.csv`) trả lời câu hỏi *mục tiêu ban đầu có đạt không* một cách **trung thực từng chế độ**, thay vì chọn τ sau khi nhìn test (tránh p-hacking, Ren & Ji, 2024). Ba điểm then chốt:

1. **Không tồn tại single τ đạt đồng thời** precision fake ≥ 0,975 và macro F1 tối đa: @ τ = 0,30 macro F1 đạt 0,9463 nhưng precision fake 0,9344 (`gap = −0,0406`, `pass = False`); @ τ = 0,50 precision 0,9699 vẫn dưới 0,975. Đây không phải thất bại pipeline mà là **đặc tính bài toán** — Luca & Zervas (2016) nhấn mạnh chi phí false accusation đòi hỏi chế độ riêng.

2. **Precision-first @ τ = 0,60 pass cả ba target** — macro F1 0,9126 (> 0,89), precision fake 0,9816 (> 0,975), ROC-AUC 0,9769 (> 0,93). Val đã đạt precision 0,9752 trước audit test (`phase5_metadata.json`, `precision_first_winner`), gap val–test precision ≈ 0,0064 — generalization tốt.

3. **ROC-AUC vượt target ở mọi τ** (+0,047 so với 0,93) — khả năng ranking xác suất ổn định; sàn có thể điều chỉnh τ theo mùa spam mà không retrain, miễn re-sweep trên validation hold-out.

Chính sách dual-threshold lấp Gap G7 (Gupta et al., 2024): hiếm paper trong 20 công trình Bảng 2.2 báo cáo đồng thời macro F1 balanced và precision-first ≥ 0,975 trên cùng pipeline có kiểm soát leakage.

---

## 4.9. Robustness và XAI — Phase 6 (dual-track, 2026-06-11)

Notebook Phase 6 (`phase6_metadata.json`, generated 2026-06-11) tách **hai luồng** có chủ đích: (i) **headline track** `final_raw_777` — SHAP/LIME trên vector fused 777-d, case chọn theo xác suất `phase5_weighted_blend`; (ii) **legacy appendix** `pca_400` — FGSM/PGD + SHAP/LIME PCA phục vụ ablation track. Thiết kế này phản ánh `known_limitations` trong artifact: adversarial attack chỉ khả thi trong không gian PCA 400-d đã fit train; XAI headline gắn pipeline báo cáo mà không ép SHAP lên toàn bộ blend black-box.

### 4.9.1. Đánh giá tính bền vững adversarial (legacy appendix)

> *Footnote:* FGSM/PGD áp dụng `final_ensemble_model.pkl` — legacy PCA ensemble, τ = 0,79, XGB 50% / LGBM 50%. **Không** phải `phase5_weighted_blend`. Nguồn: `phase6_robustness_metric_drops.csv`, subset n = 1.000, `applies_to: legacy_appendix_only`.

FGSM/PGD tấn công **PCA feature space** (Goodfellow et al., 2015), surrogate `dl_pso`, clamp theo min–max train PCA. Bảng 4.9a: metric drops tại FGSM ε = 0,03.

### Bảng 4.9a. Suy giảm hiệu năng dưới FGSM (ε = 0,03, legacy appendix)

| Model | Clean Macro F1 | FGSM Macro F1 | Δ Macro F1 | Δ Prec. Fake |
|-------|----------------|---------------|------------|--------------|
| dl_pso | 0,7724 | 0,7068 | −0,0656 | −0,0942 |
| final_ensemble (legacy) | 0,8000 | 0,7949 | −0,0052 | −0,0006 |

Legacy ensemble ổn định hơn DL-PSO đơn lẻ — phù hợp quan sát ensemble giảm phương sai (Breiman, 1996). PGD ε = 0,03 cho pattern tương tự: legacy Δ Macro F1 −0,0039; dl_pso −0,0686 (`phase6_robustness_metric_drops.csv`). Tuy nhiên, clean F1 legacy 0,80 thấp hơn nhiều so với `weighted_blend` 0,9463 (`phase7_final_metrics.csv`, τ = 0,30). **Không suy diễn** robustness của pipeline chính từ appendix này; adversarial trên raw 777-d hoặc text space được xếp hướng mở (Ren & Ji, 2024).

**Diễn giải:** Suy giảm Macro F1 của `dl_pso` (−0,0656) gấp ~12 lần `final_ensemble` legacy (−0,0052) — DL đơn trên PCA dễ bị nhiễu loạn hơn ensemble tree. Precision fake của dl_pso tụt mạnh hơn (−0,0942), cho thấy attack làm lệch ngưỡng quyết định “fake” theo hướng báo động giả.

**Kết luận (Bảng 4.9a):** Appendix chứng minh **khả năng chạy** FGSM/PGD trong không gian PCA và gợi ý ensemble ổn định hơn DL đơn — nhưng baseline clean quá thấp (0,80) so với pipeline chính (0,9463). Chiều D6 giữ nấc 3 vì adversarial **chưa** áp dụng lên `weighted_blend` (§4.14).

---

### 4.9.2. Khả năng diễn giải — final track (headline)

XAI headline (`phase6_final_metadata.json`, track `final_raw_777`) giải thích nhánh tabular **XGBoost raw 777-d** (`phase5_xgb_raw`) — chiếm **50%** trọng số blend seed 42. SHAP global (Lundberg & Lee, 2017) trên subset test stratified n = 500 (`phase6_final_shap_global_importance.csv`); LIME (Ribeiro et al., 2016) với background n = 1.000. **Chọn case** theo xác suất `weighted_blend` @ τ = 0,30 (balanced, `phase5_balanced_validation_threshold`); **giải thích cục bộ** bằng XGB raw — protocol disclose trong metadata.

#### Bảng 4.9b. SHAP global top-10 — final track (XGB raw 777-d)

| Rank | Feature | Nhóm | Mean \|SHAP\| |
|------|---------|------|---------------|
| 1 | `basic_word_count_log` | behavioral | 1,6186 |
| 2 | `basic_verified_purchase` | behavioral | 1,1504 |
| 3 | `basic_char_len_log` | behavioral | 0,2875 |
| 4 | `basic_rating_deviation` | behavioral | 0,2024 |
| 5 | `bert_278` | embedding | 0,1285 |
| 6 | `bert_079` | embedding | 0,1007 |
| 7 | `bert_402` | embedding | 0,0756 |
| 8 | `bert_108` | embedding | 0,0746 |
| 9 | `bert_323` | embedding | 0,0691 |
| 10 | `bert_370` | embedding | 0,0677 |

**Đọc kết quả:** Bốn đặc trưng behavioral cơ bản chiếm **top-4** mean \|SHAP\| — nhất quán với EDA §4.1.1 (độ dài review, verified purchase) và ablation controlled (Bảng 4.4: advanced block chỉ +0,0023 so với ref. PCA+9 feat.). Embedding ModernBERT vẫn xuất hiện từ rank 5 (`bert_278`, `bert_079`…), phản ánh dual-view: ngữ nghĩa latent **bổ sung** chứ không thay thế tín hiệu hành vi — cùng lập luận §4.7 về blend CNN+XGB.

#### Bảng 4.9c. SHAP theo khối đặc trưng (`phase6_final_shap_block_importance.csv`)

| Khối | Mean \|SHAP\| tổng hợp | Ghi chú |
|------|------------------------|---------|
| `bert_embedding` (768 chiều) | 7,1873 | Tổng theo khối; trung bình mỗi chiều thấp hơn behavioral đơn lẻ |
| `behavioral` (9 chiều) | 3,3250 | Ít chiều nhưng mật độ tín hiệu cao — khớp thiết kế 777-d |

**Diễn giải:** Tổng \|SHAP\| khối BERT (7,19) cao hơn behavioral (3,33) vì **768 chiều** cộng dồn — nhưng **mật độ** mỗi chiều behavioral (~0,37/chiều) vượt trung bình mỗi chiều BERT (~0,009/chiều). Điều này giải thích vì sao 9 feature có tên vẫn “thắng” top-4 per-feature (Bảng 4.9b) dù embedding chiếm phần lớn không gian 777-d.

**Kết luận:** XAI headline **xác nhận** thiết kế fusion: behavioral mang tín hiệu giải thích được (word count, verified), BERT bổ sung ngữ nghĩa latent — nhất quán RQ1 và ablation Model C (+0,0023 advanced).

Hình `reports/figures/phase6_final_shap_top_behavioral.png` và `phase6_final_shap_summary_behavioral.png` trực quan hóa phân bố SHAP trên 9 behavioral; 768 chiều BERT được báo cáo per-dimension và aggregate block — semantic token-level attribution cần phương pháp text-level riêng (giới hạn đã ghi trong metadata).

#### LIME case studies (6 mẫu, `phase6_final_lime_case_summary.csv`)

| Case | Loại | P(blend) | P(XGB) | Ghi chú LIME |
|------|------|----------|--------|--------------|
| 1 | TP fake | 0,99999 | 0,99998 | `basic_word_count_log` thấp đẩy hướng fake |
| 2 | TN real | 0,0045 | 0,0040 | Nhiều chiều `bert_*` âm, word count vùng trung |
| 3 | FP (real→fake) | 0,9873 | 0,9828 | Review ngắn + embedding — false alarm tiềm năng |
| 4 | FN (fake→real) | 0,0123 | 0,0153 | Embedding “giống real” che tín hiệu fake |
| 5 | Highest conf. fake | 0,99998 | 0,99997 | Cùng pattern TP: độ dài + BERT |
| 6 | Highest conf. real | 0,0056 | 0,0065 | BERT dimensions âm mạnh |

HTML: `artifacts/xai/phase6_final_lime_case_0{1..6}_*.html`. Case 3–4 liên kết trực tiếp §4.13.1 (FPR 1,06% @ τ = 0,60; FN = 160 @ τ = 0,30): LIME cho thấy **review ngắn** và **chiều embedding cực đoan** là nguồn lỗi điển hình — phù hợp Mukherjee et al. (2013) về behavioral spam và Hajek et al. (2020) về đặc trưng ngôn ngữ.

**Diễn giải LIME:** Case 1/5 (TP, P > 0,99) — mẫu “dễ”: độ dài thấp + xác suất đồng thuận blend/XGB. Case 2/6 (TN) — nhiều chiều `bert_*` âm, word count vùng trung, không kích hoạt rule ngắn. Case 3 (FP): review **thật** nhưng ngắn → mô hình gán P(fake) ≈ 0,99 — đúng nguồn FPR 1,06% @ τ = 0,60. Case 4 (FN): fake có embedding “giống real” → FN @ balanced (160 mẫu).

**Kết luận (XAI final track):** SHAP global + LIME cục bộ **cùng hướng** — lỗi tập trung ở biên phân phối (review cực ngắn, embedding outlier), không mâu thuẫn ablation. Moderation có thể dùng LIME để giải thích cờ nhầm cho seller (D7).

So với legacy appendix (`phase6_shap_global_importance.csv`: `pca_000` dominant 1,2295), XAI final track **đọc được** tên feature thật (word count, verified purchase) thay vì thành phần PCA trừu tượng — bước cần thiết để triển khai moderation có giải thích (Gupta et al., 2024).

---

## 4.10. Ablation và cross-validation (Phase 7)

**5-fold CV** trên LightGBM PCA (`phase7_cv_summary.csv`): Macro F1 **0,8659 ± 0,0036**, ROC-AUC **0,9233 ± 0,0041** — σ thấp, điểm tuyệt đối thấp hơn final blend ~0,08. CV surrogate trên PCA **không** thay thế đánh giá raw final track (hạn chế: Ch.5 §5.7).

**Controlled ablation** (`phase7_ablation_results.csv`, `phase7_ablation_delta.csv`, `ablation_evidence_map`, test @ τ = 0,50, generated 2026-06-11):

### Bảng 4.4. Đóng góp từng thành phần (Models A–E)

| Variant | Model | Macro F1 | Δ vs ref. | Diễn giải |
|---------|-------|----------|-----------|-----------|
| Full Model | `phase5_weighted_blend` | **0,9433** | — | Pipeline chính (CNN 50% + XGB 50%) |
| Model A | `phase5_xgb_raw` | 0,9059 | −0,0374 vs full | Chỉ nhánh tabular; bỏ sequence |
| Model B | `controlled_lightgbm_raw_no_pca` | 0,9058 | +0,0397 vs PCA LGBM ref. | **Raw thắng PCA** (controlled) |
| Model C | `controlled_lightgbm_bert_basic5_no_advanced` | 0,8684 | +0,0023 vs PCA+9 ref. | Advanced feat. đóng góp nhỏ |
| Model D | `phase5_cnn_bilstm_sequence` | 0,9343 | −0,0090 vs full | Chỉ nhánh sequence; bỏ tabular blend |
| Model E | `dl_baseline` (nearest) | 0,7665 | −0,1769 vs full | Baseline Phase 4 gần nhất* |

*Model E vẫn dùng PCA 400-d và 9 behavioral; **không** overclaim là full no-PCA/no-advanced retrain (`phase7_metadata.json`, `known_limitations`). PSO trên DL đơn lẻ: 0,7665 → 0,7793 (+0,0128, §4.6) — tách khỏi ablation table. Legacy PCA+PSO blend: Macro F1 0,8558 (`phase5_final_metrics.csv`).

**So sánh với tài liệu:** Deshai và Rao (2023) nhấn mạnh APSO cho CNN; đề tài định lượng PSO +0,0128 trên DL đơn (§4.6) — có ích nhưng **nhỏ hơn nhiều** so với lợi ích ensemble đa view (+0,1769 vs Model E). Zhang et al. (2020) và Duma et al. (2023) dùng hybrid/ensemble nhưng không ablation controlled trên cùng Amazon split 42k — Gap G8 mà đề tài lấp.

### Diễn giải sau Bảng 4.4

Thứ bậc đóng góp theo |Δ| giảm dần: **ensemble đa view** (+0,1769 vs Model E) >> **bỏ PCA** (+0,0397 Model B) >> **bỏ sequence** (−0,0090 Model D) >> **advanced behavioral** (+0,0023 Model C). Model A (−0,0374) cho thấy chỉ tabular không đủ — cần sequence hoặc blend. Full model (0,9433) vượt mọi variant đơn lẻ, xác nhận RQ4 và RQ6.

**Kết luận ablation:** Ba phát hiện có thể báo cáo độc lập: (1) Raw 777-d thắng PCA — **negative result** quan trọng (G5); (2) Ensemble >> PSO >> advanced feat. — ưu tiên đầu tư vào **kiến trúc** hơn tinh chỉnh DL đơn (G6 lấp một phần); (3) Controlled ablation trên **cùng split** lấp G8 — điểm mạnh phương pháp luận so với literature.

---

## 4.11. Độ ổn định multi-seed (42, 123, 456)

Bổ sung kiểm định độ tin cậy theo khuyến nghị Ren & Ji (2024): huấn luyện lại Phase 5 (`05_01`, `05_02`, `05_04a/b/c`, `05_05`) với **cùng split Phase 1 (seed 42)** nhưng **ba seed huấn luyện** 42 / 123 / 456. Artifact seed 42 giữ đường dẫn canonical; seed 123 và 456 nằm trong `artifacts/predictions/seed_{123,456}/`.

Với mỗi seed, ngưỡng τ được **sweep trên validation** (balanced: max Macro F1; precision-first: Prec. Fake ≥ 0,975 rồi max Recall), sau đó audit **một lần trên test** — nguồn: `phase7_multiseed_metrics.csv`, `phase7_multiseed_summary.csv` (generated 2026-06-11). *Ghi chú:* τ headline seed 42 trong `phase7_final_metrics.csv` (0,30 / 0,60) lấy từ `phase5_metadata.json`; multi-seed re-sweep trên val có thể chọn τ khác (42: 0,43 / 0,54) — hai protocol bổ sung nhau, không mâu thuẫn.

### Bảng 4.5a. Hiệu năng test `weighted_blend` theo seed (τ chọn trên val)

| Seed | Chế độ | τ | Macro F1 | Prec. Fake | Rec. Fake | ROC-AUC |
|------|--------|---|----------|------------|-----------|---------|
| 42 | balanced | 0,43 | 0,9492 | 0,9591 | 0,9200 | 0,9769 |
| 42 | precision-first | 0,54 | 0,9295 | 0,9761 | 0,8579 | 0,9769 |
| 123 | balanced | 0,40 | 0,9502 | 0,9499 | 0,9318 | 0,9757 |
| 123 | precision-first | 0,58 | 0,9166 | 0,9728 | 0,8312 | 0,9757 |
| 456 | balanced | 0,38 | 0,9460 | 0,9428 | 0,9291 | 0,9766 |
| 456 | precision-first | 0,63 | 0,9148 | **0,9800** | 0,8213 | 0,9766 |

*Ghi chú:* Seed 123 precision-first đạt Prec. Fake = 0,9728 — hơi dưới target 0,975 (~0,002); mean 3 seed vẫn ≥ 0,975 (Bảng 4.5b).

**Diễn giải (Bảng 4.5a):** Ba seed cho balanced Macro F1 trong biên **0,9460–0,9502** (spread 0,0042) — nhỏ hơn nhiều so với redesign legacy (+0,09). Seed 123 đạt balanced cao nhất (0,9502) nhưng precision-first yếu nhất (0,9728) — trade-off τ khác nhau theo seed là bình thường khi re-sweep val. ROC-AUC gần như không đổi (0,9757–0,9769), chứng tỏ ranking xác suất ổn định hơn metric phụ thuộc τ.

### Bảng 4.5b. Tóm tắt mean ± std (n = 3 seed)

| Chế độ | Metric | Mean | Std |
|--------|--------|------|-----|
| balanced | Macro F1 | **0,9485** | 0,0018 |
| balanced | Prec. Fake | 0,9506 | 0,0067 |
| balanced | Rec. Fake | 0,9270 | 0,0051 |
| balanced | ROC-AUC | 0,9764 | 0,0005 |
| precision-first | Macro F1 | 0,9203 | 0,0066 |
| precision-first | **Prec. Fake** | **0,9763** | 0,0029 |
| precision-first | Rec. Fake | 0,8368 | 0,0154 |
| precision-first | ROC-AUC | 0,9764 | 0,0005 |

**Kết luận multi-seed:** Macro F1 balanced dao động trong biên ±0,002; ROC-AUC ổn định (~0,976 ± 0,0005). Precision-first mean **0,9763 ± 0,0029** — chứng minh headline seed 42 không phải kết quả may mắn đơn lẻ, đáp ứng tiêu chí nấc Xuất sắc chiều D1 (Bảng 3.13).

**Trọng số blend theo seed** (CNN / LGBM / XGB) được grid chọn **riêng từng seed** — xem Bảng 4.1c (§4.7.1). Seed 42 và 123 chốt **50% / 0% / 50%**; seed 456 chốt **60% / 35% / 5%** vì trên validation của seed đó LGBM cải thiện Macro F1 so với CNN+XGB thuần. Multi-seed kiểm định **độ ổn định metric**, không yêu cầu **đồng nhất trọng số** giữa các lần huấn luyện (Ren & Ji, 2024).

---

## 4.12. So sánh SOTA theo phân tầng

### Tier A — Text/tabular (so sánh trực tiếp)

| Công trình | Dataset | Metric | Score | Đề tài (balanced) |
|------------|---------|--------|-------|-------------------|
| **Ours weighted_blend** | Amazon 42,7k | Macro F1 | **0,9463** | Headline (seed 42) |
| Veluru et al. (2025) | Multimodal 20k | F1 | 0,934 | Khác modality |
| Gupta (2021) | Yelp 1,4M | Weighted-F1 | 0,69 | Khác dataset |
| Mir et al. (2023) | General | Accuracy | 0,8781 | Khác metric |
| Bhuvaneshwari et al. (2021) | Amazon | F1/Acc | >0,90 vùng | Protocol khác; sequence đơn 0,9324 |

**Diễn giải (Tier A):** Đề tài so sánh **có điều kiện** — cùng paradigm text/tabular trên Amazon, metric Macro F1. Veluru (2025) dùng multimodal nên F1 0,934 **không** claim vượt trực tiếp; Gupta (2021) và Mir (2023) khác dataset/metric. Điểm mạnh của đề tài: protocol leakage control + ablation + multi-seed, không chỉ một con số đơn lẻ.

### Tier B — Graph (bối cảnh, không so sánh trực tiếp)

Rayana & Akoglu (2015) F1 ~0,85+; Wu et al. (2024) DOS-GNN ~0,915 — paradigm graph collective, không cùng feature space text+behavioral 777-d.

**Diễn giải (Tier B):** Graph SOTA bắt spam **tập thể** (collusion) mà pipeline text-only không mô hình hóa — hướng mở §4.13.3. Số 0,915 DOS-GNN **không** mâu thuẫn 0,9463 của đề tài vì khác bài toán, feature và split.

### Tier C — Foundational

Ott et al. (2011) F1 ~0,90 trên 400 gold; Mukherjee et al. (2013) Acc 67,8% — động lực lịch sử, không benchmark trực tiếp.

**Diễn giải (Tier C):** Công trình foundational đặt **bối cảnh** (deceptive language, behavioral Yelp) chứ không là mốc SOTA để claim “vượt Ott 2011”. Quy mô 42k và fusion 777-d vượt xa setup hotel 400 mẫu.

**Kết luận SOTA:** Trên Tier A, với Macro F1 và Amazon 42k split có kiểm soát, đề tài đạt vùng **state-of-the-art text/tabular** trong tập 20 papers đã verify — mean balanced Macro F1 **0,9485 ± 0,0018** trên 3 seed (§4.11); không bao gồm multimodal/graph.

### 4.12.1. Ma trận khoảng trống nghiên cứu — gap, bằng chứng artifact, kết luận

Nguồn khoảng trống G1–G8: `docs/00_Literature_Review_SOTA.md` §4 (20 papers đã xác minh). Bảng 4.12a liên kết từng gap với artifact tái lập — làm cơ sở nấc Xuất sắc chiều D2 (Bảng 3.13).

### Bảng 4.12a. Ma trận gap – evidence – kết luận (G1–G8)

| Gap | Mô tả khoảng trống | Bằng chứng artifact | Kết luận |
|-----|-------------------|---------------------|----------|
| **G1** | Text-only transformer chiếm đa số; ít kết hợp behavioral engineered | `behavioral_{train,val,test}.csv`; fusion **777-d** §4.3; EDA §4.1.1 | **Đã lấp** — ModernBERT + 9 behavioral; Bảng 4.4 xác nhận advanced feat. |
| **G2** | Thiếu dual-track: tabular embeddings + sequence DL | Phase 5 notebooks; §4.4 (sequence 0,9324); §4.7 (blend 0,9463) | **Đã lấp** — Tree raw 777 + sequence; `weighted_blend` headline |
| **G3** | Ensemble shallow+deep ít, thiếu protocol chọn threshold | `phase5_metadata.json`; sweep τ §4.8 (τ = 0,30 / 0,60) | **Đã lấp** — Hai chế độ vận hành có số test |
| **G4** | Thiếu audit leakage / fit policy công khai | Split §4.1; Ch.3 §3.2.3; `phase5_metadata.json` | **Đã lấp** — Fit train-only, audit test một lần |
| **G5** | PCA trên fused vector chưa optimal | §4.6, §4.10 Bảng 4.4: raw 0,9058 vs PCA 0,8661 | **Đã lấp** — Negative result PCA |
| **G6** | PSO chưa gắn full hybrid stack | §4.6 PSO +0,0128; final grid blend §4.7 | **Lấp một phần** — PSO << ensemble (+0,1769) |
| **G7** | Precision-first e-commerce hiếm | §4.8: Prec. **0,9816** @ τ=0,60; multi-seed §4.11 | **Đã lấp** — Dual-threshold + 3 seed |
| **G8** | Ablation thiếu trên cùng split | §4.10 Bảng 4.4 (Models A–E) | **Đã lấp** — Controlled ablation |

**Tổng hợp:** 7/8 gap **đã lấp**; G6 **lấp một phần**. Ma trận bổ sung §4.12 và chiều D2 (Bảng 3.13).

**Diễn giải (Bảng 4.12a):** Mỗi dòng gap liên kết **mô tả lý thuyết** (Ch.2 §2.2) với **artifact cụ thể** — tránh claim “đã lấp gap” không có bằng chứng. G6 là ngoại lệ duy nhất: PSO có số (+0,0128) nhưng chưa gắn full hybrid stack, nên đánh dấu “lấp một phần” thay vì che giấu.

**Kết luận (ma trận gap):** Khung G1–G8 cho phép độc giả đánh giá **phạm vi đóng góp** mà không nhầm với graph/multimodal SOTA. Điểm D2 đạt nấc Xuất sắc (14,0/14) nhờ ma trận này + bảng Tier A (§4.12).

---

## 4.13. Phân tích lỗi và hướng cải thiện

### 4.13.1. Confusion matrix theo chế độ τ

**Precision-first (τ = 0,60):** TN = 3.749, FP = 40, FN = 485, TP = 2.139.
- FPR = 40/3.789 = **1,06%** (tỷ lệ khóa nhầm user thật)
- FNR = 485/2.624 = **18,48%** (fake bị bỏ sót)

**Balanced (τ = 0,30):** FN = 160, Recall Fake = **93,90%**, Macro F1 = **94,63%**.

Trade-off có chủ đích: precision-first đạt target 0,975 (Prec. Fake = **0,9816**) theo yêu cầu e-commerce (Luca & Zervas, 2016 — chi phí false accusation cao); balanced phục vụ moderation cần recall cao hơn.

**Diễn giải:** Ở τ = 0,60, **40 FP** trên 3.789 real (1,06%) là chi phí chấp nhận được cho auto-flag; **485 FN** (18,48%) là fake bị lọt — phù hợp khi ưu tiên không khóa nhầm seller. Ở τ = 0,30, FN giảm còn **160** (recall 93,90%) đổi lại ~66 FP thêm (suy ra từ precision 0,9344). Hai chế độ map trực tiếp Ch.6 §6.3 (staging vs moderation queue).

**Kết luận (phân tích lỗi):** Lỗi không phân bố đều — LIME §4.9.2 chỉ ra FP tập trung ở review ngắn, FN ở fake “giống real” về embedding. Cải thiện ưu tiên: fine-tune BERT, graph collusion, adversarial trên blend (§4.13.3).

### 4.13.2. Overfit train

Train blend Macro F1 ≈ **0,976** vs test **0,9463** (balanced); val–test gap ≈ 0,0005. Val–test gần nhau cho thấy validation proxy tốt; train–test gap lớn chủ yếu do LightGBM **raw** train Macro F1 = 1,0 (test 0,9051) — cần disclose (Ren & Ji, 2024: overfitting là thách thức phổ biến FRD).

### 4.13.3. Hướng cải thiện

Fine-tune ModernBERT (LoRA); graph features (Rayana & Akoglu, 2015); adversarial robustness trên final blend / text space; SHAP trên CNN sequence branch; đánh giá cross-dataset (Yelp).

---

## 4.14. Tự đánh giá theo khung chất lượng nghiên cứu

Áp dụng khung D0–D8 tại **Bảng 3.13 (Ch.3 §3.13)**. Bảng 4.14 ghi **nấc tự chấm**, điểm đóng góp (Trọng số × Nấc/4), ghi chú và bằng chứng artifact.

### Bảng 4.14. Kết quả tự đánh giá theo khung chất lượng (2026-06-11)

| Mã | Tên chiều | Trọng số | Nấc chấm | Nhãn nấc | Điểm | Ghi chú | Bằng chứng |
|----|-----------|----------|----------|----------|------|---------|------------|
| D0 | Phân tích dữ liệu và EDA | 8% | 3 | Tốt | 6,0 | Sáu trong tám khía cạnh EDA có bảng/hình; §4.1.1 liên kết behavioral → thiết kế 777-d. Chưa đạt nấc 4 vì thiếu benchmark EDA công khai và hai mục (wordcloud, category) bỏ qua có lý do. | `phase1_eda_summary.csv`, `phase1_cleaning_report.csv`, `phase1_advanced_eda_summary.csv`, `reports/figures/phase1_*.png` |
| D1 | Hiệu năng mô hình | 16% | 4 | Xuất sắc | 16,0 | Precision-first @ τ = 0,6 pass 3/3 target (Prec. Fake = **0,9816**; AUC = **0,9769**). Balanced Macro F1 = **0,9463**; val–test gap ≈ **0,0005**. Multi-seed: balanced **0,9485 ± 0,0018**, prec.-first Prec. Fake **0,9763 ± 0,0029** (seed 123 đơn lẻ 0,9728 — disclosed). | `phase7_target_audit.csv`, `phase7_multiseed_summary.csv`, Bảng 4.5a–b |
| D2 | So sánh với nghiên cứu trước | 14% | 4 | Xuất sắc | 14,0 | 20 papers đã xác minh; ≥5 Tier A (§4.12); ma trận G1–G8 (Bảng 4.12a); không overclaim graph/multimodal. | `literature_references_20.csv`, `literature_sota_comparison.csv`, Bảng 4.12a |
| D3 | Phương pháp luận và tái lập | 12% | 4 | Xuất sắc | 12,0 | Seed 42 cố định; split 70/15/15; fit train-only; τ chọn trên val; audit test một lần; metadata Phase 5/7; Phase 8 manifest + inventory (2026-06-11); multi-seed `seed_{123,456}/`. | `phase5_metadata.json`, `phase7_metadata.json`, `phase8_submission_package_manifest.csv`, `phase8_artifact_inventory.csv`, `phase8_run_order_checklist.csv` |
| D4 | Phân tích ablation | 14% | 4 | Xuất sắc | 14,0 | §4.10 + Bảng 4.4 (Models A–E): raw vượt PCA (+0,0397); ensemble vượt baseline (+**0,1769**); advanced (+0,0023). PCA tiêu cực (§5.2, Bảng 4.12a G5/G8). | `phase7_ablation_results.csv`, `phase7_ablation_delta.csv`, Bảng 4.12a |
| D5 | Độ tin cậy và tổng quát hóa | 12% | 4 | Xuất sắc | 12,0 | Multi-seed §4.11 (n=3): σ Macro F1 = 0,0018. Val–test gap ≈ 0,0005; CV σ = 0,0036. Overfit disclosed §4.13.2. | `phase7_multiseed_summary.csv`, `phase7_cv_summary.csv`, §4.11, §4.13.2 |
| D6 | Khả năng chống tấn công và XAI | 9% | 3 | Tốt | 6,75 | XAI headline §4.9.2: SHAP top-4 behavioral, LIME 6 case. Robustness legacy §4.9.1. Chưa adversarial trên `weighted_blend`. | `phase6_final_shap_global_importance.csv`, `phase6_final_lime_case_summary.csv`, `phase6_robustness_metric_drops.csv` |
| D7 | Khả năng triển khai thực tiễn | 5% | 3 | Tốt | 3,75 | Dual-threshold: τ=0,6 FPR **1,06%**; τ=0,3 Recall **93,90%**. Chưa quant ROI (Luca & Zervas, 2016). | `phase7_final_metrics.csv`, §4.13.1, Ch.6 §6.3 |
| D8 | Tính trung thực và hoàn chỉnh | 10% | 4 | Xuất sắc | 10,0 | Dual-track nhất quán Chương 1–6; SSOT = `phase7_final_metrics.csv`; Phase 6 tách XAI final / adversarial legacy disclose; Phase 8 `phase8_report_summary.csv` đồng bộ ablation Models A–E; seed 123 công khai §5.7. | Chương 1–6, `phase6_metadata.json`, `phase8_report_summary.csv`, `phase8_submission_package_manifest.csv` |
| | **Tổng cộng** | **100%** | — | **Xuất sắc** | **94,5 / 100** | Ba chiều Tốt (D0, D6, D7); sáu chiều Xuất sắc (D1–D5, D8). Khung: **Bảng 3.13**. | |

**Diễn giải:** Điểm tổng **94,5/100** (xếp loại **Xuất sắc**, thang ≥ 90) sau rà soát đối chiếu Bảng 3.13 với artifact Phase 6–8 (cập nhật Phase 6 final track, Phase 7 ablation rerun, Phase 8 package 2026-06-11). So với lần chấm trước (**88,0**), hai chiều được nâng nấc: **D5** (3→4) vì multi-seed; **D4** (3→4) vì ablation + G5/G8. **D6** giữ nấc 3: XAI final track đã có (§4.9.2) nhưng adversarial chưa trên pipeline chính và giải thích chưa phủ full blend. Hướng nâng điểm: adversarial trên final blend (+D6), quant hóa chi phí false alarm (+D7), EDA benchmark (+D0), cross-dataset (+D5).

---

## 4.15. Gói tái lập và nộp bài (Phase 8, 2026-06-11)

Notebook `08_Final_Report_Kaggle.ipynb` tổng hợp artifact Phase 1–7 thành gói `reports/final/` — không train lại, chỉ audit và đóng gói. `phase8_artifact_inventory.csv` xác nhận **tất cả** mục `required=True` tồn tại (split, features 777-d, PCA, models, ensemble, bảng Phase 7).

### Bảng 4.15. Thành phần gói nộp (`phase8_submission_package_manifest.csv`)

| Hạng mục | Đường dẫn | Phase nguồn |
|----------|-----------|-------------|
| Notebook cuối | `notebooks/08_Final_Report_Kaggle.ipynb` | 8 |
| Báo cáo Markdown | `reports/final/Phase8_Final_Report.md` | 8 |
| Facts compact | `reports/final/phase8_report_summary.csv` | 8 ← 7 |
| Inventory / checklist | `phase8_artifact_inventory.csv`, `phase8_run_order_checklist.csv` | 8 |
| Metrics headline | `reports/tables/phase7_final_metrics.csv` | 7 |
| Target + ablation | `phase7_target_audit.csv`, `phase7_ablation_results.csv` | 7 |
| Robustness/XAI | `phase6_robustness_metrics.csv` (+ `phase6_final_*` headline, §4.9.2) | 6 |
| Model artifacts | `artifacts/models/`, `artifacts/ensemble/` | 4–5 |

`phase8_report_summary.csv` khớp số luận văn: balanced Macro F1 **0,9463**, precision-first Prec. Fake **0,9816**, ablation Model E Δ **−0,1769**, CV Macro F1 **0,8659 ± 0,0036**. Thứ tự chạy: 01→08 theo `phase8_run_order_checklist.csv` (seed 42, Colab). Bản đồ đầy đủ thư mục `artifacts/`, file SSOT và bảng tra nhanh demo: **Phụ lục** `thesis/Appendix_Artifacts_and_Folder_Structure.md`.

**Diễn giải (Bảng 4.15):** Gói Phase 8 **không train lại** — chỉ audit tồn tại artifact và đồng bộ số từ Phase 7 vào `phase8_report_summary.csv`. Mọi mục `required=True` trong manifest đều có đường dẫn cố định, cho phép hội đồng truy vết từ số trong luận văn → file CSV → notebook sinh file.

**Kết luận:** Pipeline 8 phase **khép kín** vòng reproducibility (D3, D8): từ `01_EDA` đến `08_Final_Report`, SSOT cuối là `phase7_final_metrics.csv`; Phase 8 là lớp đóng gói nộp bài, không thay đổi metric headline.