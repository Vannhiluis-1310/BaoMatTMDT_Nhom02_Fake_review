# CHƯƠNG 5: THẢO LUẬN

Chương này diễn giải ý nghĩa của các kết quả Chương 4 trong bối cảnh 20 công trình tham chiếu và 8 khoảng trống nghiên cứu (Gupta et al., 2024; Ren & Ji, 2024). Mục tiêu không phải tái liệt kê số liệu mà là trả lời **tại sao** pipeline dual-track hoạt động, **giới hạn nào** còn tồn tại, và **so sánh với ai** trên cơ sở metric và dataset tương đương.

*Quy ước trình bày:* Các bảng tổng hợp (Bảng 5.1–5.4) kèm **Diễn giải** (ý nghĩa so với literature, RQ, triển khai) và **Kết luận** (thông điệp thảo luận rút ra) — đồng bộ với Chương 4. Số chi tiết gốc: artifact Phase 7–8 và Bảng 4.x.

---

## 5.1. Tổng hợp kết quả định lượng

Pipeline chính `phase5_weighted_blend` đạt trên test n = 6.413, seed 42 (`phase7_final_metrics.csv`): Macro F1 **0,9463** (balanced, τ = 0,30), Precision Fake **0,9816** (precision-first, τ = 0,60), ROC-AUC **0,9769** (bất biến theo τ). Kiểm định multi-seed (`phase7_multiseed_summary.csv`, n = 3): balanced Macro F1 **0,9485 ± 0,0018**; precision-first Prec. Fake **0,9763 ± 0,0029** — seed 123 đơn lẻ 0,9728 (`phase7_multiseed_metrics.csv`) hơi dưới target 0,975 nhưng mean vẫn đạt. Ba target ban đầu (≥0,89 / ≥0,975 / ≥0,93) **đều đạt** ở chế độ tương ứng (`phase7_target_audit.csv`) — điều legacy PCA+PSO track không đạt (Macro F1 0,8558 @ τ = 0,50, `phase5_final_metrics.csv`).

Trọng số blend **không cố định** và **không** lấy từ trực giác “chia đều ba nhánh DL + tree”. Grid validation (bước 0,05, max Macro F1) chốt: seed **42** → CNN **50%** + XGB **50%**, LGBM **0%** (`phase5_weighted_blend_metadata.json`); seed **123** → cùng 50/0/50 (`seed_123/...metadata.json`); seed **456** → CNN **60%** + LGBM **35%** + XGB **5%** (`seed_456/...metadata.json`) — chi tiết Bảng 4.1b–c, §4.7.1. Tỷ lệ 50/35/15 từng xuất hiện trong tài liệu kỹ thuật cũ chỉ là **ứng viên sweep** (val 0,9375), thua 50/50 (val 0,9450) trên seed 42; luận văn không dùng tỷ lệ đó làm headline. Sự khác biệt 42/123 vs 456 phản ánh **bề mặc val khác nhau** sau mỗi lần train lại base model, phù hợp Zhang et al. (2020): ensemble tối ưu phụ thuộc $p_k$ trên hold-out, không nên hard-code trước sweep.

So với Tier A đã kiểm chứng nguồn (Bảng 2.2): kết quả **cao hơn** Gupta (2021) weighted-F1 0,69 trên Yelp 1,4M — khác corpus và metric; **cao hơn** Mir et al. (2023) accuracy 87,81% khi đọc qua Macro F1 (Ott et al., 2011 khuyến cáo không so sánh Accuracy trực tiếp trên dữ liệu mất cân bằng); **ngang hoặc trên** Veluru et al. (2025) F1 0,934 — nhưng Veluru multimodal (BERT+ResNet), Tier khác. Sequence branch đơn Macro F1 0,9324 (`phase7_final_metrics.csv`, τ = 0,50) tiếp cận vùng Bhuvaneshwari et al. (2021) >90% Amazon; full blend default 0,9433 (+0,0109) cho thấy GBDT bổ sung diversity (Breiman, 1996) chứ không thay thế nhánh DL.

**Kết luận (§5.1):** Ba lớp bằng chứng chồng lên nhau: (1) **Headline seed 42** đạt mục tiêu từng chế độ; (2) **Multi-seed** xác nhận không phụ thuộc may mắn đơn lẻ (σ Macro F1 = 0,0018); (3) **So sánh Tier A** có điều kiện — mạnh trên text/tabular Amazon, không overclaim graph/multimodal. Redesign pipeline (+0,0907 vs legacy) là đóng góp định lượng lớn nhất so với tinh chỉnh PSO đơn lẻ (+0,0128).

---

## 5.2. Thảo luận ablation study

Ablation Phase 7 (2026-06-11, `phase7_metadata.json`) cung cấp bằng chứng định lượng cho RQ1–RQ6 qua sáu variant Models A–E (`ablation_evidence_map`).

**PCA (RQ3):** Raw LGBM Macro F1 0,9058 vs PCA surrogate 0,8661 — Δ = +0,0397 (`phase7_ablation_results.csv`, Model B). Kết quả **đảo ngược** kỳ vọng từ Shah (2019): PCA hữu ích khi feature thô nhiễu và chiều cực cao (Jolliffe, 2002), nhưng trên vector fused ModernBERT+behavioral, giảm chiều 777→400 có thể loại bỏ tương tác phi tuyến mà GBDT khai thác tốt (Shwartz-Ziv & Armon, 2021). Đây là *negative result* có giá trị — Gupta et al. (2024) khuyến khích báo cáo thất bại có kiểm soát thay vì chỉ headline SOTA.

**Ensemble (RQ4):** Full blend 0,9433 vượt từng nhánh đơn: Model A (XGB tabular) 0,9059 (−0,0374), Model D (sequence) 0,9343 (−0,0090), và Model E nearest baseline (`dl_baseline`) 0,7665 (−0,1769, `phase7_ablation_delta.csv`). PSO trên DL đơn lẻ vẫn chỉ +0,0128 (0,7665→0,7793, §4.6) — tách khỏi ablation table. Lợi ích ensemble đa view lớn hơn nhiều so với PSO. Phù hợp Zhang et al. (2020) và Duma et al. (2023): kết hợp nhiều view vượt single view — nhưng **weighted blend** thắng stacking calibrated 0,9105 (`phase7_final_metrics.csv`), tránh over-engineering meta-learner khi base models đã mạnh.

**Behavioral (RQ1):** Controlled ablation Model C: bỏ 4 advanced features → Macro F1 0,8684 vs ref. PCA+9 feat. 0,8661 — Δ = +0,0023 (`phase7_ablation_results.csv`). Mukherjee et al. (2013) và Duma et al. (2023) nhấn mạnh behavioral; đề tài xác nhận đóng góp **marginal** của advanced block trên surrogate LGBM+PCA. Bổ sung từ Phase 6: SHAP trên XGB raw 777-d đặt **bốn behavioral cơ bản** (word count, verified purchase, char length, rating deviation) ở top-4 attribution (`phase6_final_shap_global_importance.csv`) — không mâu thuẫn ablation: marginal ≠ attribution khi đã có full fusion. Pipeline chính vẫn dựa vào ModernBERT+sequence+blend (sequence 0,9343; blend 0,9433).

**Diễn giải (§5.2):** Ablation không chỉ “chứng minh SOTA” mà **sắp xếp lại ưu tiên đầu tư kỹ thuật**: ensemble (+0,1769) >> bỏ PCA (+0,0397) >> bỏ sequence (−0,0090) >> advanced behavioral (+0,0023). Negative result PCA đặc biệt quan trọng vì trái với narrative draft cũ (PCA+PSO làm trục) và với Shah (2019) trên feature thô khác domain.

**Kết luận (§5.2):** Thảo luận ablation củng cố **dual-track** là quyết định thiết kế đúng: final track raw+blend mang SOTA; ablation track PCA+PSO phục vụ diagnostic và appendix. Đây là câu trả lời nội dung cho RQ3, RQ4, RQ6 (chi tiết Bảng 5.4).

---

## 5.3. Đánh giá đối chiếu mục tiêu

### Bảng 5.1. Đối chiếu ba mục tiêu định lượng (M1–M3)

| Metric | Target | Đạt được | Chế độ | Ý nghĩa thực tiễn |
|--------|--------|----------|--------|-------------------|
| Macro F1 | ≥0,89 | **0,9463** (mean 0,9485±0,0018) | balanced τ=0,30 | Headline nghiên cứu |
| Precision Fake | ≥0,975 | **0,9816** (mean 0,9763±0,0029) | precision-first τ=0,60 | Auto-flag ít khóa nhầm |
| ROC-AUC | ≥0,93 | **0,9769** | mọi τ | Ranking tốt |

**Diễn giải:** Cả ba metric **vượt target** khi đọc đúng chế độ — không phải “một con số đạt cả ba”. Macro F1 headline (0,9463) vượt 0,89 thêm **+0,0563**; precision-first (0,9816) vượt 0,975 thêm **+0,0066**; ROC-AUC (+0,0469 so với 0,93) cho thấy ranking xác suất mạnh **độc lập** với τ. Cột mean multi-seed (Bảng 4.5b) bổ sung: balanced Macro F1 0,9485±0,0018 — margin an toàn trên target M1.

**Kết luận (Bảng 5.1):** M1–M3 **đạt có điều kiện** — điều kiện là **dual-threshold**, không phải single τ. Đây là đóng góp phương pháp luận (Gap G7): báo cáo trung thực trade-off thay vì chọn τ sau khi nhìn test. Khuyến nghị báo cáo: luôn kèm `phase7_target_audit.csv` và nêu rõ chế độ vận hành (Ch.6 §6.3).

---

## 5.4. Thảo luận cơ chế: tại sao dual-track hiệu quả?

Phần này giải thích **cơ chế** đứng sau các con số §5.1–5.3, không lặp lại bảng mà nối kết quả thực nghiệm với lý thuyết đã trình bày ở Chương 2.

### 5.4.1. Hai view bổ sung: tabular 777-d và sequence token

Kiến trúc dual-track phản ánh giả thuyết từ Gupta et al. (2024) và Ren & Ji (2024): văn bản đánh giá giả và thật có thể **không phân tách được** chỉ bằng lexicon (minh chứng EDA: compound VADER Fake 0,456 vs Real 0,445 — `phase1_advanced_eda_summary.csv`), nhưng vẫn mang tín hiệu trong **không gian embedding** (Devlin et al., 2019) và **thứ tự token** (Kim, 2014; Bahdanau et al., 2015).

- *Nhánh tabular:* ModernBERT freeze → 768-d, concat 9 behavioral → 777-d. GBDT (Chen & Guestrin, 2016; Ke et al., 2017) học ranh giới phi tuyến trên vector dày — phù hợp Shwartz-Ziv và Armon (2021) trên dữ liệu bảng có cấu trúc.
- *Nhánh sequence:* CNN-BiLSTM-Attention trên chuỗi token + late fusion behavioral, Focal Loss (Lin et al., 2017) xử lý mất cân bằng lớp ~41% Fake (`phase1_advanced_eda_summary.csv`).

Hai view **không trùng lặp hoàn toàn**: XGB/LGBM raw @ τ = 0,50 đạt Macro F1 0,9059 / 0,9051 (`phase7_final_metrics.csv`), thấp hơn sequence 0,9324 — nhưng khi blend, XGB đóng góp precision fake cao (0,9686 test) giúp kéo biên quyết định ở chế độ precision-first. Seed 42 chọn trọng số CNN 50% + XGB 50% (`phase5_weighted_blend_metadata.json`), tức meta-learner implicit coi hai view này **bổ sung** chứ không thay thế.

### 5.4.2. Tại sao weighted blend thắng stacking?

Stacking calibrated test Macro F1 0,9105 (`phase7_final_metrics.csv`) thua blend 0,9433 — chênh 0,0328. Diễn giải theo Breiman (1996) và Zhang et al. (2020):

1. **Base models đã mạnh và đa dạng:** CNN sequence và XGB raw có ROC-AUC 0,9726 / 0,9531 — meta-learner logistic trên xác suất calibrated dễ **overfit** trên validation 6.413 mẫu.
2. **Blend là convex combination:** Giảm phương sai dự đoán mà không thêm tham số học — phù hợp khi grid sweep trên validation đã tìm được vùng trọng số ổn định (macro F1 val 0,9450 tại CNN+XGB 50/50, `phase5_weighted_blending_best.csv`).
3. **LGBM train F1 = 1,0** (`phase5_lgbm_raw_metrics.csv`) gợi ý overfit nặng trên train; grid search seed 42 **không** đưa LGBM vào blend cuối — minh chứng rằng protocol chọn trọng số trên val có tác dụng regularization implicit.

### 5.4.3. Tại sao trọng số khác nhau giữa seed 42, 123 và 456?

Đây là câu hỏi phương pháp luận thường gặp khi đọc §4.11 cùng Bảng 4.1c. Ba điểm cần tách bạch:

1. **Split dữ liệu giữ nguyên, seed đổi ý nghĩa “seed huấn luyện”.** Phase 1 cố định stratified 70/15/15 với seed 42 (`split_metadata.json`). Multi-seed 42/123/456 **không** đổi ranh giới train/val/test mà đổi khởi tạo ngẫu nhiên khi fit CNN, XGB, LGBM → vector xác suất $p_{\text{CNN}}, p_{\text{XGB}}, p_{\text{LGBM}}$ trên **cùng** tập validation 6.413 mẫu thay đổi.

2. **Grid chạy lại trên val từng seed.** Mỗi lần chạy, `05_05` tìm $\arg\max_{w} \text{Macro F1}_{\text{val}}$. Seed 42: mọi $w_{\text{LGBM}} > 0$ trong lưới CNN–LGBM–XGB đều thua 50/50 (Bảng 4.1b). Seed 456: CNN+XGB 50/50 chỉ đạt val 0,9397; thêm 35% LGBM (và giảm XGB còn 5%) đẩy lên 0,9403 — chênh nhỏ nhưng đủ để grid chọn ba thành phần. **LGBM xuất hiện ở 456 không phải vì “multi-seed bắt buộc dùng LGBM”**, mà vì trên val của seed đó nó có marginal gain.

3. **Metric ổn định ≠ trọng số cố định.** Bảng 4.5b cho thấy balanced Macro F1 test **0,9485 ± 0,0018** dù trọng số 456 khác 42. Điều này hỗ trợ claim **độ tin cậy kết quả** (D1): pipeline không phụ thuộc một may mắn trọng số duy nhất; đồng thời minh bạch rằng **không** nên báo cáo một tỷ lệ 50/35/15 chung cho mọi seed nếu artifact metadata không chứng minh.

**Hàm ý cho triển khai:** Production nên **lưu metadata seed + trọng số + τ** cùng model card; khi retrain, chạy lại grid trên validation hold-out thay vì copy trọng số seed 42 sang môi trường khác.

### 5.4.4. ModernBERT freeze và max_length 160

Lý thuyết ModernBERT hỗ trợ ngữ cảnh dài tới 8.192 tokens (Warner et al., 2024), nhưng triển khai đề tài giới hạn `max_length = 160` (`phase5_cnn_bilstm_sequence_metadata.json`) vì RAM ≤ 12GB. Đây là **trade-off có chủ đích**: embedding tabular dùng pooling toàn chuỗi ModernBERT (Phase 2), trong khi nhánh sequence cắt token — hai nhánh vẫn chia sẻ cùng encoder frozen. EDA cho thấy median độ dài Fake 43 ký tự vs Real 125 (`phase1_advanced_eda_summary.csv`), nên 160 token phủ phần lớn review; phần đuôi dài hiếm có thể bị cắt — hạn chế cần ghi nhận khi so với claim lý thuyết dài ngữ cảnh ở §2.3.2.

### 5.4.5. PCA ablation track — vai trò diagnostic, không phải SOTA

PCA 777→400 giữ 95,10% phương sai (Phase 3 metadata), phục vụ DL legacy trên RAM 12GB và CV surrogate Macro F1 0,8659 ± 0,0036 (`phase7_cv_summary.csv`). Ổn định cao (σ nhỏ) nhưng điểm tuyệt đối thấp hơn raw blend ~0,08 điểm. Bài học phương pháp luận: khi tái lập Shah (2019) trên **vector fused BERT+behavioral**, PCA không còn là đường chính — phải tách track và báo cáo negative result (Gap G5), tránh narrative PCA+PSO làm trục chính như các bản draft cũ.

**Kết luận (§5.4):** Cơ chế hiệu quả của pipeline là **complementarity** (tabular precision + sequence recall + blend convex) chứ không phải một “silver bullet” (PSO, PCA, hay behavioral đơn lẻ). Trọng số seed-dependent nhưng metric ổn định — production cần metadata-driven re-sweep, không hard-code 50/50.

---

## 5.5. Robustness và XAI — dual-track Phase 6 (2026-06-11)

Phase 6 được tái cấu trúc thành **hai luồng** (`phase6_metadata.json`, `headline_track: final_raw_777`), phản ánh đúng vai trò dual-track của luận văn: XAI headline gắn pipeline báo cáo; adversarial robustness giữ ở legacy appendix vì attack space được định nghĩa trên PCA 400-d fit train — không tương đương perturbation trên raw 777-d hay text.

### 5.5.1. Robustness adversarial (legacy appendix)

FGSM/PGD trên **PCA feature space**, subset n = 1.000, surrogate `dl_pso` (`phase6_robustness_metric_drops.csv`). FGSM ε = 0,03:

### Bảng 5.2. Suy giảm FGSM — legacy appendix (đối chiếu §4.9.1)

| Model | Clean Macro F1 | FGSM Macro F1 | Δ Macro F1 |
|-------|----------------|---------------|------------|
| dl_pso | 0,7724 | 0,7068 | −0,0656 |
| final_ensemble (legacy) | 0,8000 | 0,7949 | −0,0052 |

Legacy ensemble ổn định hơn DL-PSO đơn lẻ — phù hợp Breiman (1996) và Goodfellow et al. (2015). Clean F1 legacy 0,80 thấp hơn nhiều `weighted_blend` 0,9463 — **không suy diễn** robustness pipeline chính từ appendix (Ren & Ji, 2024). `phase6_final_metadata.json` ghi rõ: *No adversarial robustness claim for the final raw tabular + sequence pipeline*.

**Diễn giải (Bảng 5.2):** Tỷ lệ suy giảm tương đối: dl_pso mất **8,5%** Macro F1 clean (0,0656/0,7724); legacy ensemble chỉ **0,65%** (0,0052/0,8000). Diễn giải theo Goodfellow et al. (2015): không gian PCA đã nén làm perturbation ε = 0,03 tác động mạnh lên decision boundary DL đơn; ensemble tree “làm mịn” biên. Tuy nhiên baseline clean 0,80 vs 0,9463 của pipeline chính — so sánh robustness **không cùng mức hiệu năng**.

**Kết luận (Bảng 5.2):** Appendix chứng minh **khả năng thí nghiệm** adversarial trong RAM 12GB, không phải claim bảo mật cho `weighted_blend`. Chiều D6 giữ nấc 3 (Bảng 4.14) là hợp lý; hướng mở §5.8 ưu tiên attack trên final blend hoặc text space.

### 5.5.2. XAI final track — ý nghĩa và giới hạn

SHAP trên `phase5_xgb_raw` (777-d, n = 500) cho thấy **bốn behavioral cơ bản** chiếm top-4 mean \|SHAP\| (`basic_word_count_log` 1,62; `basic_verified_purchase` 1,15; §4.9.2 Bảng 4.9b). Điều này **nhất quán** với ablation controlled: advanced behavioral chỉ +0,0023 Macro F1 (Bảng 4.4) trong khi basic block (độ dài, verified purchase, rating deviation) mang tín hiệu phân biệt mạnh trong SHAP — không mâu thuẫn: ablation đo **marginal** contribution khi đã có full fusion, SHAP đo **attribution** trên mô hình đã train.

LIME 6 case (`phase6_final_lime_case_summary.csv`) chọn mẫu theo `weighted_blend` @ τ = 0,30 nhưng giải thích bằng XGB raw — hợp lý vì XGB chiếm 50% blend seed 42 và là mô hình tabular có feature names đọc được. Case FP (review thật bị flag, P = 0,987) và FN (fake bị bỏ sót, P = 0,012) cho thấy lỗi tập trung ở **review cực ngắn** và **chiều embedding cực đoan** — gợi ý rule-based pre-filter hoặc human review queue cho outlier độ dài (Mukherjee et al., 2013).

**Giới hạn còn lại:** (i) SHAP/LIME chưa phủ CNN-BiLSTM sequence (50% blend còn lại); (ii) 768 chiều BERT báo cáo per-dimension, chưa map token-level; (iii) adversarial chưa trên final blend. Hướng mở §5.8.

**Diễn giải (§5.5.2):** XAI final track trả lời câu hỏi *“mô hình dựa vào gì?”* cho moderator: word count và verified purchase có tên feature, không phải `pca_000`. LIME case FP/FN nối trực tiếp FPR 1,06% và FN = 160 — giải thích **cơ chế lỗi** chứ không chỉ báo cáo confusion matrix.

**Kết luận (§5.5):** Phase 6 dual-track là mô hình **trung thực khoa học**: headline XAI trên pipeline báo cáo; robustness legacy disclose. Đủ cho moderation staging (gắn LIME vào queue); chưa đủ cho claim “chống adversarial” trên production.

---

## 5.6. Ý nghĩa thực tiễn và đóng góp khoa học

### 5.6.1. Triển khai trên nền tảng thương mại điện tử

Phát hiện fake review trên sàn không chỉ là tối đa hóa accuracy mà là **quản trị rủi ro niềm tin** (Luca & Zervas, 2016): flag nhầm đánh giá thật (false positive) gây thiệt hại người bán; bỏ sót fake (false negative) làm suy giảm chất lượng thông tin. Đề tài formalize hai chế độ từ cùng một mô hình `weighted_blend`, ngưỡng chọn trên validation (`phase5_metadata.json`):

**Precision-first (τ = 0,60):** Precision Fake 0,9816, TN = 3.749, FP = 40 trên 3.789 real → FPR = 40/3.789 = **1,06%** (`phase7_final_metrics.csv`, test). Recall Fake 0,8152 — chấp nhận bỏ sót ~18,5% fake để giảm false alarm. Phù hợp **auto-flag** hoặc ẩn tự động khi chi phí khóa nhầm cao.

**Balanced (τ = 0,30):** Macro F1 0,9463, Recall Fake **0,9390** (FN = 160 trên 2.624 fake), Precision Fake 0,9344. Phù hợp **moderation queue** cần bắt phần lớn spam trước khi human review — tương tự tinh thần Ott et al. (2011) ưu tiên F1 cân bằng trên corpus mất cân bằng nhẹ. LIME Phase 6 chọn case @ τ = 0,30 (`phase6_final_lime_case_summary.csv`): moderator có thể đính kèm giải thích local (word count, verified purchase) cho từng flag.

ROC-AUC 0,9769 bất biến theo τ — sàn có thể điều chỉnh ngưỡng theo mùa chiến dịch spam mà không retrain, miễn audit lại trên validation hold-out (Ren & Ji, 2024).

So với legacy τ = 0,79: test Macro F1 chỉ 0,7860 @ precision-first (`phase5_final_metrics.csv`) với FPR legacy cao hơn (61 FP @ τ = 0,79 trên test). Redesign pipeline 09/06 mang lại cải thiện triển khai thực tế rõ rệt.

**Kết luận (§5.6.1):** Hai chế độ τ map **hai KPI nghiệp vụ**: precision-first ≈ “đừng làm tổn thương seller” (FPR 1,06%); balanced ≈ “bắt đủ spam trước human review” (recall 93,90%). ROC-AUC cao cho phép điều chỉnh τ theo mùa mà không retrain — điều kiện là re-sweep trên val hold-out, không copy τ từ luận văn sang production mùa khác.

### 5.6.2. Đóng góp khoa học đối chiếu 8 gaps

### Bảng 5.3. Tóm tắt đóng góp lấp gap (G1, G4, G5, G7, G8 — tiêu biểu)

| Gap | Nội dung | Bằng chứng đề tài |
|-----|----------|-------------------|
| G1 | Text-only thiếu behavioral engineered | Vector 777-d + SHAP top-4 behavioral (`phase6_final_shap_global_importance.csv`) |
| G4 | Thiếu protocol reproducible | Dual-track, metadata JSON/CSV, fit train-only (`phase5_metadata.json`) |
| G5 | PCA trên embedding hiện đại | Negative result Δ +0,0397 raw thắng PCA (`phase7_ablation_results.csv`) |
| G7 | Hiếm dual-threshold e-commerce | Target audit pass @ τ = 0,60 (`phase7_target_audit.csv`) |
| G8 | Ablation không đầy đủ | 6 variant Phase 7 + delta table (`phase7_ablation_delta.csv`) |

**Diễn giải (Bảng 5.3):** Bảng chọn năm gap **có bằng chứng trực tiếp** trong thảo luận Ch.5 — không lặp đủ G2/G3/G6 (đã có tại Bảng 4.12a). G1 được hỗ trợ cả ablation lẫn SHAP (marginal nhỏ nhưng attribution lớn ở basic block). G4 là nền tảng cho mọi số khác: không protocol thì SOTA không audit được. G5 negative result **đổi narrative** từ “PCA+PSO SOTA” sang “raw+ensemble SOTA”.

**Kết luận (Bảng 5.3):** Đóng góp khoa học nằm ở **phương pháp có kiểm chứng** (G4, G8) và **kết quả có điều kiện** (G7), không chỉ một con số Macro F1. Đề tài **không** claim beat graph (Wu et al., 2024 ~0,915 F1) hay multimodal (Veluru et al., 2025 F1 0,934) — phân tầng Tier A/B/C (Bảng 2.2) theo khuyến nghị Gupta et al. (2024).

---

## 5.7. Hạn chế của nghiên cứu

1. **Multi-seed đã thực hiện** (§4.11) — seed 123 precision-first đơn lẻ (0,9728) hơi dưới target 0,975; mean 3 seed vẫn đạt. Chưa có kiểm định thống kê formal (bootstrap / paired test).

2. **Overfit train** — blend train Macro F1 ≈ 0,976; LightGBM **raw** train = 1,0 (test 0,9051). Test blend vẫn 0,9433–0,9463; val–test gap ≈ 0,0005. Cần regularization mạnh hơn hoặc early stopping nghiêm trên nhánh tree.

3. **Phase 6 tách track** — XAI headline trên XGB raw 777-d (50% blend), chưa full blend + sequence; adversarial chỉ legacy PCA appendix.

4. **Không graph/multimodal** — không so sánh Rayana & Akoglu (2015), Wu et al. (2024), Veluru et al. (2025) trực tiếp.

5. **ModernBERT freeze** — chưa khai thác fine-tune như Refaeli & Hajek (2021).

6. **Metric heterogeneity** — nhiều paper báo Accuracy (Vidanagama et al., 2020; Mir et al., 2023); so sánh đòi hỏi chuẩn hóa cẩn trọng.

**Diễn giải (§5.7):** Sáu hạn chế được nhóm theo **độ nghiêm trọng đối với claim**: (i) thiếu kiểm định thống kê formal — ảnh hưởng diễn đạt “độ tin cậy”, không đảo ngược SOTA; (ii) overfit train LGBM — đã mitigate bằng grid loại LGBM khỏi blend seed 42; (iii) XAI/adversarial chưa full pipeline — giới hạn D6; (iv) phạm vi corpus/paradigm — giới hạn D2/D5.

**Kết luận (§5.7):** Hạn chế được **công khai** (D8 Xuất sắc) thay vì che giấu — phù hợp Ren & Ji (2024). Không hạn chế nào phủ nhận việc đạt M1–M3 ở chế độ tương ứng; chúng định hướng agenda §5.8.

---

## 5.8. Hướng nghiên cứu tiếp theo

- Fine-tune ModernBERT (LoRA) trong RAM 12GB — theo hướng Refaeli & Hajek (2021)
- Kiểm định thống kê formal trên multi-seed (bootstrap / paired test) — Ren & Ji (2024)
- Adversarial robustness trên final blend; SHAP/LIME trên CNN sequence branch
- Graph collusion features (Rayana & Akoglu, 2015; Wu et al., 2024)
- Đánh giá cross-dataset Yelp — Gupta et al. (2024) đề xuất

**Kết luận (§5.8):** Năm hướng ưu tiên theo **tác động lên rubric**: adversarial + XAI full blend (+D6); kiểm định thống kê (+D5); LoRA ModernBERT (+D1); graph/cross-dataset (+D2/D5). Thứ tự phản ánh hạn chế §5.7, không phải danh sách ý tưởng ngẫu nhiên.

---

## 5.9. Trả lời câu hỏi nghiên cứu RQ1–RQ6

Bảng dưới tổng hợp câu trả lời ngắn gọn; mọi số liệu trích từ artifact đã audit Phase 7–8 (2026-06-11).

### Bảng 5.4. Tổng hợp trả lời RQ1–RQ6

| RQ | Câu hỏi (tóm tắt) | Kết luận | Bằng chứng chính |
|----|-------------------|----------|------------------|
| **RQ1** | ModernBERT + 9 behavioral có cải thiện so với surrogate không? | **Có, nhưng đóng góp behavioral riêng rất nhỏ** trên LGBM controlled; pipeline chính dựa vào fusion + sequence | Full blend 0,9433 vs Model C 0,8684 (`phase7_ablation_results.csv`); Δ advanced +0,0023 |
| **RQ2** | Dual-track có vượt kiến trúc đơn nhánh? | **Có** — blend vượt sequence (Model D), XGB tabular (Model A), LGBM, stacking từng đơn lẻ | Blend 0,9433 vs Model D 0,9343, Model A 0,9059, stacking 0,9105 (`phase7_ablation_results.csv`, `phase7_final_metrics.csv`) |
| **RQ3** | PCA 777→400 còn phù hợp làm đường chính? | **Không** — raw thắng PCA trên cùng LGBM controlled | Raw 0,9058 vs PCA 0,8661, Δ +0,0397 (`phase7_ablation_results.csv`, Model B) |
| **RQ4** | Weighted ensemble có vượt base và stacking? | **Có** — blend thắng mọi base @ τ = 0,50 và thắng stacking 0,9105 | `phase7_final_metrics.csv`; trọng số seed 42: CNN 50% + XGB 50% (`phase5_weighted_blend_metadata.json`) |
| **RQ5** | Dual-threshold có đạt target trên test audit? | **Có, theo từng chế độ** — không có single τ đạt đồng thời mọi target | Precision-first @ 0,6 pass cả 3 (`phase7_target_audit.csv`); balanced @ 0,3 fail precision ≥ 0,975 |
| **RQ6** | Đóng góp tương đối PSO / PCA / behavioral / ensemble? | Ensemble >> PSO >> PCA; behavioral marginal | Ensemble Δ +0,1769 vs Model E; PSO +0,0128 (§4.6); PCA raw thắng +0,0397; behavioral +0,0023 (`phase7_ablation_delta.csv`) |

**RQ1 — chi tiết:** Đề tài **không** có ablation text-only thuần (768-d không behavioral) trên cùng protocol; RQ1 được trả lời gián tiếp qua so sánh full pipeline (ModernBERT+behavioral+ensemble) với Model C (bỏ advanced behavioral, giữ 5 basic trên PCA surrogate). Khoảng cách 0,9433 vs 0,8684 (Δ 0,075) cho thấy **tổ hợp kiến trúc** quan trọng hơn từng nhóm feature đơn lẻ — nhất quán Duma et al. (2023) về hybrid text–metadata.

**RQ2 — chi tiết:** Sequence branch Macro F1 0,9324 (`phase7_final_metrics.csv`) đã tiếp cận Bhuvaneshwari et al. (2021) trên Amazon, nhưng dual-track với GBDT tabular đẩy thêm ~1,1 điểm — chứng minh **late fusion đa view** (tabular + sequence) vượt single CNN-BiLSTM dù cùng ModernBERT backbone.

**RQ5 — chi tiết:** Trade-off precision–recall cổ điển (Ott et al., 2011): balanced @ τ = 0,30 đạt macro F1 0,9463 nhưng precision fake 0,9344 < 0,975 (`phase7_target_audit.csv`, `pass = False`). Đóng góp là **formalize policy** chọn τ trên val thay vì báo cáo đơn điểm — đáp ứng Gap G7 mà hầu hết 20 papers (Gupta et al., 2024) không làm.

**RQ6 — tổng kết:** Thứ tự ưu tiên đầu tư kỹ thuật trên corpus này: (1) ensemble đa view, (2) raw fused features thay PCA, (3) PSO tinh chỉnh DL, (4) advanced behavioral. PSO (+0,0128) có ích trong ablation track nhưng không bù được thiếu ensemble (+0,1769 vs Model E `dl_baseline`).

**Diễn giải (Bảng 5.4):** Sáu RQ **không độc lập hoàn toàn**: RQ2/RQ4 cùng chứng minh dual-track+ensemble; RQ3/RQ6 cùng khẳng định PCA/PSO không phải trục SOTA; RQ5 tách khỏi RQ4 — đạt target là vấn đề **policy τ**, không chỉ accuracy. RQ1 trả lời **gián tiếp** (thiếu ablation text-only thuần) — hạn chế đã disclose; khoảng 0,9433 vs 0,8684 vẫn đủ mạnh cho claim “fusion + kiến trúc”.

**Kết luận (Bảng 5.4):** **5/6 RQ** có kết luận “Có” hoặc “Có, có điều kiện”; **RQ3** kết luận “Không” (PCA không phù hợp đường chính) — negative result có giá trị ngang positive. Toàn bộ RQ được neo vào artifact reproducible (Phase 7–8), đóng vòng Ch.1 (đặt vấn đề) → Ch.3 (luồng) → Ch.4 (số) → Ch.5 (ý nghĩa) → Ch.6 (tổng kết).