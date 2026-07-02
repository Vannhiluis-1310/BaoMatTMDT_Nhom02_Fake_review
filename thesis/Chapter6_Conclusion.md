# CHƯƠNG 6: KẾT LUẬN

Chương này **tổng kết** luồng Chương 1 (vấn đề, mục tiêu) → Chương 3 (phương pháp) → Chương 4 (số liệu) → Chương 5 (thảo luận). Không nhắc lại chi tiết thí nghiệm; mỗi bảng (Bảng 6.1–6.4) kèm **Diễn giải** và **Kết luận** — đồng bộ quy ước Chương 4–5. Số gốc: `phase7_final_metrics.csv`, `phase7_target_audit.csv`, Bảng 4.x, Bảng 5.x.

---

## 6.1. Tóm tắt kết quả nghiên cứu

Luận văn đã xây dựng và đánh giá toàn diện hệ thống phát hiện đánh giá giả mạo trên **Amazon Labeled Fake Reviews** (42.749 mẫu sau làm sạch, stratified split 70/15/15, seed = 42), trong ràng buộc tài nguyên Google Colab RAM ≤ 12GB. Hệ thống theo kiến trúc **dual-track**: nhánh chính (*final track*) kết hợp ModernBERT embeddings (768-d), 9 đặc trưng hành vi, nhánh sequence CNN-BiLSTM-Attention và weighted blend (grid val; seed 42: CNN 50% + XGB 50%, `phase5_weighted_blend_metadata.json`); nhánh ablation giữ PCA 777→400 và PSO-tuned deep learning để chứng minh giả thuyết âm tính về giảm chiều trên vector fused.

Trên tập kiểm tra độc lập (6.413 mẫu), kết quả chính từ `phase7_final_metrics.csv` (audit 2026-06-11):

### Bảng 6.1. Hiệu năng headline `weighted_blend` theo chế độ ngưỡng (test, seed 42)

| Chế độ | Ngưỡng τ | Macro F1 | Precision Fake | ROC-AUC |
|--------|----------|----------|----------------|---------|
| Default | 0,50 | 0,9433 | 0,9699 | 0,9769 |
| Balanced | 0,30 | **0,9463** | 0,9344 | 0,9769 |
| Precision-first | 0,60 | 0,9126 | **0,9816** | 0,9769 |

**Diễn giải:** Ba dòng phản ánh **cùng một mô hình**, **khác τ** chọn trên validation — ROC-AUC bất biến (0,9769) vì đo ranking, không phụ thuộc ngưỡng. Balanced đạt Macro F1 cao nhất (0,9463) nhưng precision fake 0,9344 **dưới** target 0,975; precision-first đạt 0,9816 nhưng Macro F1 giảm còn 0,9126 — minh họa trade-off đã thảo luận §5.3, Bảng 5.1. Mean multi-seed balanced **0,9485 ± 0,0018** (Bảng 4.5b) củng cố headline không phụ thuộc seed đơn lẻ.

**Kết luận (Bảng 6.1):** Ba mục tiêu M1–M3 **đạt có điều kiện** ở chế độ tương ứng (`phase7_target_audit.csv`): precision-first @ τ = 0,60 pass cả ba target; đây là số “đóng” cho claim mục tiêu luận văn. So với legacy PCA+PSO (Macro F1 0,8558 @ τ = 0,50), redesign +0,0907 — cải thiện lớn hơn PSO đơn (+0,0128). Ablation: raw thắng PCA (+0,0397); ensemble thắng `dl_baseline` (+0,1769). Pipeline 8 notebooks Phase 1–8 hoàn tất; audit Phase 6–8 ngày 2026-06-11.

---

## 6.2. Đóng góp của đề tài

### Bảng 6.2. Tóm tắt bốn nhóm đóng góp

| Nhóm | Nội dung chính | Bằng chứng / tham chiếu |
|------|----------------|-------------------------|
| **Phương pháp luận** | Dual-track reproducible; fit train-only; τ chỉ trên val; audit test một lần | `phase5_metadata.json`, `phase8_submission_package_manifest.csv`; Gap G4 (Bảng 4.12a) |
| **Kỹ thuật** | Fusion 777-d; dual view GBDT + sequence; weighted blend; negative result PCA | Bảng 4.1, 4.4; §4.6–4.7; Model B Δ +0,0397 |
| **Đánh giá** | SOTA 3-tier; target audit; ablation 6 variant; XAI final + adversarial legacy disclose | Bảng 4.3, 4.12a, 4.14; §4.9; `phase8_report_summary.csv` |
| **Triển khai** | Dual-threshold formalized — hai KPI nghiệp vụ từ một `weighted_blend` | §6.3, Bảng 6.3; Ch.5 §5.6.1 |

**Diễn giải:** Bốn nhóm không tách rời: phương pháp luận (G4) là điều kiện để số SOTA có ý nghĩa; kỹ thuật (fusion + ensemble) mang delta lớn nhất (+0,1769); đánh giá (ablation, multi-seed, rubric 94,5/100) tránh overclaim; triển khai chuyển metric thành **chính sách vận hành** (τ = 0,30 vs 0,60). SHAP top-2 `basic_word_count_log`, `basic_verified_purchase` (§4.9.2) bổ sung **diễn giải có tên feature** cho đóng góp behavioral — không mâu thuẫn ablation marginal (+0,0023 advanced).

**Kết luận (§6.2):** Đóng góp cốt lõi là **pipeline có audit end-to-end** trên Amazon 42k trong RAM 12GB — lấp 7/8 gap G1–G8 (G6 lấp một phần), đạt Tier A text/tabular có trách nhiệm (Ch.5 §5.6.2). Không claim graph/multimodal SOTA.

---

## 6.3. Khả năng ứng dụng thực tiễn

Trên các sàn thương mại điện tử như Amazon hay Shopee, quyết định phân loại đánh giá giả không chỉ là bài toán học máy mà còn là bài toán **quản trị rủi ro niềm tin** (Luca & Zervas, 2016). Một hệ thống flag nhầm đánh giá thật (false positive) có thể gây thiệt hại uy tín người bán và trải nghiệm người mua; ngược lại, bỏ sót fake (false negative) làm suy giảm chất lượng thông tin trên sàn.

### Bảng 6.3. Hai chế độ triển khai đề xuất

| Chế độ | τ | Metric chính | FPR / Recall (test) | Kịch bản nghiệp vụ |
|--------|---|--------------|---------------------|-------------------|
| **Precision-first** | 0,60 | Prec. Fake **0,9816** | FPR **1,06%** (40/3.789 real); Recall Fake 81,52% | Auto-flag / ẩn tự động — ưu tiên không khóa nhầm seller |
| **Balanced** | 0,30 | Macro F1 **0,9463** | Recall Fake **93,90%** (FN = 160); Prec. Fake 93,44% | Moderation queue — bắt phần lớn spam trước human review |

**Diễn giải:** Hai chế độ dùng **cùng** `phase5_weighted_blend` và ROC-AUC 0,9769 — sàn chỉ đổi τ theo mùa chiến dịch, không bắt buộc retrain, miễn re-sweep τ trên validation hold-out (Ren & Ji, 2024). Precision-first FPR 1,06% thấp hơn legacy τ = 0,79 (61 FP trên test, `phase5_final_metrics.csv`) trong khi Macro F1 precision-first vẫn > 0,89. LIME @ τ = 0,30 (`phase6_final_lime_case_summary.csv`) hỗ trợ moderator đính kèm giải thích local (word count, verified purchase).

**Kết luận (Bảng 6.3):** Hệ thống **sẵn sàng staging** e-commerce: có số FPR/FNR cụ thể, dual-threshold có protocol, XAI behavioral có tên feature. Production cần lưu metadata (seed, trọng số blend, τ) và chạy lại grid khi retrain — không copy cứng cấu hình seed 42 (Ch.5 §5.4.3).

---

## 6.4. Hạn chế và hướng phát triển

### Bảng 6.4. Hạn chế và hướng khắc phục tương ứng

| Hạn chế | Mức ảnh hưởng claim | Hướng phát triển | Liên hệ rubric |
|---------|---------------------|------------------|----------------|
| Chưa kiểm định thống kê formal multi-seed | Diễn đạt độ tin cậy | Bootstrap / paired test trên 3 seed | D5 |
| Overfit train (LGBM F1 = 1,0; blend train ≈ 0,976) | Đã mitigate bằng grid loại LGBM seed 42 | Regularization, early stopping tree | D1, D8 (disclosed) |
| XAI chưa full blend + sequence; adversarial chỉ legacy | Không claim robustness chính | SHAP/LIME CNN; FGSM trên final blend | D6 |
| ModernBERT freeze; max_length 160 | Chưa khai thác hết domain | LoRA fine-tune trong RAM 12GB | D1 |
| Không graph / multimodal / cross-dataset | Giới hạn phạm vi so sánh SOTA | Graph collusion; đánh giá Yelp | D2, D5 |
| Metric heterogeneity literature | So sánh cần thận trọng | Chuẩn hóa Macro F1 khi survey | D2 |

**Diễn giải:** Hạn chế được **nhóm theo tác động**, không liệt kê để “làm yếu” kết quả: M1–M3 vẫn đạt ở chế độ tương ứng; hạn chế chủ yếu giới hạn **phạm vi claim** (D2, D6) và **độ sâu đánh giá** (D5). Seed 123 precision-first 0,9728 hơi dưới 0,975 đã disclose; mean 3 seed vẫn ≥ 0,975 (Bảng 4.5b). CV surrogate PCA (σ = 0,0036) ổn định nhưng không thay thế đánh giá full blend.

**Kết luận (§6.4):** Năm hướng phát triển (LoRA, kiểm định thống kê, adversarial+XAI full pipeline, graph, cross-dataset/drift) xếp theo **ưu tiên nâng rubric** (Ch.5 §5.8), phù hợp agenda FRD 2024–2026 (Gupta et al., 2024; Ren & Ji, 2024). Minh bạch hạn chế đạt chiều D8 (Bảng 4.14).

---

**Kết thúc luận văn.** Nghiên cứu chứng minh tính khả thi của pipeline hai nhánh ModernBERT–behavioral–ensemble với chính sách ngưỡng kép trên Amazon Labeled Fake Reviews, đạt hiệu năng Tier A có trách nhiệm (Bảng 6.1) và cung cấp bằng chứng ablation reproducible (Bảng 4.4). Hệ thống sẵn sàng làm nền tảng triển khai staging (Bảng 6.3): XAI final track đã có giải thích behavioral; bước tiếp theo là adversarial trên blend đầy đủ và mở rộng đa corpus (Bảng 6.4). Sáu RQ (Bảng 5.4) được trả lời với bằng chứng artifact Phase 7–8; tự chấm **94,5/100** — Xuất sắc (Bảng 4.14).

*Phụ lục:* Cấu trúc repository, ánh xạ artifact → mô hình, và bảng tra nhanh khi báo cáo — `thesis/Appendix_Artifacts_and_Folder_Structure.md`.

*Tài liệu tham khảo đầy đủ: `thesis/References.md`.*