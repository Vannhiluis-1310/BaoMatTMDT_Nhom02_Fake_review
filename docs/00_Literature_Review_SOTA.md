# Literature Review & SOTA Comparison (2026 refresh)

**Mục đích:** File nguồn (single source of truth) cho 20 tài liệu tham khảo, bảng SOTA, và khoảng trống nghiên cứu.  
**Artifact đồng bộ:** `reports/tables/literature_references_20.csv`, `reports/tables/literature_sota_comparison.csv`  
**Cập nhật:** 2026-06-10 | Pipeline chính: ModernBERT raw 777 + CNN-BiLSTM sequence + weighted ensemble (2026-06-09)

---

## 1. Kiểm tra tình trạng docs cũ

| File | Vấn đề | Hành động | Trạng thái |
|------|--------|-----------|------------|
| `docs/01_Tong_Quan_Du_An.md` | Abstract/SOTA ghi 0.856; §9 cũ | §6.0 + §9 mới; banner legacy | ✅ 2026-06-10 |
| `docs/06_Phase5_Hybrid_Ensemble.md` | Metrics pipeline 01/06 | Leaderboard 09/06 | ✅ |
| `docs/07_Phase6-8_*.md`, `docs/08_Ket_Luan_*.md` | Target gap 0.856 | Số Phase 7 CSV + hoàn tất 06–08 | ✅ 2026-06-10 |
| `thesis/Chapter1–6` | Narrative PCA+PSO là đường chính | Tách final vs ablation | ✅ 2026-06-10 |
| `docs/02–05` | Một phần đúng methodology | Giữ Phase facts | ✅ (methodology) |

### Papers bị loại khỏi bảng cũ (chưa verify đủ)

Các dòng sau trong `01_Tong_Quan` **không đưa vào bảng 20 bài** cho đến khi trích được DOI + metric từ PDF gốc:

- Geetha et al. (2025) DeBERTa+MBO F1=0.90  
- Liu et al. (2023) RoBERTa+augmentation F1=0.88  
- Liu et al. (2024) RoBERTa+behavioral F1=0.89  
- Khan et al. (2025), Jain et al. (2025), Zhu et al. (2025), IIETA (2025), Gupta et al. (2025) swarm  

---

## 2. 20 tài liệu tham khảo (đã kiểm chứng)

| ID | Tác giả (Năm) | Dataset chính | Metric báo cáo | Điểm nổi bật |
|----|---------------|---------------|----------------|--------------|
| 1 | Jindal & Liu (2008) | Amazon | Accuracy ~0.78 (LR) | Nền tảng opinion spam |
| 2 | Ott et al. (2011) | OpSpam 400 | F1 ~0.90 | Gold corpus nhỏ |
| 3 | Mukherjee et al. (2013) | Yelp | Acc 67.8% (SVM) | Behavioral pioneer |
| 4 | Rayana & Akoglu (2015) | Yelp graph | F1 ~0.85+ | Graph collective |
| 5 | Wang et al. (2017) | Yelp | F1 > baselines | Joint embedding |
| 6 | Kennedy et al. (2019) | OpSpam | Acc > ML | Contextualized embeddings |
| 7 | Shah (2019) | Amazon | Acc ~82% | PCA + Active Learning |
| 8 | Hajek et al. (2020) | Amazon+Yelp | F1/Acc cao | DL + emotion embeddings |
| 9 | Vidanagama et al. (2020) | Yelp+Amazon | Acc 92.7/97.3% | CNN; metric là Accuracy |
| 10 | Zhang et al. (2020) | Yelp | F1 > RF/SVM | Neural autoencoder forest |
| 11 | Gupta (2021) | **Yelp 1.4M** | **Weighted-F1 0.69** | RoBERTa; **không phải Amazon 0.86** |
| 12 | Bhuvaneshwari et al. (2021) | Amazon | F1/Acc >0.90 | CNN-BiLSTM-Attention |
| 13 | Refaeli & Hajek (2021) | Multi-domain | F1 competitive | Fine-tuned BERT |
| 14 | Deshai & Rao (2023) | Multi incl. Amazon | Acc up to 99.4%* | CNN+APSO (*multi-dataset) |
| 15 | Duma et al. (2023) | Amazon aspects | F1 strong | Text+ratings+aspects |
| 16 | Mir et al. (2023) | General reviews | Acc 87.81% | SVM+BERT |
| 17 | Gupta et al. (2024) | Survey 98 papers | — | Gap taxonomy 2019–2023 |
| 18 | Ren & Ji (2019) | Survey | — | FRD principles & reproducibility |
| 19 | Veluru et al. (2025) | Multimodal 20k | **F1 0.934** | BERT+ResNet; khác dataset |
| 20 | Wu DOS-GNN line (2024–25) | Graph Yelp/Amazon | F1 ~0.915 | Graph tier — non-comparable |

**Ours (2026):** Amazon Labeled 42.749 | split 70/15/15 seed 42 | train-only fit  
- Macro F1 **0.9433** @τ=0.5 | **0.9463** @τ=0.3 (balanced) | Precision Fake **0.9816** @τ=0.6  
- ROC-AUC **0.9769** | CNN-BiLSTM single **0.9324** | Multi-seed balanced **0.9485 ± 0.0018**

---

## 3. Bảng SOTA (3 tầng so sánh)

### Tier A — Text/tabular classification (so sánh trực tiếp với đề tài)

| Method | Dataset | Metric | Score | Ghi chú |
|--------|---------|--------|-------|---------|
| **Ours weighted blend (balanced)** | Amazon 42.7k | Macro F1 | **0.9463** | τ=0.3, val-select; mean 0.9485±0.0018 (3 seed) |
| **Ours weighted blend (default)** | Amazon 42.7k | Macro F1 | **0.9433** | τ=0.5 |
| **Ours CNN-BiLSTM sequence** | Amazon 42.7k | Macro F1 | **0.9324** | Single DL |
| Veluru et al. (2025) multimodal | Custom 20k | F1 | 0.934 | Text+image |
| XGB/LGBM raw 777 (ours baseline) | Amazon 42.7k | Macro F1 | ~0.906 | Tabular |
| Gupta (2021) RoBERTa | Yelp 1.4M | Weighted-F1 | 0.69 | Khác dataset & metric |
| Mir (2023) SVM+BERT | General | Accuracy | 0.8781 | Không cùng corpus |

### Tier B — Graph / collective spam (tham khảo, không claim beat trực tiếp)

| Method | Score | Lý do khác biệt |
|--------|-------|-----------------|
| Rayana & Akoglu (2015) | F1 ~0.85+ | Review graph + metadata |
| DOS-GNN / Wu line (2024–25) | F1 ~0.915 | Heterogeneous graph features |

### Tier C — Foundational / legacy

| Method | Score | Vai trò trong luận văn |
|--------|-------|----------------------|
| Ott (2011) | F1 ~0.89 | Gold 400 — historical |
| Mukherjee (2013) | Acc 67.8% | Behavioral motivation |
| **Ours legacy PCA+PSO** (01/06) | Macro F1 0.856 | Ablation / negative result PCA track |

---

## 4. Khoảng trống nghiên cứu (8 gaps) — đề tài giải quyết

| Gap | Bằng chứng từ 20 papers | Giải pháp của đề tài |
|-----|-------------------------|----------------------|
| **G1** Text-only transformer chiếm đa số; ít kết hợp behavioral engineered | Chỉ Mukherjee/Rayana/Duma có behavioral; Refaeli/Gupta/Mir text-only | ModernBERT 768 + 9 behavioral → 777-d |
| **G2** Thiếu dual-track: tabular embeddings + sequence DL | Bhuvaneshwari có CNN-BiLSTM; ít paper có cả tree + sequence | 05_01–05_04 song song; ensemble 05_05 |
| **G3** Ensemble đa họ shallow+deep ít, thiếu protocol chọn threshold | Zhang (2020) ensemble; hầu hết 1 model | Weighted blend + val sweep + 2 chế độ (balanced / precision-first) |
| **G4** Thiếu audit leakage / fit policy công khai | Hầu hết papers không mô tả train-only PCA/scaler | Seed 42, stratified 70/15/15, train-only fit, metadata JSON |
| **G5** PCA trên fused vector không được chứng minh là optimal | Shah (2019) PCA text; pipeline cũ PCA 777→400 thua raw | Ablation: raw 777 thắng PCA track (DL-PSO test 0.779) |
| **G6** Swarm opt (PSO/APSO) chưa gắn full hybrid stack | Deshai APSO+CNN; không có BERT+behavioral+ensemble | PSO giữ ở Phase 4 ablation; final dùng grid blend |
| **G7** Precision-first cho e-commerce hiếm khi báo cùng macro F1 | Papers báo accuracy/F1 đơn | Precision Fake **0.9816** @τ=0.6; mean 0.9763±0.0029 (3 seed) |
| **G8** Ablation định lượng thiếu trên cùng split | 0/20 papers có full ablation 5 thành phần | Phase 7 controlled ablation + delta table |

---

## 5. Cách trích dẫn SOTA trong luận văn (bắt buộc)

1. Luôn nêu **metric name** (Macro F1 vs Weighted-F1 vs Accuracy).  
2. Luôn nêu **dataset + n + split** trước khi so sánh số.  
3. Claim “vượt SOTA” chỉ trên **Tier A** cùng bài toán text classification Amazon.  
4. Tier B (graph) và Tier C (foundational) chỉ dùng làm **bối cảnh**, không so sánh số trực tiếp.  
5. Headline đề xuất: **Macro F1 0.9463** (balanced; mean 0.9485±0.0018) + **Precision Fake 0.9816** (e-commerce mode).

---

## 6. Tài liệu tham khảo (APA rút gọn)

1. Jindal, N., & Liu, B. (2008). Opinion spam and analysis. WSDM.  
2. Ott, M., et al. (2011). Finding deceptive opinion spam. ACL.  
3. Mukherjee, A., et al. (2013). Spotting fake reviewer groups. WWW.  
4. Rayana, S., & Akoglu, L. (2015). Collective opinion spam detection. KDD.  
5. Wang, Y., et al. (2017). Handling cold-start in review spam. ACL.  
6. Kennedy, S., et al. (2019). Fact or Factitious? ACL SRW.  
7. Shah, F. (2019). PCA and Active Learning for fake reviews. IJCA.  
8. Hajek, P., et al. (2020). Fake consumer review detection with DNN. NC&A.  
9. Vidanagama, D., et al. (2020). CNN fake review detection. IEEE Access.  
10. Zhang, Y., et al. (2020). Neural autoencoder decision forest. WSDM.  
11. Gupta, P. (2021). BERT family for fake reviews (MSc thesis). NCI.  
12. Bhuvaneshwari, P., et al. (2021). Self-attention CNN-BiLSTM. MTAP.  
13. Refaeli, O., & Hajek, P. (2021). Fine-tuned BERT fake reviews. ICAI/ACM.  
14. Deshai, N., & Rao, B. B. (2023). CNN + adaptive PSO. Soft Computing.  
15. Duma, R. A., et al. (2023). Deep hybrid text+ratings+aspects. Soft Computing.  
16. Mir, A. Q., et al. (2023). SVM + BERT fake reviews. arXiv.  
17. Gupta, R., et al. (2024). Comprehensive FRD review. Knowl. Eng. Rev.  
18. Ren, Y., & Ji, D. (2019). Learning to detect deceptive opinion spam: A survey. *IEEE Access*, *7*, 42934–42945. https://doi.org/10.1109/ACCESS.2019.2908495  
19. Veluru, S. R., et al. (2025). BERT + ResNet-50 multimodal. arXiv.  
20. Wu, F., et al. (2024–25). DOS-GNN graph spam detection. (graph line)