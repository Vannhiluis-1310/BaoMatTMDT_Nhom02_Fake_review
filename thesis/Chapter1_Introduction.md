# CHƯƠNG 1: TỔNG QUAN VỀ ĐỀ TÀI NGHIÊN CỨU

**Đề tài:** Phát hiện đánh giá giả trên Amazon bằng pipeline hai nhánh ModernBERT–đặc trưng hành vi–ensemble học sâu và GBDT với chính sách ngưỡng kép

*Dual-Track ModernBERT, Behavioral Features, and Threshold-Selected Ensemble for Fake Review Detection*

---

## 1.1. Đặt vấn đề

### 1.1.1. Bối cảnh: đánh giá trực tuyến như hạ tầng tín nhiệm

Thương mại điện tử đã trở thành kênh phân phối chủ lực: người tiêu dùng không thể sờ, dùng thử hay kiểm định chất lượng sản phẩm trước khi thanh toán, nên phải dựa vào tín hiệu gián tiếp — đặc biệt là **đánh giá và xếp hạng sao** — để giảm rủi ro mua hàng. Ở quy mô toàn cầu, doanh thu bán lẻ trực tuyến đạt **4,4 nghìn tỷ USD** năm 2023 (tăng từ 1,3 nghìn tỷ USD năm 2014), chiếm **~20%** tổng bán lẻ; dự báo đạt **6,8 nghìn tỷ USD** và **~24%** vào năm 2028 (Forrester Research, 2024). Tại Hoa Kỳ — thị trường Amazon chiếm vị thế dẫn đầu — doanh thu bán lẻ trực tuyến theo quý đã vượt ngưỡng **300 tỷ USD** vào cuối 2024, chiếm khoảng **16–17%** tổng bán lẻ (U.S. Census Bureau, 2025).

**Bảng 1.0.** Quy mô thị trường thương mại điện tử (toàn cầu và Hoa Kỳ)

| Chỉ số | Giá trị | Năm / kỳ | Nguồn |
|--------|---------|----------|-------|
| Doanh thu TMDT toàn cầu | 4,4 nghìn tỷ USD | 2023 | Forrester Research (2024) |
| Tỷ trọng bán lẻ trực tuyến / tổng bán lẻ toàn cầu | ~20% | 2023 | Forrester Research (2024) |
| Dự báo doanh thu TMDT toàn cầu | 6,8 nghìn tỷ USD | 2028 | Forrester Research (2024) |
| Dự báo tỷ trọng trực tuyến | ~24% | 2028 | Forrester Research (2024) |
| Doanh thu bán lẻ trực tuyến Hoa Kỳ (theo quý) | >300 tỷ USD | Q4/2024 | U.S. Census Bureau (2025) |
| Tỷ trọng e-commerce / bán lẻ Hoa Kỳ | ~16–17% | Q4/2024 | U.S. Census Bureau (2025) |

Trên các nền tảng quy mô lớn như Amazon, mỗi quyết định “có nên mua hay không” thường được tổng hợp từ điểm trung bình, số lượng review, nội dung văn bản, nhãn *Verified Purchase* và thời điểm đăng tải. *Coalition for Trusted Reviews* ước tính đánh giá trực tuyến ảnh hưởng tới gần **4 nghìn tỷ USD** chi tiêu tiêu dùng toàn cầu mỗi năm (trích dẫn trong Amazon Staff, 2025) — cho thấy quy mô “kinh tế đánh giá” ngang tầm một ngành công nghiệp lớn.

Nghiên cứu kinh tế học chỉ ra rằng chênh lệch nhỏ trên thang điểm năm sao có thể chuyển hóa thành thay đổi đáng kể về doanh thu và sức cạnh tranh giữa người bán: trên Yelp, tăng **1 sao** trung bình gắn với tăng doanh thu khoảng **5–9%** (Luca & Zervas, 2016). Đánh giá vì thế không còn là phản hồi phụ, mà là **tài sản kinh tế** ảnh hưởng trực tiếp đến khả năng hiển thị sản phẩm (*visibility*), thuật toán gợi ý và niềm tin thương hiệu.

**Bảng 1.1.** Chỉ số kinh tế–hành vi liên quan đến đánh giá trực tuyến

| Chỉ số | Giá trị / mô tả | Ý nghĩa với đề tài | Nguồn |
|--------|-----------------|-------------------|-------|
| Chi tiêu bị ảnh hưởng bởi review | ~4 nghìn tỷ USD/năm (toàn cầu) | Gian lận tác động hàng loạt quyết định mua | Amazon Staff (2025); Coalition for Trusted Reviews (n.d.) |
| Tăng 1 sao trung bình → doanh thu | +5% đến +9% | Review giả làm méo tín hiệu kinh tế | Luca & Zervas (2016) |
| Tỷ lệ fake trên corpus thực nghiệm | ~40% (42.749 mẫu) | Dữ liệu mất cân bằng, không dùng Accuracy đơn thuần | Amazon Labeled Fake Reviews; §1.3 |
| Bài toán hình thành sớm trên Amazon | Từ 2008 | Corpus và baseline Tier A tái sử dụng được | Jindal & Liu (2008) |

Trong hành trình mua hàng điển hình — **nhận thức nhu cầu → tìm kiếm → so sánh → đánh giá rủi ro → thanh toán → phản hồi sau mua** — đánh giá trực tuyến đóng vai trò thay thế cho trải nghiệm trực tiếp tại cửa hàng. Người mua không chỉ đọc điểm sao trung bình mà còn đọc nội dung chi tiết, so sánh tỷ lệ *Verified Purchase*, quan sát phân bổ thời gian đăng tải và đối chiếu giữa các sản phẩm cạnh tranh. Khi bất kỳ tín hiệu nào trong chuỗi này bị thao túng, quyết định mua dựa trên thông tin không trung thực — dù thuật toán gợi ý hay giao diện người dùng vẫn hoạt động bình thường về mặt kỹ thuật.

Với đồ án thuộc lĩnh vực **bảo mật thương mại điện tử**, góc nhìn then chốt không nằm ở “điểm số mô hình” đơn thuần mà ở **tính toàn vẹn thông tin** (*information integrity*): khi tín hiệu phản hồi bị làm nhiễu có chủ đích, toàn bộ chuỗi giá trị từ so sánh sản phẩm → đánh giá rủi ro → quyết định mua bị đứt gãy ở bước trung gian. Người tiêu dùng mua sai kỳ vọng, tăng tỷ lệ hoàn trả và khiếu nại; người bán trung thực mất cơ hội cạnh tranh công bằng vì bị lấn át bởi đối thủ dùng review giả; nền tảng chịu chi phí kiểm duyệt, điều tra, xử lý tranh chấp và suy giảm niềm tin dài hạn. Từ góc nhìn an ninh thông tin, đánh giá giả là dạng **tấn công vào không gian dữ liệu công khai** mà cả thuật toán lẫn người dùng đều mặc định tin cậy — không phá hệ thống bằng exploit kỹ thuật truyền thống, mà bằng **thao túng nội dung** (*content manipulation*) ở tầng ứng dụng.

**Bảng 1.2.** Tác động của đánh giá giả lên các bên liên quan

| Bên liên quan | Rủi ro khi đánh giá bị làm nhiễu | Hệ quả dài hạn |
|---------------|----------------------------------|----------------|
| Người tiêu dùng | Mua sai kỳ vọng, lãng phí chi phí | Giảm niềm tin vào TMDT nói chung |
| Người bán trung thực | Mất visibility so với đối thủ spam | Rời nền tảng, thị trường kém lành mạnh |
| Nền tảng | Chi phí kiểm duyệt, khiếu nại, pháp lý | Suy giảm uy tín thương hiệu |
| Hệ sinh thái TMDT | Thông tin thị trường bị méo | Phân bổ nguồn lực sai, cạnh tranh không công bằng |

**Bảng 1.2a.** Vai trò của đánh giá trực tuyến trong hệ sinh thái TMDT

| Khía cạnh | Chức năng | Hệ quả khi bị gian lận |
|-----------|-----------|-------------------------|
| Tín nhiệm | Thay thế trải nghiệm trực tiếp | Suy giảm niềm tin toàn nền tảng |
| Cạnh tranh | Phân biệt chất lượng người bán | Người bán trung thực bị loại |
| Khám phá | Ảnh hưởng gợi ý / xếp hạng | Sản phẩm kém được đẩy lên top |
| Chi phí thông tin | Giảm chi phí tìm kiếm | Người mua chịu sai lệch, hoàn trả tăng |

*Nguồn: Luca và Zervas (2016); Jindal và Liu (2008).*

### 1.1.2. Thực trạng gian lận và áp lực pháp lý–vận hành

Bài toán *opinion spam* — đánh giá tạo ra có chủ đích, không phản ánh trải nghiệm thật — được Jindal và Liu (2008) hình thành sớm trên chính dữ liệu Amazon. Họ phân loại spam thành ba dạng: **quảng bá** (*promotional* — khen quá mức để đẩy sản phẩm), **phá hoại** (*demotional* — hạ uy tín đối thủ) và **không mang thông tin** (*non-review*). Hơn một thập kỷ sau, hình thức gian lận đã **công nghiệp hóa**: xuất hiện chuỗi cung ứng *review broker* (mua/bán review, hoàn tiền đổi sao 5), farm tài khoản, dịch vụ viết review theo gói và gần đây là văn bản do mô hình ngôn ngữ lớn (LLM) tạo sinh — ngữ phong tự nhiên, đa dạng cấu trúc câu, khó phân biệt bằng quy tắc tĩnh hay từ điển từ khóa.

Động lực kinh tế làm gian lận bền vững: lợi ích thao túng nhận thức (tăng doanh số, đẩy sản phẩm lên top tìm kiếm) thường lớn hơn chi phí rủi ro bị phát hiện, đặc biệt với seller mới hoặc sản phẩm cạnh tranh gay gắt (Luca & Zervas, 2016). Mỗi fake negative review có thể làm giảm doanh thu seller khoảng **0,5–1%** (Luca & Zervas, 2016). Khi chi tiêu bị ảnh hưởng bởi review lên tới hàng nghìn tỷ USD (Bảng 1.1), chi phí cơ hội của việc để spam lọt qua trở nên rất lớn.

Khái niệm **review broker** — tổ chức/cá nhân bán hoặc kích thích đánh giá giả đổi tiền, quà hoặc hoàn tiền (Coalition for Trusted Reviews, n.d.) — mô tả chuỗi cung ứng: người bán đặt hàng → broker phân phối → người viết đăng bài (đôi khi kèm đơn *Verified Purchase*) → thanh toán. Chuỗi này đã **công nghiệp hóa** và là động lực chính của gian lận hiện đại (Amazon Staff, 2025).

Phía phòng thủ, Amazon công bố đã **chặn hơn 275 triệu** đánh giá nghi ngờ giả trong năm 2024, kết hợp mô hình học máy, điều tra con người và kiện hàng trăm broker (Amazon Staff, 2025). Ở tầng pháp lý, FTC (Hoa Kỳ) ban hành **Trade Regulation Rule on Consumer Reviews and Testimonials** — cấm đánh giá giả, mua bán review, suppression và chứng thực không có căn cứ — có hiệu lực từ tháng 10/2024, kèm chế tài dân sự (Federal Trade Commission, 2024, 2025). Các động lực này cho thấy phát hiện đánh giá giả không chỉ là chủ đề học thuật mà là **nhu cầu vận hành và tuân thủ** ngày càng bắt buộc.

**Bảng 1.3.** Phản ứng hệ sinh thái đối với đánh giá giả (2023–2025)

| Tổ chức / cơ chế | Hành động chính | Số liệu nổi bật | Nguồn |
|-------------------|-----------------|-----------------|-------|
| Amazon | ML + điều tra + kiện broker | Chặn **>275 triệu** review nghi ngờ giả (2024); kiện **115** broker (2 năm); **>40** website broker ngừng hoạt động (2024) | Amazon Staff (2025) |
| Coalition for Trusted Reviews | Phối hợp ngành (thành lập 2023) | Amazon, Tripadvisor, Booking.com, Expedia, Trustpilot…; định nghĩa chung *review broker* | Amazon Staff (2025); Coalition for Trusted Reviews (n.d.) |
| FTC (Hoa Kỳ) | Quy định cấm review giả | Rule có hiệu lực **10/2024**; civil penalty cho mua/bán review, suppression | Federal Trade Commission (2024, 2025) |
| Xu hướng gian lận 2023–nay | LLM-generated review, farm tài khoản | Chi phí tạo spam giảm; chi phí kiểm duyệt tăng theo quy mô catalog | Gupta et al. (2024); Ren & Ji (2024) |

*Nguồn tổng hợp: Amazon Staff (2025); Federal Trade Commission (2024, 2025).*

**Bảng 1.4.** Timeline gian lận, phản ứng hệ sinh thái và paradigm phát hiện (2008–2026)

| Giai đoạn | Gian lận tiêu biểu | Phản ứng nền tảng / pháp lý | Paradigm FRD học thuật | Nguồn |
|-----------|-------------------|------------------------------|------------------------|-------|
| **2008–2012** | Spam thủ công; promotional/demotional | Chính sách nền tảng sơ khai | Feature + LR/SVM; đặt nền *opinion spam* | Jindal & Liu (2008); Ott et al. (2011) |
| **2013–2017** | Farm review, burst theo sản phẩm | Siết *Verified Purchase* | Behavioral metadata; graph NN | Mukherjee et al. (2013); Rayana & Akoglu (2015) |
| **2018–2022** | Review broker, hoàn tiền đổi sao 5 | Amazon tăng ML + kiện broker | CNN, BiLSTM, BERT embeddings | Bhuvaneshwari et al. (2021); Shah (2019) |
| **2023** | Broker công nghiệp hóa | **Coalition for Trusted Reviews** | Hybrid text + rating + aspect | Duma et al. (2023); Amazon Staff (2025) |
| **2024** | LLM-generated review; suppression | FTC Rule (10/2024); chặn **>275 triệu** review | Khảo sát G1–G8; multimodal | Federal Trade Commission (2024); Gupta et al. (2024) |
| **2024–2025** | Spam đa phương thức, đồ thị | Kiện **115** broker; **>40** site ngừng hoạt động | Graph FRD; survey reproducibility | Wu et al. (2024); Ren & Ji (2024) |
| **2026** | Spam tinh vi + yêu cầu audit | Bối cảnh đề tài | Dual-track ModernBERT + behavioral + GBDT | Chương 3–4 |

Ba xu hướng cấu trúc: **(1) Gian lận** thủ công → broker → LLM — chi phí tạo spam giảm; **(2) Phòng thủ** chính sách nội bộ → phối hợp ngành (2023) → pháp lý hóa (FTC 2024); **(3) FRD học thuật** đơn tín hiệu → hybrid → đa paradigm, nhưng báo cáo/tái lập chưa theo kịp (Gupta et al., 2024; Ren & Ji, 2024). Chi tiết lý thuyết và so sánh thuật toán: Chương 2, §2.2–§2.3.

Tuy nhiên, áp lực pháp lý và quy mô chặn spam không đồng nghĩa với bài toán đã được giải quyết. Nền tảng hiếm công bố tỷ lệ false positive/false negative theo chuẩn thống nhất; người bán và người mua vẫn khiếu nại về review bị gỡ nhầm hoặc spam lọt qua. Hệ thống phòng thủ do đó phải đáp ứng **hai ràng buộc đối lập**: (i) **recall** đủ cao để không bỏ sót quá nhiều spam trong luồng kiểm duyệt rộng; (ii) **precision** đủ cao ở chế độ tự động gắn cờ để không penalize khách hàng thật — vì false positive tương đương thiệt hại uy tín và có thể vi phạm chính sách nền tảng (Ren & Ji, 2024). Đây chính là lý do đề tài không chỉ tối ưu một metric duy nhất mà thiết kế **chính sách ngưỡng kép** (*dual-threshold*) phục vụ hai kịch bản vận hành khác nhau (§1.2.2, M1–M2).

### 1.1.3. Khoảng trống kỹ thuật và lý do chọn hướng đề tài

Song song với áp lực thị trường, cộng đồng nghiên cứu *fake review detection* (FRD) đã phát triển từ feature thủ công (Jindal & Liu, 2008; Ott et al., 2011) sang học sâu (CNN, BiLSTM), Transformer (BERT, ModernBERT) và gần đây là graph neural network (Wu et al., 2024) hay đa phương thức (Veluru et al., 2025). Tuy nhiên, tiến bộ mô hình chưa đồng bộ với tiêu chuẩn **báo cáo và tái lập** mà bối cảnh vận hành đòi hỏi. Hai khảo sát gần đây (Gupta et al., 2024; Ren & Ji, 2024) tổng hợp rằng phần lớn công trình vẫn gặp ba hạn chế hệ thống:

**Thứ nhất — đơn tín hiệu, thiếu tích hợp có kiểm soát.** Phần lớn công trình tập trung vào văn bản (TF-IDF, CNN, BERT) hoặc metadata hành vi, nhưng ít khi kết hợp **đồng thời** embedding Transformer hiện đại, đặc trưng hành vi engineered và hai góc nhìn mô hình (tabular + sequence) trên **cùng một protocol** fit/đánh giá.

**Thứ hai — metric báo cáo lệch nhu cầu triển khai.** Nhiều paper dùng Accuracy trên tập mất cân bằng hoặc chỉ báo một F1 duy nhất; trong khi nền tảng thương mại điện tử cần ít nhất hai chế độ: **cân bằng recall–precision** (kiểm duyệt rộng) và **ưu tiên precision lớp Fake** (tự động gắn cờ, chấp nhận bỏ sót hơn là khóa nhầm).

**Thứ ba — thiếu tái lập và ablation đầy đủ.** Hiếm công trình công bố rõ scaler/PCA/reducer fit từ đâu, ngưỡng quyết định chọn trên tập nào, và đóng góp từng thành phần (PCA, behavioral, ensemble) trên **cùng split** — dẫn đến so sánh SOTA khó kiểm chứng.

Amazon được chọn làm bối cảnh thực nghiệm vì: (i) hệ sinh thái đánh giá lớn, ảnh hưởng trực tiếp *visibility* và quyết định mua; (ii) *opinion spam* hình thành sớm trên dữ liệu Amazon (Jindal & Liu, 2008); (iii) corpus *Amazon Labeled Fake Reviews* (~42.749 mẫu, ~40% fake) tái sử dụng rộng — đối chiếu Tier A (Chương 2, Bảng 2.2); (iv) phù hợp **Bảo mật TMDT** — bảo vệ tính toàn vẹn dữ liệu phản hồi.

**Vấn đề cần giải quyết** được cụ thể hóa như sau: *Làm thế nào để xây dựng pipeline phát hiện đánh giá giả trên Amazon Labeled Fake Reviews, kết hợp ModernBERT, đặc trưng hành vi, học sâu chuỗi và ensemble GBDT, có chính sách ngưỡng kép phục vụ triển khai e-commerce, đồng thời đảm bảo tái lập và ablation có kiểm soát trong ràng buộc RAM ≤ 12GB?*

Hướng trả lời của đề tài là pipeline **hai nhánh** (*dual-track*): nhánh chính (*final track*) — raw 777-d + sequence + weighted blend + dual-threshold; nhánh ablation — PCA/PSO song song cho *negative result* và so sánh có trách nhiệm. Kiến trúc chi tiết tại Chương 3; bằng chứng số liệu tại Chương 4; cơ sở lý thuyết và 20 công trình tham chiếu tại Chương 2.

---

## 1.2. Mục tiêu nghiên cứu

### 1.2.1. Mục tiêu tổng quát

Xây dựng và đánh giá hệ thống phát hiện đánh giá giả trên *Amazon Labeled Fake Reviews*, trong ràng buộc Google Colab (RAM ≤ 12GB, GPU Tesla T4), với artifact đầy đủ phục vụ tái lập.

### 1.2.2. Mục tiêu cụ thể và cơ sở đặt ra

Mục tiêu tổng quát (§1.2.1) mô tả *điều cần làm*; mục tiêu cụ thể M1–M6 mô tả *điều cần đạt* và *cách biết đã đạt*. Việc đặt sáu mục tiêu thay vì một con số F1 duy nhất xuất phát từ nhận định ở §1.1: bài toán FRD trên TMDT vừa là bài toán **học máy trên dữ liệu mất cân bằng**, vừa là bài toán **vận hành có chi phí sai khác nhau giữa false positive và false negative**, vừa là bài toán **báo cáo khoa học cần audit được**.

Các mục tiêu cụ thể (M1–M6) không chọn ngẫu nhiên mà bám **ba trục**:

1. **Trục năng lực phân loại (M1–M3):** Corpus Amazon có ~40% fake — không cực đoan như spam email nhưng đủ mất cân bằng để Accuracy gây hiểu nhầm (Ott et al., 2011). Cần ít nhất một metric cân bằng hai lớp (M1), một metric ưu tiên an toàn khi auto-flag (M2) và một metric đo chất lượng xếp hạng xác suất trước khi chọn ngưỡng (M3) — định nghĩa tại Chương 3, §3.2.1.
2. **Trục vận hành e-commerce (M2, RQ5):** Hai chế độ τ balanced và τ precision-first không phải “tùy chọn thêm” mà phản ánh hai luồng moderation thực tế đã nêu ở §1.1.2.
3. **Trục phương pháp luận (M4–M6):** Gupta et al. (2024) và Ren & Ji (2024) chỉ ra các gap G4 (leakage), G7 (thiếu metric đa chiều), G8 (thiếu ablation) — M4–M6 là cam kết lấp các gap này *trước khi* claim kết quả.

**Nguyên tắc đặt ngưỡng số (0,89 / 0,975 / 0,93):** Đây là các **ngưỡng sàn** (*floor targets*) đặt ra **trước thực nghiệm**, không phải con số “chọn sau khi chạy xong”. Logic đặt như sau:

- **0,89 Macro F1:** Cao hơn pipeline legacy PCA+PSO của chính đề tài (Macro F1 ≈ 0,86 trên test) và nằm trong vùng các baseline CNN-BiLSTM/BERT trên Amazon (~0,82–0,93 tùy metric/paper — Chương 2, Bảng 2.2). Mức này đủ thách thức nhưng khả thi với ensemble dual-track trong RAM 12GB.
- **0,975 Precision Fake:** Tương đương FPR ≤ ~2,5% trên lớp Genuine nếu recall giữ ở mức hợp lý — phù hợp kịch bản “chỉ gắn cờ khi chắc chắn”. Con số này **cố ý cao** vì đại diện cho chi phí niềm tin khi khóa nhầm người dùng thật (Luca & Zervas, 2016; Ren & Ji, 2024).
- **0,93 ROC-AUC:** Đảm bảo mô hình có khả năng ranking tốt *độc lập* với τ; nếu AUC < 0,90, việc sweep τ trên validation dễ overfit mà không generalize sang test.

M4–M6 **không có ngưỡng F1 cố định** vì là mục tiêu **định tính có tiêu chí kiểm chứng**: protocol tái lập (M4), phạm vi so sánh Tier A (M5), và Δ Macro F1 có kiểm soát trong ablation (M6).

**Bảng 1.5.** Mục tiêu cụ thể, ngưỡng và lý do đặt ra

| Mã | Mục tiêu | Ngưỡng | Lý do đặt mục tiêu như vậy |
|----|----------|--------|----------------------------|
| **M1** | Phân loại cân bằng hai lớp | Macro F1 ≥ 0,89 | Dùng **Macro F1** thay Accuracy vì tập có tỷ lệ fake ~40% — Accuracy dễ “đẹp” khi thiên lớp đa số (Ott et al., 2011; Ren & Ji, 2024). Macro F1 trung bình đều F1 từng lớp, phản ánh khả năng bắt cả review giả lẫn giữ đúng review thật. Ngưỡng **0,89** đặt **cao hơn** nhiều baseline Tier A đã kiểm chứng trên Amazon (ví dụ vùng ~0,82–0,90 Accuracy/F1 trong các paper cũ) nhưng **khả thi** trong ràng buộc Colab 12GB — vừa là thước đo nội bộ đồ án, vừa là mốc so sánh có ý nghĩa với CNN-BiLSTM/BERT trên cùng corpus (Bhuvaneshwari et al., 2021; Refaeli & Hajek, 2021). |
| **M2** | Ưu tiên precision lớp Fake | Precision Fake ≥ 0,975 | Trên nền tảng, **false positive** (gắn nhãn giả cho review thật) gây thiệt hại trực tiếp: khách hàng hợp pháp bị penalize, uy tín seller bị tổn hại (Luca & Zervas, 2016). Chế độ *precision-first* phục vụ kịch bản **auto-flag** — chỉ đánh dấu khi mô hình rất chắc là fake. Gupta et al. (2024) (Gap **G7**) chỉ ra hiếm paper báo cáo đồng thời macro F1 balanced và precision cao cho lớp spam; ngưỡng **0,975** thể hiện cam kết “ưu tiên không khóa nhầm” ở mức vận hành nghiêm ngặt. |
| **M3** | Khả năng phân biệt xác suất | ROC-AUC ≥ 0,93 | ROC-AUC đo khả năng **xếp hạng** xác suất, **không phụ thuộc** một ngưỡng τ cố định — bổ sung cho M1/M2 vốn gắn với τ chọn trên validation. Ngưỡng **0,93** đảm bảo mô hình tách được hai lớp ở mức ranking tốt trước khi áp dụng dual-threshold; nếu AUC thấp, việc tinh chỉnh τ trên val không có ý nghĩa ổn định. |
| **M4** | Tái lập và kiểm toán | Seed 42; split 70/15/15; fit train-only; test audit 1 lần | Lấp Gap **G4** (Gupta et al., 2024): nhiều công trình không mô tả leakage (scaler/PCA fit trên toàn bộ data, test bị “nhìn” nhiều lần khi chỉnh τ). Đề tài **công khai** seed, tỷ lệ split, và quy tắc **test chỉ đánh giá một lần** sau khi đóng băng cấu hình — để giảng viên/bên thứ ba có thể audit hoặc rerun notebook. |
| **M5** | So sánh SOTA có trách nhiệm | Tier A text/tabular | Không claim “SOTA tuyệt đối” trên mọi paradigm: chỉ so **cùng loại bài toán** — phân loại text/tabular trên Amazon — với 20 papers đã verify (Chương 2, Bảng 2.2). Graph (Wu et al., 2024) và multimodal (Veluru et al., 2025) chỉ là bối cảnh Tier B — tránh so sánh số sai metric/dataset (Gupta et al., 2024). |
| **M6** | Ablation có kiểm soát | Δ Macro F1 theo thành phần | Lấp Gap **G8**: novelty cần **bằng chứng định lượng** — PCA có thực sự cần trên fused 777-d? Behavioral đóng góp bao nhiêu khi đã có ModernBERT? Ensemble có vượt single model? PSO có justify chi phí? Ablation Models A–E trên **cùng split** trả lời các câu hỏi này, tránh claim “hybrid tốt hơn” không có số liệu hỗ trợ. |

**Ánh xạ M1–M6 ↔ RQ ↔ Gaps:** Mỗi mục tiêu trả lời một “câu hỏi kiểm chứng” riêng, tránh trùng lặp hoặc mơ hồ:

| Nhóm | Mục tiêu | Câu hỏi kiểm chứng | RQ liên quan | Gap lý thuyết |
|------|----------|-------------------|--------------|---------------|
| Năng lực | M1, M2, M3 | “Mô hình có đủ tốt ở cả hai chế độ vận hành và ranking không?” | RQ2, RQ4, RQ5 | G7 |
| Tin cậy | M4 | “Kết quả có audit được, không leakage không?” | (toàn pipeline) | G4 |
| Vị trí | M5 | “So với Tier A cùng corpus, đứng ở đâu?” | RQ2 | G1–G3 |
| Đóng góp | M6 | “Thành phần nào thực sự mang lại Δ, thành phần nào có thể bỏ?” | RQ1, RQ3, RQ6 | G8 |

Tóm lại, **M1–M3** đo *năng lực mô hình*; **M4** đo *độ tin cậy quy trình*; **M5** đo *vị trí tương đối trong tài liệu*; **M6** đo *đóng góp từng thành phần*. Bốn nhóm này khớp với câu hỏi nghiên cứu RQ1–RQ6 (§1.2.3) và được kiểm chứng tại Chương 4. Việc đặt mục tiêu **trước** khi chạy Phase 7 (audit `phase7_target_audit.csv`) đảm bảo đồ án không “điều chỉnh mục tiêu theo kết quả” (*post-hoc goal shifting*) — một lỗi phương pháp luận mà Gupta et al. (2024) cảnh báo trong khảo sát reproducibility.

### 1.2.3. Câu hỏi nghiên cứu

- **RQ1:** Fusion ModernBERT + behavioral có cải thiện hiệu năng so với từng nhánh đơn lẻ không?
- **RQ2:** Kiến trúc dual-track có vượt các mô hình đơn nhánh đã báo cáo trên Amazon không?
- **RQ3:** PCA trên vector fused có còn phù hợp so với raw 777-d không?
- **RQ4:** Weighted ensemble có vượt từng base model không?
- **RQ5:** Chính sách ngưỡng kép có đạt đồng thời các target M1–M3 không?
- **RQ6:** Đóng góp tương đối của từng thành phần là bao nhiêu (ablation)?

---

## 1.3. Đối tượng và phạm vi nghiên cứu

**Đối tượng:** Các mô hình và kỹ thuật phân loại nhị phân (fake / genuine) cho đánh giá sản phẩm trên thương mại điện tử.

**Phạm vi dữ liệu:** *Amazon Labeled Fake Reviews* — 42.749 mẫu sau tiền xử lý; tiếng Anh; tỷ lệ fake ~40%; stratified split 70/15/15 (seed = 42).

**Phạm vi kỹ thuật:** ModernBERT (feature extractor), 9 đặc trưng hành vi, CNN-BiLSTM-Attention, XGBoost/LightGBM, weighted ensemble, dual-threshold; ablation PCA/PSO; SHAP/LIME trên subset.

**Giới hạn:** Không triển khai graph NN quy mô lớn, fine-tune Transformer end-to-end, đa phương thức (ảnh + text), cross-dataset hay ứng dụng production realtime — các hướng này chỉ dùng làm bối cảnh tham chiếu (Chương 2, §2.4; so sánh số tại Chương 4).

---

## 1.4. Phương pháp nghiên cứu

Nghiên cứu theo hướng **thực nghiệm định lượng** kết hợp **ablation có kiểm soát**, bám khuyến nghị reproducibility của Gupta et al. (2024) và Ren & Ji (2024). Mọi scaler, reducer, grid blend và ngưỡng τ chỉ fit/học từ tập train; validation dùng để chọn cấu hình; tập test (n = 6.413) chỉ đánh giá **một lần** sau khi đóng băng.

Pipeline gồm: tiền xử lý → trích xuất ModernBERT + behavioral → huấn luyện dual-track (tabular GBDT + sequence DL) → ensemble và chọn ngưỡng → đánh giá, XAI và ablation. Nhánh ablation (PCA, PSO) chạy song song phục vụ so sánh, không thay thế nhánh chính ở inference.

*Sơ đồ chi tiết, hyperparameter và artifact từng phase: Chương 3.*

---

## 1.5. Ý nghĩa và cấu trúc luận văn

### 1.5.1. Ý nghĩa nghiên cứu

**Ý nghĩa về mặt lý luận và phương pháp luận**

Đề tài đóng góp theo hướng **phòng thủ có kiểm chứng** (*verifiable defense*) trong FRD: không dừng ở việc báo cáo một con số F1 cao, mà cung cấp **chuỗi bằng chứng** có thể truy vết từng bước. Cụ thể, bốn đóng góp phương pháp luận có thể liệt kê:

1. **Protocol leakage-controlled (M4):** Mọi scaler, PCA reducer, grid trọng số blend và ngưỡng τ chỉ học từ train; validation chỉ dùng chọn cấu hình; test (n = 6.413) chỉ audit một lần. Metadata từng phase được lưu dạng JSON/CSV — cho phép bên thứ ba rerun hoặc phát hiện leakage mà không cần đọc toàn bộ notebook.
2. **Metric đa chiều gắn kịch bản triển khai (M1–M3, M2):** Báo cáo đồng thời Macro F1 balanced, Precision Fake precision-first và ROC-AUC — phản hồi Gap G7 (Gupta et al., 2024): hiếm paper FRD báo cáo cả “hiệu năng tổng thể” lẫn “an toàn auto-flag”.
3. **Dual-track có negative result (M6, RQ3):** Nhánh ablation PCA/PSO chạy song song nhánh chính raw 777-d không phải để “thêm cho nhiều” mà để **kiểm chứng giả thuyết** Shah (2019) trên vector fused BERT+behavioral. Kết quả âm tính (PCA thua raw) có giá trị học thuật ngang kết quả dương tính — tránh narrative một chiều.
4. **So sánh SOTA có phạm vi (M5):** Chỉ claim vị trí tương đối trong Tier A (text/tabular, Amazon); graph và multimodal là bối cảnh Tier B — giảm nguy cơ so sánh sai metric/dataset (Ren & Ji, 2024).

Về mặt kiến trúc, pipeline **dual-track** (final + ablation) cho phép vừa triển khai đường chính tối ưu cho điều kiện RAM hạn chế, vừa tách biệt rõ “đường inference production” và “đường thí nghiệm so sánh” — mô hình mà các hệ thống ML vận hành thực tế cũng áp dụng (canary/ablation path) nhưng ít được ghi nhận đầy đủ trong đồ án tốt nghiệp.

**Ý nghĩa về mặt thực tiễn**

Trong bối cảnh FTC (2024) siết chặt đánh giá giả và nền tảng như Amazon đầu tư hàng trăm triệu lượt chặn spam mỗi năm (Amazon Staff, 2025), đề tài minh chứng một pipeline **có thể triển khai trên hạ tầng phổ thông** (Colab 12GB, GPU Tesla T4) mà vẫn đáp ứng hai nhu cầu vận hành song song:

- **Moderation cân bằng** (τ balanced, M1): duy trì recall hợp lý để không bỏ sót quá nhiều spam trong luồng kiểm duyệt rộng — tương tự hàng đợi (*queue*) cần human reviewer xem xét.
- **Auto-flag an toàn** (τ precision-first, M2): chỉ gắn cờ khi chắc chắn cao, giảm rủi ro khóa nhầm người bán/người mua thật — phù hợp logic kinh tế học hành vi về chi phí niềm tin (Luca & Zervas, 2016).

**Bảng 1.6.** Ý nghĩa thực tiễn theo bên liên quan

| Bên liên quan | Lợi ích cụ thể từ đề tài | Giới hạn cần nhận thức |
|---------------|--------------------------|------------------------|
| Nền tảng TMDT | Dual-threshold map trực tiếp hai luồng moderation; artifact audit được | Chưa triển khai realtime/production; một corpus tiếng Anh |
| Người bán trung thực | Giảm spam đối thủ; precision-first hạn chế false accusation | Không thay thế khiếu nại thủ công |
| Người tiêu dùng | Thông tin review đáng tin hơn qua lớp lọc | Không đảm bảo 100% spam-free |
| Nhóm R&D / SV | Mẫu pipeline 8 phase có notebook + JSON tái sử dụng | Ràng buộc RAM 12GB; chưa cross-dataset |

Đối với sinh viên và nhóm nghiên cứu bảo mật TMDT, đề tài còn là **mẫu quy trình** từ dữ liệu gán nhãn → feature engineering → dual-track training → weighted ensemble → threshold audit → XAI (SHAP/LIME) — mỗi bước có artifact đầu ra (`phase7_final_metrics.csv`, `phase7_ablation_results.csv`, …) phục vụ báo cáo, bảo vệ đồ án hoặc mở rộng sang corpus khác.

**Ý nghĩa đối với môn học Bảo mật Thương mại điện tử**

Đề tài liên kết trực tiếp chủ đề **toàn vẹn dữ liệu và chống gian lận trên nền tảng số**: đánh giá giả là dạng tấn công/manipulation vào không gian dữ liệu công khai mà thuật toán và người dùng cùng tin cậy — không cần exploit lỗ hổng phần mềm truyền thống vẫn gây thiệt hại kinh tế–xã hội. Pipeline đề xuất là lớp **phát hiện và giảm thiểu** (*detect & mitigate*) ở tầng dữ liệu, nằm trong mô hình phòng thủ nhiều lớp (*defense-in-depth*):

- **Lớp chính sách–pháp lý:** FTC Rule (2024), kiện broker (Amazon Staff, 2025)
- **Lớp vận hành nền tảng:** ML + human investigation
- **Lớp nghiên cứu/đồ án (đề tài):** Pipeline có audit, ablation, giải thích (XAI)

Đề tài **không** claim thay thế hoàn toàn hệ thống của Amazon hay đạt SOTA trên mọi paradigm (graph, multimodal); ý nghĩa nằm ở chứng minh **có thể** xây dựng phòng thủ có kiểm chứng trong ràng buộc tài nguyên sinh viên, với metric và protocol đủ chặt để hội đồng và độc giả đánh giá được mức độ tin cậy của kết quả (Chương 5 thảo luận hạn chế chi tiết).

### 1.5.2. Cấu trúc luận văn

| Chương | Nội dung | Liên hệ với mục tiêu |
|--------|----------|----------------------|
| **1** | Tổng quan đề tài | Đặt vấn đề, M1–M6, phạm vi |
| **2** | Cơ sở lý thuyết: §2.1 bài toán → §2.2 tài liệu & gaps → §2.3 lý thuyết chọn thành phần → §2.4 tính mới | Cơ sở cho M5; gaps G1–G8 (Bảng 2.3–2.4); metric/protocol → Ch. 3 §3.2 |
| **3** | Phương pháp — **logic luồng**: §3.1 (Hình 3.1–3.3) → §3.2 protocol *tại sao* → §3.12–3.13 môi trường + khung D0–D8 | Triển khai + số liệu → Ch. 4 §4.1–4.15 |
| **4** | Triển khai thực nghiệm (§4.1–4.6) + kết quả, SOTA, ablation (§4.7–4.15) | Kiểm chứng M1–M6; tự chấm §4.14 |
| **5** | Thảo luận, hạn chế, trả lời RQ | Diễn giải ý nghĩa M1–M6 |
| **6** | Kết luận | Tổng kết đóng góp |

*Bối cảnh thị trường và timeline: §1.1 (Bảng 1.0–1.4). Lý thuyết thuật toán: Chương 2. Metric và phương pháp: Chương 3.*