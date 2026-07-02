# Original User Request

## 2026-06-06T13:44:39Z

Viết bản nháp chi tiết mang văn phong hàn lâm cho Chương 2 (Cơ Sở Lý Thuyết và Tổng Quan Tài Liệu) của Khóa luận Phát hiện Fake Review, hướng tới đối tượng người đọc là Giảng viên chấm thi.

Working directory: c:/Users/vanhi/Desktop/HCMUTE_TMDT/HKII_Nam_3/Bao_Mat_TMDT/Fake_reviews
Integrity mode: benchmark

## Requirements

### R1. Tuân thủ Cấu trúc Tuyệt đối (2.1 đến 2.11)
Phải bám sát chính xác 11 mục lục từ 2.1 đến 2.11 đã được định nghĩa. Không tự ý gộp, xóa hay thêm mục mới. 

### R2. Trích dẫn & Web Search (Bắt buộc)
Đội ngũ AI BẮT BUỘC sử dụng công cụ tìm kiếm web để tìm các bài báo khoa học thực tế nhằm làm nền tảng lý thuyết cho các công nghệ (ModernBERT, PCA, PSO, Focal Loss, Ensemble, XAI...). Tuyệt đối không sao chép nguyên văn khóa luận mẫu.

### R3. Nguyên tắc "Cái Gì" (WHAT) - Tránh lặp ý Chương 3
Chương 2 chỉ tập trung giải thích lý thuyết nền tảng (Công nghệ này là gì? Ưu nhược điểm?). Tuyệt đối KHÔNG mô tả việc công nghệ đó được áp dụng vào đồ án như thế nào (số liệu, cấu hình), vì phần đó thuộc về Chương 3. Viết thành đoạn văn nghị luận dài, phân tích sâu, không dùng gạch đầu dòng cộc lốc.

## Acceptance Criteria

### Xác minh bằng Agent-as-judge (Rubric)
- [ ] Báo cáo nháp phải chứa chính xác 11 tiêu đề mục (từ 2.1 đến 2.11).
- [ ] Báo cáo phải chứa ít nhất 5 trích dẫn khoa học có thật (tên tác giả, năm) thu thập được từ web, rải đều cho các phần công nghệ lõi.
- [ ] KHÔNG chứa các câu mô tả chi tiết thực nghiệm (ví dụ: "Trong đồ án này, chúng tôi cấu hình PCA 400 chiều" -> Câu này vi phạm R3).
- [ ] Không có đoạn văn nào chỉ bao gồm các gạch đầu dòng cụt lủn.
