import os
from datetime import datetime

file_path = r"c:\Users\vanhi\Desktop\HCMUTE_TMDT\HKII_Nam_3\Bao_Mat_TMDT\Fake_reviews\.agents\ORIGINAL_REQUEST.md"
new_request = """
## 2026-06-06T14:20:00Z

Viết bản nháp chi tiết mang văn phong hàn lâm cho Chương 3 (Phương Pháp Nghiên Cứu) của Khóa luận Phát hiện Fake Review, hướng tới đối tượng người đọc là Giảng viên chấm thi.

Working directory: c:/Users/vanhi/Desktop/HCMUTE_TMDT/HKII_Nam_3/Bao_Mat_TMDT/Fake_reviews
Integrity mode: benchmark

## Requirements

### R1. Tuân thủ Cấu trúc Tuyệt đối (3.1 đến 3.11)
Phải bám sát chính xác 11 mục lục từ 3.1 đến 3.11 đã được định nghĩa. Không tự ý gộp, xóa hay thêm mục mới. 

### R2. Nguyên tắc "Như thế nào" (HOW) - Tránh lặp ý Chương 2
Tuyệt đối KHÔNG nhắc lại định nghĩa lý thuyết (ví dụ không giải thích PCA là gì nữa). Chỉ tập trung mô tả chi tiết cách hệ thống vận hành.

### R3. Chi tiết Dữ liệu & Kiến trúc cốt lõi (BẮT BUỘC ĐƯA VÀO BÀI)
Nhóm viết bài bắt buộc phải lồng ghép các số liệu và quyết định kỹ thuật sau vào đúng các mục tương ứng:
- **Dữ liệu:** Tập Amazon Labeled Fake Reviews, cân bằng 50/50 (CG vs OR).
- **ModernBERT:** Được dùng để xử lý văn bản dài, kết hợp với **9 đặc trưng hành vi** tạo ra không gian **777 chiều**.
- **PCA:** Ép không gian 777 chiều xuống **400 chiều** (giữ 96.19% phương sai) nhằm giải quyết bài toán tràn bộ nhớ vì giới hạn phần cứng cứng ngắc là **12GB RAM**.
- **PSO:** Chỉ chạy trên **20% subset** (tập con) để tối ưu hóa 12 siêu tham số liên tục/rời rạc, tránh tràn RAM.
- **Ensemble:** Sử dụng Stacking meta-learner với trọng số thực tế là DL (0.1) - XGB (0.0) - LGBM (0.9). Giải thích tại sao LGBM áp đảo (do PCA biến dữ liệu thành dạng bảng tabular).
- **Ngưỡng (Threshold):** Điều chỉnh từ 0.5 (dùng cho Flagging kiểm duyệt thủ công) lên **0.79** (dùng cho Auto-Removal tự động xóa) nhằm ép Precision Fake lên **96.04%** (sai số khóa nhầm chỉ 1.5%).

## Acceptance Criteria

### Xác minh bằng Agent-as-judge (Rubric)
- [ ] Báo cáo nháp phải chứa chính xác 11 tiêu đề mục (từ 3.1 đến 3.11).
- [ ] KHÔNG chứa các câu định nghĩa lý thuyết hàn lâm thuần túy (ví dụ: "Focal loss là một hàm mất mát do Lin et al...").
- [ ] Bắt buộc phải xuất hiện chính xác các con số: "777 chiều", "400 chiều", "96.19%", "12GB RAM", "subset 20%", "trọng số LGBM 0.9", "Precision 96.04%".
- [ ] Mọi luận điểm thiết kế phải nhất quán với `Fake_Review_Deep_Dive.md`.
"""

if os.path.exists(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    if "Chương 3 (Phương Pháp Nghiên Cứu)" not in content:
        with open(file_path, "a", encoding="utf-8") as f:
            f.write("\n" + new_request)
else:
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("# Original User Request\n" + new_request)
