"""Rebuild expandChapter.md: Ch2 + restructured Ch3/Ch4 per agreed outline."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CH2_SRC = ROOT / "Chapter2_Theory.md"
EXPAND = ROOT / "expandChapter.md"
THESIS = ROOT / "Thesis_Full.md"
OUT = EXPAND

PART1 = re.compile(r"^#### \(1\)", re.M)
PART2 = re.compile(r"^#### \(2\)", re.M)
PART3 = re.compile(r"^#### \(3\)", re.M)
SEC23 = re.compile(r"^### 2\.3\.(\d+)\.", re.M)


def split_23_subsections(text: str):
    """Return list of (num, title, part1, part2, part3)."""
    start = text.find("## 2.3.")
    end = text.find("## 2.4.")
    block = text[start:end]
    chunks = SEC23.split(block)
    # chunks[0] is header preamble
    results = []
    it = iter(chunks[1:])
    for num_s, rest in zip(it, it):
        num = int(num_s)
        lines = rest.strip().split("\n")
        title = lines[0].strip()
        body = "\n".join(lines[1:]).strip()
        p1 = p2 = p3 = ""
        m2 = PART2.search(body)
        m3 = PART3.search(body)
        m1 = PART1.search(body)
        if m1:
            if m2:
                p1 = body[m1.end() : m2.start()].strip()
            elif m3:
                p1 = body[m1.end() : m3.start()].strip()
            else:
                p1 = body[m1.end() :].strip()
        if m2:
            if m3:
                p2 = body[m2.end() : m3.start()].strip()
            else:
                p2 = body[m2.end() :].strip()
        if m3:
            p3 = body[m3.end() :].strip()
        p3 = re.sub(r"\n---\s*$", "", p3).strip()
        results.append((num, title, p1, p2, p3))
    return results


def build_ch2_foundation(subs):
    lines = [
        "## 2.3. Nền tảng thuật toán liên quan",
        "",
        "*Phần này trình bày **định nghĩa và cơ chế** các họ thuật toán dùng trong pipeline — không biện luận lựa chọn cụ thể của đề tài. So sánh với tài liệu và kết luận chọn thành phần (neo G1–G8, EDA): **Chương 3, §3.4**. Chi tiết triển khai: §3.5–3.6.*",
        "",
    ]
    for num, title, p1, _p2, _p3 in subs:
        lines.append(f"### 2.3.{num}. {title}")
        lines.append("")
        if p1:
            lines.append(p1)
        lines.append("")
        if num < len(subs):
            lines.append("---")
            lines.append("")
    return "\n".join(lines)


def build_ch3_argument(subs):
    titles = {
        1: "Khung tổng thể: dual-track và hai không gian tín hiệu",
        2: "Biểu diễn văn bản: ModernBERT freeze",
        3: "Đặc trưng hành vi: 9 feature và fusion",
        4: "Nhánh tabular: XGBoost và LightGBM",
        5: "Nhánh sequence: CNN-BiLSTM-Attention và Focal Loss",
        6: "Tổng hợp dự đoán: weighted blend",
        7: "Ablation: PCA và PSO (track phụ)",
        8: "XAI và đánh giá độ bền",
    }
    lines = [
        "## 3.4. Biện luận lựa chọn thành phần pipeline",
        "",
        "Sau EDA và Bảng 3.5 (§3.3), mục này trả lời **vì sao đề tài chọn** từng thành phần — so sánh với Bảng 2.1–2.2, ánh xạ Gap Bảng 2.4, và bằng chứng qualitative từ EDA. Nền tảng thuật toán (cơ chế, công thức): **Ch.2 §2.3**. Kiến trúc vận hành và sơ đồ: **§3.5**.",
        "",
    ]
    for num, title, _p1, p2, p3 in subs:
        t = titles.get(num, title)
        lines.append(f"### 3.4.{num}. {t}")
        lines.append("")
        if p2:
            lines.append("#### So sánh với tài liệu và phương án thay thế")
            lines.append("")
            lines.append(p2)
            lines.append("")
        if p3:
            lines.append("#### Kết luận lựa chọn")
            lines.append("")
            # cross-ref updates inside part3
            p3 = p3.replace("§2.3.1", "§2.3.1 và §3.4.1")
            p3 = p3.replace("(§2.3.1)", "(§2.3.1; biện luận §3.4.1)")
            p3 = p3.replace("§2.3.3", "§2.3.3; EDA Bảng 3.5")
            p3 = p3.replace("Chương 3", "Chương 3 §3.5–3.6")
            lines.append(p3)
            lines.append("")
        if num == 3:
            lines.append(
                "*Bằng chứng EDA:* median độ dài fake 7 từ / 43 ký tự; verified fake 74,02% — justify khối basic và `basic_verified_purchase` (Bảng 3.5, §4.1). *Ablation thực tế:* advanced chỉ +0,0023 Macro F1 (§4.10) — không overclaim EDA checklist.*"
            )
            lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def patch_ch2_preamble(ch2: str) -> str:
    old_intro = (
        "*Chương này trả lời ba câu hỏi học thuật: (1) Bài toán FRD được định nghĩa và phân loại thế nào? "
        "(2) Tài liệu hiện có đi đến đâu, còn thiếu gì? (3) **Vì sao** pipeline đề tài chọn từng họ thuật toán — "
        "dựa trên **so sánh lý thuyết**, không mô tả triển khai (metric, split, τ: Chương 3, §3.2; kiến trúc: Chương 3, §3.1). "
        "Bối cảnh thị trường: Chương 1, §1.1.*"
    )
    new_intro = (
        "*Chương này trả lời ba câu hỏi học thuật: (1) Bài toán FRD được định nghĩa và phân loại thế nào? "
        "(2) Tài liệu hiện có đi đến đâu, còn thiếu gì (G1–G8)? (3) **Nền tảng thuật toán** các họ dùng trong pipeline là gì? "
        "**Biện luận lựa chọn** cụ thể của đề tài (neo EDA + gap): **Chương 3, §3.4**; triển khai: §3.5–3.6. Bối cảnh thị trường: Ch.1 §1.1.*"
    )
    ch2 = ch2.replace(old_intro, new_intro)
    ch2 = ch2.replace(
        "| **§2.3** | Cơ sở lý thuyết lựa chọn từng thành phần pipeline (so sánh & lập luận) |",
        "| **§2.3** | Nền tảng thuật toán (định nghĩa, cơ chế) — biện luận chọn → Ch.3 §3.4 |",
    )
    ch2 = ch2.replace(
        "Metric và protocol đánh giá được trình bày tại **Chương 3, §3.2** (không thuộc lý thuyết thuật toán chương này).",
        "Metric và protocol đánh giá: **Chương 3, §3.6**.",
    )
    ch2 = ch2.replace("dẫn tới thiết kế fusion ở §2.3.", "dẫn tới thiết kế fusion (§2.3; biện luận §3.4).")
    # Bảng 2.4 mapping
    ch2 = ch2.replace("| G1 | §2.3.2–2.3.3 ModernBERT + behavioral fusion |", "| G1 | §2.3.2–2.3.3 + biện luận §3.4.2–3.4.3 |")
    ch2 = ch2.replace("| G2 | §2.3.1 dual-track; §2.3.4–2.3.5 GBDT + sequence |", "| G2 | §2.3.1, §2.3.4–2.3.5 + biện luận §3.4.1, §3.4.4–3.4.5 |")
    ch2 = ch2.replace("| G3, G7 | §2.3.6 ensemble; protocol τ → Ch. 3 §3.2 |", "| G3, G7 | §2.3.6 + biện luận §3.4.6; protocol τ → Ch.3 §3.6 |")
    ch2 = ch2.replace("| G4 | — (phương pháp Ch. 3 §3.2.3) |", "| G4 | — (phương pháp Ch.3 §3.6.3) |")
    ch2 = ch2.replace("| G5, G6 | §2.3.7 PCA/PSO ablation |", "| G5, G6 | §2.3.7 + biện luận §3.4.7 |")
    ch2 = ch2.replace("| G8 | Toàn §2.3 ablation logic |", "| G8 | Biện luận §3.4 + ablation Ch.4 |")
    ch2 = ch2.replace(
        "§2.3 trình bày **lý thuyết và so sánh** để giải thích vì sao Bảng 2.4 chọn đúng các họ thuật toán đó — trước khi Chương 3 mô tả cách cài đặt.",
        "Bảng 2.4 ánh xạ gap → hướng thuật toán (§2.3) và **biện luận lựa chọn** (Ch.3 §3.4) sau EDA (§3.2–3.3).",
    )
    return ch2


def patch_ch2_24(ch2_24: str) -> str:
    ch2_24 = ch2_24.replace("(§2.3.1, 2.3.4–2.3.5)", "(§2.3 + biện luận §3.4.1, §3.4.4–3.4.5)")
    ch2_24 = ch2_24.replace("(§2.3.2–2.3.3)", "(§2.3.2–2.3.3; biện luận §3.4.2–3.4.3)")
    ch2_24 = ch2_24.replace("protocol Ch. 3 (§2.3.6)", "§2.3.6; protocol §3.6 (biện luận §3.4.6)")
    ch2_24 = ch2_24.replace("(§2.3.7)", "(§2.3.7; biện luận §3.4.7)")
    ch2_24 = ch2_24.replace("(§2.3.8)", "(§2.3.8; biện luận §3.4.8)")
    ch2_24 = ch2_24.replace("*Kiến trúc và metric: Chương 3. Số liệu: Chương 4–5. Mục tiêu: Chương 1.*",
        "*Biện luận chọn: Ch.3 §3.4. Kiến trúc: §3.5. Protocol: §3.6. Số liệu: Ch.4–5. Mục tiêu: Ch.1.*")
    return ch2_24


def extract_env_312(thesis: str) -> str:
    m = re.search(r"(## 3\.12\. Môi trường thực nghiệm.*?)(?=\n---\n\n## 3\.13\.)", thesis, re.S)
    if not m:
        return ""
    block = m.group(1)
    block = block.replace("## 3.12.", "## 3.7.")
    block = block.replace("Bảng 3.3", "Bảng 3.6")
    return block.strip() + "\n\n---\n\n"


def renumber_architecture_block(block: str) -> str:
    """Old §3.4 architecture → §3.5 (subsections only within this block)."""
    block = block.replace("## 3.4. Kiến trúc dual-track", "## 3.5. Kiến trúc dual-track")
    block = re.sub(r"^### 3\.4\.(\d+)", r"### 3.5.\1", block, flags=re.M)
    block = block.replace("Hình 3.3, §3.4.3)", "Hình 3.3, §3.5.3)")
    block = block.replace("Bảng 3.3 liệt kê notebook", "Bảng 3.6 liệt kê notebook")
    return block


def renumber_protocol_block(block: str) -> str:
    """Old §3.5 protocol (+ §3.13 rubric) → §3.6 / §3.8."""
    block = block.replace("## 3.5. Quy trình thực nghiệm", "## 3.6. Quy trình thực nghiệm")
    block = re.sub(r"^### 3\.5\.(\d+)", r"### 3.6.\1", block, flags=re.M)
    block = block.replace("## 3.13. Khung đánh giá", "## 3.8. Khung đánh giá")
    block = block.replace("Bảng 3.13", "Bảng 3.8")
    block = block.replace("§3.13", "§3.8")
    return block


def renumber_ch3(ch3_rest: str) -> str:
    """ch3_rest starts at old ## 3.4 architecture through §3.13 rubric."""
    proto_marker = "## 3.5. Quy trình thực nghiệm"
    proto_start = ch3_rest.find(proto_marker)
    if proto_start < 0:
        raise RuntimeError("Expected old ## 3.5. Quy trình in ch3_rest")
    arch = ch3_rest[:proto_start]
    proto = ch3_rest[proto_start:]
    return renumber_architecture_block(arch) + renumber_protocol_block(proto)


def patch_ch3_intro_and_tables(ch3: str) -> str:
    old = (
        "Chương này trình bày phương pháp theo **trình tự logic khoa học**: (1) **mô tả bộ dữ liệu** — nguồn gốc, quy mô, schema (§3.1); "
        "(2) **phân tích khám phá (EDA)** — chiến lược, checklist và insight dẫn đến thiết kế (§3.2); "
        "(3) **thiết kế pipeline** — tiền xử lý, đặc trưng, ánh xạ EDA → quyết định kỹ thuật (§3.3); "
        "(4) **kiến trúc và lựa chọn mô hình** — dual-track, ModernBERT, GBDT, ensemble (§3.4); "
        "(5) **quy trình thực nghiệm** — split, metric, ngưỡng, CV, multi-seed (§3.5). "
        "Lý do *tại sao chọn* từng họ thuật toán: Chương 2 §2.3; Chương 3 mô tả *cách thiết kế và vận hành* để kiểm chứng RQ. "
        "**Số liệu và kết quả** → Chương 4; **khung tự chấm** → §3.13."
    )
    new = (
        "Chương này trình bày phương pháp theo **trình tự logic khoa học**: (1) **mô tả bộ dữ liệu** (§3.1); "
        "(2) **EDA** (§3.2); (3) **thiết kế pipeline từ EDA** (§3.3); (4) **biện luận lựa chọn** thành phần — neo G1–G8 và Bảng 3.5 (§3.4); "
        "(5) **kiến trúc dual-track và sơ đồ** (§3.5); (6) **protocol thực nghiệm** (§3.6); (7) **môi trường** (§3.7); "
        "(8) **khung đánh giá** D0–D8 (§3.8). Nền tảng thuật toán: **Ch.2 §2.3**. **Số liệu** → Ch.4; **điểm tự chấm** → §4.14."
    )
    ch3 = ch3.replace(old, new)
    ch3 = ch3.replace("chi tiết môi trường §4.0", "chi tiết môi trường §3.7, §4.0")

    # Bảng 3.1
    ch3 = ch3.replace("| §2.3.1 | Dual-track | §3.4 |", "| §2.3.1 + §3.4.1 | Dual-track | §3.5 |")
    ch3 = ch3.replace("| §2.3.2 | ModernBERT | §3.4, Hình 3.3 |", "| §2.3.2 + §3.4.2 | ModernBERT | §3.5, Hình 3.3 |")
    ch3 = ch3.replace("| §2.3.3 | 9 behavioral; 777-d | §3.2, §3.3 |", "| §2.3.3 + §3.4.3 | 9 behavioral; 777-d | §3.2, §3.3 |")
    ch3 = ch3.replace("| §2.3.4–2.3.6 | GBDT, sequence, blend | §3.4–3.5 |", "| §2.3.4–2.3.6 + §3.4.4–3.4.6 | GBDT, sequence, blend | §3.5–3.6 |")
    ch3 = ch3.replace("| §2.3.7 | PCA/PSO ablation | §3.4 |", "| §2.3.7 + §3.4.7 | PCA/PSO ablation | §3.5 |")
    ch3 = ch3.replace("| §2.3.8 | XAI | §3.4 |", "| §2.3.8 + §3.4.8 | XAI | §3.5 |")
    ch3 = ch3.replace("| — | Protocol, metric | §3.5 |", "| — | Protocol, metric | §3.6 |")
    ch3 = ch3.replace("| — | Tái lập | §3.5.3 |", "| — | Tái lập | §3.6.3 |")

    # Bảng 3.2 RQ
    ch3 = ch3.replace("| RQ1 | §3.3–3.4 — fusion 777-d |", "| RQ1 | §3.3–3.4.3 — fusion 777-d |")
    ch3 = ch3.replace("| RQ2 | §3.4 — PSO ablation |", "| RQ2 | §3.4.7 — PSO ablation |")
    ch3 = ch3.replace("| RQ3 | §3.4 — PCA vs raw |", "| RQ3 | §3.4.7 — PCA vs raw |")
    ch3 = ch3.replace("| RQ4 | §3.4–3.5 — blend |", "| RQ4 | §3.4.6, §3.5–3.6 — blend |")
    ch3 = ch3.replace("| RQ5 | §3.5.2 — dual-threshold |", "| RQ5 | §3.6.2 — dual-threshold |")
    ch3 = ch3.replace("| RQ6 | §3.5 — ablation protocol |", "| RQ6 | §3.6 — ablation protocol |")

    ch3 = ch3.replace("Khung tổng hợp *đánh giá chất lượng nghiên cứu*: §3.13 (D0–D8).", "Khung đánh giá: §3.8 (D0–D8).")
    ch3 = ch3.replace("(Ch. 2, §2.3.1)", "(Ch.2 §2.3.1; biện luận §3.4.1)")
    old_link = (
        "Câu hỏi *tại sao* dual-track, freeze, blend thay stacking, PCA ablation… → Ch. 2, §2.3 và Bảng 2.4 (G1–G8). "
        "Chương 3 trả lời *luồng thực hiện thế nào* để các lựa chọn lý thuyết đó được kiểm chứng có kiểm soát."
    )
    new_link = (
        "Nền tảng thuật toán → **Ch.2 §2.3**; biện luận *tại sao chọn* (neo EDA + G1–G8) → **§3.4**; "
        "*luồng vận hành và sơ đồ* → **§3.5**; *protocol đo* → **§3.6**."
    )
    ch3 = ch3.replace(old_link, new_link)
    ch3 = ch3.replace("### Bảng 3.3. Lộ trình notebook", "### Bảng 3.6. Lộ trình notebook")
    ch3 = ch3.replace("nguyên tắc dual-track §3.4.", "nguyên tắc dual-track §3.5.")
    ch3 = ch3.replace(
        "đặc trưng** và **kiến trúc mô hình** (§3.4).",
        "đặc trưng**; biện luận và kiến trúc mô hình (§3.4–3.5).",
    )
    ch3 = ch3.replace("Split §3.3.1; Ch.3 §3.5.3;", "Split §3.3.1; Ch.3 §3.6.3;")

    # Ch4 intro
    ch3 = ch3.replace(
        "Phương pháp: bộ dữ liệu §3.1; EDA §3.2; thiết kế pipeline §3.3; kiến trúc §3.4; quy trình §3.5; rubric §3.13.",
        "Phương pháp: §3.1 dataset; §3.2 EDA; §3.3 thiết kế; §3.4 biện luận; §3.5 kiến trúc; §3.6 protocol; rubric §3.8.",
    )
    ch3 = ch3.replace("Bảng 3.13 (Ch.3 §3.13)", "Bảng 3.8 (Ch.3 §3.8)")
    ch3 = ch3.replace("khung D0–D8 tại **Bảng 3.13", "khung D0–D8 tại **Bảng 3.8")
    ch3 = ch3.replace("Điểm D2 đạt nấc Xuất sắc (14,0/14) nhờ ma trận này + bảng Tier A (§4.12).", 
                        "Điểm D2 đạt nấc Xuất sắc (14,0/14) nhờ ma trận này + bảng Tier A (§4.12).")
    ch3 = ch3.replace("rubric D0 (Bảng 3.13)", "rubric D0 (Bảng 3.8)")
    ch3 = ch3.replace("nấc D0 ≥ 3 (Bảng 3.13)", "nấc D0 ≥ 3 (Bảng 3.8)")
    ch3 = ch3.replace("chiều D1 (Bảng 3.13)", "chiều D1 (Bảng 3.8)")
    ch3 = ch3.replace("chiều D2 (Bảng 3.13)", "chiều D2 (Bảng 3.8)")
    ch3 = ch3.replace("Khung: **Bảng 3.13**", "Khung: **Bảng 3.8**")
    ch3 = ch3.replace("đối chiếu Bảng 3.13 với artifact", "đối chiếu Bảng 3.8 với artifact")
    return ch3


def main():
    ch2_full = CH2_SRC.read_text(encoding="utf-8")
    expand = EXPAND.read_text(encoding="utf-8")
    thesis = THESIS.read_text(encoding="utf-8")

    subs = split_23_subsections(ch2_full)
    if len(subs) != 8:
        raise RuntimeError(f"Expected 8 subsections in 2.3, got {len(subs)}")

    # Ch2: everything before ## 2.3 + new 2.3 foundation + 2.4
    idx_23 = ch2_full.find("## 2.3.")
    idx_24 = ch2_full.find("## 2.4.")
    ch2_head = patch_ch2_preamble(ch2_full[:idx_23].rstrip())
    ch2_foundation = build_ch2_foundation(subs)
    ch2_24 = patch_ch2_24(ch2_full[idx_24:].strip())

    # Ch3 from expand: split at ## 3.1 and at old ## 3.4
    ch3_start = expand.find("# CHƯƠNG 3:")
    ch4_start = expand.find("# CHƯƠNG 4:")
    ch3_block = expand[ch3_start:ch4_start]
    # Support re-run: architecture may already be §3.5
    old_34_start = ch3_block.find("## 3.4. Kiến trúc dual-track")
    if old_34_start < 0:
        old_34_start = ch3_block.find("## 3.5. Kiến trúc dual-track")
    old_35_start = ch3_block.find("## 3.5. Quy trình thực nghiệm")
    if old_35_start < 0:
        old_35_start = ch3_block.find("## 3.6. Quy trình thực nghiệm")
    ch3_prefix = ch3_block[:old_34_start].rstrip()  # intro + 3.1-3.3
    ch3_arch = ch3_block[old_34_start:old_35_start].rstrip()  # old 3.4 architecture
    ch3_suffix = ch3_block[old_35_start:].rstrip()  # old 3.5 + 3.13

    ch3_argument = build_ch3_argument(subs)
    env = extract_env_312(thesis)

    # Renumber architecture and suffix before merge
    ch3_arch_renum = renumber_ch3(ch3_arch + "\n\n")
    ch3_suffix_renum = renumber_ch3(ch3_suffix)

    # Insert env before 3.8 in suffix
    if "## 3.8. Khung" in ch3_suffix_renum:
        ch3_suffix_renum = ch3_suffix_renum.replace(
            "## 3.8. Khung đánh giá chất lượng nghiên cứu",
            env + "## 3.8. Khung đánh giá chất lượng nghiên cứu",
        )

    ch3_combined = (
        ch3_prefix
        + "\n\n---\n\n"
        + ch3_argument
        + "---\n\n"
        + ch3_arch_renum
        + ch3_suffix_renum
    )
    ch3_combined = patch_ch3_intro_and_tables(ch3_combined)

    ch4_block = expand[ch4_start:].strip()

    out = ch2_head + "\n\n---\n\n" + ch2_foundation + "\n\n---\n\n" + ch2_24 + "\n\n---\n\n" + ch3_combined + "\n\n---\n\n" + ch4_block + "\n"
    OUT.write_text(out, encoding="utf-8")
    print(f"Wrote {OUT} ({len(out.splitlines())} lines)")


if __name__ == "__main__":
    main()