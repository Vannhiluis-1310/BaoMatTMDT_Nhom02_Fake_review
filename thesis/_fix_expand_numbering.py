"""Fix section numbering and cross-refs in expandChapter.md."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXPAND = ROOT / "expandChapter.md"


def fix_architecture_block(text: str) -> str:
    arch_marker = "## 3.5. Kiến trúc dual-track"
    proto_marker = "## 3.6. Quy trình thực nghiệm"
    arch_start = text.find(arch_marker)
    proto_start = text.find(proto_marker)
    if arch_start < 0 or proto_start < 0:
        raise RuntimeError("Could not locate §3.5 / §3.6 boundaries")

    head = text[:arch_start]
    arch = text[arch_start:proto_start]
    tail = text[proto_start:]

    # Architecture subsections were wrongly renumbered to 3.6.x
    for i in range(5, 0, -1):
        arch = arch.replace(f"### 3.6.{i}.", f"### 3.5.{i}.")

    # Intra-architecture cross-refs (Hình 3.3 lives in §3.5.3)
    arch = arch.replace("Hình 3.3, §3.6.3)", "Hình 3.3, §3.5.3)")
    arch = arch.replace("Bảng 3.3 liệt kê notebook", "Bảng 3.6 liệt kê notebook")

    return head + arch + tail


def fix_global_refs(text: str) -> str:
    text = text.replace("Bảng 3.13", "Bảng 3.8")
    text = text.replace("§3.13", "§3.8")

    text = text.replace(
        "Phương pháp: bộ dữ liệu §3.1; EDA §3.2; thiết kế pipeline §3.3; "
        "kiến trúc §3.4; quy trình §3.5; rubric §3.8.",
        "Phương pháp: §3.1 dataset; §3.2 EDA; §3.3 thiết kế; §3.4 biện luận; "
        "§3.5 kiến trúc; §3.6 protocol; rubric §3.8.",
    )

    text = text.replace(
        "Split §3.3.1; Ch.3 §3.5.3; `phase5_metadata.json`",
        "Split §3.3.1; Ch.3 §3.6.3; `phase5_metadata.json`",
    )

    text = text.replace(
        "đặc trưng** và **kiến trúc mô hình** (§3.4).",
        "đặc trưng**; biện luận và kiến trúc mô hình (§3.4–3.5).",
    )

    text = text.replace("Điểm neo Bảng 3.3", "Điểm neo §3.3.1 / §3.6.3")
    text = text.replace(
        "đã giải thích ở §3.4 và Ch.2 §2.3.",
        "đã giải thích ở §3.7 và Ch.2 §2.3.",
    )
    text = text.replace(
        "protocol chọn metric/ngưỡng đã định nghĩa ở §3.5.",
        "protocol chọn metric/ngưỡng đã định nghĩa ở §3.6.",
    )

    return text


def verify(text: str) -> list[str]:
    errors = []
    arch_start = text.find("## 3.5. Kiến trúc")
    proto_start = text.find("## 3.6. Quy trình")
    arch = text[arch_start:proto_start]
    if re.search(r"### 3\.6\.\d+", arch):
        errors.append("Architecture block still contains ### 3.6.x")
    if "§3.13" in text or "Bảng 3.13" in text:
        errors.append("Stale §3.13 / Bảng 3.13 references remain")
    if not text.startswith("# CHƯƠNG 2:"):
        errors.append("Missing Chapter 2 preamble")
    return errors


def main():
    text = EXPAND.read_text(encoding="utf-8")
    text = fix_architecture_block(text)
    text = fix_global_refs(text)
    errs = verify(text)
    if errs:
        raise RuntimeError("Verification failed: " + "; ".join(errs))
    EXPAND.write_text(text, encoding="utf-8")
    print(f"Fixed {EXPAND} ({len(text.splitlines())} lines)")


if __name__ == "__main__":
    main()