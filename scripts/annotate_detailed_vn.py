"""
Chèn comment # tiếng Việt chi tiết trước TỪNG dòng code thực thi (mức 01_EDA).
Chỉ thêm comment; không đổi logic code.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"

# Chỉ bỏ qua dòng thuần cú pháp không mang ngữ nghĩa
SKIP_LINE_RE = re.compile(r"^(\.\.\.|pass|break|continue)$")

CALL_HINTS = {
    "read_csv": "đọc file CSV vào DataFrame",
    "to_csv": "ghi DataFrame ra file CSV",
    "read_json": "đọc file JSON",
    "json.load": "phân tích nội dung JSON",
    "json.dump": "ghi từ điển ra file JSON",
    "json.dumps": "chuyển từ điển sang chuỗi JSON",
    "groupby": "nhóm dữ liệu theo cột",
    "merge": "ghép hai DataFrame",
    "concat": "nối nhiều DataFrame",
    "dropna": "xóa dòng/cột có giá trị thiếu",
    "fillna": "điền giá trị thay thế cho NaN",
    "astype": "ép kiểu dữ liệu cột",
    "value_counts": "đếm tần suất từng giá trị",
    "describe": "thống kê mô tả",
    "head": "xem vài dòng đầu",
    "display": "hiển thị bảng/kết quả trên notebook",
    "print": "in thông tin ra console",
    "plt.figure": "tạo figure matplotlib mới",
    "plt.savefig": "lưu biểu đồ ra file hình",
    "plt.show": "hiển thị biểu đồ",
    "plt.title": "đặt tiêu đề biểu đồ",
    "plt.xlabel": "đặt nhãn trục X",
    "plt.ylabel": "đặt nhãn trục Y",
    "plt.tight_layout": "tự chỉnh lề biểu đồ",
    "plt.legend": "hiển thị chú thích",
    "plt.axhline": "vẽ đường ngang tham chiếu",
    "plt.axvline": "vẽ đường dọc tham chiếu",
    "plt.plot": "vẽ đồ thị đường",
    "plt.bar": "vẽ biểu đồ cột",
    "sns.set_theme": "đặt theme seaborn",
    "sns.histplot": "vẽ histogram phân phối",
    "sns.boxplot": "vẽ boxplot so sánh phân phối",
    "sns.barplot": "vẽ barplot seaborn",
    "sns.countplot": "vẽ biểu đồ đếm",
    "sns.lineplot": "vẽ line plot seaborn",
    "sns.catplot": "vẽ faceted categorical plot",
    "np.save": "lưu mảng numpy ra file .npy",
    "np.load": "nạp mảng từ file .npy",
    "np.concatenate": "nối các mảng numpy",
    "np.unique": "lấy giá trị duy nhất",
    "np.isfinite": "kiểm tra phần tử hữu hạn",
    "train_test_split": "chia tập train/test (có thể stratify)",
    "fit_transform": "fit scaler trên train và transform",
    "transform": "transform dữ liệu bằng object đã fit",
    "fit": "fit model/reducer trên dữ liệu train",
    "predict": "dự đoán nhãn/xác suất",
    "torch.device": "chọn thiết bị GPU hoặc CPU",
    "torch.manual_seed": "cố định seed torch",
    "joblib.dump": "lưu object (scaler/model) ra disk",
    "joblib.load": "tải object đã lưu",
    "mkdir": "tạo thư mục nếu chưa có",
    "exists": "kiểm tra file/thư mục tồn tại",
    "open": "mở file để đọc/ghi",
    "write_text": "ghi nội dung text ra file",
    "raise": "ném lỗi và dừng cell",
    "return": "trả kết quả từ hàm",
    "len": "đếm số phần tử",
    "range": "tạo dãy số cho vòng lặp",
    "enumerate": "lặp kèm chỉ số",
    "zip": "ghép song song nhiều iterable",
    "sorted": "sắp xếp danh sách",
    "list": "chuyển/chiếu thành list",
    "dict": "tạo từ điển",
    "set": "tạo tập hợp",
    "int": "ép kiểu số nguyên",
    "float": "ép kiểu số thực",
    "str": "ép kiểu chuỗi",
    "bool": "ép kiểu boolean",
    "round": "làm tròn số",
    "sum": "tính tổng",
    "min": "lấy giá trị nhỏ nhất",
    "max": "lấy giá trị lớn nhất",
    "mean": "tính trung bình",
    "median": "tính trung vị",
    "random.seed": "cố định seed random",
    "np.random.seed": "cố định seed numpy",
    "pd.set_option": "cấu hình hiển thị pandas",
    "drive.mount": "mount Google Drive trên Colab",
    "gc.collect": "giải phóng bộ nhớ",
    "del ": "xóa biến để giải phóng RAM/VRAM",
    "isinstance": "kiểm tra kiểu dữ liệu",
    "append": "thêm phần tử vào danh sách",
}

VAR_HINTS = {
    "SEED": "hằng số seed tái lập kết quả ngẫu nhiên",
    "PROJECT_ROOT": "thư mục gốc dự án trên Drive",
    "DEVICE": "thiết bị chạy model (cuda/cpu)",
    "THRESHOLD": "ngưỡng xác suất phân loại fake",
    "FAKE_LABEL": "giá trị nhãn lớp fake (positive)",
    "REAL_LABEL": "giá trị nhãn lớp real (negative)",
    "IN_COLAB": "cờ đánh dấu đang chạy trên Colab",
    "df": "DataFrame chứa dữ liệu chính",
    "clean_df": "DataFrame sau bước làm sạch",
    "train_df": "tập train sau split",
    "val_df": "tập validation sau split",
    "test_df": "tập test sau split",
    "X": "dictionary/mảng đặc trưng đầu vào",
    "y": "dictionary/mảng nhãn",
    "schema": "ánh xạ tên các cột quan trọng",
    "metadata": "từ điển metadata tái lập pipeline",
}

IMPORT_HINTS = {
    "json": "thư viện đọc/ghi JSON",
    "random": "thư viện sinh số ngẫu nhiên",
    "gc": "giải phóng bộ nhớ",
    "os": "thao tác hệ điều hành/biến môi trường",
    "sys": "tham số runtime Python",
    "time": "đo thời gian thực thi",
    "math": "hàm toán học",
    "re": "biểu thức chính quy",
    "subprocess": "chạy lệnh hệ thống (pip install)",
    "pathlib": "quản lý đường dẫn",
    "datetime": "xử lý ngày giờ",
    "collections": "cấu trúc dữ liệu mở rộng",
    "copy": "sao chép object",
    "itertools": "công cụ iterator",
    "typing": "kiểu type hint",
    "numpy": "tính toán mảng số",
    "pandas": "xử lý bảng dữ liệu",
    "matplotlib": "vẽ biểu đồ",
    "seaborn": "biểu đồ thống kê",
    "sklearn": "thư viện machine learning scikit-learn",
    "torch": "deep learning PyTorch",
    "joblib": "serialize model/scaler",
    "transformers": "HuggingFace transformers",
    "google.colab": "module Google Colab",
    "lightgbm": "thư viện LightGBM",
    "xgboost": "thư viện XGBoost",
    "shap": "giải thích mô hình SHAP",
    "platform": "thông tin nền tảng hệ thống",
    "nn": "module neural network PyTorch",
    "F": "hàm functional PyTorch",
    "DataLoader": "tải batch dữ liệu PyTorch",
    "TensorDataset": "dataset tensor cho PyTorch",
    "Dataset": "lớp dataset tùy chỉnh PyTorch",
    "AutoModel": "model HuggingFace tự động",
    "AutoTokenizer": "tokenizer HuggingFace tự động",
}

CLOSE_HINTS = {
    ")": "đóng ngoặc gọi hàm hoặc biểu thức",
    ")": "đóng ngoặc gọi hàm",
    "])": "đóng list comprehension hoặc danh sách",
    "})": "đóng từ điển hoặc DataFrame constructor",
    "},": "đóng phần tử từ điển (còn phần tử sau)",
    "],": "đóng phần tử danh sách (còn phần tử sau)",
    "}": "đóng khối từ điển",
    "]": "đóng khối danh sách",
    ",": "phân tách tham số hoặc phần tử",
}

VN_WORDS = (
    "thư viện", "thu vien", "hằng", "hang", "biến", "bien", "hàm", "ham",
    "đọc", "doc", "ghi", "tạo", "tao", "kiểm tra", "kiem tra", "vòng lặp",
    "vong lap", "điều kiện", "dieu kien", "nhánh", "nhanh", "đóng", "dong",
    "mảng", "mang", "dữ liệu", "du lieu", "từ điển", "tu dien", "thực thi",
    "thuc thi", "lệnh", "lenh", "trong", "pipeline", "của", "cua", "để", "de",
    "và", "va", "hoặc", "hoac", "theo", "cho", "với", "voi", "khi", "nếu", "neu",
)


def has_vn_diacritic(text: str) -> bool:
    return any(ord(c) > 127 for c in text)


def is_vn_comment(line: str) -> bool:
    s = line.strip()
    if not s.startswith("#"):
        return False
    if has_vn_diacritic(s):
        return True
    low = s.lower()
    return any(w in low for w in VN_WORDS)


def has_vn_comment_above(prev_lines: list[str]) -> bool:
    j = len(prev_lines) - 1
    while j >= 0 and not prev_lines[j].strip():
        j -= 1
    if j < 0:
        return False
    return is_vn_comment(prev_lines[j])


def should_skip(stripped: str) -> bool:
    if not stripped or stripped.startswith("#"):
        return True
    if SKIP_LINE_RE.match(stripped):
        return True
    return False


def snake_to_vi(name: str) -> str:
    parts = name.strip("_").split("_")
    return " ".join(parts) if parts else name


def call_hint(code: str) -> str | None:
    for key, hint in sorted(CALL_HINTS.items(), key=lambda x: -len(x[0])):
        if key in code:
            return hint
    return None


def ensure_diacritic(text: str) -> str:
    """Đảm bảo chuỗi giải thích có ít nhất một ký tự tiếng Việt có dấu."""
    if has_vn_diacritic(text):
        return text
    replacements = {
        "thu vien": "thư viện",
        "thuc thi": "thực thi",
        "lenh": "lệnh",
        "bien": "biến",
        "ham": "hàm",
        "doc": "đọc",
        "ghi": "ghi",
        "tao": "tạo",
        "kiem tra": "kiểm tra",
        "vong lap": "vòng lặp",
        "dieu kien": "điều kiện",
        "dong": "đóng",
        "du lieu": "dữ liệu",
        "tu dien": "từ điển",
        "dictionary": "từ điển",
        "trong pipeline": "trong pipeline dự án",
        "thuc thi lenh Python trong pipeline": "thực thi lệnh trong pipeline dự án",
        "thuc thi lenh Python": "thực thi lệnh Python",
    }
    low = text.lower()
    for old, new in replacements.items():
        if old in low:
            idx = low.index(old)
            return text[:idx] + new + text[idx + len(old):]
    return text + " trong pipeline dự án"


def explain_line(line: str) -> str:
    s = line.strip()
    core = s.split("#")[0].strip()

    # Dòng đóng ngoặc / phân tách
    if core in CLOSE_HINTS:
        return ensure_diacritic(f"{core}: {CLOSE_HINTS[core]}")
    if re.match(r"^[)\]},]+$", core):
        return ensure_diacritic(f"{core}: đóng khối cú pháp Python")

    if core.startswith(("import ", "from ")):
        parts = core.replace("from ", "").replace("import ", "").split()
        mod = parts[0].split(".")[0] if parts else "module"
        hint = IMPORT_HINTS.get(mod, f"import thư viện {mod}")
        return ensure_diacritic(f"{core}: {hint}")

    m = re.match(r"^def\s+([A-Za-z_][\w]*)\s*\(", core)
    if m:
        return ensure_diacritic(f"{m.group(1)}: định nghĩa hàm {snake_to_vi(m.group(1))}")

    m = re.match(r"^class\s+([A-Za-z_][\w]*)", core)
    if m:
        return ensure_diacritic(f"class {m.group(1)}: định nghĩa lớp {m.group(1)}")

    m = re.match(r"^([A-Z][A-Z0-9_]*)\s*=", core)
    if m:
        name = m.group(1)
        hint = VAR_HINTS.get(name, f"biến cấu hình {snake_to_vi(name.lower())}")
        return ensure_diacritic(f"{name}: {hint}")

    m = re.match(r"^([a-zA-Z_][\w]*)\s*=", core)
    if m and not core.startswith(("if ", "for ", "with ")):
        name = m.group(1)
        ch = call_hint(core) or f"gán giá trị cho biến {snake_to_vi(name)}"
        return ensure_diacritic(f"{name} = ...: {ch}")

    if core.startswith("if "):
        return ensure_diacritic(f"if: kiểm tra điều kiện — {core[:70]}")
    if core.startswith("elif "):
        return ensure_diacritic(f"elif: nhánh điều kiện phụ — {core[:70]}")
    if core.startswith("else:"):
        return ensure_diacritic("else: nhánh còn lại của điều kiện")
    if core.startswith("for "):
        return ensure_diacritic(f"for: vòng lặp — {core[:70]}")
    if core.startswith("while "):
        return ensure_diacritic(f"while: lặp khi điều kiện đúng — {core[:70]}")
    if core.startswith("try:"):
        return ensure_diacritic("try: bắt đầu khối thử/catch ngoại lệ")
    if core.startswith("except"):
        return ensure_diacritic(f"except: xử lý ngoại lệ — {core[:70]}")
    if core.startswith("with "):
        return ensure_diacritic(f"with: context manager — {core[:70]}")
    if core.startswith("return "):
        ch = call_hint(core) or "trả kết quả từ hàm"
        return ensure_diacritic(f"return: {ch}")

    ch = call_hint(core)
    if ch:
        short = core[:80] + ("..." if len(core) > 80 else "")
        return ensure_diacritic(f"{short}: {ch}")

    short = core[:80] + ("..." if len(core) > 80 else "")
    return ensure_diacritic(f"{short}: thực thi lệnh trong pipeline dự án")


def upgrade_existing_comment(comment_line: str) -> str:
    """Nâng cấp comment cũ thiếu dấu tiếng Việt."""
    s = comment_line.strip()
    if not s.startswith("#"):
        return comment_line
    body = s[1:].strip()
    if is_vn_comment(comment_line):
        return comment_line
    upgraded = ensure_diacritic(body)
    indent = comment_line[: len(comment_line) - len(comment_line.lstrip())]
    return f"{indent}# {upgraded}"


def annotate_source(source: str) -> str:
    lines = source.splitlines()
    out: list[str] = []

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("#"):
            out.append(upgrade_existing_comment(line))
            continue

        if should_skip(stripped):
            out.append(line)
            continue

        prev = [l.rstrip("\n") for l in out]
        if not has_vn_comment_above(prev):
            indent = line[: len(line) - len(line.lstrip())]
            explanation = explain_line(line)
            out.append(f"{indent}# {explanation}")

        out.append(line)

    result = "\n".join(out)
    if source.endswith("\n"):
        result += "\n"
    return result


def annotate_notebook(path: Path, dry_run: bool = False) -> tuple[int, int]:
    nb = json.loads(path.read_text(encoding="utf-8"))
    changed = added = 0
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        old = "".join(cell.get("source", []))
        new = annotate_source(old)
        if new != old:
            changed += 1
            added += max(0, new.count("\n") - old.count("\n"))
            if not dry_run:
                cell["source"] = [ln + "\n" for ln in new.splitlines()]
    if changed and not dry_run:
        path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    return changed, added


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--only", nargs="*")
    args = p.parse_args()

    paths = sorted(NOTEBOOK_DIR.rglob("*.ipynb"))
    if args.only:
        names = set(args.only)
        paths = [x for x in paths if x.name in names]

    total_cells = total_lines = 0
    for path in paths:
        if path.name == "05_04_CNN_BiLSTM_Sequence.ipynb":
            continue
        c, a = annotate_notebook(path, dry_run=args.dry_run)
        if c:
            tag = "[dry] " if args.dry_run else ""
            print(f"{tag}{path.name}: {c} cells, ~{a} comment lines added")
            total_cells += c
            total_lines += a
    print(f"Done: {total_cells} cells, ~{total_lines} new lines")


if __name__ == "__main__":
    main()