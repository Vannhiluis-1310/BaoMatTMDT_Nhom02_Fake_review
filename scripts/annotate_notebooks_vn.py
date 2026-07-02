"""Thêm comment # tiếng Việt trước import, def, và biến hằng trong cell code notebook."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"

# Notebook đã comment thủ công đầy đủ — bỏ qua hoặc chỉ bổ sung thiếu
SKIP_FULL = {
    "01_EDA_Preprocessing.ipynb",
    "02_Feature_Engineering.ipynb",
}

IMPORT_HINTS = {
    "google.colab": "module kiểm tra / mount Colab",
    "drive": "API Google Drive trên Colab",
    "json": "đọc/ghi JSON metadata",
    "random": "cố định seed ngẫu nhiên",
    "numpy": "tính toán mảng số",
    "np": "tính toán mảng số",
    "pandas": "xử lý DataFrame",
    "pd": "xử lý DataFrame",
    "torch": "deep learning PyTorch",
    "matplotlib": "vẽ biểu đồ",
    "seaborn": "biểu đồ thống kê",
    "sklearn": "machine learning sklearn",
    "joblib": "lưu/tải object đã fit",
    "pathlib": "quản lý đường dẫn",
    "gc": "giải phóng bộ nhớ",
    "time": "đo thời gian thực thi",
    "os": "biến môi trường hệ thống",
    "sys": "tham số Python runtime",
    "subprocess": "chạy lệnh pip/cài package",
    "transformers": "model HuggingFace (BERT/ModernBERT)",
    "lightgbm": "thư viện LightGBM",
    "xgboost": "thư viện XGBoost",
    "shap": "giải thích mô hình SHAP",
}

FUNC_HINTS = {
    "seed_everything": "cố định seed Python/numpy/torch",
    "set_global_seed": "cố định seed random và numpy",
    "configure_seed_artifacts": "thiết lập đường dẫn artifact theo seed",
    "load_raw_arrays": "nạp feature/label .npy từ Phase 2",
    "evaluate_predictions": "tính metric classification từ xác suất",
    "save_probability": "lưu xác suất fake ra file .npy",
    "write_metrics": "ghi bảng metric ra CSV",
    "ensure_package": "import hoặc pip install package",
    "read_json": "đọc file JSON",
    "utc_now": "timestamp UTC ISO",
    "make_loader": "tạo PyTorch DataLoader",
    "build_model": "khởi tạo kiến trúc mô hình",
    "fit_model": "vòng lặp train với early stopping",
    "train_one_epoch": "một epoch training",
    "predict_probabilities": "suy xác suất fake",
    "infer_column": "tự nhận diện tên cột dataset",
    "_normalize_name": "chuẩn hóa tên cột",
    "record_advanced_artifact": "ghi nhận artifact đã tạo",
    "parse_review_datetime": "parse cột thời gian sang datetime",
    "load_sentiment_scorer": "tải backend sentiment VADER/TextBlob",
    "add_basic_behavioral_features": "tính 5 đặc trưng hành vi cơ bản",
    "add_advanced_behavioral_features": "tính 4 đặc trưng hành vi nâng cao",
    "extract_or_load_bert_embeddings": "trích hoặc load embedding BERT",
    "load_bert_model": "tải tokenizer và model BERT",
    "mean_pool_last_hidden_state": "mean pooling hidden state có attention mask",
    "safe_roc_auc": "ROC-AUC an toàn",
    "safe_pr_auc": "PR-AUC an toàn",
    "capped_batch_size": "giới hạn batch size tối đa",
    "array_memory_mb": "ước lượng RAM mảng (MB)",
    "process_memory_mb": "RAM tiến trình hiện tại (MB)",
}


def has_vn_comment_prev(prev_lines: list[str]) -> bool:
    j = len(prev_lines) - 1
    while j >= 0 and not prev_lines[j].strip():
        j -= 1
    if j < 0:
        return False
    prev = prev_lines[j].strip()
    return prev.startswith("#") and any(ord(c) > 127 for c in prev)


def import_comment(line: str) -> str | None:
    s = line.strip()
    if s.startswith("import "):
        mod = s.split()[1].split(".")[0].split(" as ")[0]
    elif s.startswith("from "):
        mod = s.split()[1].split(".")[0]
    else:
        return None
    hint = IMPORT_HINTS.get(mod, f"import thư viện {mod}")
    return f"# {s.split(' as ')[0]}: {hint}"


def const_comment(line: str) -> str | None:
    m = re.match(r"^([A-Z][A-Z0-9_]*)\s*=", line.strip())
    if not m:
        return None
    name = m.group(1)
    return f"# {name}: biến cấu hình/hằng số của notebook"


def func_comment(line: str) -> str | None:
    m = re.match(r"^def\s+([a-zA-Z_][\w]*)\s*\(", line.strip())
    if not m:
        return None
    name = m.group(1)
    hint = FUNC_HINTS.get(name, f"hàm xử lý {name.replace('_', ' ')}")
    return f"# {name}: {hint}"


def annotate_source(source: str) -> str:
    lines = source.splitlines(keepends=True)
    if not lines:
        return source
    out: list[str] = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            out.append(line)
            continue
        if has_vn_comment_prev([l.rstrip("\n") for l in out]):
            out.append(line)
            continue

        comment = None
        if stripped.startswith(("import ", "from ")):
            comment = import_comment(line)
        elif stripped.startswith("def "):
            comment = func_comment(line)
        elif re.match(r"^[A-Z][A-Z0-9_]*\s*=", stripped):
            comment = const_comment(line)
        elif stripped.startswith(("if ", "for ", "try:", "with ", "class ")):
            key = stripped.split("(")[0].split(":")[0].split()[0]
            if key == "try":
                comment = "# try/except: khối xử lý ngoại lệ"
            elif key == "for":
                comment = f"# for: vòng lặp — {stripped[:60]}"
            elif key == "if":
                comment = f"# if: điều kiện — {stripped[:60]}"
            elif key == "with":
                comment = f"# with: context manager — {stripped[:60]}"
            elif key == "class":
                m = re.match(r"class\s+(\w+)", stripped)
                if m:
                    comment = f"# class {m.group(1)}: định nghĩa lớp"

        if comment and not any(ord(c) > 127 for c in comment):
            comment = comment  # noqa — giữ nguyên nếu không có tiếng Việt (shouldn't happen)

        if comment:
            indent = line[: len(line) - len(line.lstrip())]
            out.append(indent + comment + "\n")
        out.append(line)

    result = "".join(out)
    if not result.endswith("\n") and source.endswith("\n"):
        result += "\n"
    return result


def annotate_notebook(path: Path, dry_run: bool = False) -> int:
    nb = json.loads(path.read_text(encoding="utf-8"))
    changed = 0
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        old = "".join(cell.get("source", []))
        new = annotate_source(old)
        if new != old:
            changed += 1
            if not dry_run:
                cell["source"] = [ln if ln.endswith("\n") else ln + "\n" for ln in new.splitlines()]
                if new.endswith("\n") and (not cell["source"] or not cell["source"][-1].endswith("\n")):
                    pass
    if changed and not dry_run:
        path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    return changed


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only", nargs="*", help="tên file ipynb cụ thể")
    args = parser.parse_args()

    paths = sorted(NOTEBOOK_DIR.rglob("*.ipynb"))
    if args.only:
        names = set(args.only)
        paths = [p for p in paths if p.name in names]

    total = 0
    for path in paths:
        if path.name in SKIP_FULL:
            print(f"skip (manual): {path.relative_to(ROOT)}")
            continue
        n = annotate_notebook(path, dry_run=args.dry_run)
        if n:
            print(f"{'[dry] ' if args.dry_run else ''}updated {n} cells: {path.relative_to(ROOT)}")
            total += n
    print(f"Done. cells touched: {total}")


if __name__ == "__main__":
    main()