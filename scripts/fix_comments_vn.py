"""
Chuẩn hóa comment # tiếng Việt: xóa trùng, đảm bảo mỗi dòng code có đúng 1 comment VN phía trên.
Mức chuẩn: 01_EDA — mỗi dòng thực thi có comment tiếng Việt có dấu ngay phía trên.
"""
from __future__ import annotations

import json
from pathlib import Path

from annotate_detailed_vn import (
    explain_line,
    has_vn_comment_above,
    is_vn_comment,
    should_skip,
    upgrade_existing_comment,
)

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"


def is_exec_line(line: str) -> bool:
    return not should_skip(line.strip()) and bool(line.strip()) and not line.strip().startswith("#")


def normalize_cell(source: str) -> str:
    lines = source.splitlines()

    # Bước 1: loại comment trùng liên tiếp; nâng cấp comment thiếu dấu
    cleaned: list[str] = []
    for line in lines:
        if line.strip().startswith("#"):
            if cleaned and cleaned[-1].strip() == line.strip():
                continue
            cleaned.append(upgrade_existing_comment(line))
            continue
        cleaned.append(line)

    # Bước 2: chèn comment trước mỗi dòng thực thi nếu thiếu
    out: list[str] = []
    for line in cleaned:
        if not is_exec_line(line):
            out.append(line)
            continue
        prev = [l.rstrip("\n") for l in out]
        if not has_vn_comment_above(prev):
            indent = line[: len(line) - len(line.lstrip())]
            out.append(f"{indent}# {explain_line(line)}")
        out.append(line)

    result = "\n".join(out)
    if source.endswith("\n"):
        result += "\n"
    return result


def process_notebook(path: Path) -> int:
    nb = json.loads(path.read_text(encoding="utf-8"))
    changed = 0
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        old = "".join(cell.get("source", []))
        new = normalize_cell(old)
        if new != old:
            changed += 1
            cell["source"] = [ln + "\n" for ln in new.splitlines()]
    if changed:
        path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    return changed


def main():
    total = 0
    for path in sorted(NOTEBOOK_DIR.rglob("*.ipynb")):
        if path.name == "05_04_CNN_BiLSTM_Sequence.ipynb":
            continue
        n = process_notebook(path)
        if n:
            print(f"{path.name}: {n} cells fixed")
            total += n
    print(f"Done: {total} cells")


if __name__ == "__main__":
    main()