"""Đo % dòng code có comment tiếng Việt ngay phía trên (mức 01_EDA)."""
import json
from pathlib import Path

from annotate_detailed_vn import has_vn_diacritic, is_vn_comment, should_skip

root = Path(__file__).resolve().parents[1] / "notebooks"


def is_exec(line: str) -> bool:
    return not should_skip(line.strip()) and bool(line.strip()) and not line.strip().startswith("#")


def has_vn_above(lines, idx):
    j = idx - 1
    while j >= 0 and not lines[j].strip():
        j -= 1
    if j < 0:
        return False
    return is_vn_comment(lines[j])


print("FILE | cells | exec_lines | vn_comments | covered% | gaps")
for p in sorted(root.rglob("*.ipynb")):
    if p.name == "05_04_CNN_BiLSTM_Sequence.ipynb":
        continue
    nb = json.loads(p.read_text(encoding="utf-8"))
    cells = gaps = 0
    exec_total = vn_total = covered = 0
    gap_cells = []
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        cells += 1
        lines = "".join(c.get("source", [])).splitlines()
        cell_gaps = 0
        for j, line in enumerate(lines):
            if not is_exec(line):
                continue
            exec_total += 1
            if has_vn_above(lines, j):
                covered += 1
            else:
                cell_gaps += 1
        vn_total += sum(
            1 for l in lines if l.strip().startswith("#") and is_vn_comment(l)
        )
        if cell_gaps:
            gap_cells.append((i, cell_gaps))
            gaps += cell_gaps
    pct = 100 * covered / exec_total if exec_total else 100
    gap_str = str(gap_cells[:3]) if gap_cells else "none"
    print(f"{p.name} | {cells} | {exec_total} | {vn_total} | {pct:.1f}% | {gaps} gaps {gap_str}")