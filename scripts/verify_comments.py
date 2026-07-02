import json
from pathlib import Path

root = Path(__file__).resolve().parents[1] / "notebooks"
total_vn = 0
print("Notebook | code_cells | vn_comments")
for p in sorted(root.rglob("*.ipynb")):
    nb = json.loads(p.read_text(encoding="utf-8"))
    code_cells = vn = 0
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        code_cells += 1
        src = "".join(c.get("source", []))
        vn += sum(
            1
            for line in src.splitlines()
            if line.strip().startswith("#") and any(ord(ch) > 127 for ch in line)
        )
    total_vn += vn
    print(f"{p.name} | {code_cells} | {vn}")
print("TOTAL vn_comments:", total_vn)