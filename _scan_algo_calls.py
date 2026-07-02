"""Scan notebooks for algorithm function definitions and call sites."""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NB_DIR = ROOT / "notebooks"

ALGO_PATTERNS = re.compile(
    r"pso|pca|svd|bert|lgbm|xgb|lightgbm|xgboost|cnn|bilstm|lstm|ensemble|blend|"
    r"stack|calibr|fgsm|pgd|shap|lime|mlp|train_model|predict|particle|swarm|"
    r"adversar|robust|threshold|feature|behavioral|modernbert|tokenizer|embedding|"
    r"forward|fit_model|build_model|evaluate|metric|f1|roc|auc|lime|shap",
    re.I,
)

SKIP_NAMES = {
    "display", "show", "print", "len", "range", "str", "int", "float", "list", "dict",
    "set", "tuple", "min", "max", "sum", "abs", "round", "sorted", "enumerate",
    "zip", "map", "filter", "open", "Path", "json", "pd", "np", "plt", "sns",
}


def is_algo_related(name: str, line: str = "") -> bool:
    text = f"{name} {line}"
    return bool(ALGO_PATTERNS.search(text))


def parse_notebook(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        nb = json.load(f)

    defs: dict[str, list[dict]] = defaultdict(list)
    all_calls: list[dict] = []

    for ci, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell.get("source", []))
        if not source.strip():
            continue
        lines = source.splitlines()
        local_defs: set[str] = set()

        for li, line in enumerate(lines, 1):
            m_def = re.match(r"^(\s*)def\s+(\w+)\s*\(", line)
            if m_def:
                name = m_def.group(2)
                local_defs.add(name)
                defs[name].append(
                    {
                        "cell": ci + 1,
                        "line": li,
                        "sig": line.strip()[:100],
                        "nb": path.name,
                    }
                )

        for li, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith("def "):
                continue
            # function calls: name(...)
            for m in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", line):
                name = m.group(1)
                if name in SKIP_NAMES or name in local_defs:
                    continue
                if name[0].isupper() and name not in {"LGBMClassifier", "XGBClassifier"}:
                    # likely class ctor - keep if algo
                    if not is_algo_related(name):
                        continue
                all_calls.append(
                    {
                        "name": name,
                        "cell": ci + 1,
                        "line": li,
                        "code": stripped[:100],
                        "nb": path.name,
                    }
                )

    return {"defs": dict(defs), "calls": all_calls, "path": path}


def main() -> None:
    notebooks = sorted(NB_DIR.rglob("*.ipynb"))
    global_defs: dict[str, list[dict]] = defaultdict(list)
    all_notebooks: list[dict] = []

    for nb_path in notebooks:
        parsed = parse_notebook(nb_path)
        all_notebooks.append(parsed)
        for name, sites in parsed["defs"].items():
            global_defs[name].extend(sites)

    # Focus on algorithm functions defined IN notebooks
    algo_funcs = {
        name: sites
        for name, sites in global_defs.items()
        if is_algo_related(name, sites[0]["sig"])
    }

    print("=" * 80)
    print("ALGORITHM FUNCTIONS DEFINED IN NOTEBOOKS")
    print("=" * 80)

    for nb_data in all_notebooks:
        nb_name = nb_data["path"].name
        nb_algo_defs = {
            name: sites
            for name, sites in nb_data["defs"].items()
            if is_algo_related(name, sites[0]["sig"])
        }
        if not nb_algo_defs:
            continue
        print(f"\n### {nb_name}")
        for name in sorted(nb_algo_defs):
            for site in nb_algo_defs[name]:
                print(f"  DEFINE  Cell {site['cell']:2d}, dòng {site['line']:3d}: def {name}()")
                print(f"          {site['sig']}")

    print("\n" + "=" * 80)
    print("CALL SITES FOR NOTEBOOK-DEFINED ALGORITHM FUNCTIONS")
    print("=" * 80)

    for name in sorted(algo_funcs):
        def_sites = algo_funcs[name]
        def_nb_set = {s["nb"] for s in def_sites}
        print(f"\n## {name}()")
        print("  Định nghĩa:")
        for s in def_sites:
            print(f"    - {s['nb']} | Cell {s['cell']}, dòng {s['line']}")

        # find calls across all notebooks
        calls = []
        for nb_data in all_notebooks:
            for c in nb_data["calls"]:
                if c["name"] == name:
                    calls.append(c)
        if not calls:
            print("  Gọi: (không tìm thấy lời gọi trực tiếp — có thể gọi qua cell chạy tuần tự)")
            continue
        print("  Gọi tại:")
        for c in calls:
            print(f"    - {c['nb']} | Cell {c['cell']}, dòng {c['line']}: {c['code']}")

    # Also report key sklearn/pytorch API usage per notebook
    print("\n" + "=" * 80)
    print("KEY LIBRARY ALGORITHM CALLS (fit/predict/train) BY NOTEBOOK")
    print("=" * 80)

    key_apis = re.compile(r"\.(fit|predict|predict_proba|fit_transform|transform|forward)\s*\(")
    for nb_data in all_notebooks:
        nb_path = nb_data["path"]
        hits = []
        with open(nb_path, encoding="utf-8") as f:
            nb = json.load(f)
        for ci, cell in enumerate(nb["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell.get("source", []))
            for li, line in enumerate(source.splitlines(), 1):
                if key_apis.search(line):
                    hits.append((ci + 1, li, line.strip()[:110]))
        if hits:
            print(f"\n### {nb_path.name}")
            for cell, line, code in hits[:30]:
                print(f"  Cell {cell:2d}, dòng {line:3d}: {code}")
            if len(hits) > 30:
                print(f"  ... và {len(hits)-30} dòng khác")


if __name__ == "__main__":
    main()