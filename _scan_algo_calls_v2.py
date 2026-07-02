"""Detailed scan: algorithm defs + all call sites with cell/line."""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NB_DIR = ROOT / "notebooks"

# Group algorithms by category for the report
CATEGORIES = {
    "ModernBERT & Feature": [
        "load_bert_model", "extract_or_load_bert_embeddings", "valid_cached_embedding",
        "add_basic_behavioral_features", "add_advanced_behavioral_features",
        "fit_reviewer_behavior_map", "build_basic_behavioral_features",
        "train_only_reduce_features",
    ],
    "PCA / SVD": ["array_memory_mb", "process_memory_mb"],
    "PSO + DL (CNN-BiLSTM)": [
        "build_model", "fit_model", "evaluate_model", "predict_probabilities",
        "evaluate_predictions", "safe_roc_auc", "safe_pr_auc",
        "decode_particle", "compute_pso_objective_score", "pso_objective",
        "fallback_pso", "normalize_pso_result",
    ],
    "LightGBM / XGBoost / MLP": ["write_metrics", "lightgbm_config", "fit_lightgbm_predict"],
    "CNN-BiLSTM Sequence": ["predict"],
    "Ensemble": ["threshold_sweep", "stack_matrix", "selected_weighted_blend"],
    "Adversarial & XAI": [
        "fgsm_attack_batch", "pgd_attack_batch", "clamp_features", "evaluate_condition",
        "predict_dl_proba", "predict_final_ensemble_fake_proba", "predict_xgb_raw_proba",
        "predict_xgb_raw_fake_proba", "select_final_lime_cases", "select_lime_cases",
        "build_raw_feature_names", "build_raw_feature_metadata",
    ],
    "Evaluation & Ablation": [
        "evaluate_probabilities", "evaluate_test_probability", "error_case_table",
        "load_multiseed_blend_probs", "select_multiseed_thresholds", "multiseed_prediction_dir",
    ],
    "Benchmark & Viz": ["compute_metrics", "metric_bundle", "choose_threshold_by_precision",
                        "train_model", "predict_probs", "ghep_metric_tu_nhieu_file", "loc_metric_test"],
}


def parse_nb(path: Path) -> tuple[dict[str, list], list[tuple[int, int, str]]]:
    with open(path, encoding="utf-8") as f:
        nb = json.load(f)

    defs: dict[str, list] = defaultdict(list)
    all_lines: list[tuple[int, int, str]] = []  # cell, line, text

    for ci, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell.get("source", []))
        for li, line in enumerate(source.splitlines(), 1):
            all_lines.append((ci + 1, li, line))
            m = re.match(r"^\s*def\s+(\w+)\s*\(", line)
            if m:
                defs[m.group(1)].append({"cell": ci + 1, "line": li, "sig": line.strip()[:90]})

    return dict(defs), all_lines


def find_calls(name: str, lines: list[tuple[int, int, str]], skip_def_line: set) -> list[dict]:
    pattern = re.compile(rf"\b{re.escape(name)}\s*\(")
    hits = []
    for cell, line, text in lines:
        if (cell, line) in skip_def_line:
            continue
        if pattern.search(text) and not re.match(rf"^\s*def\s+{name}\s*\(", text):
            hits.append({"cell": cell, "line": line, "code": text.strip()[:100]})
    return hits


def find_pattern(pattern: str, lines: list, label: str) -> list[dict]:
    rx = re.compile(pattern)
    return [
        {"cell": c, "line": l, "code": t.strip()[:100], "label": label}
        for c, l, t in lines if rx.search(t)
    ]


def main() -> None:
    notebooks = sorted(NB_DIR.rglob("*.ipynb"))
    nb_data = {}
    for p in notebooks:
        defs, lines = parse_nb(p)
        nb_data[p.name] = {"path": p, "defs": defs, "lines": lines}

    # Build global index
    global_defs: dict[str, list] = defaultdict(list)
    for name, data in nb_data.items():
        for fn, sites in data["defs"].items():
            for s in sites:
                global_defs[fn].append({**s, "nb": name})

    print("BÁO CÁO: HÀM THUẬT TOÁN — ĐỊNH NGHĨA & LỜI GỌI (Cell, Dòng)")
    print("=" * 90)

    reported = set()
    for cat, funcs in CATEGORIES.items():
        print(f"\n## {cat}\n")
        for fn in funcs:
            if fn not in global_defs:
                continue
            reported.add(fn)
            print(f"### `{fn}()`")
            for s in global_defs[fn]:
                print(f"  **Định nghĩa:** `{s['nb']}` → Cell **{s['cell']}**, dòng **{s['line']}**")
                print(f"    `{s['sig']}`")

            # find calls in all notebooks
            all_calls = []
            for nb_name, data in nb_data.items():
                skip = {(s["cell"], s["line"]) for s in data["defs"].get(fn, [])}
                calls = find_calls(fn, data["lines"], skip)
                for c in calls:
                    c["nb"] = nb_name
                    all_calls.append(c)

            if all_calls:
                print("  **Gọi tại:**")
                for c in all_calls:
                    print(f"    - `{c['nb']}` → Cell **{c['cell']}**, dòng **{c['line']}**: `{c['code']}`")
            else:
                print("  **Gọi tại:** *(gọi nội bộ class / callback / cùng cell — xem bên dưới)*")
            print()

    # Library-level algorithm calls (PSO, SHAP, LIME, etc.)
    print("\n## Gọi thư viện thuật toán (không tự định nghĩa hàm)\n")

    lib_patterns = [
        (r"pyswarm|from pyswarm|import pso", "PSO (pyswarm)"),
        (r"pso\.pmin|pmin\s*\(", "PSO pmin()"),
        (r"LGBMClassifier|lgb\.train|lightgbm", "LightGBM"),
        (r"XGBClassifier|xgb\.train|xgboost", "XGBoost"),
        (r"TruncatedSVD|PCA\s*\(", "PCA / TruncatedSVD"),
        (r"SHAP|shap\.|TreeExplainer|KernelExplainer", "SHAP"),
        (r"LimeTextExplainer|lime\.|explain_instance", "LIME"),
        (r"CalibratedClassifierCV|calibrat", "Calibration"),
        (r"RandomForestClassifier|LogisticRegression|LinearSVC", "Sklearn baseline"),
        (r"BiLSTM|nn\.LSTM|Conv1d", "CNN-BiLSTM architecture"),
        (r"ModernBERT|AutoModel|AutoTokenizer", "ModernBERT / HuggingFace"),
    ]

    for nb_name, data in sorted(nb_data.items()):
        if nb_name.startswith("08") or nb_name.startswith("09"):
            pass
        hits_by_label: dict[str, list] = defaultdict(list)
        for pat, label in lib_patterns:
            for h in find_pattern(pat, data["lines"], label):
                hits_by_label[label].append(h)
        if not hits_by_label:
            continue
        # Only main pipeline notebooks
        if not re.match(r"0[1-7]|10|05|tests", nb_name):
            continue
        print(f"### `{nb_name}`")
        for label, hits in hits_by_label.items():
            # show first 3 + count
            shown = hits[:4]
            print(f"  **{label}** ({len(hits)} lần):")
            for h in shown:
                print(f"    - Cell **{h['cell']}**, dòng **{h['line']}**: `{h['code']}`")
            if len(hits) > 4:
                print(f"    - ... +{len(hits)-4} dòng khác")
        print()

    # Phase 5 weighted blend inline (no function)
    print("\n## Logic ensemble không bọc hàm (code trực tiếp)\n")
    for nb in ["05_05_Weighted_Blending.ipynb", "05_Hybrid_Ensemble.ipynb"]:
        if nb not in nb_data:
            continue
        lines = nb_data[nb]["lines"]
        for pat, desc in [
            (r"blend|weighted|w_cnn|w_xgb|w_lgbm", "Weighted blending"),
            (r"np\.(average|dot|sum).*prob|prob.*\*.*weight", "Tính xác suất blend"),
        ]:
            hits = find_pattern(pat, lines, desc)
            if hits:
                print(f"### `{nb}` — {desc}")
                for h in hits[:6]:
                    print(f"  - Cell **{h['cell']}**, dòng **{h['line']}**: `{h['code']}`")
                print()


if __name__ == "__main__":
    main()