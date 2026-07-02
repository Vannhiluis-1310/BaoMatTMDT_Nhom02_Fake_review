# -*- coding: utf-8 -*-
"""Generate complete algorithm function map: def + calls with cell/line."""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NB_DIR = ROOT / "notebooks"
OUT = ROOT / "BANG_HAM_THUAT_TOAN_CELL_DONG.md"

# All user-defined algorithm functions to track
TRACK = None  # None = all algo-related


def parse_nb(path: Path):
    with open(path, encoding="utf-8") as f:
        nb = json.load(f)
    defs = defaultdict(list)
    lines = []
    for ci, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell.get("source", []))
        for li, line in enumerate(src.splitlines(), 1):
            lines.append((ci + 1, li, line))
            m = re.match(r"^\s*def\s+(\w+)\s*\(", line)
            if m:
                defs[m.group(1)].append({"cell": ci + 1, "line": li, "sig": line.strip()})
    return defs, lines


def calls_of(name: str, lines, skip):
    pat = re.compile(rf"\b{re.escape(name)}\s*\(")
    out = []
    for c, l, t in lines:
        if (c, l) in skip:
            continue
        if pat.search(t) and not re.match(rf"^\s*def\s+{name}\s*\(", t):
            out.append((c, l, t.strip()))
    return out


def main():
    nbs = sorted(NB_DIR.rglob("*.ipynb"))
    all_nb = {}
    global_defs = defaultdict(list)

    for p in nbs:
        defs, lines = parse_nb(p)
        all_nb[p.name] = {"defs": defs, "lines": lines}
        for fn, sites in defs.items():
            for s in sites:
                global_defs[fn].append({**s, "nb": p.name})

    # Phase-grouped report structure
    phases = [
        ("Phase 1 — EDA", ["01_EDA_Preprocessing.ipynb"]),
        ("Phase 2 — ModernBERT & Behavioral", ["02_Feature_Engineering.ipynb"]),
        ("Phase 3 — PCA/SVD", ["03_PCA_Feature_Selection.ipynb"]),
        ("Phase 4 — PSO + DL", ["04_PSO_Model_Training.ipynb"]),
        ("Phase 5 — Model Zoo & Ensemble", [
            "05_00_Phase5_Run_Order.ipynb", "05_01_LightGBM_Raw.ipynb",
            "05_02_XGBoost_Raw.ipynb", "05_03_MLP_Raw.ipynb",
            "05_04a_CNN_BiLSTM_Sequence_seed42.ipynb",
            "05_04b_CNN_BiLSTM_Sequence_seed123.ipynb",
            "05_04c_CNN_BiLSTM_Sequence_seed456.ipynb",
            "05_05_Weighted_Blending.ipynb", "05_06_Stacking_Calibration.ipynb",
            "05_Hybrid_Ensemble.ipynb",
        ]),
        ("Phase 6 — Adversarial & XAI", ["06_Adversarial_XAI.ipynb"]),
        ("Phase 7 — Evaluation & Ablation", ["07_Evaluation_Ablation.ipynb"]),
        ("Phase 9 — Diagnostic", ["tests/01_DL_PCA_Diagnostic_Test.ipynb"]),
        ("Benchmark & Viz", ["10_Baseline_Algorithm_Benchmark.ipynb", "09_Bieu_Do_So_Sanh_Mo_Hinh.ipynb"]),
    ]

    lines_out = []
    lines_out.append("# Bảng hàm thuật toán — Cell & Dòng gọi\n")
    lines_out.append("> Số **Cell** đếm từ 1 (cell code đầu tiên trong notebook, bỏ qua markdown).\n")
    lines_out.append("> **Dòng** là số dòng trong source của cell đó.\n\n---\n")

    reported_fns = set()

    for phase_name, nb_list in phases:
        lines_out.append(f"\n## {phase_name}\n")
        phase_has = False
        for nb_name in nb_list:
            key = nb_name.split("/")[-1]
            if key not in all_nb:
                continue
            data = all_nb[key]
            algo_fns = sorted(data["defs"].keys())
            if not algo_fns:
                continue
            lines_out.append(f"\n### `{nb_name}`\n")
            lines_out.append("| Hàm | Định nghĩa (Cell, Dòng) | Gọi tại (Cell, Dòng) |")
            lines_out.append("|-----|------------------------|----------------------|")

            for fn in algo_fns:
                reported_fns.add(fn)
                def_site = data["defs"][fn][0]
                def_str = f"**{def_site['cell']}**, dòng **{def_site['line']}**"
                skip = {(s["cell"], s["line"]) for s in data["defs"][fn]}
                local_calls = calls_of(fn, data["lines"], skip)
                # also cross-notebook calls
                cross = []
                for other_name, other in all_nb.items():
                    if other_name == key:
                        continue
                    oskip = {(s["cell"], s["line"]) for s in other["defs"].get(fn, [])}
                    for c, l, t in calls_of(fn, other["lines"], oskip):
                        cross.append(f"`{other_name}` C**{c}** L**{l}**")
                call_parts = [f"C**{c}** L**{l}**" for c, l, _ in local_calls[:5]]
                if cross:
                    call_parts.extend(cross[:3])
                if not call_parts:
                    call_str = "*(method/callback — xem mục bổ sung)*"
                else:
                    extra = ""
                    if len(local_calls) > 5:
                        extra = f" (+{len(local_calls)-5})"
                    call_str = "; ".join(call_parts) + extra
                lines_out.append(f"| `{fn}()` | {def_str} | {call_str} |")
            phase_has = True

        if not phase_has:
            lines_out.append("*(không có hàm tự định nghĩa)*\n")

    # Library calls section
    lines_out.append("\n---\n\n## Gọi thư viện thuật toán (không bọc hàm riêng)\n")
    lib_entries = [
        ("04_PSO_Model_Training.ipynb", "PSO `pmin`", r"pmin\s*\("),
        ("04_PSO_Model_Training.ipynb", "PSO import", r"pyswarm|from pyswarm"),
        ("02_Feature_Engineering.ipynb", "ModernBERT forward", r"model\(|AutoModel|last_hidden_state"),
        ("03_PCA_Feature_Selection.ipynb", "PCA/SVD fit", r"\.fit\(|TruncatedSVD|PCA\("),
        ("05_01_LightGBM_Raw.ipynb", "LGBM fit/predict", r"LGBMClassifier|\.fit\(|predict_proba"),
        ("05_02_XGBoost_Raw.ipynb", "XGB fit/predict", r"XGBClassifier|\.fit\(|predict_proba"),
        ("05_06_Stacking_Calibration.ipynb", "Stacking fit", r"\.fit\(|CalibratedClassifierCV"),
        ("06_Adversarial_XAI.ipynb", "SHAP", r"shap\.|TreeExplainer|KernelExplainer"),
        ("06_Adversarial_XAI.ipynb", "LIME", r"LimeTextExplainer|explain_instance"),
        ("10_Baseline_Algorithm_Benchmark.ipynb", "Sklearn fit", r"\.fit\(|predict_proba"),
    ]
    lines_out.append("| Notebook | Thuật toán | Cell | Dòng | Code |")
    lines_out.append("|----------|------------|------|------|------|")
    for nb_name, label, pat in lib_entries:
        if nb_name not in all_nb:
            continue
        rx = re.compile(pat, re.I)
        shown = 0
        for c, l, t in all_nb[nb_name]["lines"]:
            if rx.search(t):
                code = t.strip()[:60].replace("|", "/")
                lines_out.append(f"| `{nb_name}` | {label} | **{c}** | **{l}** | `{code}` |")
                shown += 1
                if shown >= 3:
                    break

    # Supplement: missed calls found manually
    lines_out.append("\n---\n\n## Bổ sung — lời gọi quan trọng (đã xác minh thủ công)\n")
    manual = [
        ("02_Feature_Engineering.ipynb", "load_bert_model()", "Định nghĩa", 10, 32),
        ("02_Feature_Engineering.ipynb", "load_bert_model()", "Gọi", 11, 2),
        ("02_Feature_Engineering.ipynb", "extract_or_load_bert_embeddings()", "Gọi train", 11, 16),
        ("02_Feature_Engineering.ipynb", "extract_or_load_bert_embeddings()", "Gọi val", 11, 22),
        ("02_Feature_Engineering.ipynb", "extract_or_load_bert_embeddings()", "Gọi test", 11, 28),
        ("02_Feature_Engineering.ipynb", "add_basic_behavioral_features()", "Gọi (dict comp)", 15, 8),
        ("02_Feature_Engineering.ipynb", "fit_reviewer_behavior_map()", "Gọi", 16, 2),
        ("02_Feature_Engineering.ipynb", "add_advanced_behavioral_features()", "Gọi (dict comp)", 17, 8),
        ("04_PSO_Model_Training.ipynb", "pso_objective()", "Callback PSO", 20, 8),
        ("04_PSO_Model_Training.ipynb", "fallback_pso()", "Gọi pmin", 20, 120),
        ("04_PSO_Model_Training.ipynb", "normalize_pso_result()", "Gọi", 20, 155),
        ("04_PSO_Model_Training.ipynb", "predict_probabilities()", "Gọi trong evaluate_model", 14, 118),
        ("05_03_MLP_Raw.ipynb", "predict()", "Gọi val mỗi epoch", 7, 97),
        ("05_03_MLP_Raw.ipynb", "predict()", "Gọi prob_map", 7, 132),
        ("05_04a_*", "predict()", "Gọi prob_map", 9, 103),
        ("05_05_Weighted_Blending.ipynb", "threshold_sweep()", "Gọi sweep", 6, 56),
        ("05_05_Weighted_Blending.ipynb", "threshold_sweep()", "Gọi grid", 7, 54),
        ("05_06_Stacking_Calibration.ipynb", "stack_matrix()", "Gọi build X_stack", 7, 4),
        ("05_Hybrid_Ensemble.ipynb", "threshold_sweep()", "Gọi", 8, 14),
        ("06_Adversarial_XAI.ipynb", "fgsm_attack_batch()", "Gọi", 20, 13),
        ("06_Adversarial_XAI.ipynb", "pgd_attack_batch()", "Gọi", 20, 17),
        ("06_Adversarial_XAI.ipynb", "evaluate_condition()", "Gọi clean/fgsm/pgd", 20, 45),
        ("06_Adversarial_XAI.ipynb", "select_final_lime_cases()", "Gọi", 13, 2),
        ("06_Adversarial_XAI.ipynb", "select_lime_cases()", "Gọi", 27, 2),
        ("07_Evaluation_Ablation.ipynb", "selected_weighted_blend()", "Gọi ablation", 19, 60),
        ("07_Evaluation_Ablation.ipynb", "fit_lightgbm_predict()", "Gọi CV fold", 15, 110),
        ("07_Evaluation_Ablation.ipynb", "error_case_table()", "Gọi", 23, 2),
        ("07_Evaluation_Ablation.ipynb", "load_multiseed_blend_probs()", "Gọi", 27, 8),
        ("07_Evaluation_Ablation.ipynb", "select_multiseed_thresholds()", "Gọi", 27, 15),
        ("09_Bieu_Do_So_Sanh_Mo_Hinh.ipynb", "ghep_metric_tu_nhieu_file()", "Gọi", 7, 2),
        ("09_Bieu_Do_So_Sanh_Mo_Hinh.ipynb", "loc_metric_test()", "Gọi trong ghep_metric", 6, 35),
    ]
    lines_out.append("| Notebook | Hàm | Loại | Cell | Dòng |")
    lines_out.append("|----------|-----|------|------|------|")
    for nb, fn, typ, cell, line in manual:
        lines_out.append(f"| `{nb}` | `{fn}` | {typ} | **{cell}** | **{line}** |")

    OUT.write_text("\n".join(lines_out), encoding="utf-8")
    print(f"Written: {OUT}")
    print(f"Functions tracked: {len(reported_fns)}")


if __name__ == "__main__":
    main()