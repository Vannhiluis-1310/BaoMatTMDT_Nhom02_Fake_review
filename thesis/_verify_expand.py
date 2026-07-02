"""Verify expandChapter.md claims against artifacts and processed data."""
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables"
PROCESSED = ROOT / "data" / "processed"

TOL = 0.0006  # allow rounding to 4 decimals


def r4(x):
    return round(float(x), 4)


def check(name, claimed, actual, tol=TOL):
    ok = abs(float(claimed) - float(actual)) <= tol
    return {
        "check": name,
        "claimed": claimed,
        "actual": round(float(actual), 6),
        "actual_r4": r4(actual),
        "ok": ok,
    }


def load_test_row(csv_name, model_variant, threshold=0.5, extra=None):
    df = pd.read_csv(TABLES / csv_name)
    mask = (df["split"] == "test") & (df["model_variant"] == model_variant)
    if "threshold" in df.columns:
        mask &= df["threshold"].astype(float) == threshold
    if extra:
        for k, v in extra.items():
            mask &= df[k] == v
    rows = df[mask]
    if len(rows) != 1:
        raise ValueError(f"{csv_name} {model_variant}: expected 1 row, got {len(rows)}")
    return rows.iloc[0]


results = []

# --- Phase 1 cleaning ---
clean = pd.read_csv(TABLES / "phase1_cleaning_report.csv")
clean_map = dict(zip(clean["metric"], clean["value"]))
for metric, claimed in [
    ("original_rows", 50000),
    ("final_rows", 42749),
    ("duplicate_text_label_before_drop", 7194),
    ("rows_dropped_missing_text_or_label", 57),
    ("rows_removed_total", 7251),
]:
    results.append(check(f"phase1_{metric}", claimed, clean_map[metric], tol=0))

# --- Length stats ---
length = pd.read_csv(TABLES / "phase1_length_by_label.csv")
for label, col, claimed in [
    ("fake", "char_len_median", 43),
    ("real", "char_len_median", 125),
    ("fake", "word_count_median", 7),
    ("real", "word_count_median", 24),
]:
    row = length[length["label_name"] == label].iloc[0]
    results.append(check(f"length_{label}_{col}", claimed, row[col], tol=0))

# --- Verified / helpful from processed CSVs ---
frames = []
for split in ("train", "val", "test"):
    p = PROCESSED / f"{split}.csv"
    if p.exists():
        frames.append(pd.read_csv(p))
if frames:
    full = pd.concat(frames, ignore_index=True)
    assert len(full) == 42749, f"expected 42749 rows, got {len(full)}"

    def norm_label(x):
        if isinstance(x, str):
            return x.strip().lower()
        return x

    full["label_norm"] = full["label"].apply(norm_label)
    fake = full[full["label_norm"].isin(["fake", "1", 1])]
    real = full[full["label_norm"].isin(["real", "0", 0])]

    vp_col = None
    for c in ("verified_purchase", "basic_verified_purchase", "Verified"):
        if c in full.columns:
            vp_col = c
            break

    if vp_col:
        def to_bool(v):
            if pd.isna(v):
                return False
            if isinstance(v, bool):
                return v
            s = str(v).strip().lower()
            return s in ("true", "1", "yes", "t")

        real_vp = real[vp_col].apply(to_bool).mean() * 100
        fake_vp = fake[vp_col].apply(to_bool).mean() * 100
        all_vp = full[vp_col].apply(to_bool).mean() * 100
        results.append(check("verified_real_pct", 100.0, real_vp, tol=0.1))
        results.append(check("verified_fake_pct", 74.02, fake_vp, tol=0.05))
        results.append(check("verified_corpus_pct", 89.4, all_vp, tol=0.1))

    hv_col = None
    for c in ("helpful_vote", "helpful_votes", "total_votes", "helpful", "Helpful"):
        if c in full.columns:
            hv_col = c
            break
    if hv_col:
        real_h = real[hv_col].astype(float).mean()
        fake_h = fake[hv_col].astype(float).mean()
        results.append(check("helpful_real_mean", 1.05, real_h, tol=0.02))
        results.append(check("helpful_fake_mean", 0.868, fake_h, tol=0.02))

# --- phase7_final_metrics test @0.5 ---
fm = pd.read_csv(TABLES / "phase7_final_metrics.csv")
test = fm[(fm["split"] == "test") & (fm["threshold"].astype(float) == 0.5)]

claims = {
    "phase5_weighted_blend": {"macro_f1": 0.9433, "precision_fake": 0.9699, "recall_fake": 0.8956},
    "phase5_cnn_bilstm_sequence": {
        "macro_f1": 0.9343,
        "precision_fake": 0.9465,
        "recall_fake": 0.8967,
        "roc_auc": 0.9726,
    },
    "phase5_xgb_raw": {"macro_f1": 0.9059, "precision_fake": 0.9686, "recall_fake": 0.8106},
    "phase5_lgbm_raw": {"macro_f1": 0.9051, "precision_fake": 0.9677, "recall_fake": 0.8095},

    "dl_baseline": {"macro_f1": 0.7665},
    "dl_pso": {"macro_f1": 0.7793},
}
for model, metrics in claims.items():
    row = test[test["model_variant"] == model]
    if len(row) != 1:
        results.append({"check": f"{model}_row", "claimed": "1 row", "actual": len(row), "ok": False})
        continue
    row = row.iloc[0]
    for m, c in metrics.items():
        results.append(check(f"{model}_{m}", c, row[m]))

# balanced / precision-first thresholds
for thresh, strategy, macro, prec in [
    (0.3, "phase5_balanced_macro_f1_threshold", 0.9463, 0.9344),
    (0.6, "phase5_selected_precision_threshold", 0.9126, 0.9816),
]:
    row = fm[
        (fm["split"] == "test")
        & (fm["model_variant"] == "phase5_weighted_blend")
        & (fm["threshold"].astype(float) == thresh)
        & (fm["threshold_strategy"] == strategy)
    ]
    if len(row) == 1:
        results.append(check(f"blend_balanced_macro_f1_t{thresh}", macro, row.iloc[0]["macro_f1"]))
        if prec:
            results.append(check(f"blend_prec_fake_t{thresh}", prec, row.iloc[0]["precision_fake"]))

# --- ablation ---
abl = pd.read_csv(TABLES / "phase7_ablation_results.csv")
model_c = abl[abl["ablation_variant"] == "Model C"]
if len(model_c) == 1:
    results.append(check("model_c_delta_macro_f1", 0.0023, model_c.iloc[0]["controlled_delta_macro_f1"], tol=0.0001))

# --- stacking calibrated (separate artifact) ---
stack_cal = pd.read_csv(TABLES / "phase5_stacking_calibrated_metrics.csv")
sc_row = stack_cal[(stack_cal["split"] == "test") & (stack_cal["threshold"].astype(float) == 0.5)].iloc[0]
for m, c in {
    "macro_f1": 0.9105,
    "precision_fake": 0.9728,
    "recall_fake": 0.8175,
    "roc_auc": 0.9731,
}.items():
    results.append(check(f"stacking_calibrated_{m}", c, sc_row[m]))

# --- multiseed ---
ms = pd.read_csv(TABLES / "phase7_multiseed_summary.csv")
bal = ms[(ms["mode"] == "balanced") & (ms["metric"] == "macro_f1")].iloc[0]
results.append(check("multiseed_balanced_macro_f1_mean", 0.9485, bal["mean"], tol=0.0002))
results.append(check("multiseed_balanced_macro_f1_std", 0.0018, bal["std"], tol=0.0002))

# --- print report ---
failed = [r for r in results if not r["ok"]]
passed = [r for r in results if r["ok"]]
print(f"PASS: {len(passed)}  FAIL: {len(failed)}\n")
if failed:
    print("=== FAILURES ===")
    for r in failed:
        print(
            f"  {r['check']}: claimed={r['claimed']} actual={r['actual']} (r4={r.get('actual_r4', r['actual'])})"
        )
print("\n=== ALL CHECKS ===")
for r in results:
    mark = "OK" if r["ok"] else "FAIL"
    print(f"  [{mark}] {r['check']}: {r['claimed']} vs {r.get('actual_r4', r['actual'])}")

out = ROOT / "thesis" / "_verify_expand_report.json"
out.write_text(json.dumps(results, indent=2), encoding="utf-8")
print(f"\nWrote {out}")