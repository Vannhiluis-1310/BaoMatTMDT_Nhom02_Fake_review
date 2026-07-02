"""Train sklearn baselines on 777-d features and merge Phase 7 artifact metrics."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

SEED = 42
THRESHOLD = 0.5
THRESHOLD_STRATEGY = "default_0.5"
SPLIT = "test"

SKLEARN_MODELS = {
    "sklearn_logistic_regression": {
        "display_name": "Logistic Regression",
        "group": "classical",
        "order": 1,
        "estimator": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=SEED,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    },
    "sklearn_linear_svc": {
        "display_name": "Linear SVM (calibrated)",
        "group": "classical",
        "order": 2,
        "estimator": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    CalibratedClassifierCV(
                        LinearSVC(
                            max_iter=3000,
                            class_weight="balanced",
                            dual="auto",
                            random_state=SEED,
                        ),
                        cv=3,
                        method="sigmoid",
                    ),
                ),
            ]
        ),
    },
    "sklearn_random_forest": {
        "display_name": "Random Forest",
        "group": "classical",
        "order": 3,
        "estimator": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=300,
                        class_weight="balanced_subsample",
                        random_state=SEED,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    },
}

ARTIFACT_MODELS = [
    {
        "model_variant": "phase5_xgb_raw",
        "display_name": "XGBoost raw 777-d",
        "group": "pipeline",
        "order": 4,
        "highlight": 0,
    },
    {
        "model_variant": "phase5_cnn_bilstm_sequence",
        "display_name": "CNN-BiLSTM sequence",
        "group": "pipeline",
        "order": 5,
        "highlight": 0,
    },
    {
        "model_variant": "phase5_weighted_blend",
        "display_name": "Weighted blend (de xuat)",
        "group": "pipeline",
        "order": 6,
        "highlight": 1,
    },
]


def find_project_root() -> Path:
    candidates = [
        Path(__file__).resolve().parents[1],
        Path.cwd(),
        Path("/content/drive/MyDrive/BaoMatCuoiKy/Fake_reviews"),
    ]
    for path in candidates:
        if (path / "artifacts" / "features" / "features_raw_train.npy").exists():
            return path
    raise FileNotFoundError("Could not locate Fake_reviews project root.")


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "precision_fake": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_fake": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_fake": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "support_real": int((y_true == 0).sum()),
        "support_fake": int((y_true == 1).sum()),
    }


def row_from_metrics(
    *,
    generated_at: str,
    model_variant: str,
    display_name: str,
    model_group: str,
    display_order: int,
    highlight: int,
    evidence_type: str,
    metrics: dict,
    probability_path: str,
    notes: str,
    tier: str,
) -> dict:
    return {
        "generated_at_utc": generated_at,
        "seed": SEED,
        "split": SPLIT,
        "tier": tier,
        "model_variant": model_variant,
        "display_name": display_name,
        "model_group": model_group,
        "display_order": display_order,
        "highlight": highlight,
        "evidence_type": evidence_type,
        "threshold": THRESHOLD,
        "threshold_strategy": THRESHOLD_STRATEGY,
        "probability_path": probability_path,
        "notes": notes,
        **metrics,
    }


def main() -> None:
    root = find_project_root()
    feat_dir = root / "artifacts" / "features"
    pred_dir = root / "artifacts" / "predictions"
    table_dir = root / "reports" / "tables"
    eval_dir = root / "artifacts" / "evaluation"
    table_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    x_train = np.load(feat_dir / "features_raw_train.npy")
    x_test = np.load(feat_dir / "features_raw_test.npy")
    y_train = np.load(feat_dir / "labels_train.npy")
    y_test = np.load(feat_dir / "labels_test.npy")

    generated_at = datetime.now(timezone.utc).isoformat()
    rows: list[dict] = []

    for model_variant, spec in SKLEARN_MODELS.items():
        print(f"[TRAIN] {model_variant} ...")
        estimator = spec["estimator"]
        estimator.fit(x_train, y_train)
        y_prob = estimator.predict_proba(x_test)[:, 1]
        prob_path = pred_dir / f"{model_variant}_{SPLIT}_prob.npy"
        np.save(prob_path, y_prob)
        metrics = compute_metrics(y_test, y_prob, THRESHOLD)
        rows.append(
            row_from_metrics(
                generated_at=generated_at,
                model_variant=model_variant,
                display_name=spec["display_name"],
                model_group=spec["group"],
                display_order=spec["order"],
                highlight=0,
                evidence_type="trained_sklearn_baseline",
                metrics=metrics,
                probability_path=str(prob_path),
                notes="Trained on features_raw_777 train-only StandardScaler; fair Tier-1 benchmark @ tau=0.5.",
                tier="tier1_fair_internal",
            )
        )
        print(f"  macro_f1={metrics['macro_f1']:.4f} roc_auc={metrics['roc_auc']:.4f}")

    phase7_path = table_dir / "phase7_final_metrics.csv"
    if not phase7_path.exists():
        raise FileNotFoundError(f"Missing {phase7_path}")
    df_phase7 = pd.read_csv(phase7_path)
    mask = (
        (df_phase7["seed"] == SEED)
        & (df_phase7["split"] == SPLIT)
        & (df_phase7["threshold"] == THRESHOLD)
        & (df_phase7["threshold_strategy"] == THRESHOLD_STRATEGY)
    )

    for spec in ARTIFACT_MODELS:
        subset = df_phase7.loc[df_phase7["model_variant"] == spec["model_variant"]]
        subset = subset.loc[mask] if not subset.empty else subset
        if subset.empty:
            raise ValueError(f"Missing Phase 7 row for {spec['model_variant']}")
        row = subset.iloc[0]
        rows.append(
            row_from_metrics(
                generated_at=generated_at,
                model_variant=spec["model_variant"],
                display_name=spec["display_name"],
                model_group=spec["group"],
                display_order=spec["order"],
                highlight=spec["highlight"],
                evidence_type="phase7_artifact_reuse",
                metrics={
                    "accuracy": float(row["accuracy"]),
                    "macro_f1": float(row["macro_f1"]),
                    "precision_fake": float(row["precision_fake"]),
                    "recall_fake": float(row["recall_fake"]),
                    "f1_fake": float(row["f1_fake"]),
                    "roc_auc": float(row["roc_auc"]),
                    "pr_auc": float(row["pr_auc"]),
                    "brier_score": float(row["brier_score"]),
                    "tn": int(row["tn"]),
                    "fp": int(row["fp"]),
                    "fn": int(row["fn"]),
                    "tp": int(row["tp"]),
                    "support_real": int(row["support_real"]),
                    "support_fake": int(row["support_fake"]),
                },
                probability_path=str(row["probability_path"]),
                notes="Reused from phase7_final_metrics.csv; no retrain.",
                tier="tier1_fair_internal",
            )
        )

    df_out = pd.DataFrame(rows).sort_values("display_order").reset_index(drop=True)
    out_csv = table_dir / "baseline_benchmark_metrics.csv"
    df_out.to_csv(out_csv, index=False)

    config_rows = [
        {
            "model_variant": r["model_variant"],
            "ten_hien_thi": r["display_name"],
            "nhom": r["model_group"],
            "thu_tu": int(r["display_order"]),
            "highlight": int(r["highlight"]),
            "tier": r["tier"],
        }
        for r in rows
    ]
    pd.DataFrame(config_rows).to_csv(table_dir / "baseline_benchmark_model_config.csv", index=False)

    metadata = {
        "generated_at_utc": generated_at,
        "seed": SEED,
        "split": SPLIT,
        "threshold": THRESHOLD,
        "threshold_strategy": THRESHOLD_STRATEGY,
        "feature_dim": int(x_train.shape[1]),
        "train_rows": int(x_train.shape[0]),
        "test_rows": int(x_test.shape[0]),
        "tier1_model_count": len(rows),
        "sklearn_models": list(SKLEARN_MODELS.keys()),
        "artifact_models": [s["model_variant"] for s in ARTIFACT_MODELS],
        "outputs": {
            "metrics_csv": str(out_csv),
            "model_config_csv": str(table_dir / "baseline_benchmark_model_config.csv"),
            "metadata_json": str(eval_dir / "baseline_benchmark_metadata.json"),
        },
        "known_limitations": [
            "Tier-1 fair comparison uses the same 777-d features and tau=0.5 for all models.",
            "Literature SOTA rows are exported separately and must not share the same bar chart axis.",
            "Linear SVM uses CalibratedClassifierCV(sigmoid, cv=3) for probability outputs.",
        ],
    }
    meta_path = eval_dir / "baseline_benchmark_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    lit_path = table_dir / "literature_sota_comparison.csv"
    if lit_path.exists():
        df_lit = pd.read_csv(lit_path)
        df_lit.head(10).to_csv(table_dir / "baseline_benchmark_literature_tier2.csv", index=False)

    print(f"[OK] Wrote {out_csv} ({len(df_out)} models)")
    print(df_out[["model_variant", "macro_f1", "roc_auc"]].to_string(index=False))


if __name__ == "__main__":
    main()