---
phase: 02-feature-engineering
verified: 2026-06-01T08:58:35Z
status: verified_with_gap
score: 5/6 must-haves verified
mode: initial_verification
constraints_observed:
  - "Did not run notebook cells."
  - "Did not run EDA, training, tuning, or dataset processing locally."
  - "Inspected notebook JSON, planning docs, CSV/JSON reports, file listings, and lightweight array shapes only."
gaps:
  - truth: "BERT embeddings are cached for all train/validation/test splits."
    status: partial
    reason: "Validation and test standalone BERT caches exist, and fused train features exist, but artifacts/features/bert_train.npy is absent even though metadata and reports claim it was generated."
    artifacts:
      - path: "artifacts/features/bert_train.npy"
        issue: "Missing from workspace."
      - path: "artifacts/features/feature_metadata.json"
        issue: "References /content/drive/MyDrive/BaoMatCuoiKy/Fake_reviews/artifacts/features/bert_train.npy."
      - path: "reports/tables/phase2_bert_embedding_report.csv"
        issue: "Reports train BERT rows=29923, dim=768, dtype=float32, status=generated, path ending in bert_train.npy."
      - path: "reports/tables/phase2_memory_report.csv"
        issue: "Reports bert_train.npy shape=(29923, 768), dtype=float32, memory_mb=87.665, but the file is not present."
    missing:
      - "Restore or regenerate artifacts/features/bert_train.npy, or update metadata/reports to state that only fused train features are retained."
  - truth: "BERT cache is reusable without forced recomputation."
    status: partial
    reason: "The notebook contains cache validation/loading logic, but the checked notebook default sets FORCE_REBUILD_BERT = True, so rerunning it skips cache reuse unless manually edited."
    artifacts:
      - path: "notebooks/02_Feature_Engineering.ipynb"
        issue: "Cell phase2_imports_config sets FORCE_REBUILD_BERT = True; cell defining valid_cached_embedding returns False when FORCE_REBUILD_BERT is true."
    missing:
      - "Set the final notebook default to reuse valid cache artifacts, or document the force-rebuild setting as an intentional one-time rebuild mode."
---

# Phase 2: Feature Engineering Verification Report

**Phase goal:** Produce BERT embeddings plus 9 behavioral features, leakage-aware feature artifacts, and metadata usable by Phase 3 PCA/SVD.

**Verdict:** `verified_with_gap`

Phase 2 is usable by Phase 3: fused raw feature matrices, labels, behavioral feature files, feature dictionary, leakage report, and metadata are present and internally consistent. The blocking model-input path is not broken. The remaining gap is the standalone BERT cache contract: `bert_train.npy` is missing locally, and the notebook default forces BERT cache rebuilds.

## Verification Scope

- Read `AGENTS.md`, `.planning/PROJECT.md`, `.planning/REQUIREMENTS.md`, `.planning/ROADMAP.md`, `.planning/TASKLIST.md`, `.planning/NOTEBOOK_PLAN.md`, and `.planning/PROJECT_STRUCTURE.md`.
- No previous Phase 2 verification existed; `.planning/phases/02-feature-engineering/` was absent before this report.
- `gsd-sdk` was not available in this PowerShell environment, so roadmap/requirements were read directly.
- No project skill directories were present under `.claude/skills/` or `.agents/skills/`.
- No notebook cells, EDA, training, tuning, or dataset-processing jobs were run.

## Must-Haves

| # | Must-have | Status | Evidence |
|---|---|---|---|
| 1 | Phase 2 notebook contains the executable pipeline source and follows Colab/local guard policy. | VERIFIED | `notebooks/02_Feature_Engineering.ipynb` exists; it has a Colab guard that raises on non-Colab execution, reads Phase 1 split files, and contains BERT, behavioral, fusion, report, and metadata code. No main pipeline `.py` files were found by `rg --files -g "*.py"`. |
| 2 | BERT embeddings are produced and cached by split. | VERIFIED_WITH_GAP | Model metadata: `answerdotai/ModernBERT-base`, pooling `masked_mean_last_hidden_state`, max_length 160, batch_size 8, embedding_dim 768. `artifacts/features/bert_val.npy` and `artifacts/features/bert_test.npy` exist with shape `(6413, 768)` float32. `artifacts/features/features_raw_train.npy` exists with shape `(29923, 777)`, showing train BERT-derived vectors were fused. Gap: `artifacts/features/bert_train.npy` is missing despite reports claiming it exists. |
| 3 | Five basic and four advanced behavioral features exist with formulas. | VERIFIED | `artifacts/features/feature_dictionary.csv` has 9 rows; `artifacts/features/behavioral_{train,val,test}.csv` each has `row_id` plus 9 feature columns. Feature groups are 5 basic and 4 advanced. |
| 4 | Behavioral features are scaled train-only and fused with BERT into raw feature matrices. | VERIFIED | `notebooks/02_Feature_Engineering.ipynb` cell 19 fits `StandardScaler()` with `fit_transform` on train and uses `transform` for val/test, then concatenates BERT and behavioral arrays. Fused shapes are `(29923, 777)`, `(6413, 777)`, `(6413, 777)`. |
| 5 | Leakage checks and train-only policies are documented. | VERIFIED | `reports/tables/phase2_leakage_check_report.csv` has 9 pass rows, including label formulas not using labels, train-only rating mean, train-only reviewer behavior map, scaler fit only in fusion, and val/test unknown reviewer fallback counts. |
| 6 | Metadata/artifacts are usable by Phase 3. | VERIFIED | `artifacts/pca/phase3_metadata.json` snapshots Phase 2 with seed 42, bert_dim 768, behavioral_count 9, fused_dim 777, and row counts 29923/6413/6413. `reports/tables/phase3_input_validation.csv` passes for all splits using Phase 2 raw features and labels. |

**Score:** 5/6 must-haves verified. The failed portion is limited to standalone BERT train-cache reuse, not the Phase 3 feature handoff.

## Artifact Evidence

### Phase 2 Notebook

| Artifact | Status | Evidence |
|---|---|---|
| `notebooks/02_Feature_Engineering.ipynb` | VERIFIED | 21 cells. Contains Colab-only guard, Drive mount, Phase 1 split loading, BERT extraction/loading, 9 behavioral features, train-only scaling, fused feature saves, reports, and metadata. |
| `notebooks/03_PCA_Feature_Selection.ipynb` | VERIFIED AS CONSUMER | Reads `artifacts/features/features_raw_train.npy`, `features_raw_val.npy`, `features_raw_test.npy`, and labels; fits scaler/reducer on train and transforms val/test. |

### Split Inputs

| Path | Rows | Columns | Status |
|---|---:|---:|---|
| `data/processed/train.csv` | 29923 | 16 | VERIFIED |
| `data/processed/val.csv` | 6413 | 16 | VERIFIED |
| `data/processed/test.csv` | 6413 | 16 | VERIFIED |

`reports/tables/phase2_input_validation.csv` confirms 0 missing text and 0 missing labels for all splits, with label counts:

- train: label 0 = 17677, label 1 = 12246
- val: label 0 = 3789, label 1 = 2624
- test: label 0 = 3789, label 1 = 2624

### BERT Embeddings

| Path | Shape | Dtype | Size / Reported Memory | Status |
|---|---:|---|---:|---|
| `artifacts/features/bert_train.npy` | expected `(29923, 768)` | expected float32 | reported 87.665 MB | MISSING |
| `artifacts/features/bert_val.npy` | `(6413, 768)` | float32 | 19700864 bytes / 18.7881 MB | VERIFIED |
| `artifacts/features/bert_test.npy` | `(6413, 768)` | float32 | 19700864 bytes / 18.7881 MB | VERIFIED |

`reports/tables/phase2_bert_embedding_report.csv` reports all three splits as generated:

| Split | Rows | Dim | Dtype | Elapsed Seconds | Model |
|---|---:|---:|---|---:|---|
| train | 29923 | 768 | float32 | 299.748 | `answerdotai/ModernBERT-base` |
| val | 6413 | 768 | float32 | 62.628 | `answerdotai/ModernBERT-base` |
| test | 6413 | 768 | float32 | 65.102 | `answerdotai/ModernBERT-base` |

The report and `feature_metadata.json` both reference `bert_train.npy`, but the file listing under `artifacts/features/` does not contain it.

### Behavioral Features

| Path | Rows | Columns | Status |
|---|---:|---:|---|
| `artifacts/features/behavioral_train.csv` | 29923 | 10 | VERIFIED |
| `artifacts/features/behavioral_val.csv` | 6413 | 10 | VERIFIED |
| `artifacts/features/behavioral_test.csv` | 6413 | 10 | VERIFIED |
| `artifacts/features/behavioral_scaler.joblib` | n/a | n/a | VERIFIED, exists, 1295 bytes |
| `artifacts/features/feature_dictionary.csv` | 9 | 6 | VERIFIED |

Behavioral feature columns:

- `basic_char_len_log`
- `basic_word_count_log`
- `basic_rating_deviation`
- `basic_sentiment_compound`
- `basic_verified_purchase`
- `adv_review_velocity_30d`
- `adv_product_burst_7d`
- `adv_reviewer_behavior_score`
- `adv_time_gap_hours_log`

`feature_metadata.json` records `behavioral_features.count = 9`, `sentiment_backend = nltk_vader`, `train_rating_mean = 3.9354342813220597`, and reviewer behavior metadata with `reviewer_count = 29628`.

### Fused Raw Features And Labels

| Path | Shape | Dtype | Size / Reported Memory | Status |
|---|---:|---|---:|---|
| `artifacts/features/features_raw_train.npy` | `(29923, 777)` | float32 | 93000812 bytes / 88.6924 MB | VERIFIED |
| `artifacts/features/features_raw_val.npy` | `(6413, 777)` | float32 | 19931732 bytes / 19.0083 MB | VERIFIED |
| `artifacts/features/features_raw_test.npy` | `(6413, 777)` | float32 | 19931732 bytes / 19.0083 MB | VERIFIED |
| `artifacts/features/labels_train.npy` | `(29923,)` | int64 | 239512 bytes | VERIFIED |
| `artifacts/features/labels_val.npy` | `(6413,)` | int64 | 51432 bytes | VERIFIED |
| `artifacts/features/labels_test.npy` | `(6413,)` | int64 | 51432 bytes | VERIFIED |
| `artifacts/features/row_ids_train.csv` | 29923 rows | 4 columns | 1832739 bytes | VERIFIED |
| `artifacts/features/row_ids_val.csv` | 6413 rows | 4 columns | 387644 bytes | VERIFIED |
| `artifacts/features/row_ids_test.csv` | 6413 rows | 4 columns | 387544 bytes | VERIFIED |

`reports/tables/phase2_feature_quality_report.csv` confirms:

| Split | Rows | BERT Dim | Behavioral Dim | Fused Dim | Missing Behavioral | Non-Finite Fused |
|---|---:|---:|---:|---:|---:|---:|
| train | 29923 | 768 | 9 | 777 | 0 | 0 |
| val | 6413 | 768 | 9 | 777 | 0 | 0 |
| test | 6413 | 768 | 9 | 777 | 0 | 0 |

### Metadata And Reports

| Path | Status | Key Evidence |
|---|---|---|
| `artifacts/features/feature_metadata.json` | VERIFIED_WITH_GAP | Seed 42; BERT model and dimensions recorded; 9 behavioral features recorded; fusion order and fused_dim 777 recorded; leakage policy recorded. Gap: references missing `bert_train.npy`. |
| `reports/tables/phase2_input_validation.csv` | VERIFIED | 3 pass rows for train/val/test. |
| `reports/tables/phase2_bert_embedding_report.csv` | VERIFIED_WITH_GAP | Reports all BERT splits generated; conflicts with missing local `bert_train.npy`. |
| `reports/tables/phase2_feature_quality_report.csv` | VERIFIED | Rows and dimensions match expected Phase 2 outputs. |
| `reports/tables/phase2_leakage_check_report.csv` | VERIFIED | 9 pass rows. |
| `reports/tables/phase2_memory_report.csv` | VERIFIED_WITH_GAP | Reports memory for all BERT and fused artifacts; conflicts with missing local `bert_train.npy`. |

## Leakage Review

`reports/tables/phase2_leakage_check_report.csv` contains all pass statuses:

| Check | Status | Detail |
|---|---|---|
| behavioral_feature_count | pass | 9 |
| label_not_used_in_feature_formulas | pass | No behavioral feature formula uses labels. |
| rating_deviation_fit_policy | pass | Uses train_rating_mean only. |
| reviewer_behavior_map_fit_policy | pass | Generated train-fitted reviewer map without labels. |
| timestamp_parse_rates | pass | train/val/test parse rate 1.0 |
| sentiment_backend | pass | nltk_vader |
| scaler_not_fit_before_fusion_section | pass | Scaler fit occurs only in fusion section. |
| val_unknown_reviewer_fallback_count | pass | 6299 |
| test_unknown_reviewer_fallback_count | pass | 6307 |

Notebook wiring supports this report:

- Basic rating deviation uses `train_rating_mean`.
- Advanced temporal features use prior windows and train history for validation/test.
- Reviewer behavior score map is fit from train-derived behavior summaries and labels are not used.
- `StandardScaler` uses train `fit_transform` and val/test `transform`.

## Phase 3 Handoff

Phase 3 can consume Phase 2 outputs.

| Evidence | Status |
|---|---|
| `artifacts/pca/phase3_metadata.json` has `source_phase2_metadata_path`, Phase 2 snapshot, input feature paths, and row counts. | VERIFIED |
| Phase 2 snapshot in `phase3_metadata.json`: seed 42, bert_dim 768, behavioral_count 9, fused_dim 777, row counts train 29923 / val 6413 / test 6413. | VERIFIED |
| `reports/tables/phase3_input_validation.csv` passes for train/val/test with feature_dim 777 and matching label rows. | VERIFIED |
| Phase 3 reduced outputs exist: `features_final_train.npy` `(29923, 400)`, `features_final_val.npy` `(6413, 400)`, `features_final_test.npy` `(6413, 400)`, all float32. | VERIFIED |

This confirms the Phase 2 feature artifacts are usable by Phase 3 despite the standalone train BERT cache gap.

## Anti-Pattern Scan

| Check | Result | Impact |
|---|---|---|
| Main pipeline `.py` files | None found by `rg --files -g "*.py"`. | OK |
| Placeholder/TODO markers in Phase 2 notebook and artifacts | No Phase 2 placeholder blocker found. TODO markers are in later placeholder notebooks `06`, `07`, and `08`; not Phase 2. | Not a Phase 2 gap |
| Hardcoded empty output artifacts | No empty Phase 2 output artifacts found in inspected CSV/JSON/shape checks. | OK |
| Cache consistency | `bert_train.npy` missing while metadata/report tables claim it exists. | Gap |
| Cache reuse default | `FORCE_REBUILD_BERT = True` prevents automatic reuse of valid BERT cache on rerun. | Gap |

## Behavioral Spot-Checks

No runnable pipeline commands were executed, by project and user constraint. Lightweight non-mutating checks performed:

- File existence/listing under `notebooks/`, `artifacts/features/`, `artifacts/pca/`, and `reports/tables/`.
- Notebook JSON inspection for code wiring.
- CSV header/row counts for Phase 2 reports and feature files.
- JSON metadata inspection.
- NumPy `.npy` shape/dtype inspection with read-only mmap.

## Gaps Summary

Phase 2 achieves the model-input and Phase 3 handoff goals: `features_raw_*`, labels, behavioral features, leakage reports, and metadata are present and internally consistent.

The phase should not be marked fully verified because the standalone train BERT cache is absent and the notebook default disables cache reuse:

1. `artifacts/features/bert_train.npy` is missing. Metadata and reports claim it exists with `(29923, 768)` float32.
2. `notebooks/02_Feature_Engineering.ipynb` sets `FORCE_REBUILD_BERT = True`, so the BERT cache path is not reused by default.

Recommended closure: restore/regenerate `artifacts/features/bert_train.npy` or explicitly document that only fused train features are retained; then set the final notebook cache default to reuse valid cached embeddings unless a rebuild is requested.

---

Verified: 2026-06-01T08:58:35Z
Verifier: Codex (GSD phase verifier)
