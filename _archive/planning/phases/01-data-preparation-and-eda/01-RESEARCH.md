# Phase 1 Research: Data Preparation and EDA

## Research Complete

This research is scoped to planning Phase 1. It is based on project constraints and standard notebook/data-science practice; no local dataset processing was performed.

## What Matters For Planning

### Reproducible data contracts

Phase 1 must produce stable artifact names so later notebooks do not need to guess. The minimum contract is:

- `data/processed/clean_reviews.csv`
- `data/processed/train.csv`
- `data/processed/val.csv`
- `data/processed/test.csv`
- `data/processed/split_metadata.json`
- `reports/tables/phase1_eda_summary.csv`
- `reports/tables/phase1_cleaning_report.csv`
- `reports/figures/phase1_label_distribution.png`
- `reports/figures/phase1_review_length_distribution.png`

### Schema uncertainty

The dataset schema still needs confirmation in Colab. The notebook should support explicit override variables and candidate-based inference for:

- review text column
- label column
- reviewer id column
- product id column
- rating column
- timestamp column

The text and label columns are mandatory. Reviewer/product/rating/time are optional for Phase 1 but should be recorded because Phase 2 behavioral features may depend on them.

### Cleaning policy

Cleaning should be transparent and reversible through reports:

- Drop rows missing text or label.
- Strip whitespace and normalize empty strings to missing.
- Normalize labels to a consistent binary representation when safe.
- Detect duplicate rows and duplicate review-text/label pairs.
- Only drop duplicates by a documented default rule, with counts saved.

### Split policy

Use stratified split on the normalized label. A robust 70/15/15 split can be implemented as:

1. Train/test split with `test_size=0.15`.
2. Split remaining temporary set into train/validation using `val_size / (1 - test_size)`.
3. Use `random_state=42` and `stratify=y` in both calls.

The metadata should store exact row counts and label distribution for every split.

### RAM and Colab notes

For roughly 50,000 rows, pandas CSV loading is expected to fit in 12GB, but the notebook should still:

- load only one main dataframe at a time where practical,
- use `low_memory=False` for schema consistency,
- avoid creating unnecessary deep copies,
- write outputs before Phase 2 heavy processing.

## Planning Implications

- Phase 1 should be sequential because all work modifies one notebook and later cells depend on earlier config/helper cells.
- Verification for the planning phase should not run the notebook. It should only check notebook JSON validity and required cell content.
- Execution verification after user runs Colab should include checking that expected output files exist in Drive.

## Research Limitations

- Exact dataset column names are unknown until the notebook is run in Colab.
- If labels are not binary or have unexpected string values, the notebook should stop and ask the user to set label mapping explicitly.
