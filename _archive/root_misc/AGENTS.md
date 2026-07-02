# Project Agent Guide

## Non-Negotiable Rules

- Do not run notebook cells, EDA, training, tuning, or dataset processing locally unless the user explicitly approves it.
- Keep executable pipeline source inside the 8 phase notebooks under `notebooks/`.
- Do not create separate `.py` modules for the main ML pipeline.
- Preserve `data/final_labeled_fake_reviews.csv`; do not move or rewrite it without permission.
- Treat RAM 12GB as a hard design constraint.

## Project Priorities

1. Phase 2 feature extraction and Phase 3 PCA/SVD must be completed early.
2. Ablation study is the strongest evidence component and should not be postponed until the end.
3. Every metric must be tied to a seed, split and model variant.
4. Robustness and XAI may use subset evaluation if full evaluation exceeds RAM.

## Planning Files

- Project context: `.planning/PROJECT.md`
- Requirements: `.planning/REQUIREMENTS.md`
- Roadmap: `.planning/ROADMAP.md`
- Tasklist: `.planning/TASKLIST.md`
- Structure: `.planning/PROJECT_STRUCTURE.md`
- State: `.planning/STATE.md`

## Expected Output

The final project should include 8 clean Colab notebooks, model artifacts, ablation tables, robustness/XAI outputs and a final report suitable for submission.
