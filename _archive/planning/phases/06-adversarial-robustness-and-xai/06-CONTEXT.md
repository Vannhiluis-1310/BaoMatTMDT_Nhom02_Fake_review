# Phase 6: Adversarial Robustness and XAI - Context

**Gathered:** 2026-06-01  
**Status:** Ready for planning  
**Source:** Roadmap + verified Phase 2-5 artifacts

<domain>

## Phase Boundary

Phase 6 builds `notebooks/06_Adversarial_XAI.ipynb` into a complete Colab notebook for adversarial robustness and explainability. It must not retrain Phase 2-5. It consumes existing artifacts and writes robustness/XAI outputs.

</domain>

<decisions>

## Implementation Decisions

### Runtime

- Run on Google Colab only.
- Do not run notebook cells, EDA, training, tuning, adversarial evaluation, SHAP or LIME locally.
- Keep executable ML code inside `notebooks/06_Adversarial_XAI.ipynb`.

### Robustness

- Generate FGSM/PGD adversarial perturbations in PCA feature space.
- Use the Phase 4 PyTorch DL model as the differentiable surrogate for adversarial attacks.
- Evaluate robustness on a seeded subset to respect RAM/time.
- Report clean and adversarial metrics side by side for both DL and final ensemble where possible.

### XAI

- Use SHAP on LightGBM/tree model for efficient global explanations.
- Use LIME Tabular on the final ensemble prediction function for local explanations.
- Map important PCA components back to raw feature groups where possible.

### Evidence

- Phase 4/5 current metrics are below target. Phase 6 should present robustness/XAI honestly and include limitations.

</decisions>

<canonical_refs>

## Canonical References

### Planning

- `.planning/ROADMAP.md` - Phase 6 goal, tasks and success criteria.
- `.planning/REQUIREMENTS.md` - ROB-01, ROB-02, XAI-01, XAI-02.
- `.planning/STATE.md` - Current Phase 4/5 metric gaps and next actions.

### Inputs

- `artifacts/pca/phase3_metadata.json` - PCA reducer metadata.
- `artifacts/models/phase4_metadata.json` - DL model config and output paths.
- `artifacts/models/best_model_dl.pth` - Phase 4 model checkpoint.
- `artifacts/ensemble/phase5_metadata.json` - selected ensemble candidate and threshold.
- `artifacts/ensemble/final_ensemble_model.pkl` - final ensemble bundle.

### Notebook

- `notebooks/06_Adversarial_XAI.ipynb` - target notebook to implement.

</canonical_refs>

<specifics>

## Specific Ideas

- Default subset sizes: robustness 1000, SHAP 500, LIME background 1000, LIME cases 6.
- Output robustness tables under `reports/tables/`.
- Output XAI artifacts under `artifacts/xai/` and figures under `reports/figures/`.
- Save `phase6_metadata.json` under `artifacts/xai/`.

</specifics>

<deferred>

## Deferred Ideas

- Full-dataset SHAP/LIME is deferred unless Colab RAM/time is clearly sufficient.
- Adversarial training is deferred; Phase 6 focuses on adversarial evaluation/evidence, not retraining.

</deferred>
