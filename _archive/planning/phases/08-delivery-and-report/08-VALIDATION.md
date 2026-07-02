# Phase 8 Validation Plan

> **Updated 2026-06-10.** Current report SSOT: `0.9463`, `0.9816`, `0.9769` in `Phase8_Final_Report.md`. Legacy checks for `0.855820` below are historical.

## Nyquist Validation Targets

Phase 8 is valid when it proves three things:

1. The final notebook can guide a reader through the whole project without rerunning heavy experiments.
2. The final report uses current verified results and does not contain stale or inflated claims.
3. The delivery package clearly identifies required artifacts and run order.

## Static Checks

| Check | Method | Pass Criteria |
|-------|--------|---------------|
| Notebook JSON parses | `json.load(...)` | No parse error. |
| Required notebook sections exist | Inspect markdown/source cells | Colab run order, artifact inventory, metrics, target audit, ablation, XAI, limitations and checklist are present. |
| Final report exists | `Test-Path` | `reports/final/Phase8_Final_Report.md` exists. |
| Current metrics are present | `Select-String` | Report contains `0.9463` or `0.9433`, `0.9816`, `0.9769` (final track). Legacy `0.855820` must not be headline. |
| Stale claims absent | `Select-String` | No final-report matches for stale Phase 9 values or target-achieved wording. |
| Artifact inventory exists | `Test-Path` | `phase8_artifact_inventory.csv` exists. |
| Manifest exists | `Test-Path` | `phase8_submission_package_manifest.csv` exists. |
| Run order exists | `Test-Path` | `phase8_run_order_checklist.csv` exists. |

## Content Checks

| Topic | Required Evidence |
|-------|-------------------|
| Target gap honesty | Report states that all original target metrics remain missed. |
| PCA caveat | Report says controlled no-PCA Model B was strong and PCA is primarily RAM/pipeline stabilization evidence. |
| Ablation | Report includes Full, Model A, Model B, Model C, Model D and Model E. |
| Reproducibility | Report/notebook includes seed, split and run order. |
| RAM constraint | Report mentions 12GB RAM and why PCA/SVD was used. |

## Explicit Non-Goals

- Do not execute notebook cells locally.
- Do not rerun feature extraction, PCA, training, PSO, ensemble, adversarial attacks, SHAP/LIME or CV locally.
- Do not delete old report/artifact files automatically.
