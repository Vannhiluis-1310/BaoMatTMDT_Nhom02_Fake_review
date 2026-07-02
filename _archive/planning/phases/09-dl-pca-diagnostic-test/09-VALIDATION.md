# Phase 9 Validation Plan

## Validation Goal

Phase 9 is valid when it provides evidence for the DL failure mode without changing the main pipeline.

The test should answer:

1. Does PCA reduce DL performance?
2. Is CNN-BiLSTM-Attention mismatched with static PCA vectors?
3. Can any DL branch make a non-zero contribution to the final ensemble?

## Static Validation Before Colab Run

| Check | Pass Criteria |
|---|---|
| Notebook exists | `notebooks/tests/01_DL_PCA_Diagnostic_Test.ipynb` exists. |
| Notebook JSON parses | JSON load succeeds. |
| Code cells parse | Python AST parse succeeds. |
| Main notebooks untouched | No changes to `notebooks/01_*.ipynb` through `08_*.ipynb`. |
| Scope guard present | Notebook states diagnostics-only and no overwrite of Phase 1-8 artifacts. |
| Output paths isolated | Notebook writes only to diagnostics paths. |

## Colab Output Validation

| Output | Pass Criteria |
|---|---|
| `input_validation.csv` | Required inputs exist and row counts match. |
| `baseline_snapshot.csv` | Existing DL/tree/final metrics are captured. |
| `mlp_raw_vs_pca_metrics.csv` | Contains MLP raw and MLP PCA metrics. |
| `architecture_mismatch_metrics.csv` | Contains MLP PCA and CNN-BiLSTM PCA comparison. |
| `constrained_dl_ensemble_sweep.csv` | Contains candidates with non-zero DL weights. |
| `constrained_dl_ensemble_best.csv` | Selects best constrained candidates. |
| `runtime_summary.csv` | Reports runtime for diagnostic components. |
| `diagnostic_decision_table.csv` | Answers PCA/architecture/DL-contribution questions. |
| `diagnostic_summary.md` | Gives recommendation: keep title with constrained rerun, redesign DL, or change title/framing. |

## Interpretation Rules

| Condition | Decision |
|---|---|
| MLP raw beats MLP PCA by >= 0.02 Macro F1 | PCA likely hurts DL signal. |
| MLP PCA beats CNN-BiLSTM PCA by >= 0.02 Macro F1 | CNN-BiLSTM likely mismatched with static PCA vector. |
| Non-zero DL ensemble is within 0.01 Macro F1/ROC-AUC of unconstrained best | DL contribution is viable. |
| Non-zero DL ensemble is clearly worse | Current title should be changed or DL redesigned. |

## Non-Goals

- Do not fix Phase 4/5/7 inside this phase.
- Do not update final report inside this phase.
- Do not delete the legacy Phase 9 report.
- Do not run locally.
