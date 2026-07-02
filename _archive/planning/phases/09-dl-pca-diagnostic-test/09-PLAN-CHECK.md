# Phase 9 Plan Check

**Verdict:** VERIFICATION PASSED

## User Constraint Coverage

| User Requirement | Covered By | Notes |
|---|---|---|
| Create a separate test | Task 1 | New notebook under `notebooks/tests/`, not the main 01-08 notebooks. |
| Do not touch main source | Scope guard | Plan forbids modifying notebooks 01-08 and existing Phase 1-8 artifacts. |
| Diagnose whether PCA causes weak DL | Task 3 | Controlled MLP raw vs PCA comparison. |
| Diagnose whether architecture is the problem | Task 4 | MLP PCA vs CNN-BiLSTM PCA comparison. |
| Check whether DL can contribute non-zero | Task 5 | Constrained ensemble sweep with `dl_weight >= 0.10` and `>= 0.20`. |

## Project Constraint Check

| Constraint | Status |
|---|---|
| Colab-only execution | pass |
| No local notebook execution or training | pass |
| No `.py` module for main pipeline | pass |
| Existing dataset preserved | pass |
| RAM 12GB respected | pass via lightweight models, early stopping and batch guard |
| Main Phase 1-8 source untouched | pass |

## Risk Review

| Risk | Handling |
|---|---|
| Test turns into another broad retune | Plan limits scope to controlled comparisons and constrained blend sweep. |
| Diagnostic outputs overwrite main artifacts | Output folders are diagnostics-only. |
| MLP raw overfits | Use validation early stopping and report train/val/test metrics. |
| CNN-BiLSTM diagnostic differs from Phase 4 | It is intentionally lightweight; existing Phase 4 DL PSO remains part of baseline snapshot. |
| Results still inconclusive | Decision table supports `inconclusive` and recommends redesign or title change. |

## Execution Shape

One plan only. The diagnostic notebook is self-contained and should not be parallelized across multiple executors because they would conflict on the same new notebook.
