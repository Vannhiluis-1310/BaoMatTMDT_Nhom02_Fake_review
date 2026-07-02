# Phase 7 UAT

> **Superseded metrics note (2026-06-10):** Rows below verified the **legacy PCA** run. Final headline: `phase7_final_metrics.csv` — targets pass on `weighted_blend`. See `.planning/STATE.md`.

Status: Complete (re-audited 2026-06-10 on final track)

## User-Facing Acceptance Tests

| Test | Result | Notes |
|---|---|---|
| Can verify all required Phase 7 outputs after Colab run | PASS | Tables, figures and metadata are present locally after sync |
| Final metrics are report-ready | PASS | `phase7_final_metrics.csv` has seed, split, model, threshold and full metric set |
| Target audit is honest | PASS | `phase7_target_audit.csv` reports all target misses with gaps |
| CV evidence exists | PASS | `phase7_cv_metrics.csv` has 5 full folds |
| Ablation A-E exists | PASS | Full, A, B, C, D and E all present with evidence labels |
| Error analysis is usable | PASS | 200 rows covering false positives, false negatives, high-confidence wrong and borderline cases |
| Reproducibility metadata exists | PASS | `artifacts/evaluation/phase7_metadata.json` records seed, source artifacts, CV config, fallbacks and limitations |

## Notes for Phase 8 (updated)

- **Final track** (`phase5_weighted_blend`): all three targets pass in appropriate modes (`phase7_target_audit.csv`).
- Legacy PCA ensemble (0.8558 @0.5) remains ablation evidence only.
- Controlled Model B raw LGBM 0.9058 vs PCA ref. 0.8661 (+0.0397); advanced behavioral +0.0008 only — report honestly.
- Phase 6 robustness/XAI = legacy scope; do not attribute to final blend without rerun.

## Final UAT Decision

Approved for Phase 8 report finalization.
