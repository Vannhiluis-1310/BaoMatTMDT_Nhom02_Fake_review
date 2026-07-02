# Archive — workspace cleanup 2026-06-11

Các file/folder ở đây **không nằm trong gói nộp** (`phase8_submission_package_manifest.csv`).
Được chuyển vào archive thay vì xóa hẳn để có thể khôi phục nếu cần.

## Nội dung

| Path | Lý do archive |
|------|----------------|
| `.agents/` | Log làm việc AI agent (handoff, draft, scripts) |
| `root_misc/` | `prompt_draft.md`, `ORIGINAL_REQUEST.md`, `AGENTS.md`, `evaluation_report.md`, PDF nhóm cũ |
| `planning/` | Toàn bộ `.planning/` trừ `PROJECT.md` + `STATE.md` (UAT/VERIFICATION legacy 01/06) |
| `reports/tables/pso_trials/` | 90 file `phase4_pso_trial_XXX_history.csv` (giữ aggregate tại `reports/tables/phase4_pso_trial_history.csv`) |
| `reports/tables/seed_123/`, `seed_456/` | Bảng Phase 5 chi tiết theo seed (tổng hợp tại `phase7_multiseed_summary.csv`) |

## Không archive (theo yêu cầu)

- Notebooks `05_04a/b/c_*_seed*.ipynb` và `tests/01_DL_PCA_Diagnostic_Test.ipynb`
- `artifacts/predictions/*` legacy (`final_ensemble_*`, `dl_pso_*`, …)

## Khôi phục

Di chuyển ngược từ `_archive/` về vị trí gốc tương ứng.