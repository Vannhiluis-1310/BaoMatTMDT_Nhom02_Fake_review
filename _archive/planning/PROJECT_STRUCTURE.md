# Project Structure

## Folder Tree

```text
Fake_reviews/
├── AGENTS.md
├── README.md
├── data/
│   ├── final_labeled_fake_reviews.csv
│   ├── interim/
│   └── processed/
├── notebooks/
│   ├── 01_EDA_Preprocessing.ipynb
│   ├── 02_Feature_Engineering.ipynb
│   ├── 03_PCA_Feature_Selection.ipynb
│   ├── 04_PSO_Model_Training.ipynb
│   ├── 05_Hybrid_Ensemble.ipynb
│   ├── 06_Adversarial_XAI.ipynb
│   ├── 07_Evaluation_Ablation.ipynb
│   ├── 08_Final_Report_Kaggle.ipynb
│   ├── 05_00_Phase5_Run_Order.ipynb
│   ├── 05_01_LightGBM_Raw.ipynb
│   ├── 05_02_XGBoost_Raw.ipynb
│   ├── 05_03_MLP_Raw.ipynb
│   ├── 05_04_CNN_BiLSTM_Sequence.ipynb
│   ├── 05_05_Weighted_Blending.ipynb
│   ├── 05_06_Stacking_Calibration.ipynb
│   └── tests/
│       └── 01_DL_PCA_Diagnostic_Test.ipynb
├── artifacts/
│   ├── features/
│   ├── pca/
│   ├── models/
│   ├── ensemble/
│   ├── diagnostics/
│   ├── xai/
│   └── figures/
├── reports/
│   ├── tables/
│   ├── figures/
│   └── final/
├── logs/
└── .planning/
    ├── PROJECT.md
    ├── REQUIREMENTS.md
    ├── ROADMAP.md
    ├── TASKLIST.md
    ├── STATE.md
    ├── config.json
    └── research/
```

## Directory Purpose

| Path | Purpose |
|------|---------|
| `data/` | Dataset gốc và các phiên bản clean/split nếu cần. |
| `data/interim/` | File trung gian từ cleaning/EDA. Không dùng làm source chính. |
| `data/processed/` | Train/validation/test split đã clean. |
| `notebooks/` | Chứa 8 notebook phase chính và các notebook phụ thuộc Phase 5 khi cần chạy từng model riêng. |
| `notebooks/05_*.ipynb` | Phase 5 notebook family: mỗi model/ensemble step có notebook riêng, còn `05_Hybrid_Ensemble.ipynb` là orchestrator/summary. |
| `notebooks/tests/` | Chứa notebook diagnostic độc lập, không thuộc submission chain chính. |
| `artifacts/features/` | BERT embeddings, behavioral features, fused features. |
| `artifacts/pca/` | PCA/SVD object, reduced feature matrices, explained variance; dùng cho diagnostic/ablation theo pipeline mới. |
| `artifacts/models/` | MLP raw, sequence CNN-BiLSTM, legacy DL checkpoints và training history. |
| `artifacts/ensemble/` | LightGBM/XGBoost/model zoo/weighted blend/stacking/calibration artifacts. |
| `artifacts/diagnostics/` | Output thử nghiệm cô lập như DL-vs-PCA diagnostic. |
| `artifacts/xai/` | SHAP/LIME outputs và robustness subset results. |
| `artifacts/figures/` | Hình sinh từ experiments. |
| `reports/tables/` | CSV/Markdown tables cho metrics và ablation. |
| `reports/figures/` | Hình cuối dùng trong báo cáo. |
| `reports/final/` | Báo cáo cuối, phụ lục, bản nộp. |
| `logs/` | Log experiment nhẹ, ví dụ PSO history và run summary. |
| `.planning/` | Tài liệu quản lý dự án, requirements, roadmap, tasklist. |

## Source Code Policy

- Không tạo pipeline chính dưới dạng `.py`.
- Notebook có thể lưu/load artifacts nhưng code tạo artifacts vẫn nằm trong các file `.ipynb`.
- Các file Markdown/JSON ở đây là tài liệu kế hoạch, không phải source thực thi.

## Artifact Naming Convention

```text
{phase}_{artifact}_{split}_{date-or-seed}.{ext}
```

Examples:

```text
features/phase2_bert_embeddings_train_seed42.npy
pca/phase3_truncated_svd_seed42.joblib
models/phase4_pso_cnn_bilstm_attention_best_seed42.pt
ensemble/phase5_stacking_calibrated_seed42.joblib
tables/phase7_ablation_metrics_seed42.csv
```

## Revised Pipeline Naming Notes

- Raw fused tabular features should use names like `features_raw_777_{split}_seed42.*`.
- Sequence DL artifacts should use names like `phase4_cnn_bilstm_sequence_{split}_prob_seed42.npy`.
- Legacy PCA/CNN-BiLSTM artifacts must be labeled as diagnostic or ablation if reused in the revised report.
