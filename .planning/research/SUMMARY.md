# Research Summary

This is a project-local synthesis from the provided brief, not a web literature review.

## Stack

- Notebook runtime: Google Colab.
- Text representation: HuggingFace BERT embeddings.
- Behavioral features: pandas groupby/time windows, sentiment via VADER/TextBlob if useful.
- Feature selection: scikit-learn PCA, IncrementalPCA or TruncatedSVD fallback.
- Deep model: PyTorch or Keras CNN-BiLSTM-Attention.
- Optimization: pyswarm PSO, subset-first.
- Ensemble: XGBoost, LightGBM, stacking meta-learner, probability calibration.
- Explanation: SHAP and LIME on subset.

## Table Stakes

- Strict train-only fitting for scaler/PCA/model selection.
- Reproducible split and seed.
- Metrics beyond accuracy, especially Macro F1, Precision Fake, ROC-AUC and PR-AUC.
- Ablation table that directly maps to novelty claims.
- Memory-aware batch extraction and artifact caching.

## Watch Out For

- Behavioral leakage from reviewer/product aggregate features.
- Misleading high Precision Fake if threshold destroys Recall Fake.
- PCA fitted before split or on all data.
- PSO search space too wide for 18-22 days.
- SHAP/LIME memory usage on full dataset.
- Dataset missing timestamp/user columns, requiring fallback definitions for advanced behavioral features.
