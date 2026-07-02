# Stack Notes

## Recommended Stack

| Area | Tooling | Notes |
|------|---------|-------|
| Notebook | Google Colab | Primary runtime, source lives in `.ipynb` |
| Data | pandas, numpy | Use dtype optimization and column pruning |
| EDA | matplotlib, seaborn, ydata-profiling optional | Keep profiling optional due RAM |
| BERT | HuggingFace Transformers | Batch extraction and cache |
| Feature selection | scikit-learn PCA/IncrementalPCA/TruncatedSVD | Fit on train only |
| Deep learning | PyTorch or Keras | Pick one, avoid mixing unless necessary |
| PSO | pyswarm | Start with 20% subset |
| Ensemble | XGBoost, LightGBM, scikit-learn StackingClassifier | Use reduced features |
| XAI | SHAP, LIME | Subset with fixed seed |
| Export | joblib, pickle, torch/keras checkpoint, ONNX optional | ONNX only if time allows |

## Implementation Preference

Use PyTorch if custom attention, focal loss and adversarial perturbation are central. Use Keras only if faster implementation matters more than custom control. Do not maintain both deep learning implementations unless there is a clear need.
