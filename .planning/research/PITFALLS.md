# Pitfalls

## Data Leakage

**Warning signs:** Unrealistically high validation/test metrics, feature aggregates computed using all rows, PCA fitted before split.  
**Prevention:** Split first, fit scaler/PCA on train only, compute behavioral aggregates with clear train/test policy.

## RAM Overrun

**Warning signs:** Colab crashes during BERT extraction, PCA, SHAP or PSO loop.  
**Prevention:** Batch extraction, write artifacts, use IncrementalPCA/SVD, run XAI on subset.

## PSO Cost Explosion

**Warning signs:** Trial count grows into hundreds without clear improvement.  
**Prevention:** Narrow search space, subset 20%, cap particles/iterations, stop if improvement plateaus.

## Weak Ablation Evidence

**Warning signs:** Only full model reported, no delta metrics, ablation variants not comparable.  
**Prevention:** Lock split, use same reduced feature artifacts where applicable, record delta vs full model.

## Precision-Recall Imbalance

**Warning signs:** Precision Fake meets target but Recall Fake collapses.  
**Prevention:** Report PR-AUC, Recall Fake and threshold curves, not only single precision number.

## Missing Behavioral Columns

**Warning signs:** Dataset lacks timestamp/reviewer/product fields.  
**Prevention:** Define fallback features and document limitation. Do not fabricate behavior signals.
