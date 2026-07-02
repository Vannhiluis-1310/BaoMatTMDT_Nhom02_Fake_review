# Phase 4: PSO Model Training - Chi tiết

> **Vai trò trong báo cáo 2026:** Ablation / diagnostic track (PCA 400 + DL-PSO). **Không phải** pipeline SOTA cuối. Pipeline chính dùng `05_04_CNN_BiLSTM_Sequence` trên raw features + ensemble Phase 5 (09/06).

## 1. Lý do chọn phương án (từ .planning/phases/04-pso-tuned-cnn-bilstm-attention/)

### 1.1. Vấn đề cần giải quyết

- 12 hyperparameters cần tối ưu đồng thời
- Grid Search/Random Search/Optuna vs PSO
- CNN-BiLSTM-Attention architecture cần thiết kế
- Focal Loss vs Cross-Entropy

### 1.2. Quyết định thiết kế

**Tại sao PSO thay vì Grid Search?**
- Grid Search: Tuần tự, exponential time với 12+ params
- PSO: Bầy đàn song song, tìm kiếm đồng thời nhiều vùng, hội tụ nhanh

**Tại sao PSO thay vì Random Search?**
- Random Search: Độc lập, thiếu hướng
- PSO: Chia sẻ thông tin giữa trials (social/cognitive velocity)

**Tại sao PSO thay vì Optuna (TPE)?**
- Optuna: Tốt nhưng surrogate-based
- PSO: Tự nhiên cho continuous space (4 block weights), tối ưu feature weights

**PSO objective function:**
```
Score = 0.5 × Macro F1 + 0.3 × ROC-AUC + 0.2 × Precision Fake
```
- Ưu tiên F1 (0.5) vì cân bằng precision-recall
- ROC-AUC (0.3) cho discrimination ability
- Precision Fake (0.2) cho yêu cầu e-commerce

**12 hyperparameters:**
- Learning rate: [10⁻⁴, 5×10⁻³]
- Dropout: [0.1, 0.4]
- CNN filters: [64, 192]
- Kernel size: [3, 7]
- LSTM hidden: [64, 128]
- Attention dim: [64, 128]
- Focal gamma: [1.0, 3.0]
- Batch size: [32, 64]
- Block weight 1-4: [0.6, 1.5]

## 2. Luồng code (notebooks/04_PSO_Model_Training.ipynb)

### 2.1. CNN-BiLSTM-Attention Architecture

```
Input: 400-d PCA features
Reshape: 1 × 400
1D-CNN: 96 filters, kernel=7, ReLU + BatchNorm
BiLSTM: 64 hidden, 2 directions
Attention: 96-d attention dim
Dropout: 0.1447
FC Layer: 1 output
Sigmoid: P(fake)
```

### 2.2. PSO Configuration

```
Particles: 10
Iterations: 8
Trial epochs: 5
Subset: 20% train
Total trials: 90
Best params: artifacts/models/best_params.json
```

### 2.3. PSO Best Parameters (best_params.json)

```
lr: 3.95e-4
dropout: 0.1447
cnn_filters: 96
kernel_size: 7
lstm_hidden: 64
attention_dim: 96
focal_gamma: 1.609
batch_size: 32
block_weight_1: 0.916
block_weight_2: 1.197
block_weight_3: 1.368
block_weight_4: 1.399
PSO objective score: 0.7953 (val subset)
```

## 3. Kết quả và phân tích (reports/tables/)

### 3.1. Số liệu chính xác

**Baseline DL (Phase 4)** — `phase7_final_metrics.csv`, model `dl_baseline`, test @τ=0.5:
- Macro F1: 0.7665
- Precision Fake: 0.7818
- ROC-AUC: 0.8389

**PSO-tuned DL (Phase 4)** — `dl_pso`, test @τ=0.5:
- Macro F1: 0.7793 (**+0.0128** vs baseline)
- Precision Fake: 0.7819
- ROC-AUC: 0.8517 (**+0.0128** vs baseline)

### 3.2. Insight kinh doanh

**Nỗi đau từ manual hyperparameter tuning:**
- 12 hyperparameters → 12! combinations (impossible)
- Grid Search: Rất lâu trên Colab 12GB
- Random Search: Thiếu hướng, hội tụ chậm

**Ý nghĩa PSO:**
- Tự động hóa việc dò tìm 12 hyperparameters
- Cải thiện +0.0128 Macro F1 so với baseline (ablation track PCA 400-d)
- Hội tụ nhanh trên subset 20% (tiết kiệm 80-90% thời gian)

**Focal Loss insight:**
- PSO chọn focal_gamma = 1.609 (thấp hơn default γ=2.0)
- Dataset ít cần down-weight extreme examples so với object detection
- Class balance (1.02) làm focal loss ít cần thiết

### 3.3. Lý do chuyển sang Phase 5

- DL PSO-tuned (Macro F1 = 0.7793) yếu hơn LightGBM PCA track (0.8601, cùng `phase7_final_metrics.csv`)
- Cần stacking ensemble để kết hợp DL + tree-based models
- Threshold optimization cho Precision Fake target

## 4. Ràng buộc và quyết định cho phase sau

**Ràng buộc cho Phase 5:**
- 3 base models: DL-PSO, XGBoost, LightGBM
- 71 candidate sweep (weighted blend combinations)
- Threshold sweep trên val_calibration
- Selected: blend_dl01_xgb00_lgbm09, threshold=0.79

**Quyết định cho Phase 6/7:**
- Adversarial robustness (FGSM/PGD)
- XAI (SHAP/LIME)
- Ablation study
