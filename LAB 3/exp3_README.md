# ♻️ Experiment 3 — Garbage Classification using AlexNet CNN

![Type](https://img.shields.io/badge/Task-Image%20Classification-blue)
![Model](https://img.shields.io/badge/Model-AlexNet%20(CNN)-purple)
![Dataset](https://img.shields.io/badge/Dataset-Garbage%20Dataset-yellow)
![Accuracy](https://img.shields.io/badge/Accuracy-68.43%25-orange)
![Classes](https://img.shields.io/badge/Classes-6-lightblue)

> Classify waste images into 6 recyclable categories using a custom AlexNet-inspired Convolutional Neural Network trained from scratch.

---

## 📌 Objective

Build a CNN-based image classifier to identify the type of garbage in a photo, supporting automated waste sorting across 6 material categories.

---

## 📂 Files

```
exp3.ipynb
Garbage_Dataset/
├── train/        (2101 images)
├── validation/   (446 images)
└── test/         (453 images)
```

---

## 📊 Dataset

| Property | Value |
|---|---|
| **Classes** | 6 |
| **Train Images** | 2,101 |
| **Validation Images** | 446 |
| **Test Images** | 453 |
| **Image Size** | 224 × 224 (RGB) |
| **Class Mode** | Categorical |

### Class Labels

| Index | Class |
|---|---|
| 0 | `battery` |
| 1 | `biological` |
| 2 | `brown-glass` |
| 3 | `cardboard` |
| 4 | `clothes` |
| 5 | `plastic` |

---

## 🧠 Model Architecture — Custom AlexNet

```
Input (224×224×3)
    ↓
Conv2D(96, 11×11, stride=4, ReLU)  →  BatchNorm  →  MaxPool(3×3, stride=3)
    ↓
Conv2D(256, 5×5, same, ReLU)       →  BatchNorm  →  MaxPool(3×3, stride=2)
    ↓
Conv2D(384, 3×3, same, ReLU)
Conv2D(384, 3×3, same, ReLU)
Conv2D(384, 3×3, same, ReLU)       →  MaxPool(3×3, stride=2)
    ↓
GlobalAveragePooling2D
    ↓
Dense(512, ReLU)  →  Dropout(0.5)
    ↓
Dense(6, Softmax)
```

| Layer | Output Shape | Parameters |
|---|---|---|
| Conv2D (96 filters) | (54, 54, 96) | 34,944 |
| BatchNorm | (54, 54, 96) | 384 |
| MaxPool | (18, 18, 96) | 0 |
| Conv2D (256 filters) | (18, 18, 256) | 614,656 |
| BatchNorm | (18, 18, 256) | 1,024 |
| MaxPool | (8, 8, 256) | 0 |
| Conv2D (384 filters) ×3 | (8, 8, 384) | 2,540,096 |
| MaxPool | (3, 3, 384) | 0 |
| GlobalAvgPool | (384,) | 0 |
| Dense + Dropout | (512,) | 197,120 |
| Dense (output) | (6,) | 3,078 |
| **Total** | | **4,391,302** |

---

## ⚙️ Training Configuration

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss Function | Categorical Cross-Entropy |
| Metric | Accuracy |
| Epochs | 5 |
| Batch Size | 32 |

### Data Augmentation (Train Only)

| Augmentation | Value |
|---|---|
| Rescale | 1/255 |
| Rotation Range | 20° |
| Shear Range | 0.2 |
| Zoom Range | 0.2 |
| Horizontal Flip | True |

---

## 📈 Training History

| Epoch | Train Acc | Train Loss | Val Acc | Val Loss |
|---|---|---|---|---|
| 1 | 45.79% | 1.5418 | 29.82% | 1.7662 |
| 2 | 61.26% | 1.1512 | 27.35% | 1.9537 |
| 3 | 64.06% | 1.0684 | 34.53% | 2.4801 |
| 4 | 67.82% | 0.9499 | 63.23% | 1.0319 |
| 5 | 71.54% | 0.8672 | **68.39%** | **0.9457** |

---

## 📊 Evaluation Results (Test Set)

| Metric | Value |
|---|---|
| **Test Accuracy** | **68.43%** |
| **Test Loss** | 0.9393 |

### Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| battery | 0.62 | 0.89 | 0.73 | 72 |
| biological | 0.60 | 0.57 | 0.59 | 72 |
| brown-glass | 0.84 | 0.62 | 0.71 | 78 |
| cardboard | 0.73 | 0.71 | 0.72 | 82 |
| clothes | 0.63 | 0.86 | 0.73 | 66 |
| plastic | 0.78 | 0.51 | 0.61 | 83 |
| **Macro Avg** | **0.70** | **0.69** | **0.68** | 453 |

---

## 🔄 Pipeline

```
ImageDataGenerator (augment + rescale)
    → flow_from_directory (train / val / test)
    → Build AlexNet (custom, from scratch)
    → Compile (Adam, categorical_crossentropy)
    → Fit (5 epochs, generator)
    → Evaluate on test set
    → Confusion Matrix (seaborn heatmap)
    → Classification Report
```

---

## 💡 Key Takeaways

- `biological` and `plastic` are the hardest classes (low recall/F1), likely due to visual overlap with other materials.
- Training from scratch on only ~2,100 images limits performance — pretrained weights would significantly improve accuracy.
- Validation loss was unstable early (epochs 1–3) due to limited data; it stabilized by epoch 4.
- Using `GlobalAveragePooling2D` instead of `Flatten` + dense blocks reduces overfitting.

---

## 📦 Dependencies

```python
tensorflow, keras, numpy, matplotlib, seaborn, scikit-learn
```
