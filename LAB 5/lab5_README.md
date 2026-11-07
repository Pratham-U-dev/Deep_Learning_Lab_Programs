# 🔁 Lab 5 — Garbage Classification using ResNet50 Transfer Learning

![Type](https://img.shields.io/badge/Task-Transfer%20Learning-blue)
![Model](https://img.shields.io/badge/Model-ResNet50-purple)
![Dataset](https://img.shields.io/badge/Dataset-Garbage%20Dataset-yellow)
![Classes](https://img.shields.io/badge/Classes-6-lightblue)

> Apply transfer learning using a ResNet50 backbone (feature extractor) with a custom classification head for 6-class garbage image classification.

---

## 📌 Objective

Demonstrate the transfer learning pipeline by using ResNet50 as a frozen feature extractor and training only the final classification layers on a small garbage dataset.

---

## 📂 Files

```
lab5.ipynb
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

## 🧠 Model Architecture — ResNet50 + Custom Head

```
Input (224×224×3)
    ↓
ResNet50 (frozen, no pretrained weights, include_top=False)
    ↓
GlobalAveragePooling2D  →  (2048,)
    ↓
Dense(512, ReLU)
    ↓
Dropout(0.5)
    ↓
Dense(6, Softmax)
```

| Layer | Output Shape | Parameters |
|---|---|---|
| ResNet50 (frozen) | (7, 7, 2048) | 23,587,712 |
| GlobalAvgPool | (2048,) | 0 |
| Dense | (512,) | 1,049,088 |
| Dropout | (512,) | 0 |
| Dense (output) | (6,) | 3,078 |
| **Total Params** | | **24,639,878** |
| **Trainable Params** | | **1,052,166** |
| **Frozen Params** | | **23,587,712** |

---

## ⚙️ Training Configuration

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam (default lr) |
| Loss Function | Categorical Cross-Entropy |
| Metric | Accuracy |
| Epochs | 2 |
| Batch Size | 32 |

### Data Augmentation (Train Only)

| Augmentation | Value |
|---|---|
| Rescale | 1/255 |
| Zoom Range | 0.3 |
| Rotation Range | 20° |
| Shear Range | 0.3 |
| Horizontal Flip | True |

---

## 📈 Training History

| Epoch | Train Acc | Train Loss | Val Acc | Val Loss |
|---|---|---|---|---|
| 1 | 18.80% | 1.7848 | 14.80% | 1.7860 |
| 2 | 19.13% | 1.7718 | 17.04% | 1.7621 |

---

## ⚠️ Notes & Observations

- **`weights=None`** — No pretrained ImageNet weights were loaded. The ResNet50 backbone was initialized randomly and then immediately frozen, which explains the near-random (~17%) accuracy. This defeats the purpose of transfer learning.
- **Expected fix:** Use `weights='imagenet'` to load pretrained ImageNet features before freezing the backbone.
- **F1 Score computation** was interrupted mid-run (`6/15` batches), so the final weighted F1 score was not captured in the notebook.
- Despite this, the pipeline correctly demonstrates feature extraction, frozen layers, and a trainable classification head.

---

## 🔄 Pipeline

```
ImageDataGenerator (augment + rescale)
    → flow_from_directory (train / val / test)
    → Load ResNet50 (include_top=False, weights=None)
    → Freeze backbone (resnet_base.trainable = False)
    → Add GlobalAvgPool + Dense head
    → Compile (Adam, categorical_crossentropy)
    → Fit (2 epochs)
    → Predict on test set
    → Compute weighted F1 Score
```

---

## 🔧 Recommended Improvements

| Issue | Fix |
|---|---|
| Random weights (`weights=None`) | Use `weights='imagenet'` for pretrained features |
| Only 2 epochs | Train for 10–20 epochs |
| Frozen backbone | Fine-tune last few ResNet blocks after initial training |
| No LR schedule | Add `ReduceLROnPlateau` callback |

---

## 💡 Key Takeaways

- Transfer learning is most effective when pretrained weights are used — randomly initialized frozen layers provide no meaningful features.
- With `weights='imagenet'`, ResNet50 is expected to reach 80–90%+ accuracy even with minimal fine-tuning on small datasets.
- The pipeline structure (frozen backbone + trainable head) is architecturally correct and production-ready.

---

## 📦 Dependencies

```python
tensorflow, keras, numpy, matplotlib, scikit-learn
```
