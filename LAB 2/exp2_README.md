# ✍️ Experiment 2 — MNIST Digit Classification using ANN

![Type](https://img.shields.io/badge/Task-Classification-blue)
![Model](https://img.shields.io/badge/Model-ANN%20(Dense)-purple)
![Dataset](https://img.shields.io/badge/Dataset-MNIST-yellow)
![Accuracy](https://img.shields.io/badge/Accuracy-95.09%25-brightgreen)

> Classify handwritten digits (0–9) from the MNIST dataset using a fully connected deep neural network.

---

## 📌 Objective

Train a multi-class classification ANN to recognize handwritten digits from 28×28 grayscale images by flattening them into 784-dimensional vectors.

---

## 📂 File

```
exp2.ipynb
```

---

## 📊 Dataset

| Property | Value |
|---|---|
| **Source** | `tensorflow.keras.datasets.mnist` |
| **Training Samples** | 60,000 |
| **Test Samples** | 10,000 |
| **Image Size** | 28 × 28 pixels (Grayscale) |
| **Input Dimensions** | 784 (flattened) |
| **Classes** | 10 (digits 0–9) |
| **Label Encoding** | One-Hot (via `to_categorical`) |

---

## 🧠 Model Architecture

```
Input Layer  →  (784 features)
Dense        →  512 units, ReLU
Dense        →  512 units, ReLU
Dense        →  512 units, ReLU
Output       →  10 units, Softmax
```

| Layer | Output Shape | Parameters |
|---|---|---|
| Dense | (None, 512) | 401,920 |
| Dense | (None, 512) | 262,656 |
| Dense | (None, 512) | 262,656 |
| Dense (output) | (None, 10) | 5,130 |
| **Total** | | **932,362** |

---

## ⚙️ Training Configuration

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 0.01 |
| Loss Function | Categorical Cross-Entropy |
| Metric | Accuracy |
| Epochs | 4 |
| Batch Size | 32 (default) |
| Normalization | Divide by 255.0 |

---

## 📈 Training History

| Epoch | Train Accuracy | Train Loss | Val Accuracy | Val Loss |
|---|---|---|---|---|
| 1 | 83.46% | 0.4862 | 93.29% | 0.2340 |
| 2 | 93.61% | 0.2108 | 94.44% | 0.1848 |
| 3 | 95.19% | 0.1617 | 94.69% | 0.1780 |
| 4 | 95.70% | 0.1433 | **95.09%** | **0.1687** |

---

## 📊 Evaluation Metrics

| Metric | Value |
|---|---|
| **Accuracy** | **95.09%** |
| **Macro Precision** | 95.20% |
| **Macro Recall** | 95.07% |
| **Macro F1-Score** | 95.05% |

### Per-Class Performance

| Digit | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| 0 | 0.97 | 0.98 | 0.98 | 980 |
| 1 | 0.99 | 0.97 | 0.98 | 1135 |
| 2 | 0.92 | 0.98 | 0.95 | 1032 |
| 3 | 0.95 | 0.94 | 0.94 | 1010 |
| 4 | 0.99 | 0.85 | 0.92 | 982 |
| 5 | 0.93 | 0.97 | 0.95 | 892 |
| 6 | 0.96 | 0.97 | 0.96 | 958 |
| 7 | 0.98 | 0.96 | 0.97 | 1028 |
| 8 | 0.94 | 0.93 | 0.94 | 974 |
| 9 | 0.88 | 0.96 | 0.92 | 1009 |

> **Note:** Digit `4` has the lowest recall (85%), likely confused with `9`. Digit `9` has the lowest precision (88%), picking up misclassified `4`s.

---

## 🔄 Pipeline

```
Load MNIST → Reshape (60000, 28×28) → Flatten (60000, 784)
    → Normalize (÷255) → One-Hot Encode Labels
    → Build ANN (3×512 Dense) → Compile
    → Train (4 epochs) → Predict → Evaluate
    → Confusion Matrix → Classification Report
```

---

## 🖼️ Sample Prediction (Index 15)

```
Predicted:  5
True Label: 5  ✅
Confidence: 99.92%
```

---

## 💡 Key Takeaways

- Flattening 28×28 images loses spatial information but still achieves 95%+ accuracy.
- 3 hidden layers of 512 units provide sufficient capacity for 10-class digit recognition.
- The model converges quickly — 4 epochs is sufficient; additional epochs risk overfitting.
- A CNN-based approach would likely push accuracy above 99% by leveraging spatial features.

---

## 📦 Dependencies

```python
numpy, tensorflow, keras, scikit-learn, matplotlib, seaborn
```
