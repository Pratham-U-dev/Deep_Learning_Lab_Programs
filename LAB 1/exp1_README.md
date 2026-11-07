# 🏠 Experiment 1 — House Price Prediction using ANN

![Type](https://img.shields.io/badge/Task-Regression-blue)
![Model](https://img.shields.io/badge/Model-ANN%20(Dense)-purple)
![Dataset](https://img.shields.io/badge/Dataset-KC%20House%20Data-yellow)
![Score](https://img.shields.io/badge/R²%20Score-0.807-brightgreen)

> Predict house prices in King County, Washington using a fully connected Artificial Neural Network with MinMax feature scaling.

---

## 📌 Objective

Build an ANN regression model to estimate residential house prices based on 17 property features such as square footage, number of bedrooms, location coordinates, and construction quality.

---

## 📂 File

```
exp1.ipynb
kc_house_data.csv
```

---

## 📊 Dataset

| Property | Value |
|---|---|
| **Source** | King County House Sales (Kaggle) |
| **Total Samples** | ~21,613 |
| **Train / Test Split** | 70% / 30% |
| **Train Samples** | 15,127 |
| **Test Samples** | 6,484 |
| **Input Features** | 17 (after dropping `id`, `zipcode`, `date`) |
| **Target Variable** | `price` (continuous) |

### Feature Correlation with Price (Top 10)

| Feature | Correlation |
|---|---|
| `sqft_living` | 0.702 |
| `grade` | 0.667 |
| `sqft_above` | 0.606 |
| `sqft_living15` | 0.585 |
| `bathrooms` | 0.525 |
| `view` | 0.397 |
| `sqft_basement` | 0.324 |
| `bedrooms` | 0.308 |
| `lat` | 0.307 |
| `waterfront` | 0.266 |

---

## 🧠 Model Architecture

```
Input Layer  →  (17 features)
Dense        →  17 units, ReLU
Dense        →  17 units, ReLU
Dense        →  17 units, ReLU
Output       →  1 unit, Linear
```

| Layer | Output Shape | Parameters |
|---|---|---|
| Dense (input) | (None, 17) | 306 |
| Dense | (None, 17) | 306 |
| Dense | (None, 17) | 306 |
| Dense (output) | (None, 1) | 18 |
| **Total** | | **936** |

---

## ⚙️ Training Configuration

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 0.1 |
| Loss Function | Mean Squared Error (MSE) |
| Epochs | 50 |
| Batch Size | 128 |
| Scaling | MinMaxScaler (0–1) |

---

## 📈 Results

| Metric | Value |
|---|---|
| **R² Score** | **0.807** |

### Sample Predictions

| Sample | Actual Price ($) | Predicted Price ($) |
|---|---|---|
| 1 | 570,000 | 613,022 |
| 2 | 720,001 | 623,210 |
| 3 | 267,000 | 200,739 |
| 4 | 320,000 | 369,553 |
| 5 | 397,000 | 398,363 |

---

## 🔄 Pipeline

```
Load CSV → Drop Irrelevant Columns → Check Nulls → Drop NaNs
    → Feature/Target Split → Train-Test Split
    → MinMax Scale Features → Build ANN
    → Train (50 epochs) → Evaluate R² → Visualize
```

---

## 📉 Training Curves

Loss (MSE) decreased steadily from ~1.7×10¹¹ (epoch 1) to ~2.7×10¹⁰ (epoch 50), with validation loss tracking closely, indicating no severe overfitting.

---

## 💡 Key Takeaways

- `sqft_living`, `grade`, and `sqft_above` are the strongest predictors of price.
- MinMax scaling is essential — raw price values span several orders of magnitude.
- A simple 3-layer ANN achieves a solid R² of 0.807 on this tabular dataset.
- Higher learning rate (0.1) with MSE loss converges reasonably for this problem size.

---

## 📦 Dependencies

```python
numpy, pandas, tensorflow, keras, scikit-learn, matplotlib
```
