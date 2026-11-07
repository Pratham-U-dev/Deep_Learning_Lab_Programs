# 🧠 Deep Learning Lab Programs

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-API-red?logo=keras&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

> A collection of deep learning experiments covering regression, classification, computer vision, NLP, and generative models — implemented using TensorFlow/Keras.

---

## 📁 Repository Structure

```
Deep_Learning_Lab_Programs/
│
├── 📓 exp1.ipynb              # ANN for House Price Regression
├── 📓 exp2.ipynb              # ANN for MNIST Digit Classification
├── 📓 exp3.ipynb              # AlexNet CNN for Garbage Classification
├── 📓 lab5.ipynb              # ResNet50 Transfer Learning (Garbage)
├── 📓 lab6.ipynb              # LSTM for IMDB Sentiment Analysis
├── 📓 lab7.ipynb              # Convolutional Autoencoder (Denoising)
│
├── 📂 Garbage_Dataset/
│   ├── train/                 # Training images (2101 images, 6 classes)
│   ├── validation/            # Validation images (446 images)
│   └── test/                  # Test images (453 images)
│
├── 📄 kc_house_data.csv       # King County House Sales dataset
├── 📄 imdb.npz                # IMDB Movie Reviews dataset
└── 📄 imdb_word_index.json    # IMDB vocabulary index
```

---

## 🗂️ Experiments Overview

| # | Notebook | Topic | Model | Dataset | Metric | Result |
|---|----------|-------|-------|---------|--------|--------|
| 1 | `exp1.ipynb` | House Price Prediction | ANN (Dense) | KC House Data | R² Score | **0.807** |
| 2 | `exp2.ipynb` | Digit Classification | ANN (Dense) | MNIST | Accuracy | **95.09%** |
| 3 | `exp3.ipynb` | Garbage Classification | AlexNet (CNN) | Garbage Dataset | Accuracy | **68.43%** |
| 4 | `lab5.ipynb` | Garbage Classification | ResNet50 (Transfer) | Garbage Dataset | F1 Score | In Progress |
| 5 | `lab6.ipynb` | Sentiment Analysis | LSTM + Embedding | IMDB Reviews | Accuracy | **83.66%** |
| 6 | `lab7.ipynb` | Image Denoising | Convolutional Autoencoder | MNIST | Val Loss | **0.0859** |

---

## 🔧 Setup & Installation

### Prerequisites

```bash
Python >= 3.11
```

### Install Dependencies

```bash
pip install tensorflow keras numpy pandas matplotlib seaborn scikit-learn opencv-python psutil
```

### Clone the Repository

```bash
git clone https://github.com/<your-username>/Deep_Learning_Lab_Programs.git
cd Deep_Learning_Lab_Programs
```

### Launch Jupyter

```bash
jupyter notebook
```

---

## 📦 Datasets

### King County House Sales (`kc_house_data.csv`)
- **Source:** Kaggle — KC House Data
- **Rows:** ~21,613 | **Features:** 20
- **Target:** `price`

### MNIST Handwritten Digits
- **Source:** `tensorflow.keras.datasets.mnist`
- **Train:** 60,000 | **Test:** 10,000
- **Classes:** 10 (digits 0–9)

### Garbage Classification Dataset (`Garbage_Dataset/`)
- **Classes:** 6 — `battery`, `biological`, `brown-glass`, `cardboard`, `clothes`, `plastic`
- **Train:** 2,101 | **Validation:** 446 | **Test:** 453
- **Image Size:** 224×224 (RGB)

### IMDB Movie Reviews (`imdb.npz`)
- **Source:** `tensorflow.keras.datasets.imdb`
- **Train:** 25,000 | **Test:** 25,000
- **Task:** Binary sentiment classification (positive / negative)
- **Vocabulary:** Top 20,000 words

---

## 🧪 Experiment Summaries

### 🏠 Exp 1 — House Price Prediction (ANN)
Tabular regression using a 3-layer fully connected network with MinMax scaling and Adam optimizer. Achieves an R² of **0.807** on the King County dataset.

### ✍️ Exp 2 — MNIST Digit Classification (ANN)
Flattened image classification using a 3×512 Dense network trained with categorical cross-entropy. Reaches **95.09% accuracy** in just 4 epochs.

### ♻️ Exp 3 — Garbage Classification (AlexNet CNN)
Custom AlexNet-inspired CNN with BatchNorm and GlobalAveragePooling trained from scratch on the 6-class garbage dataset. Achieves **68.43% accuracy** in 5 epochs.

### 🔁 Lab 5 — Garbage Classification (ResNet50 Transfer Learning)
ResNet50 backbone (frozen, no pretrained weights) with custom classification head. Demonstrates transfer learning pipeline setup with feature extraction.

### 🎬 Lab 6 — IMDB Sentiment Analysis (LSTM)
Embedding + LSTM model for binary sentiment classification. Achieves **83.66% test accuracy** after a single epoch of training on 25,000 reviews.

### 🖼️ Lab 7 — Denoising Autoencoder (Conv AE)
Convolutional encoder-decoder trained to reconstruct clean MNIST digits from Gaussian-noised inputs. Validation loss drops to **0.0859** in 2 epochs.

---

## 📊 Key Concepts Covered

| Concept | Covered In |
|---|---|
| Data Preprocessing & Feature Scaling | Exp 1 |
| Correlation Analysis & Feature Selection | Exp 1 |
| Multi-class Classification | Exp 2, Exp 3 |
| Confusion Matrix & Classification Report | Exp 2, Exp 3 |
| Data Augmentation | Exp 3, Lab 5 |
| CNN Architecture (from scratch) | Exp 3 |
| Transfer Learning | Lab 5 |
| Word Embeddings | Lab 6 |
| Recurrent Neural Networks (LSTM) | Lab 6 |
| Autoencoders & Representation Learning | Lab 7 |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| **TensorFlow / Keras** | Model building & training |
| **NumPy / Pandas** | Data manipulation |
| **Matplotlib / Seaborn** | Visualization |
| **scikit-learn** | Metrics, preprocessing, splits |
| **OpenCV** | Image processing |

---

## 👤 Author

**Pratham** · [pratham.u05@gmail.com](mailto:pratham.u05@gmail.com)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
