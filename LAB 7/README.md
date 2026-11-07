# 🖼️ Lab 7 — Image Denoising using Convolutional Autoencoder

![Type](https://img.shields.io/badge/Task-Generative%20%7C%20Denoising-blue)
![Model](https://img.shields.io/badge/Model-Convolutional%20Autoencoder-purple)
![Dataset](https://img.shields.io/badge/Dataset-MNIST-yellow)
![Val Loss](https://img.shields.io/badge/Val%20Loss-0.0859-brightgreen)

> Train a convolutional encoder-decoder (autoencoder) to remove Gaussian noise from MNIST handwritten digit images.

---

## 📌 Objective

Learn a compressed latent representation of clean digit images, then use the decoder to reconstruct clean images from artificially corrupted (noisy) inputs — demonstrating unsupervised representation learning and image restoration.

---

## 📂 File

```
lab7.ipynb
```

---

## 📊 Dataset

| Property | Value |
|---|---|
| **Source** | `tensorflow.keras.datasets.mnist` |
| **Training Samples** | 60,000 |
| **Test Samples** | 10,000 |
| **Image Size** | 28 × 28 × 1 (Grayscale) |
| **Normalization** | Divide by 255.0 (range 0–1) |

### Noise Generation

```python
noisy_factor = 0.3
x_noisy = np.clip(x + noisy_factor * np.random.randn(*x.shape), 0., 1.)
```

| Parameter | Value |
|---|---|
| **Noise Type** | Gaussian (additive) |
| **Noise Factor** | 0.3 |
| **Clipping** | [0.0, 1.0] |

---

## 🧠 Model Architecture — Convolutional Autoencoder

```
Input (28×28×1)
━━━━━━━━━━━━ ENCODER ━━━━━━━━━━━━
Conv2D(32, 3×3, same, ReLU)   →  (28×28×32)
MaxPooling2D(2×2, same)        →  (14×14×32)
Conv2D(32, 3×3, same, ReLU)   →  (14×14×32)
MaxPooling2D(2×2, same)        →  (7×7×32)   ← Latent Space

━━━━━━━━━━━━ DECODER ━━━━━━━━━━━━
Conv2D(32, 3×3, same, ReLU)   →  (7×7×32)
UpSampling2D(2×2)              →  (14×14×32)
Conv2D(32, 3×3, same, ReLU)   →  (14×14×32)
UpSampling2D(2×2)              →  (28×28×32)
Conv2D(1, 3×3, same, Sigmoid)  →  (28×28×1)  ← Reconstructed Image
```

### Model Details

| Component | Type | Output Shape |
|---|---|---|
| Input | `layers.Input` | (28, 28, 1) |
| Encoder Conv 1 | Conv2D(32) + ReLU | (28, 28, 32) |
| Encoder Pool 1 | MaxPooling2D | (14, 14, 32) |
| Encoder Conv 2 | Conv2D(32) + ReLU | (14, 14, 32) |
| **Latent Space** | MaxPooling2D | **(7, 7, 32)** |
| Decoder Conv 1 | Conv2D(32) + ReLU | (7, 7, 32) |
| Decoder Up 1 | UpSampling2D | (14, 14, 32) |
| Decoder Conv 2 | Conv2D(32) + ReLU | (14, 14, 32) |
| Decoder Up 2 | UpSampling2D | (28, 28, 32) |
| **Output** | Conv2D(1) + Sigmoid | **(28, 28, 1)** |

---

## ⚙️ Training Configuration

| Hyperparameter | Value |
|---|---|
| Model Type | Functional API (`models.Model`) |
| Optimizer | Adam |
| Loss Function | Binary Cross-Entropy |
| Epochs | 2 |
| Batch Size | 128 |
| Shuffle | True |
| Input (X) | Noisy images |
| Target (Y) | Clean images |

---

## 📈 Training Results

| Epoch | Train Loss | Val Loss |
|---|---|---|
| 1 | 0.1392 | 0.0916 |
| 2 | 0.0894 | **0.0859** |

> Loss converges rapidly — the autoencoder learns to suppress Gaussian noise effectively within just 2 epochs.

---

## 🔄 Pipeline

```
Load MNIST → Normalize (÷255) → Expand dims to (28,28,1)
    → Add Gaussian Noise → Clip to [0,1]
    → Build Encoder (Conv + MaxPool)
    → Build Decoder (Conv + UpSample)
    → Construct Autoencoder model (noisy_input → clean_output)
    → Compile (adam, binary_crossentropy)
    → Fit (noisy_train → clean_train, 2 epochs)
    → Predict on noisy_test
    → Visualize noisy vs. denoised side-by-side
```

---

## 🖼️ Visualization

The notebook produces a 2-row comparison figure (n=10 samples):

```
Row 1: [noisy digit 1] [noisy digit 2] ... [noisy digit 10]
Row 2: [clean digit 1] [clean digit 2] ... [clean digit 10]
```

Each column shows the same digit: corrupted input (top) vs. autoencoder reconstruction (bottom).

---

## 🔬 How It Works

| Stage | Description |
|---|---|
| **Encoding** | The encoder compresses (28×28) images into a (7×7×32) latent representation — capturing the most important structural features while discarding noise. |
| **Latent Space** | The 7×7×32 = 1,568-dimensional bottleneck forces the model to learn a compact, denoised representation. |
| **Decoding** | The decoder upsamples back to (28×28), reconstructing the original clean image from the compressed representation. |

---

## 💡 Key Takeaways

- Autoencoders learn **unsupervised** — they only need paired (noisy, clean) samples, no manual labels.
- The bottleneck (7×7×32) is the key: it forces the network to retain structure and discard random noise.
- `binary_crossentropy` works well for normalized pixel values in [0,1], treating each pixel as an independent Bernoulli variable.
- Extending to a **Variational Autoencoder (VAE)** would enable generative capabilities beyond just denoising.

---

## 📦 Dependencies

```python
tensorflow, keras, numpy, matplotlib, pandas
```
