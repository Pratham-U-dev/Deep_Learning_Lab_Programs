# 🎬 Lab 6 — IMDB Sentiment Analysis using LSTM

![Type](https://img.shields.io/badge/Task-NLP%20%7C%20Sentiment%20Analysis-blue)
![Model](https://img.shields.io/badge/Model-LSTM%20%2B%20Embedding-purple)
![Dataset](https://img.shields.io/badge/Dataset-IMDB%20Reviews-yellow)
![Accuracy](https://img.shields.io/badge/Accuracy-83.66%25-brightgreen)

> Classify IMDB movie reviews as positive or negative using a Long Short-Term Memory (LSTM) network with learned word embeddings.

---

## 📌 Objective

Build a sequence model that understands the sentiment of natural language movie reviews by learning word representations and capturing long-range dependencies in text using LSTM.

---

## 📂 Files

```
lab6.ipynb
imdb.npz              ← IMDB dataset (pre-tokenized)
imdb_word_index.json  ← Vocabulary index for decoding
```

---

## 📊 Dataset

| Property | Value |
|---|---|
| **Source** | `tensorflow.keras.datasets.imdb` |
| **Task** | Binary Sentiment Classification |
| **Train Samples** | 25,000 |
| **Test Samples** | 25,000 |
| **Max Vocabulary** | 20,000 words |
| **Max Sequence Length** | 250 tokens (padded) |
| **Label Encoding** | 0 = Negative, 1 = Positive |

### Preprocessing Constants

| Constant | Value |
|---|---|
| `MAX_FEATURES` | 20,000 |
| `MAXLEN` | 250 |
| `INDEX_OFFSET` | 3 |
| `BATCH_SIZE` | 64 |
| `EPOCHS` | 1 (demonstrated) |

### Special Tokens

| Token ID | Meaning |
|---|---|
| 0 | Padding |
| 1 | `<START>` |
| 2 | `<OOP>` (Out of vocabulary) |

---

## 🧠 Model Architecture

```
Input  →  (250,) integer token sequence
    ↓
Embedding(20000, 128)   ←  Learns dense word vectors
    ↓
LSTM(128, dropout=0.3, recurrent_dropout=0.3)
    ↓
Dense(1, sigmoid)       ←  Binary probability output
```

| Layer | Output Shape | Parameters |
|---|---|---|
| Embedding | (None, 250, 128) | 2,560,000 |
| LSTM | (None, 128) | 131,584 |
| Dense (output) | (None, 1) | 129 |
| **Total** | | **2,691,713** |

---

## ⚙️ Training Configuration

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Loss Function | Binary Cross-Entropy |
| Metric | Accuracy |
| Epochs | 1 |
| Batch Size | 64 |
| LSTM Dropout | 0.3 |
| LSTM Recurrent Dropout | 0.3 |

---

## 📈 Training Results

| Epoch | Train Accuracy | Train Loss | Val Accuracy | Val Loss |
|---|---|---|---|---|
| 1 | **95.40%** | 0.1242 | **83.66%** | 0.5153 |

> High train / moderate val accuracy after 1 epoch indicates slight overfitting — adding more epochs or reducing model capacity would help generalize better.

---

## 📊 Evaluation

| Metric | Value |
|---|---|
| **Test Loss** | 0.5153 |
| **Test Accuracy** | **83.66%** |

---

## 🔍 Sample Prediction

**Review (index 5):**
> *"I'm absolutely disgusted this movie isn't being sold... everything and everybody in this movie did a good job... this is a wicked good movie and should be seen by all..."*

| Output | Value |
|---|---|
| **Predicted Probability** | 0.9791 (Positive) |
| **True Label** | 1 (Positive) ✅ |

---

## 🔄 Pipeline

```
Load IMDB (imdb.npz) → Pad Sequences to MAXLEN=250
    → Load imdb_word_index.json
    → Build inv_idx (token → word decoder)
    → Build LSTM model (Embedding + LSTM + Dense)
    → Compile (adam, binary_crossentropy)
    → Train (1 epoch, batch_size=64)
    → Evaluate on test set
    → Predict single review
```

---

## 🔤 Text Decoder Utility

The notebook implements a `s_to_t()` function to decode token sequences back to human-readable text:

```python
def s_to_t(sequence):
    words = []
    for token in sequence:
        if token == 0: continue        # skip padding
        if token == 1: words.append('<START>'); continue
        if token == 2: words.append('<OOP>'); continue
        words.append(inv_idx.get(token - INDEX_OFFSET, '?'))
    return " ".join(words)
```

---

## 💡 Key Takeaways

- LSTM + Embedding achieves 83.66% accuracy in a **single epoch** — demonstrating strong sequential learning capability.
- The gap between train (95.4%) and test (83.7%) accuracy suggests overfitting; more dropout or fewer LSTM units would help.
- `recurrent_dropout` regularizes the LSTM's hidden state transitions, reducing overfitting on longer sequences.
- For further improvement, consider: Bidirectional LSTM, GRU, or fine-tuning a pretrained language model (e.g., BERT).

---

## 📦 Dependencies

```python
tensorflow, keras, numpy, json
```
