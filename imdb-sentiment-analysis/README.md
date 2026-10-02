# IMDB Sentiment Analysis

Binary sentiment classification (positive / negative) of IMDB movie reviews, comparing three model types:

1. **Logistic Regression** (scikit-learn)
2. **Feed-Forward Neural Network** (scikit-learn `MLPClassifier`)
3. **1-D Convolutional Neural Network** with word embeddings (TensorFlow/Keras)

The script also runs a **stopword-removal experiment**, retraining the CNN on reviews with NLTK English stopwords removed and comparing accuracy.

## Dataset

The IMDB dataset bundled with Keras (`tensorflow.keras.datasets.imdb`):

- 50,000 reviews total, split into 25,000 train / 25,000 test
- Balanced classes (positive / negative)
- Reviews are pre-tokenized into integer word indices
- Only the top 5,000 most frequent words are kept; reviews are padded/truncated to 500 tokens

The dataset downloads automatically on first run.

## Requirements

- Python 3.8+
- TensorFlow / Keras 2.x
- scikit-learn
- NLTK
- NumPy

Install dependencies:

```bash
pip install "tensorflow<2.16" scikit-learn nltk numpy
```

> The script uses `Embedding(input_length=...)`, which is not supported in Keras 3 (TensorFlow 2.16+). Use TensorFlow 2.15 or earlier, or remove the `input_length` argument.

The NLTK stopword list is downloaded automatically at runtime.

## Usage

```bash
python main.py
```

Replace `main.py` with whatever you named the script. A GPU is helpful for the CNN but not required. The scikit-learn models run on CPU and may take a few minutes.

## Pipeline

| Step | Description |
|------|-------------|
| 1 | Import libraries, set hyperparameters, load and pad the dataset |
| 2 | Use the canonical Keras train/test split (25k / 25k) |
| 3a | Train and evaluate Logistic Regression |
| 3b | Train and evaluate Feed-Forward NN (hidden layers: 256 → 128, ReLU, Adam) |
| 4–5 | Build, train, and evaluate the CNN |
| 6 | Remove stopwords, retrain the CNN, compare |
| 7 | Print a results summary and analysis |

## Hyperparameters

| Name | Value | Meaning |
|------|-------|---------|
| `TOP_WORDS` | 5000 | Vocabulary size |
| `MAX_REVIEW_LEN` | 500 | Padded/truncated review length |
| `EMBEDDING_DIM` | 32 | Embedding vector size |
| `BATCH_SIZE` | 64 | CNN batch size |
| `EPOCHS` | 10 | Max CNN epochs (early stopping enabled) |

## CNN Architecture

```
Embedding (5000 → 32, length 500)
Conv1D (128 filters, kernel 5, ReLU) → MaxPooling1D (2)
Conv1D (64 filters, kernel 5, ReLU)  → MaxPooling1D (2)
GlobalMaxPooling1D
Dense (128, ReLU) → Dropout (0.5)
Dense (1, sigmoid)
```

- Optimizer: Adam (lr = 1e-3)
- Loss: binary cross-entropy
- Validation split: 10% of training data
- Early stopping: monitors `val_loss`, patience 3, restores best weights

## Stopword Experiment

Each review's integer indices are mapped back to words using the IMDB word index. Any word in NLTK's English stopword list is replaced with the padding index (0), then sequences are re-padded and the CNN is retrained from scratch.

Note that the NLTK list includes negations such as "not" and "no", so removing stopwords can discard sentiment-relevant information.

## Output

The script prints:

- Dataset and padded-shape information
- Accuracy and a classification report for Logistic Regression and the Feed-Forward NN
- The CNN model summary, per-epoch training logs, and test accuracy
- Test accuracy for the CNN trained without stopwords
- A bar-style summary comparing all four configurations, followed by a written analysis

## Notes and Limitations

- **Logistic Regression and the MLP use raw padded word indices as features.** Word indices are arbitrary IDs, not meaningful numeric values, so these two models are weak baselines and typically perform far below the CNN. A bag-of-words or TF-IDF representation would be a fairer baseline.
- The written analysis printed at the end of the script contains **hard-coded accuracy ranges**, not measured results. Rely on the numbers printed by the run itself.
- Results vary between runs because of random initialization (the scikit-learn models use `random_state=42`; the Keras models are not seeded).
- The test set is used for final evaluation only; the CNN's early stopping uses a validation split from the training data.

## Possible Extensions

- Use TF-IDF features for the Logistic Regression and MLP baselines
- Add pretrained embeddings (GloVe, word2vec)
- Try an LSTM/GRU or Transformer model
- Use a custom stopword list that preserves negations
- Set random seeds for reproducible Keras runs