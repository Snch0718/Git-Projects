# Sample Output

This file shows the output from one complete run of the IMDB sentiment analysis script, so you can check that your own run looks similar.

Exact numbers will differ slightly between runs (the Keras models are not seeded, and timings depend on hardware). Treat the values below as a reference for **what to expect**, not exact targets.

## Quick Verification Checklist

| Check | Expected |
|-------|----------|
| Train / test samples | 25,000 / 25,000 |
| Padded shapes | `(25000, 500)` for both train and test |
| Class balance (train) | 12,500 positive / 12,500 negative |
| Logistic Regression accuracy | ~50% (near chance) |
| Feed-Forward NN accuracy | ~50% (near chance) |
| CNN (with stopwords) accuracy | ~83-85% |
| CNN (stopwords removed) accuracy | ~83-86% |
| Stopword example output | First 10 indices are all `0` (expected, see notes) |

## Results Summary

| Model | Test Accuracy |
|-------|---------------|
| Logistic Regression | 50.88% |
| Feed-Forward NN (sklearn) | 50.24% |
| CNN (with stopwords) | 83.83% |
| CNN (stopwords removed) | 85.22% |

## Full Console Output

### Dataset and split

```
============================================================
  IMDB Sentiment Analysis
============================================================

[Dataset]
  Training samples : 25000
  Test samples     : 25000
  Vocabulary size  : 5000
  Sample review (indices): [1, 14, 22, 16, 43, 530, 973, 1622, 1385, 65] ...

[Padded shapes]  X_train: (25000, 500)  |  X_test: (25000, 500)

[Split]  Train=25000  Test=25000
  Positive reviews (train): 12500
  Negative reviews (train): 12500
```

### Model 1: Logistic Regression

```
============================================================
  MODEL 1: Logistic Regression
============================================================

Logistic Regression Accuracy : 50.88%

Classification Report:
              precision    recall  f1-score   support

    Negative       0.51      0.57      0.54     12500
    Positive       0.51      0.45      0.48     12500

    accuracy                           0.51     25000
   macro avg       0.51      0.51      0.51     25000
weighted avg       0.51      0.51      0.51     25000
```

### Model 2: Feed-Forward Neural Network

```
============================================================
  MODEL 2: Feed-Forward Neural Network
============================================================

Feed-Forward NN Accuracy : 50.24%

Classification Report:
              precision    recall  f1-score   support

    Negative       0.50      0.60      0.55     12500
    Positive       0.50      0.41      0.45     12500

    accuracy                           0.50     25000
   macro avg       0.50      0.50      0.50     25000
weighted avg       0.50      0.50      0.50     25000
```

### Model 3: CNN (with stopwords)

```
============================================================
  MODEL 3: Convolutional Neural Network (Keras)
============================================================
I0000 00:00:1790951928.701488    3768 cpu_feature_guard.cc:227] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ embedding (Embedding)                │ ?                           │     0 (unbuilt) │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv1d (Conv1D)                      │ ?                           │     0 (unbuilt) │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling1d (MaxPooling1D)         │ ?                           │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv1d_1 (Conv1D)                    │ ?                           │     0 (unbuilt) │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling1d_1 (MaxPooling1D)       │ ?                           │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ global_max_pooling1d                 │ ?                           │               0 │
│ (GlobalMaxPooling1D)                 │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense (Dense)                        │ ?                           │     0 (unbuilt) │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout (Dropout)                    │ ?                           │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (Dense)                      │ ?                           │     0 (unbuilt) │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 0 (0.00 B)
 Trainable params: 0 (0.00 B)
 Non-trainable params: 0 (0.00 B)
Epoch 1/10
352/352 ━━━━━━━━━━━━━━━━━━━━ 18s 49ms/step - accuracy: 0.7294 - loss: 0.5022 - val_accuracy: 0.8324 - val_loss: 0.3820
Epoch 2/10
352/352 ━━━━━━━━━━━━━━━━━━━━ 16s 44ms/step - accuracy: 0.8841 - loss: 0.2886 - val_accuracy: 0.8436 - val_loss: 0.4081
Epoch 3/10
352/352 ━━━━━━━━━━━━━━━━━━━━ 16s 46ms/step - accuracy: 0.9194 - loss: 0.2140 - val_accuracy: 0.8272 - val_loss: 0.4798
Epoch 4/10
352/352 ━━━━━━━━━━━━━━━━━━━━ 16s 46ms/step - accuracy: 0.9398 - loss: 0.1656 - val_accuracy: 0.8512 - val_loss: 0.4192

CNN Test Accuracy : 83.83%
```

### Step 6: Stopword removal and CNN retrain

```
============================================================
  STEP 6: Stopword Removal Experiment
============================================================
Removing stopwords from sequences …
  Done. Example (first 10 indices after SW removal): [0 0 0 0 0 0 0 0 0 0]
Epoch 1/10
352/352 ━━━━━━━━━━━━━━━━━━━━ 17s 47ms/step - accuracy: 0.7343 - loss: 0.5079 - val_accuracy: 0.8088 - val_loss: 0.4145
Epoch 2/10
352/352 ━━━━━━━━━━━━━━━━━━━━ 18s 50ms/step - accuracy: 0.8709 - loss: 0.3127 - val_accuracy: 0.8076 - val_loss: 0.4516
Epoch 3/10
352/352 ━━━━━━━━━━━━━━━━━━━━ 18s 50ms/step - accuracy: 0.9056 - loss: 0.2409 - val_accuracy: 0.8492 - val_loss: 0.3851
Epoch 4/10
352/352 ━━━━━━━━━━━━━━━━━━━━ 17s 47ms/step - accuracy: 0.9331 - loss: 0.1823 - val_accuracy: 0.8624 - val_loss: 0.3644
Epoch 5/10
352/352 ━━━━━━━━━━━━━━━━━━━━ 17s 47ms/step - accuracy: 0.9495 - loss: 0.1456 - val_accuracy: 0.8608 - val_loss: 0.4126
Epoch 6/10
352/352 ━━━━━━━━━━━━━━━━━━━━ 16s 46ms/step - accuracy: 0.9551 - loss: 0.1202 - val_accuracy: 0.8516 - val_loss: 0.4779
Epoch 7/10
352/352 ━━━━━━━━━━━━━━━━━━━━ 17s 47ms/step - accuracy: 0.9752 - loss: 0.0736 - val_accuracy: 0.8548 - val_loss: 0.5408

CNN (no stopwords) Test Accuracy : 85.22%
```

### Results summary

```
============================================================
  RESULTS SUMMARY
============================================================
  Logistic Regression                  50.88%  ████████████████████
  Feed-Forward NN (sklearn)            50.24%  ████████████████████
  CNN (with stopwords)                 83.83%  █████████████████████████████████
  CNN (stopwords removed)              85.22%  ██████████████████████████████████
```

The script then prints a written `[Analysis]` section. That text is hard-coded in the script and is not computed from the results (see notes below).

## Notes on Reading the Output

**Logistic Regression and FFNN at ~50%.** These models are trained on raw padded word-index IDs, which are arbitrary labels rather than meaningful numeric values. With two balanced classes, ~50% is chance level, so these two models have effectively learned nothing useful. This is a limitation of the feature representation, not a bug in the run.

**CNN early stopping.** Early stopping monitors `val_loss` with patience 3 and restores the best weights.
- *With stopwords:* the best `val_loss` was in epoch 1 (0.3820), so training stopped after epoch 4 and epoch 1 weights were restored.
- *Without stopwords:* the best `val_loss` was in epoch 4 (0.3644), so training stopped after epoch 7 and epoch 4 weights were restored.

The final test accuracy comes from these restored weights, not from the last epoch shown.

**Overfitting signal.** Training accuracy climbs above 93% while validation accuracy stays around 83-86% and `val_loss` rises. This is the overfitting that early stopping guards against.

**Stopword example showing all zeros.** `[0 0 0 0 0 0 0 0 0 0]` is expected. Sequences are padded at the *front* by default, and most reviews are shorter than 500 tokens, so the first indices are padding.

**Stopword removal result.** The stopword-removed CNN scored 85.22% vs 83.83%, a difference of about 1.4 points. Because the models are unseeded and validation accuracy fluctuates between epochs, a gap this size may not be reproducible. Run several times before drawing a conclusion.

**Model summary shows `0 (unbuilt)`.** In this run the summary was printed before the model had seen any data, so the layers show `?` shapes and 0 parameters. Training works normally. Passing `input_length` to `Embedding` is ignored or deprecated in newer Keras versions. To get a populated summary, call `model.build(input_shape=(None, MAX_REVIEW_LEN))` or add an `Input` layer.

**TensorFlow `I0000 ... cpu_feature_guard` message.** This is an informational log about CPU instruction sets and is harmless.

**Hard-coded analysis text.** The closing `[Analysis]` block claims ~85-87% for Logistic Regression and 88-91% for the CNN. Neither matches the measured results above (50.88% and 83.83% / 85.22%). The "exceeds the 85% target" conclusion also holds only for the stopword-removed CNN. Rely on the numbers printed in the Results Summary.