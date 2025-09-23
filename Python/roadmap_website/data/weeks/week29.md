---
number: 29
title: Transfer and Text in Keras
phase: Deep Learning Fundamentals
bundles:
- bundle_keras_tf
- bundle_nlp_foundations
project:
  title: “Two Ways to Sentiment”
  dataset: IMDB 50k reviews.
  dataset_links:
  - Keras/TensorFlow IMDB; Kaggle mirrors for comparison. (https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
  metrics:
  - Accuracy, macro-F1, ECE; latency on CPU.
  nuances:
  - Sequence length trade-offs; tokeniser training artifacts; handling class imbalance
    if you subset.
code_focus:
- Transfer learning with `keras.applications` (e.g., MobileNetV2) and fine-tuning
  strategies; layer freezing schedules.
- Regularisation in Keras (Dropout, BatchNorm, weight decay via `l2`).
- 'First text task in Keras: IMDB sentiment with `TextVectorization`; compare to a
  compact Transformer from `keras_nlp` or Hugging Face (lite) for the same task.'
math_stats:
- Pretrained features as priors; catastrophic forgetting; effective learning rate
  under layer freezing.
- 'Tokenisation: subword models and OOV handling; calibration in text classifiers.'
docs:
- '[Keras Applications](https://keras.io/api/applications/)'
- '[Keras NLP Guides](https://keras.io/guides/keras_nlp/)'
- '[TensorFlow Datasets IMDB](https://www.tensorflow.org/datasets/catalog/imdb_reviews)'
bibliography:
- Howard & Gugger — *Natural Language Processing with Transformers* (selected chapters
  for fine-tuning).
- Zhang et al. (2016) *Understanding Deep Learning Requires Rethinking Generalization*
  (for healthy scepticism).
---

## Summary

You connect transfer learning and lightweight NLP, seeing how pretraining and regularisation change the optimisation landscape. The aim is not state-of-the-art scores but sound engineering and honest evaluation.
## Project Description

Build two sentiment classifiers: (1) bag-of-words + logistic regression baseline, (2) Keras LSTM or small Transformer. Compare calibration and error modes; produce a confusion matrix by review length decile.
## Code Focus

- Transfer learning with `keras.applications` (e.g., MobileNetV2) and fine-tuning strategies; layer freezing schedules.
- Regularisation in Keras (Dropout, BatchNorm, weight decay via `l2`).
- First text task in Keras: IMDB sentiment with `TextVectorization`; compare to a compact Transformer from `keras_nlp` or Hugging Face (lite) for the same task.
## Math & Stats

- Pretrained features as priors; catastrophic forgetting; effective learning rate under layer freezing.
- Tokenisation: subword models and OOV handling; calibration in text classifiers.
## Docs

- [Keras Applications](https://keras.io/api/applications/)
- [Keras NLP Guides](https://keras.io/guides/keras_nlp/)
- [TensorFlow Datasets IMDB](https://www.tensorflow.org/datasets/catalog/imdb_reviews)
## Bibliography

- Howard & Gugger — *Natural Language Processing with Transformers* (selected chapters for fine-tuning).
- Zhang et al. (2016) *Understanding Deep Learning Requires Rethinking Generalization* (for healthy scepticism).
