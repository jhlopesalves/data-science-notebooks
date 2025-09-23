---
number: 27
title: Make It Reliable
phase: Deep Learning Fundamentals
bundles:
- bundle_pytorch_core
- bundle_mlops_hygiene
project:
  title: “Reproduce Yourself”
  dataset: Fashion-MNIST (Keras or TFDS mirror for cross-framework parity).
  description: Train identical architectures in PyTorch and Keras, lock random seeds,
    and document remaining variation. Package your training script with CLI args and
    a `requirements.txt`/`poetry.lock`.
  dataset_links:
  - Kaggle Fashion-MNIST; Keras dataset page; TFDS catalogue. (https://www.kaggle.com/datasets/zalando-research/fashionmnist)
  metrics:
  - Variance in test accuracy over 10 runs; time-to-train; hash of preprocessed train
    split.
  nuances:
  - Nondeterminism of GPU kernels; library version drift; dataset shuffling sources.
code_focus:
- 'Reproducibility: seeds, determinism flags, version pinning, and hashing datasets.'
- Checkpointing best practices; resume-training and evaluation time augmentation invariance
  checks.
- TorchMetrics for robust evaluation; confusion matrices and per-class metrics; early
  stopping with patience.
- 'Optional: gradient accumulation; mixed precision pitfalls; gradient norm logging.'
math_stats:
- 'Generalisation monitoring: validation curves; bootstrap confidence intervals for
  accuracy and F1.'
- Hyperparameter search as experimental design; multiple testing pitfalls.
docs:
- '[TorchMetrics Classification](https://torchmetrics.readthedocs.io/en/stable/classification/overview.html)'
- '[PyTorch Reproducibility Notes](https://pytorch.org/docs/stable/notes/randomness.html)'
bibliography:
- Bouthillier et al. (2019) “Unreproducible Research is Reproducible.”
- Raff (2019) “A Step Toward Quantifying Independently Reproducible Machine Learning
  Research.”
- D2L training-tricks sections.
---

## Summary

Models that cannot be reproduced cannot be trusted. You will learn to control randomness, track artefacts, and document exactly what produced a result.
## Code Focus

- Reproducibility: seeds, determinism flags, version pinning, and hashing datasets.
- Checkpointing best practices; resume-training and evaluation time augmentation invariance checks.
- TorchMetrics for robust evaluation; confusion matrices and per-class metrics; early stopping with patience.
- Optional: gradient accumulation; mixed precision pitfalls; gradient norm logging.
## Math & Stats

- Generalisation monitoring: validation curves; bootstrap confidence intervals for accuracy and F1.
- Hyperparameter search as experimental design; multiple testing pitfalls.
## Docs

- [TorchMetrics Classification](https://torchmetrics.readthedocs.io/en/stable/classification/overview.html)
- [PyTorch Reproducibility Notes](https://pytorch.org/docs/stable/notes/randomness.html)
## Bibliography

- Bouthillier et al. (2019) “Unreproducible Research is Reproducible.”
- Raff (2019) “A Step Toward Quantifying Independently Reproducible Machine Learning Research.”
- D2L training-tricks sections.
