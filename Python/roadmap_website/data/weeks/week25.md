---
number: 25
title: The Gradient Engine
phase: Deep Learning Fundamentals
bundles:
- bundle_pytorch_core
- bundle_dl_texts
project:
  title: “From Linear to MLP”
  dataset: MNIST (TorchVision); Adult tabular.
  description: Re-implement your Week-9 logistic classifier as an MLP; compare calibration
    and decision quality. On MNIST, reach ≥98% test accuracy with a small MLP; log
    experiments with MLflow.
  dataset_links:
  - MNIST TorchVision; alternative TFDS where useful. (https://docs.pytorch.org/vision/main/generated/torchvision.datasets.MNIST.html)
  metrics:
  - Accuracy/AUROC; Brier score; training time; parameter count.
  nuances:
  - Initialisation, learning-rate schedules; overfitting diagnostics with learning
    curves.
code_focus:
- '**PyTorch** essentials: tensors, broadcasting, `autograd`, `nn.Module`, `DataLoader`.'
- 'Implement a clean training loop: gradient clipping, early stopping, checkpointing,
  and seeding.'
- Train MLPs on tabular sets (Adult, Higgs small), plus **MNIST** for classification;
  start using TorchMetrics.
math_stats:
- Chain rule and backpropagation; cross-entropy vs MSE; softmax and log-sum-exp stability.
- Capacity, regularisation (weight decay, dropout), and bias–variance revisited in
  deep nets.
docs:
- '[PyTorch Tensors & Autograd](https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html)'
- '[TorchVision Datasets](https://pytorch.org/vision/stable/datasets.html)'
- '[TorchMetrics](https://lightning.ai/docs/torchmetrics/stable/index.html)'
bibliography:
- '*Dive into Deep Learning* (D2L), Chapters 1–4.'
- Goodfellow, Bengio, Courville — *Deep Learning* (selected sections on backprop and
  optimisation).
- Bishop — *Pattern Recognition and Machine Learning* (Chapter 5, neural networks).
---

## Summary

You will internalise the mechanics of tensors and gradients and learn to write a robust, inspectable training loop. You also begin to see when deep nets are overkill for tabular data.
## Code Focus

- **PyTorch** essentials: tensors, broadcasting, `autograd`, `nn.Module`, `DataLoader`.
- Implement a clean training loop: gradient clipping, early stopping, checkpointing, and seeding.
- Train MLPs on tabular sets (Adult, Higgs small), plus **MNIST** for classification; start using TorchMetrics.
## Math & Stats

- Chain rule and backpropagation; cross-entropy vs MSE; softmax and log-sum-exp stability.
- Capacity, regularisation (weight decay, dropout), and bias–variance revisited in deep nets.
## Docs

- [PyTorch Tensors & Autograd](https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html)
- [TorchVision Datasets](https://pytorch.org/vision/stable/datasets.html)
- [TorchMetrics](https://lightning.ai/docs/torchmetrics/stable/index.html)
## Bibliography

- *Dive into Deep Learning* (D2L), Chapters 1–4.
- Goodfellow, Bengio, Courville — *Deep Learning* (selected sections on backprop and optimisation).
- Bishop — *Pattern Recognition and Machine Learning* (Chapter 5, neural networks).
