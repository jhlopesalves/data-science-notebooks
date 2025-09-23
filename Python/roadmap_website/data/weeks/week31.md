---
number: 31
title: Reproducibility, Checkpointing, Profiling, and Experiment Tracking
phase: Deep Learning & LLMs
bundles:
- bundle_pytorch_core
- bundle_mlops_hygiene
project:
  title: Tracked Image Classifier with Reproducible Runs
  dataset: CIFAR-10
  description: Refactor your Week-25/26 CNN or a new ResNet18 on CIFAR-10 into a script
    with config, seeds, MLflow logging, and robust checkpointing. Profile and fix
    your top bottleneck.
  dataset_links:
  - https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html
  metrics:
  - Top-1 accuracy ≥ 90%
  - End-to-end run reproducible within ±0.5% across 3 seeds
  - Profiler report highlighting ≥ 1 resolved hotspot
  nuances:
  - Distinguish IO vs compute bottlenecks.
  - Keep runs comparable by freezing transforms and seedable splits.
code_focus:
- 'Determinism: seeds across NumPy/PyTorch; cudnn.deterministic; dataloader workers;
  run capture scripts.'
- 'Checkpointing: periodic and best-metric; resume training runs; keep artifacts minimal.'
- 'Profiling & performance: PyTorch Profiler; trace key bottlenecks (CPU/GPU util,
  dataloader, augmentation).'
- 'Experiment tracking: MLflow projects, parameters, metrics, and artifacts; model
  registry; compare runs.'
- 'Packaging: simple train.py with Hydra-like config or argparse, reproducible environment
  files (requirements.txt/pyproject.toml).'
math_stats:
- 'Runtime complexity: throughput vs latency; mini-batch variance effects.'
- Variance of estimators across seeds; confidence intervals via repeated runs.
- Numerical stability with mixed precision; grad scaling.
docs:
- https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
- https://pytorch.org/docs/stable/profiler.html
- https://mlflow.org/docs/latest/index.html
- https://pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html
bibliography:
- Goodfellow, Bengio, Courville, “Deep Learning” — optimisation & generalisation chapters.
- Ozsvald & Gorelick, “High Performance Python” (O’Reilly, 2e) — profiling chapters.
---

## Summary

This is your reliability week. You will turn ad-hoc notebooks into auditable, reproducible training runs with artefacts, metrics, and profiled performance so future you—and colleagues—can reproduce claims precisely.
## Code Focus

- Determinism: seeds across NumPy/PyTorch; cudnn.deterministic; dataloader workers; run capture scripts.
- Checkpointing: periodic and best-metric; resume training runs; keep artifacts minimal.
- Profiling & performance: PyTorch Profiler; trace key bottlenecks (CPU/GPU util, dataloader, augmentation).
- Experiment tracking: MLflow projects, parameters, metrics, and artifacts; model registry; compare runs.
- Packaging: simple train.py with Hydra-like config or argparse, reproducible environment files (requirements.txt/pyproject.toml).
## Math & Stats

- Runtime complexity: throughput vs latency; mini-batch variance effects.
- Variance of estimators across seeds; confidence intervals via repeated runs.
- Numerical stability with mixed precision; grad scaling.
## Docs

- https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
- https://pytorch.org/docs/stable/profiler.html
- https://mlflow.org/docs/latest/index.html
- https://pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html
## Bibliography

- Goodfellow, Bengio, Courville, “Deep Learning” — optimisation & generalisation chapters.
- Ozsvald & Gorelick, “High Performance Python” (O’Reilly, 2e) — profiling chapters.
