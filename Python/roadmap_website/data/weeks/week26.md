---
number: 26
title: Convolutions, Augmentations, and Mixed Precision
phase: Deep Learning Fundamentals
bundles:
- bundle_pytorch_core
- bundle_dl_texts
project:
  title: “CIFAR-10 to Tiny-ImageNet”
  dataset: CIFAR-10; Tiny-ImageNet.
  description: 'Train two models on CIFAR-10: (1) from scratch CNN, (2) transfer-learned
    ResNet-18. Then attempt Tiny-ImageNet with careful augmentation and early stopping.
    Use AMP and log throughput.'
  dataset_links:
  - CIFAR official page; TFDS mirror; Tiny-ImageNet mirrors. (https://www.cs.toronto.edu/~kriz/cifar.html)
  metrics:
  - Top-1 accuracy ≥85% on CIFAR-10; training images/sec; GPU memory footprint.
  nuances:
  - Data loader bottlenecks; augmentations that harm vs help; correct evaluation (no
    augmentation at test time).
code_focus:
- Implement CNNs for **CIFAR-10**; use data augmentation, weight decay, cosine LR
  schedules.
- Use **AMP** (autocast + GradScaler) for faster training on GPU; profile and compare
  wall-clock.
- Try **Tiny-ImageNet** as a stretch dataset; use transfer learning from ResNet-18.
math_stats:
- Convolution as sparse, weight-sharing linear operator; receptive fields; padding/stride
  effects.
- Regularisation in deep vision models; effect of augmentation as data-dependent prior.
docs:
- '[TorchVision CIFAR-10](https://pytorch.org/vision/main/generated/torchvision.datasets.CIFAR10.html)'
- '[PyTorch AMP Guide](https://pytorch.org/docs/stable/amp.html)'
- '[Tiny-ImageNet Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)'
bibliography:
- He et al. (2015) “Deep Residual Learning for Image Recognition.”
- Smith (2017) “Cyclical Learning Rates for Training Neural Networks.”
- D2L chapters on CNNs and training tricks.
---

## Summary

You will transition from textbook CNNs to production-like training where throughput and stability matter. This cements habits that will transfer to NLP and multimodal work.
## Code Focus

- Implement CNNs for **CIFAR-10**; use data augmentation, weight decay, cosine LR schedules.
- Use **AMP** (autocast + GradScaler) for faster training on GPU; profile and compare wall-clock.
- Try **Tiny-ImageNet** as a stretch dataset; use transfer learning from ResNet-18.
## Math & Stats

- Convolution as sparse, weight-sharing linear operator; receptive fields; padding/stride effects.
- Regularisation in deep vision models; effect of augmentation as data-dependent prior.
## Docs

- [TorchVision CIFAR-10](https://pytorch.org/vision/main/generated/torchvision.datasets.CIFAR10.html)
- [PyTorch AMP Guide](https://pytorch.org/docs/stable/amp.html)
- [Tiny-ImageNet Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
## Bibliography

- He et al. (2015) “Deep Residual Learning for Image Recognition.”
- Smith (2017) “Cyclical Learning Rates for Training Neural Networks.”
- D2L chapters on CNNs and training tricks.
