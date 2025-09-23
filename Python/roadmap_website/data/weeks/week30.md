---
number: 30
title: PEFT/LoRA Fine-Tuning for Classification & QA
phase: Deep Learning & LLMs
bundles:
- bundle_peft_quant
- bundle_responsible_ai
project:
  title: 'Adapter-Tuned BERT: Classifier + QA'
  dataset: GLUE/SST-2 for classification; SQuAD v1.1 or v2.0 for QA
  dataset_links:
  - https://huggingface.co/datasets/glue
  - https://huggingface.co/datasets/squad
  metrics:
  - 'SST-2: accuracy ≥ 90% with PEFT on dev'
  - 'SQuAD: EM ≥ 75, F1 ≥ 83 (v1.1) or competitive scores on v2.0'
  - 'Inference: ≥ 30% memory reduction vs full fine-tune; latency reported'
  nuances:
  - Monitor trainable parameter count vs rank r.
  - Check class calibration on SST-2; examine threshold effects.
  - 'SQuAD v2.0: no-answer threshold tuning and evaluation script parity.'
code_focus:
- Set up Hugging Face Transformers + Datasets + Accelerate + PEFT; choose a compact
  base model (e.g., distilbert-base-uncased) then apply LoRA adapters on attention
  projections.
- 'Train 2 tasks end-to-end: (A) Sentiment classification (GLUE/SST-2) with Trainer
  API; (B) Extractive QA (SQuAD v1.1 or v2.0) with default run_qa script or Trainer.'
- 'Memory efficiency: 8-bit/4-bit loading via bitsandbytes; gradient accumulation;
  mixed precision (fp16/bf16 on Colab Pro).'
- 'Hyperparameters: rank r for LoRA, alpha, dropout; learning rate warmup and linear
  decay; early stopping; seed fixing for reproducibility.'
- 'Evaluation: accuracy (SST-2); EM/F1 (SQuAD) using evaluate; calibration check with
  reliability curve on classifier.'
- 'Export & reuse: save adapter weights; optionally merge & export to ONNX Runtime
  for inference; compare latency and memory between full FT vs PEFT.'
math_stats:
- Cross-entropy, softmax, label smoothing; calibration and proper scoring rules.
- 'Low-rank factorisation: parameter count reduction by decomposing ΔW ≈ A·B with
  rank r; trace/nuclear norm intuition.'
- Backpropagation refresher and gradient flow through adapter modules; effects of
  mixed precision on numerical stability.
docs:
- https://huggingface.co/docs/transformers/index
- https://huggingface.co/docs/datasets/
- https://huggingface.co/docs/peft/en/index
- https://huggingface.co/docs/peft/main/en/conceptual_guides/lora
- https://github.com/TimDettmers/bitsandbytes
- https://huggingface.co/docs/evaluate/en/index
- https://arxiv.org/abs/1706.03762
bibliography:
- Vaswani et al., “Attention Is All You Need,” NeurIPS 2017.
- 'Hu et al., “LoRA: Low-Rank Adaptation of Large Language Models,” 2021.'
- 'Jurafsky & Martin, “Speech and Language Processing” (3e draft): Transformers &
  QA chapters.'
- Tunstall, von Werra, Wolf, “Natural Language Processing with Transformers” (O’Reilly,
  2022) — fine-tuning chapters.
---

## Summary

You will learn to adapt a pretrained encoder efficiently with LoRA, proving that small adapter matrices can approach full fine-tune quality at a fraction of the cost. You will compare training curves, memory footprints, and export paths so that you can defend PEFT choices in production settings.
## Project Description

Implement two PEFT pipelines: (1) fine-tune a sentiment classifier with LoRA on a compact BERT; (2) fine-tune extractive QA with LoRA. Track memory, wall-clock, and validation metrics; export adapters; test ONNX inference.
## Code Focus

- Set up Hugging Face Transformers + Datasets + Accelerate + PEFT; choose a compact base model (e.g., distilbert-base-uncased) then apply LoRA adapters on attention projections.
- Train 2 tasks end-to-end: (A) Sentiment classification (GLUE/SST-2) with Trainer API; (B) Extractive QA (SQuAD v1.1 or v2.0) with default run_qa script or Trainer.
- Memory efficiency: 8-bit/4-bit loading via bitsandbytes; gradient accumulation; mixed precision (fp16/bf16 on Colab Pro).
- Hyperparameters: rank r for LoRA, alpha, dropout; learning rate warmup and linear decay; early stopping; seed fixing for reproducibility.
- Evaluation: accuracy (SST-2); EM/F1 (SQuAD) using evaluate; calibration check with reliability curve on classifier.
- Export & reuse: save adapter weights; optionally merge & export to ONNX Runtime for inference; compare latency and memory between full FT vs PEFT.
## Math & Stats

- Cross-entropy, softmax, label smoothing; calibration and proper scoring rules.
- Low-rank factorisation: parameter count reduction by decomposing ΔW ≈ A·B with rank r; trace/nuclear norm intuition.
- Backpropagation refresher and gradient flow through adapter modules; effects of mixed precision on numerical stability.
## Docs

- https://huggingface.co/docs/transformers/index
- https://huggingface.co/docs/datasets/
- https://huggingface.co/docs/peft/en/index
- https://huggingface.co/docs/peft/main/en/conceptual_guides/lora
- https://github.com/TimDettmers/bitsandbytes
- https://huggingface.co/docs/evaluate/en/index
- https://arxiv.org/abs/1706.03762
## Bibliography

- Vaswani et al., “Attention Is All You Need,” NeurIPS 2017.
- Hu et al., “LoRA: Low-Rank Adaptation of Large Language Models,” 2021.
- Jurafsky & Martin, “Speech and Language Processing” (3e draft): Transformers & QA chapters.
- Tunstall, von Werra, Wolf, “Natural Language Processing with Transformers” (O’Reilly, 2022) — fine-tuning chapters.
