---
number: 36
title: Transformer Fine-Tuning (GLUE/SQuAD) with PEFT and Evaluation Best Practice
phase: NLP & GenAI
bundles:
- bundle_peft_quant
- bundle_serving_llm
project:
  title: Transformer Baselines You Can Trust
  dataset: GLUE (SST-2 or MRPC) and SQuAD v1.1/v2.0
  description: Produce robust, well-evaluated fine-tuned transformers with PEFT, including
    reproducible metrics and calibrated outputs.
  dataset_links:
  - https://huggingface.co/datasets/glue
  - https://huggingface.co/datasets/squad
  metrics:
  - 'SST-2: accuracy ≥ 93%; MRPC: F1 ≥ 88%'
  - 'SQuAD v1.1: EM ≥ 80, F1 ≥ 88 (targets scale with base model size)'
  nuances:
  - Beware data leakage via text normalisation; ensure evaluation scripts match official
    ones.
code_focus:
- Full HF fine-tuning flow with Trainer on GLUE (SST-2 or MRPC) and SQuAD v1.1/v2.0;
  integrate PEFT for efficiency.
- 'Robust evaluation: use evaluate for standard metrics; implement EM/F1 script parity
  for QA; track runs in MLflow.'
- 'Error analysis: calibration curves for classifiers; answer length and null-answer
  thresholds for SQuAD2.'
math_stats:
- Self-attention recap; positional encodings; warmup schedules and generalisation.
- Proper scoring rules; reliability diagrams; AUROC vs PR-AUC trade-offs with class
  imbalance.
docs:
- https://huggingface.co/docs/transformers/index
- https://huggingface.co/docs/datasets/
- https://huggingface.co/docs/evaluate/en/index
- https://rajpurkar.github.io/SQuAD-explorer/
- https://huggingface.co/docs/peft/en/index
bibliography:
- Vaswani et al., “Attention Is All You Need” (2017).
- Tunstall, von Werra, Wolf, “NLP with Transformers” — GLUE/SQuAD chapters.
---

## Summary

You will turn transformer fine-tuning into a disciplined craft: consistent metrics, calibrated decisions, and defensible comparisons.
## Code Focus

- Full HF fine-tuning flow with Trainer on GLUE (SST-2 or MRPC) and SQuAD v1.1/v2.0; integrate PEFT for efficiency.
- Robust evaluation: use evaluate for standard metrics; implement EM/F1 script parity for QA; track runs in MLflow.
- Error analysis: calibration curves for classifiers; answer length and null-answer thresholds for SQuAD2.
## Math & Stats

- Self-attention recap; positional encodings; warmup schedules and generalisation.
- Proper scoring rules; reliability diagrams; AUROC vs PR-AUC trade-offs with class imbalance.
## Docs

- https://huggingface.co/docs/transformers/index
- https://huggingface.co/docs/datasets/
- https://huggingface.co/docs/evaluate/en/index
- https://rajpurkar.github.io/SQuAD-explorer/
- https://huggingface.co/docs/peft/en/index
## Bibliography

- Vaswani et al., “Attention Is All You Need” (2017).
- Tunstall, von Werra, Wolf, “NLP with Transformers” — GLUE/SQuAD chapters.
