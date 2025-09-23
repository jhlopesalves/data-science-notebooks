---
number: 32
title: 'Phase-D Project: Multimodal or Text-Only Production Model'
phase: Deep Learning & LLMs
bundles:
- bundle_serving_api
- bundle_orchestration_tracking
project:
  title: Deployed Classifier (Multimodal or Text)
  dataset: 'Option A: MM-IMDb (posters + plots, multi-label); Option B: AG News (4-class
    text).'
  dataset_links:
  - https://huggingface.co/datasets/sxj1215/mmimdb
  - https://huggingface.co/datasets/ag_news
  metrics:
  - 'A: macro-F1 ≥ 0.60 across genres; B: accuracy ≥ 92%'
  - p50 latency ≤ 50 ms (local), p95 ≤ 150 ms for 1-sentence inputs
  - 'Parity tests: ONNX vs PyTorch prediction agreement ≥ 99.5% on 1k samples'
  nuances:
  - Multi-label thresholds vs macro/micro averaging.
  - Schema validation at the service boundary; strict error handling.
code_focus:
- 'Choose one: (A) multimodal late-fusion (image encoder + text features) or (B) text
  classifier with ONNX/TensorRT export.'
- MLflow model registry; promote from staging to production; smoke tests for on-nx
  inference equivalence.
- Basic FastAPI inference service; batch and single-request endpoints; pydantic schema;
  latency logging.
math_stats:
- Feature concatenation vs attention pooling; calibration drift post-export.
- 'Throughput modelling: QPS vs batch size; cold vs warm latency.'
docs:
- https://onnxruntime.ai/docs/
- https://fastapi.tiangolo.com/
- https://mlflow.org/docs/latest/index.html
- https://huggingface.co/datasets/sxj1215/mmimdb
- https://huggingface.co/datasets/ag_news
bibliography:
- Raschka, Mirjalili, “Machine Learning with PyTorch and Scikit-Learn” (Packt, 2022)
  — deployment chapters.
- 'Ivanov, “High Performance Python: Practical Performant Programming for Humans”
  (Apress, 2021) — selected sections.'
---

## Summary

You will ship something. Either a multimodal predictor or an optimised text model with an HTTP API and a repeatable release path. This consolidates the deep-learning phase into a demonstrable, runnable service.
## Project Description

Train, register, and serve a model. For A, extract CNN features from posters and TF-IDF or embeddings from plots; fuse and train a multi-label classifier. For B, fine-tune a compact transformer, export to ONNX, and stand up a FastAPI service.
## Code Focus

- Choose one: (A) multimodal late-fusion (image encoder + text features) or (B) text classifier with ONNX/TensorRT export.
- MLflow model registry; promote from staging to production; smoke tests for on-nx inference equivalence.
- Basic FastAPI inference service; batch and single-request endpoints; pydantic schema; latency logging.
## Math & Stats

- Feature concatenation vs attention pooling; calibration drift post-export.
- Throughput modelling: QPS vs batch size; cold vs warm latency.
## Docs

- https://onnxruntime.ai/docs/
- https://fastapi.tiangolo.com/
- https://mlflow.org/docs/latest/index.html
- https://huggingface.co/datasets/sxj1215/mmimdb
- https://huggingface.co/datasets/ag_news
## Bibliography

- Raschka, Mirjalili, “Machine Learning with PyTorch and Scikit-Learn” (Packt, 2022) — deployment chapters.
- Ivanov, “High Performance Python: Practical Performant Programming for Humans” (Apress, 2021) — selected sections.
