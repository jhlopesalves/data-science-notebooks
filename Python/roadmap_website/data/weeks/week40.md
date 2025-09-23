---
number: 40
title: 'Phase-E Project: RAG Assistant or Multilingual NER at Production Quality'
phase: NLP & GenAI
bundles:
- bundle_capstone_core
project:
  title: Deployed NLP Capstone
  dataset: 'A: Wikipedia or your domain docs + custom QA; B: CoNLL-2003 or multilingual
    variant'
  description: Build a user-facing assistant (A) or a packaged NER model (B). Include
    a README, Model Card, evaluation notebook, and a FastAPI app with one-click run
    scripts.
  dataset_links:
  - https://huggingface.co/datasets/wikipedia
  - https://huggingface.co/datasets/squad
  - https://huggingface.co/datasets/conll2003
  metrics:
  - 'A: RAGAS faithfulness ≥ 0.75; grounded citation rate ≥ 95%'
  - 'B: entity-F1 ≥ 0.88 English and ≥ 0.80 on one non-English language'
  - Service p95 latency targets met; ≥ 90% test coverage on utility modules
  nuances:
  - Dataset licences and attribution; privacy in logs; prompt injection defences for
    RAG.
code_focus:
- 'Choose one capstone: (A) RAG assistant with citations and guardrails; (B) Multilingual
  NER (fine-tuned transformer) with spaCy wrapper.'
- 'End-to-end: ingestion, indexing, retriever-reader, evaluation (RAGAS); or token
  classification training and packaging.'
- Serve with FastAPI; write a Model Card; add basic monitoring (request logs, drift
  snapshot).
math_stats:
- RAG evaluation sampling design; confidence intervals for faithfulness metrics.
- 'Sequence-labelling evaluation: macro vs micro F1; per-entity confusion.'
docs:
- https://docs.ragas.io/
- https://python.langchain.com/docs/get_started/introduction
- https://docs.llamaindex.ai/
- https://fastapi.tiangolo.com/
- https://spacy.io/usage
bibliography:
- Mitchell et al., “Model Cards for Model Reporting,” 2019.
- Rothman, “Transformers for NLP” (Packt, 3e) — deployment and applications.
---

## Summary

This capstone demonstrates a production-level NLP system: defensible evaluation, documentation, and an API. The artefacts are portfolio-ready and auditable.
## Code Focus

- Choose one capstone: (A) RAG assistant with citations and guardrails; (B) Multilingual NER (fine-tuned transformer) with spaCy wrapper.
- End-to-end: ingestion, indexing, retriever-reader, evaluation (RAGAS); or token classification training and packaging.
- Serve with FastAPI; write a Model Card; add basic monitoring (request logs, drift snapshot).
## Math & Stats

- RAG evaluation sampling design; confidence intervals for faithfulness metrics.
- Sequence-labelling evaluation: macro vs micro F1; per-entity confusion.
## Docs

- https://docs.ragas.io/
- https://python.langchain.com/docs/get_started/introduction
- https://docs.llamaindex.ai/
- https://fastapi.tiangolo.com/
- https://spacy.io/usage
## Bibliography

- Mitchell et al., “Model Cards for Model Reporting,” 2019.
- Rothman, “Transformers for NLP” (Packt, 3e) — deployment and applications.
