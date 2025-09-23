---
number: 38
title: 'Serving LLMs Efficiently: vLLM, TGI, Quantisation, and FastAPI'
phase: NLP & GenAI
bundles:
- bundle_responsible_ai
project:
  title: LLM Text-Gen Microservice with Benchmarks
  dataset: N/A (use curated prompt sets + small public QA lists)
  description: Deploy a generation server, wrap with FastAPI, and deliver a benchmark
    report with latency/throughput curves under different batch sizes and quantisation
    settings.
  metrics:
  - p50 ≤ 150 ms for short prompts (cached), p95 ≤ 600 ms
  - Throughput tokens/sec reported for 3 batch sizes
  - Error rate (timeouts) < 1% in a 10-minute load test
  nuances:
  - Memory vs batch trade-offs; request deduplication; deterministic generation for
    regression tests.
code_focus:
- 'Stand up a local text-generation server: pick vLLM or Hugging Face TGI; configure
  tensor parallelism; enable streaming.'
- 'Client service: wrap generation behind FastAPI; implement batching & timeouts;
  log prompts, seeds, and model SHA.'
- 'Optimisation: 4-bit/8-bit quantisation where supported; prompt caching; max tokens
  and stop sequences.'
- 'Benchmarking: load-test QPS vs latency; measure throughput under batch sizes; record
  token/sec.'
math_stats:
- 'Throughput models: tokens/sec vs context length; effect of kv-cache and batch.'
- Quantisation error basics; impact on perplexity and task accuracy.
docs:
- https://docs.vllm.ai/
- https://huggingface.co/docs/text-generation-inference/index
- https://fastapi.tiangolo.com/
- https://onnxruntime.ai/docs/
bibliography:
- Kleppmann, “Designing Data-Intensive Applications” — service SLAs & back-pressure.
- 'HuggFace LLM course: deployment & evaluation chapters.'
---

## Summary

You will learn to make LLMs behave like systems: observable, efficient, and predictable under load, with hard numbers—not vibes.
## Code Focus

- Stand up a local text-generation server: pick vLLM or Hugging Face TGI; configure tensor parallelism; enable streaming.
- Client service: wrap generation behind FastAPI; implement batching & timeouts; log prompts, seeds, and model SHA.
- Optimisation: 4-bit/8-bit quantisation where supported; prompt caching; max tokens and stop sequences.
- Benchmarking: load-test QPS vs latency; measure throughput under batch sizes; record token/sec.
## Math & Stats

- Throughput models: tokens/sec vs context length; effect of kv-cache and batch.
- Quantisation error basics; impact on perplexity and task accuracy.
## Docs

- https://docs.vllm.ai/
- https://huggingface.co/docs/text-generation-inference/index
- https://fastapi.tiangolo.com/
- https://onnxruntime.ai/docs/
## Bibliography

- Kleppmann, “Designing Data-Intensive Applications” — service SLAs & back-pressure.
- HuggFace LLM course: deployment & evaluation chapters.
