---
number: 37
title: 'RAG Fundamentals: Ingestion, Indexing, Retrieval, and Evaluation'
phase: NLP & GenAI
bundles:
- bundle_rag_orchestration
- bundle_rag_eval
project:
  title: Grounded Q&A over Wikipedia
  dataset: Wikipedia snapshot + held-out QA (SQuAD or custom)
  description: Build a RAG pipeline over a constrained Wikipedia slice; evaluate retrieval
    and generation quality; compare dense vs dense+rerank.
  dataset_links:
  - https://huggingface.co/datasets/wikipedia
  - https://huggingface.co/datasets/squad
  metrics:
  - RAGAS faithfulness ≥ 0.7 on dev
  - Retrieval recall@5 ≥ 0.9 on synthetic oracle questions
  - Latency p95 ≤ 500 ms per query locally
  nuances:
  - Citations in output; guard against truncated spans; store hashes for doc versioning.
code_focus:
- 'Document ingestion: chunking with overlap, metadata, citation retention.'
- 'Vector store options: FAISS (CPU/GPU) and Chroma; build HNSW/IVF indices; persistence.'
- 'RAG orchestration: implement a simple retriever-reader with LangChain or LlamaIndex;
  add reranking (cross-encoder) vs pure dense.'
- 'Evaluation: use RAGAS on held-out QA pairs; track faithfulness and answer relevance.'
math_stats:
- ANN search (HNSW/IVF/PQ) intuition; recall/latency trade-offs.
- Chunking as a bias-variance knob for retrieval; effects on grounding & hallucinations.
docs:
- https://github.com/facebookresearch/faiss
- https://docs.trychroma.com/
- https://python.langchain.com/docs/get_started/introduction
- https://docs.llamaindex.ai/
- https://docs.ragas.io/
- https://huggingface.co/docs/datasets/
bibliography:
- Lewis et al., “Retrieval-Augmented Generation for Knowledge-Intensive NLP” (2020).
- Gao et al., “Rethinking with Retrieval” (selected sections) — optional.
---

## Summary

You will connect indexing theory to grounded generation practice, including rigorous RAG evaluation so you can iterate intelligently rather than guess.
## Code Focus

- Document ingestion: chunking with overlap, metadata, citation retention.
- Vector store options: FAISS (CPU/GPU) and Chroma; build HNSW/IVF indices; persistence.
- RAG orchestration: implement a simple retriever-reader with LangChain or LlamaIndex; add reranking (cross-encoder) vs pure dense.
- Evaluation: use RAGAS on held-out QA pairs; track faithfulness and answer relevance.
## Math & Stats

- ANN search (HNSW/IVF/PQ) intuition; recall/latency trade-offs.
- Chunking as a bias-variance knob for retrieval; effects on grounding & hallucinations.
## Docs

- https://github.com/facebookresearch/faiss
- https://docs.trychroma.com/
- https://python.langchain.com/docs/get_started/introduction
- https://docs.llamaindex.ai/
- https://docs.ragas.io/
- https://huggingface.co/docs/datasets/
## Bibliography

- Lewis et al., “Retrieval-Augmented Generation for Knowledge-Intensive NLP” (2020).
- Gao et al., “Rethinking with Retrieval” (selected sections) — optional.
