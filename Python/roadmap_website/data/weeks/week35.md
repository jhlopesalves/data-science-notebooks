---
number: 35
title: 'Distributional Semantics: word2vec, fastText, and Sentence Embeddings'
phase: NLP & GenAI
bundles:
- bundle_hf_transformers
- bundle_serving_llm
project:
  title: Semantic Index of a Domain Corpus
  dataset: Use your Week-33 corpus (e.g., Wikipedia slice) + optional domain corpus
  description: Train or load embeddings, index documents, and implement nearest-neighbour
    search with qualitative demos and quantitative recall@k.
  dataset_links:
  - https://huggingface.co/datasets/wikipedia
  metrics:
  - 'Intrinsic: word similarity/analogy on standard lists where applicable'
  - 'Extrinsic: retrieval precision@10 ≥ defined baseline; latency for k-NN queries'
  nuances:
  - Tokenisation consistency between training and inference; memory layout of large
    indices.
code_focus:
- Train word2vec (CBOW/SGNS) with gensim; evaluate with analogy/similarity; explore
  subword fastText embeddings; load pre-trained vectors.
- Sentence embeddings with sentence-transformers (SBERT); semantic search and clustering;
  dimensionality reduction for visualisation.
- Build a retrieval baseline over your Week-33 corpus.
math_stats:
- Negative sampling as NCE approximation; PMI connections; subword modelling for OOV.
- Cosine distance vs Euclidean; hubness in high-dimensional spaces and its mitigation.
docs:
- https://radimrehurek.com/gensim/auto_examples/index.html
- https://sbert.net/
- https://fasttext.cc/
- https://huggingface.co/sentence-transformers
bibliography:
- Mikolov et al., “Efficient Estimation of Word Representations in Vector Space” (2013).
- Reimers & Gurevych, “Sentence-BERT” (2019).
---

## Summary

You will move from bag-of-words to meaning-aware representations and assemble your first semantic search system, a stepping stone to RAG.
## Code Focus

- Train word2vec (CBOW/SGNS) with gensim; evaluate with analogy/similarity; explore subword fastText embeddings; load pre-trained vectors.
- Sentence embeddings with sentence-transformers (SBERT); semantic search and clustering; dimensionality reduction for visualisation.
- Build a retrieval baseline over your Week-33 corpus.
## Math & Stats

- Negative sampling as NCE approximation; PMI connections; subword modelling for OOV.
- Cosine distance vs Euclidean; hubness in high-dimensional spaces and its mitigation.
## Docs

- https://radimrehurek.com/gensim/auto_examples/index.html
- https://sbert.net/
- https://fasttext.cc/
- https://huggingface.co/sentence-transformers
## Bibliography

- Mikolov et al., “Efficient Estimation of Word Representations in Vector Space” (2013).
- Reimers & Gurevych, “Sentence-BERT” (2019).
