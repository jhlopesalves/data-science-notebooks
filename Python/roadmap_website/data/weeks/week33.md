---
number: 33
title: 'NLP Foundations: Tokenisation, Tagging, Parsing'
phase: NLP & GenAI
bundles:
- bundle_embeddings_search
- bundle_serving_llm
project:
  title: Linguistic Pipeline & Corpus Explorer
  dataset: English Wikipedia subset or WikiText-103; optional UD treebanks for evaluation
  description: Build a robust text-preprocessing pipeline that outputs token, POS,
    lemma, entities, and dependencies. Provide frequency plots, collocations, and
    a small evaluation on a held-out labelled slice.
  dataset_links:
  - https://huggingface.co/datasets/wikipedia
  - https://huggingface.co/datasets/wikitext
  metrics:
  - Pipeline throughput (docs/sec), memory footprint
  - Tagger/NER spot-checks vs gold slices; coverage and OOV analysis
  nuances:
  - Unicode normalisation; sentence segmentation errors; domain drift when moving
    beyond encyclopaedic text.
code_focus:
- 'spaCy/Stanza pipelines: tokeniser, POS, lemma, dependency parse, NER; customise
  rules for domains.'
- Corpus handling with Hugging Face Datasets; efficient text cleaning; sentence segmentation;
  document stores.
- 'Comparative tokenisation: whitespace, WordPiece/BPE (SentencePiece), byte-level
  BPE for later LLM work.'
math_stats:
- n-gram models and sparsity; Zipf’s law and Heaps’ law; evaluation with perplexity
  (classical models).
- 'Sequence labelling as structured prediction: Markov assumptions, CRF intuition
  (will use later).'
docs:
- https://spacy.io/usage
- https://stanfordnlp.github.io/stanza/
- https://huggingface.co/docs/datasets/
bibliography:
- Manning & Schütze, “Foundations of Statistical NLP” (MIT Press, 1999) — n-grams,
  tagging, parsing.
- Jurafsky & Martin, “Speech and Language Processing” (3e draft) — tokenisation, POS,
  parsing.
---

## Summary

You will ground NLP in concrete pipelines, not magic. By the end of the week you will be able to ingest, normalise, and linguistically annotate large corpora reproducibly, with awareness of tokenisation trade-offs that later affect model performance.
## Code Focus

- spaCy/Stanza pipelines: tokeniser, POS, lemma, dependency parse, NER; customise rules for domains.
- Corpus handling with Hugging Face Datasets; efficient text cleaning; sentence segmentation; document stores.
- Comparative tokenisation: whitespace, WordPiece/BPE (SentencePiece), byte-level BPE for later LLM work.
## Math & Stats

- n-gram models and sparsity; Zipf’s law and Heaps’ law; evaluation with perplexity (classical models).
- Sequence labelling as structured prediction: Markov assumptions, CRF intuition (will use later).
## Docs

- https://spacy.io/usage
- https://stanfordnlp.github.io/stanza/
- https://huggingface.co/docs/datasets/
## Bibliography

- Manning & Schütze, “Foundations of Statistical NLP” (MIT Press, 1999) — n-grams, tagging, parsing.
- Jurafsky & Martin, “Speech and Language Processing” (3e draft) — tokenisation, POS, parsing.
