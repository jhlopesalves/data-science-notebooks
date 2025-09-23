---
number: 39
title: 'Multilingual & Domain Adaptation: Corpora, Tokenisation, and X-Transfer'
phase: NLP & GenAI
bundles:
- bundle_nlp_foundations
project:
  title: Multilingual Text Classifier or QA
  dataset: XNLI (NLI task) or XQuAD (QA translations)
  description: Evaluate multilingual transfer with XLM-R (or similar). Train on English,
    test on two non-English languages; report per-language results and tokenisation
    diagnostics.
  dataset_links:
  - https://huggingface.co/datasets/xnli
  - https://huggingface.co/datasets/xquad
  metrics:
  - 'XNLI: accuracy ≥ 75% on at least two languages; XQuAD: F1 and EM comparable to
    English baseline minus ≤ 10 points'
  - Tokenisation OOV rate and subword length statistics reported
  nuances:
  - Script and whitespace differences (e.g., Thai); right-to-left rendering; normalisation
    before hashing.
code_focus:
- 'Datasets: process OSCAR/OPUS slices; build multilingual tokenisation pipelines;
  normalise UTF-8, NFC/NFKC.'
- Fine-tune or evaluate a multilingual model (e.g., XLM-R base) on XNLI or XQuAD;
  explore zero-shot transfer.
- 'Domain adaptation: continued pretraining (language-modeling) vs task-tuning; vocabulary
  drift checks.'
math_stats:
- Subword segmentation effects across languages; type-token growth; OOV handling.
- 'Cross-lingual evaluation design: macro-averaging across languages; stratified sampling.'
docs:
- https://oscar-project.github.io/documentation/
- https://opus.nlpl.eu/
- https://huggingface.co/datasets/xnli
- https://huggingface.co/datasets/xquad
- https://huggingface.co/docs/datasets/
bibliography:
- Koehn, “Statistical Machine Translation” — tokenisation and multilingual corpora.
- Eisenstein, “Introduction to NLP” (MIT Press, 2019) — multilingual chapters.
---

## Summary

You will operationalise multilingual modelling, with concrete data wrangling and fair evaluation across languages, not just English-only anecdotes.
## Code Focus

- Datasets: process OSCAR/OPUS slices; build multilingual tokenisation pipelines; normalise UTF-8, NFC/NFKC.
- Fine-tune or evaluate a multilingual model (e.g., XLM-R base) on XNLI or XQuAD; explore zero-shot transfer.
- Domain adaptation: continued pretraining (language-modeling) vs task-tuning; vocabulary drift checks.
## Math & Stats

- Subword segmentation effects across languages; type-token growth; OOV handling.
- Cross-lingual evaluation design: macro-averaging across languages; stratified sampling.
## Docs

- https://oscar-project.github.io/documentation/
- https://opus.nlpl.eu/
- https://huggingface.co/datasets/xnli
- https://huggingface.co/datasets/xquad
- https://huggingface.co/docs/datasets/
## Bibliography

- Koehn, “Statistical Machine Translation” — tokenisation and multilingual corpora.
- Eisenstein, “Introduction to NLP” (MIT Press, 2019) — multilingual chapters.
