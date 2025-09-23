---
number: 34
title: 'Classical NLP: TF-IDF, Linear Models, and CRFs for NER'
phase: NLP & GenAI
bundles:
- bundle_embeddings_search
- bundle_serving_llm
project:
  title: 'Two Baselines: News Classifier + CRF NER'
  dataset: AG News (doc classification) and CoNLL-2003 (NER) or OntoNotes-style substitute
  description: Ship a strong TF-IDF + linear baseline for AG News and a CRF-based
    NER with carefully engineered token features.
  dataset_links:
  - https://huggingface.co/datasets/ag_news
  - https://huggingface.co/datasets/conll2003
  metrics:
  - 'AG News: accuracy ≥ 92%; NER: entity-level F1 ≥ 0.85 on dev'
  - Error analysis with confusion tables; per-entity breakdown; failure modes documented
  nuances:
  - Feature leakage via token normalisation; BIO tag consistency; class imbalance
    handling.
code_focus:
- 'Feature extraction: TF-IDF with character and word n-grams; hashing trick; stopword
  and sublinear TF options.'
- 'Linear baselines: LogisticRegression/LinearSVC for document classification; calibration
  check.'
- 'Sequence labelling: sklearn-crfsuite for NER on CoNLL-style BIO tags; feature templates
  (prefix/suffix, capitalisation, word shape).'
math_stats:
- TF-IDF derivation; cosine similarity; margin-based losses.
- CRF objective (log-likelihood), Viterbi decoding; regularisation (L1/L2) and feature
  selection.
docs:
- https://scikit-learn.org/stable/user_guide.html
- https://sklearn-crfsuite.readthedocs.io/
- https://python-crfsuite.readthedocs.io/en/latest/
- https://huggingface.co/datasets/ag_news
- https://huggingface.co/datasets/conll2003
bibliography:
- Bird, Klein, Loper, “Natural Language Processing with Python” (O’Reilly) — classic
  vector-space chapters.
- Sutton & McCallum, “An Introduction to Conditional Random Fields” (foundational
  tutorial).
---

## Summary

You will establish competitive classical baselines that often rival naive deep models on modest data. This builds judgement about when deep models are justified.
## Code Focus

- Feature extraction: TF-IDF with character and word n-grams; hashing trick; stopword and sublinear TF options.
- Linear baselines: LogisticRegression/LinearSVC for document classification; calibration check.
- Sequence labelling: sklearn-crfsuite for NER on CoNLL-style BIO tags; feature templates (prefix/suffix, capitalisation, word shape).
## Math & Stats

- TF-IDF derivation; cosine similarity; margin-based losses.
- CRF objective (log-likelihood), Viterbi decoding; regularisation (L1/L2) and feature selection.
## Docs

- https://scikit-learn.org/stable/user_guide.html
- https://sklearn-crfsuite.readthedocs.io/
- https://python-crfsuite.readthedocs.io/en/latest/
- https://huggingface.co/datasets/ag_news
- https://huggingface.co/datasets/conll2003
## Bibliography

- Bird, Klein, Loper, “Natural Language Processing with Python” (O’Reilly) — classic vector-space chapters.
- Sutton & McCallum, “An Introduction to Conditional Random Fields” (foundational tutorial).
