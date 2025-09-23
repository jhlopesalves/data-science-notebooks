---
number: 7
title: Buffer & Utilities
phase: Foundations
bundles:
- bundle_foundations
project:
  title: Utilities Package
  dataset: California Housing, Adult (for quick visual tests)
  dataset_links:
  - https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html
  - https://archive.ics.uci.edu/dataset/2/adult
  metrics:
  - '>=85% line coverage on package code'
  - Docstring coverage reported
  - Example notebook demonstrating each utility
  nuances:
  - Use semantic versioning (0.y.z while unstable)
  - 'Keep plotting functions pure: accept axes, return axes; no global state'
code_focus:
- 'Refactor common transformers into a reusable package: e.g., DatePartExtractor,
  RareCategoryGrouper, OutlierClipper, Winsoriser, TargetEncoder wrapper (sklearn-compatible).'
- 'Visualisation utilities (Matplotlib OO + seaborn): make reusable functions for
  histogram/ECDF, violin+swarm, correlation heatmap, calibration curves, learning/validation
  curves, residual plots. Each function returns axes objects and accepts styling kwargs.'
- 'Packaging: create a src/ package layout with pyproject.toml (setuptools), include
  tests, type hints, and docstrings; build wheel locally; install in editable mode.
  Pre-commit hooks enforce black/ruff/isort and docstring checks; measure docstring
  coverage.'
math_stats:
- 'Review Weeks 1–6 identities and pitfalls: (i) leakage via fit outside CV; (ii)
  overfitting visual EDA; (iii) imbalance-induced misleading accuracy; (iv) variance
  of CV estimates; (v) when robust statistics beat means.'
- 'Practical review method: write short ‘bug stories’ reproducing each pitfall and
  the fix.'
docs:
- '[setuptools + pyproject config](https://setuptools.pypa.io/en/stable/userguide/pyproject_config.html)'
- '[coverage.py (devguide pointer)](https://devguide.python.org/testing/coverage/)'
- '[Matplotlib gallery](https://matplotlib.org/stable/gallery/index.html)'
- '[seaborn tutorial](https://seaborn.pydata.org/tutorial.html)'
bibliography:
- '*Python packaging 101 (pyOpenSci)* — pyOpenSci — (2024) — https://www.pyopensci.org/python-package-guide/tutorials/intro.html'
- '*Semantic Versioning 2.0.0* — semver.org — (2013) — https://semver.org/spec/v2.0.0.html'
- '*docstr-coverage (measure docstring coverage)* — PyPI — (2024) — https://pypi.org/project/docstr-coverage/'
---

## Summary

This breathing week reduces duplication by giving you a personal toolbox: composable transformers and plotting helpers you will drop into every project. You also rehearse packaging and documentation, which accelerates all future work.
## Project Description

Create a small Python package (src/yourutils/) with sklearn-compatible transformers and plotting utilities. Include doctests and pytest suites. Publish a local wheel and install via pip. Provide a README with usage snippets and a CHANGELOG using Keep a Changelog conventions.
## Code Focus

- Refactor common transformers into a reusable package: e.g., DatePartExtractor, RareCategoryGrouper, OutlierClipper, Winsoriser, TargetEncoder wrapper (sklearn-compatible).
- Visualisation utilities (Matplotlib OO + seaborn): make reusable functions for histogram/ECDF, violin+swarm, correlation heatmap, calibration curves, learning/validation curves, residual plots. Each function returns axes objects and accepts styling kwargs.
- Packaging: create a src/ package layout with pyproject.toml (setuptools), include tests, type hints, and docstrings; build wheel locally; install in editable mode. Pre-commit hooks enforce black/ruff/isort and docstring checks; measure docstring coverage.
## Math & Stats

- Review Weeks 1–6 identities and pitfalls: (i) leakage via fit outside CV; (ii) overfitting visual EDA; (iii) imbalance-induced misleading accuracy; (iv) variance of CV estimates; (v) when robust statistics beat means.
- Practical review method: write short ‘bug stories’ reproducing each pitfall and the fix.
## Docs

- [setuptools + pyproject config](https://setuptools.pypa.io/en/stable/userguide/pyproject_config.html)
- [coverage.py (devguide pointer)](https://devguide.python.org/testing/coverage/)
- [Matplotlib gallery](https://matplotlib.org/stable/gallery/index.html)
- [seaborn tutorial](https://seaborn.pydata.org/tutorial.html)
## Bibliography

- *Python packaging 101 (pyOpenSci)* — pyOpenSci — (2024) — https://www.pyopensci.org/python-package-guide/tutorials/intro.html
- *Semantic Versioning 2.0.0* — semver.org — (2013) — https://semver.org/spec/v2.0.0.html
- *docstr-coverage (measure docstring coverage)* — PyPI — (2024) — https://pypi.org/project/docstr-coverage/
