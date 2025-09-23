---
number: 1
title: Environment, Reproducibility, and DataFrames
phase: Foundations
bundles:
- bundle_foundations
- bundle_sklearn_core
project:
  title: Reproducible EDA Starter
  dataset: 'Kaggle Titanic: Machine Learning from Disaster'
  dataset_links:
  - https://www.kaggle.com/c/titanic
  metrics:
  - Zero notebook state errors on rerun
  - Passing Pandera checks
  - Lint/format clean
  - Readable README with environment + run steps
  nuances:
  - Explain any imputation choices in README
  - Seed all stochastic steps
  - Compare pandas vs Polars runtimes on join/groupby (optional)
code_focus:
- 'Set up Python 3.11+, pyenv (optional), and VS Code; configure virtual environments
  (venv) and a project scaffold with pyproject.toml (build backend: setuptools).'
- 'Git discipline: initialise repo, feature branches, atomic commits, conventional
  commit messages, protected main branch.'
- 'Jupyter + Colab workflows; notebook hygiene: parameter cells, deterministic seeds,
  %load_ext autoreload.'
- 'pandas vs Polars: ingestion (CSV/Parquet), schema inspection, type casting, datetime
  parsing, joins/merges, groupby-agg, window ops, reshaping (melt/pivot).'
- 'NumPy essentials: ndarray creation, slicing, broadcasting, vectorised ops, linear
  algebra primitives (dot, svd, eigh).'
- 'Basic validation: column ranges and dtypes with Pandera; quick assertions in tests.'
- 'Project plumbing: formatter (black), linter (ruff), import sorter (isort), pre-commit
  hooks, Makefile or tasks.json.'
- 'EDA scaffolding: profile report template; quick-check visualisations (histogram,
  ECDF, violin, box, heatmap) using seaborn and Matplotlib OO API.'
math_stats:
- Sets, random variables, expectation/variance; law of large numbers (intuition) to
  justify train/validation splits.
- Vector/matrix notation; shapes; norms (L1, L2, Linf) and what they imply computationally.
- Sampling vs population; measurement error; missingness mechanisms (MCAR/MAR/MNAR)
  and practical implications for imputation.
docs:
- '[NumPy](https://numpy.org/doc/)'
- '[pandas](https://pandas.pydata.org/docs/)'
- '[Polars](https://docs.pola.rs/)'
- '[pytest (parametrisation)](https://docs.pytest.org/en/stable/how-to/parametrize.html)'
- '[Matplotlib gallery (OO)](https://matplotlib.org/stable/gallery/index.html)'
- '[seaborn tutorial](https://seaborn.pydata.org/tutorial.html)'
bibliography:
- '*Python Data Science Handbook (2e, online)* — Jake VanderPlas — (2022) — https://jakevdp.github.io/PythonDataScienceHandbook/'
- '*Fluent Python (2e)* — Luciano Ramalho — (2022) — https://www.oreilly.com/library/view/fluent-python-2nd/9781492056355/'
- '*MIT 18.06 Linear Algebra (selected lectures)* — Gilbert Strang — (2010) — https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/'
- '*An Introduction to Statistical Learning with Applications in Python* — James,
  Witten, Hastie, Tibshirani — (2023) — https://www.statlearning.com/'
---

## Summary

You establish the working habits of a professional: version control, deterministic environments, and fast, idiomatic DataFrame work. The goal is not pretty plots per se but a repeatable EDA skeleton you will reuse all year. You also begin a light-touch mathematical vocabulary, random variables, expectation, norms, so later regularisers and losses are less mysterious.
## Project Description

Build a clean EDA package: one notebook and one Python module. Ingest Titanic CSV, coerce schema, create summary tables and 6–8 core plots (hist/ECDF of age/fare; stacked bars for Pclass×Survived; heatmap of correlations; missingness heatmap). Add Pandera checks for key columns. Save figures to ./reports/figures. Commit with pre-commit hooks.
## Code Focus

- Set up Python 3.11+, pyenv (optional), and VS Code; configure virtual environments (venv) and a project scaffold with pyproject.toml (build backend: setuptools).
- Git discipline: initialise repo, feature branches, atomic commits, conventional commit messages, protected main branch.
- Jupyter + Colab workflows; notebook hygiene: parameter cells, deterministic seeds, %load_ext autoreload.
- pandas vs Polars: ingestion (CSV/Parquet), schema inspection, type casting, datetime parsing, joins/merges, groupby-agg, window ops, reshaping (melt/pivot).
- NumPy essentials: ndarray creation, slicing, broadcasting, vectorised ops, linear algebra primitives (dot, svd, eigh).
- Basic validation: column ranges and dtypes with Pandera; quick assertions in tests.
- Project plumbing: formatter (black), linter (ruff), import sorter (isort), pre-commit hooks, Makefile or tasks.json.
- EDA scaffolding: profile report template; quick-check visualisations (histogram, ECDF, violin, box, heatmap) using seaborn and Matplotlib OO API.
## Math & Stats

- Sets, random variables, expectation/variance; law of large numbers (intuition) to justify train/validation splits.
- Vector/matrix notation; shapes; norms (L1, L2, Linf) and what they imply computationally.
- Sampling vs population; measurement error; missingness mechanisms (MCAR/MAR/MNAR) and practical implications for imputation.
## Docs

- [NumPy](https://numpy.org/doc/)
- [pandas](https://pandas.pydata.org/docs/)
- [Polars](https://docs.pola.rs/)
- [pytest (parametrisation)](https://docs.pytest.org/en/stable/how-to/parametrize.html)
- [Matplotlib gallery (OO)](https://matplotlib.org/stable/gallery/index.html)
- [seaborn tutorial](https://seaborn.pydata.org/tutorial.html)
## Bibliography

- *Python Data Science Handbook (2e, online)* — Jake VanderPlas — (2022) — https://jakevdp.github.io/PythonDataScienceHandbook/
- *Fluent Python (2e)* — Luciano Ramalho — (2022) — https://www.oreilly.com/library/view/fluent-python-2nd/9781492056355/
- *MIT 18.06 Linear Algebra (selected lectures)* — Gilbert Strang — (2010) — https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/
- *An Introduction to Statistical Learning with Applications in Python* — James, Witten, Hastie, Tibshirani — (2023) — https://www.statlearning.com/
