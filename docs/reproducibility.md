# Reproducibility

This repository is a heterogeneous learning archive, not a single application.
It does not claim that every notebook executes from top to bottom in one
environment or that historical outputs can be reproduced exactly.

## Observed environments

Notebook metadata records:

| Python version | Notebook count |
| --- | ---: |
| 3.13.7 | 36 |
| 3.12.7 | 27 |
| 3.12.11 | 9 |
| 3.12.0 | 4 |
| 3.12.8 | 3 |
| Not recorded | 5 |

Kernel display names are mainly datacamp (69 notebooks), DS (9), and Python 3
(3). These names refer to historical local environments and are not dependency
specifications.

There is no current requirements file, lockfile or pyproject.toml. A single
generated dependency list would be misleading because the notebooks span
different periods and optional stacks.

## Common and specialised dependencies

The common scientific stack is:

- NumPy;
- pandas;
- Matplotlib;
- Seaborn;
- SciPy;
- scikit-learn.

Material in this archive also uses statsmodels, pingouin, NLTK, spaCy,
ucimlrepo, TensorFlow, imbalanced-learn, Polars, yfinance, geopandas,
contextily, missingno, recordlinkage, thefuzz, python-dateutil and joblib.
Notebook-local modules are stored beside several analyses.

Suggested starting point for a selected notebook:

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install jupyterlab numpy pandas matplotlib seaborn scipy scikit-learn

Install additional packages only after inspecting that notebook’s imports. No
versions are pinned here because the current tree does not supply evidence for
reliable pins. For historical behaviour, the Python version recorded in the
notebook metadata is a clue, not a guarantee.

## Working directory and data paths

Most notebooks assume that the process starts in the notebook’s directory.
Relative paths such as data/example.csv are resolved from there. Starting
Jupyter at the repository root may require changing the working directory
before running a notebook.

Data access falls into four patterns:

1. tracked files in a nearby data or datasets directory;
2. raw GitHub URLs pointing to files in this repository;
3. external datasets fetched through OpenML, ucimlrepo, yfinance or another
   remote service;
4. external-platform setup cells that clone or download files in Colab.

Resolvable repository URLs were updated to current tracked paths during
curation. This does not remove the network dependency. Where a tracked local
copy exists, a future project-specific cleanup may prefer local relative paths,
but this task does not rewrite established notebook workflows.

## Selected analyses

| Notebook | Main dependencies | Data | Expected status | Lightweight validation boundary |
| --- | --- | --- | --- | --- |
| Ames Housing | pandas, NumPy, Matplotlib, Seaborn, scikit-learn, joblib | OpenML/network plus notebook source references | Partial | Imports, source URL/path checks and pipeline construction; do not run full tuning |
| Online Retail II | pandas, NumPy, Matplotlib, Seaborn, scikit-learn | Tracked 43.51 MiB Excel workbook | Partial but self-contained | Confirm workbook path and imports; skip full clustering/plot regeneration |
| Telco Customer Churn | pandas, plotting stack, scikit-learn, optional TensorFlow, local data.plot_cm_comparison | Tracked CSV and local helper | Partial | Validate helper import/path and scikit-learn setup; skip neural-network training |
| Vehicle Price Modelling | pandas, plotting stack, scikit-learn, local utils.py, joblib | Tracked Kaggle CSV | Partial but comparatively self-contained | Import custom transformers and validate data path; skip model search/training |
| Spotify Songs | pandas, NumPy, plotting stack, SciPy, scikit-learn | Remote TidyTuesday-derived data access | Partial | Validate source and imports; do not refit all models |

The Ames and vehicle notebooks still contain cells that generate saved model
files. The generated joblib/pickle files were removed from the working tree;
rerun the relevant training and save cells if those artefacts are needed.

## Guided projects

Comparatively lightweight guided projects with tracked data include customer
analytics, Airbnb trends, football hypothesis testing, agriculture, car
insurance, crime in Los Angeles and Titanic. They are likely to run after
installing their direct imports, but their external-platform assumptions and
historical library versions have not been reconstructed.

California Housing has network, archive extraction and geospatial dependencies.
MAGIC Gamma uses imbalanced-learn and TensorFlow. The Nobel notebook and several
DataCamp notes use only the common stack but may depend on historical output or
course conventions.

## Known missing or fragile components

| Path | Dependency/problem | Consequence |
| --- | --- | --- |
| Python/nlp/nlp_in_python/nlp.ipynb | spaCy model en_core_web_lg is not installed by the base spaCy package | Model-loading cell fails until the model is installed |
| Python/statistics/exploratory_data_analysis/eda_modules/module_2.ipynb | imports helpers.olist_analyzer, absent from current tree | Olist helper cells cannot run unchanged |
| archive/study_sessions/data_reboot_airbnb_prices/airbnb.ipynb | requires nbresult and external challenge conventions | Tests/result cells are not portable without the original environment |
| Python/visualisation/seaborn/intro_to_seaborn/introduction_to_seaborn.ipynb | air_quality_no2_long.csv URL returned 404 on 2026-07-15 and no local copy exists | The related lesson cell cannot run unchanged |
| Python/visualisation/seaborn/intermediate/intermediate_seaborn.ipynb | contains example filenames such as wines.csv and data.csv not supplied locally | Some examples are illustrative or require external setup |
| Python/projects/bike_sharing/bike.ipynb | retained source includes an unfinished `y_train_temp` assignment and a feature-shape mismatch that previously produced an error | Full sequential execution needs analytical review, not an automatic fix |
| Python/machine_learning/supervised_learning/decision_trees/module_3/exercises_3-0.ipynb | previous run was interrupted | Runtime/cost and completion are uncertain |
| Python/projects/heart_disease/heart_disease.ipynb | only three code cells and no narrative; dynamically fetches a helper | Runnable pieces may work, but the project is incomplete |

The old heart-disease helper URL was updated to the tracked
eda_modules/helpers/missing.py implementation, which exposes the same
missing_data_summary function used by the notebook.

## Helpers and imports

Local imports with corresponding current files include:

- data.florida_hurricane_data for dates and times;
- data.* arrays for unsupervised learning;
- helper.plotting, helper.report, helper.selection and helper.stability for
  unsupervised_learning/unsupervised_2.ipynb;
- data.plot_cm_comparison for Telco churn;
- utils for the vehicle project;
- EDA module helpers other than helpers.olist_analyzer.

Imports from Python/helpers may require launching from the Python directory or
adding that directory to PYTHONPATH. This implicit assumption is not normalised
globally because changing import structure would require wider testing.

## Outputs and execution counts

Analytical plots, tables and model summaries are generally retained. During
curation:

- 25 output objects containing errors, local user paths or rendered email
  fields were removed;
- execution counts were reset in 22 affected cells;
- redundant Colab/Data Wrangler MIME payloads and editor metadata were removed;
- non-sequential counts elsewhere were preserved as historical state;
- notebooks were not globally re-executed.

Retained output is evidence of an earlier run, not proof that the same code
will reproduce the same values today. Network data, library defaults, random
state, locale and platform differences may all affect results.

## Validation scope

Repository validation covers JSON parsing, available notebook schema tools,
Python compilation, imports/path checks for selected projects, relative
Markdown links, current-repository URLs, large/generated file inventories,
tracked-ignore state and a redacted current-tree secret scan.

It deliberately excludes long model training, large downloads, global notebook
execution, reconstruction of external course platforms and exact numerical
comparison with historical outputs. Final commands and results are recorded in
[the curation report](curation_report.md).
