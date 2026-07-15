# Content index

This index is a navigational layer over the archive, not a quality ranking or
claim of authorship. Paths and source notes reflect the current tree and the
evidence available during curation.

Reproducibility labels:

- Likely: tracked data and local code appear sufficient for a lightweight run,
  subject to installing the named libraries.
- Partial: network access, uncommon dependencies, local helpers, large data or
  historical-version differences are expected.
- Unlikely unchanged: a required component is missing, the notebook is
  interrupted, or external platform infrastructure is assumed.
- Not assessed: execution is not the purpose of the item or evidence is too
  limited.

“Outputs retained” means useful output remains visible. Narrow removal of
private paths, error traces, rendered contact fields or redundant editor data
does not change that label.

## Selected analyses

These notebooks are prominent because they are comparatively substantial or
self-contained. “Selected” does not mean polished, production-ready or proven
independent.

| Title | Path | Category | Topic | Status | Provenance | Expected reproducibility | Outputs retained | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Ames Housing | [notebook](../Python/projects/ames_house/ames_house.ipynb) | Selected modelling project | Regression | Substantial | De Cock paper and OpenML links; Kaggle references | Partial | Yes | Regenerable joblib output was removed; the generation cell remains |
| Online Retail II | [notebook](../Python/projects/online_retail_ii/online_retail_ii.ipynb) | Selected exploratory project | Customer segmentation | Substantial | UCI DOI 10.24432/C5CG6D | Partial | Yes | Requires a 43.51 MiB workbook; plots and profiles are part of the result |
| Telco Customer Churn | [notebook](../Python/projects/telco_customer_churn/telco_customer_churn.ipynb) | Selected modelling project | Classification | Substantial | Dataset source is not adequately recorded | Partial | Yes | Includes local plotting helper and optional/heavy TensorFlow use |
| Vehicle Price Modelling | [notebook](../Python/projects/vehicles/vehicles.ipynb) | Selected modelling project | Regression and text features | Substantial | Kaggle CarDekho source stated | Partial | Yes | Regenerable large model output was removed; notebook results remain visible |
| Spotify Songs | [notebook](../Python/projects/spotify/spotify_dataset.ipynb) | Selected exploratory project | Regression and feature analysis | Substantial/mixed | TidyTuesday 2020-01-21 source stated | Partial | Yes | Later cells include study material as well as the main analysis |

Independent-project status has not been assigned because the current tree does
not consistently distinguish supplied scaffold, guided instructions and
personal additions. That classification remains a human decision.

## Guided projects

| Title | Path | Category | Topic | Status | Provenance | Expected reproducibility | Outputs retained | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Crime in Los Angeles | [notebook](../Python/projects/analyzing_crime_in_los_angeles/crime_los_angeles.ipynb) | Guided project | EDA | Substantial exercise | Supplied prompt; DataCamp evidence | Likely | Yes | Large tracked crimes dataset |
| Customer Analytics Data Preparation | [notebook](../Python/projects/customer_analytics_preparing_data_for_modeling/customer_analytics.ipynb) | Guided project | Data types and memory | Complete exercise | Supplied project instructions; DataCamp evidence | Likely | Yes | Generic notebook name corrected during curation |
| Exploring Airbnb Market Trends | [notebook](../Python/projects/exploring_airbnb_market_trends/airbnb_market_trends.ipynb) | Guided project | Data manipulation | Complete exercise | Supplied prompt; DataCamp evidence | Likely | Yes | Uses CSV, TSV and Excel inputs |
| Nobel Prize Winners | [notebook](../Python/projects/history_of_nobel_prize_winners/nobel_winners.ipynb) | Guided project with extensions | EDA | Substantial | DataCamp/supplied-project evidence | Likely | Yes | Obvious filename spelling error corrected during curation |
| Football Match Hypothesis Test | [notebook](../Python/projects/hypothesis_test_football_matches/hypothesis_football.ipynb) | Guided project | Hypothesis testing | Complete exercise | Supplied prompt; DataCamp evidence | Likely | Yes | Men’s and women’s match data are tracked and URLs are current |
| MAGIC Gamma Classification | [notebook](../Python/projects/magic_gamma/magic.ipynb) | Guided/exploratory project | Classification | Substantial | UCI; old fcc path suggests curriculum origin | Partial | Yes | TensorFlow and imbalanced-learn increase setup cost |
| Sowing Success | [notebook](../Python/projects/modeling_agriculture/agriculture.ipynb) | Guided project | Feature selection | Complete exercise | Supplied prompt; DataCamp evidence | Likely | Yes | Small tracked dataset |
| Car Insurance Claim Outcomes | [notebook](../Python/projects/modeling_car_insurance_claim_outcomes/car_insurance.ipynb) | Guided project | Logistic regression | Complete exercise | Supplied prompt; DataCamp evidence | Likely | Yes | Repository URL is current; source details remain incomplete |
| Titanic | [notebook](../Python/projects/titanic/titanic.ipynb) | Competition/guided project | Classification | Substantial | Kaggle explicitly stated | Likely | Yes | Submission CSV is retained as a small documentary output |
| California Housing Prices | [notebook](../Python/projects/california_house_pricing/california.ipynb) | Guided/textbook-derived project | Regression | Substantial | Downloads from ageron/data; exact source not stated | Partial | Yes | Network and geospatial dependencies |

## Exploratory, partial and unresolved projects

| Title | Path | Category | Topic | Status | Provenance | Expected reproducibility | Outputs retained | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Seoul Bike Sharing | [notebook](../Python/projects/bike_sharing/bike.ipynb) | Exploratory/guided project | Regression | Partial | UCI and Seoul sources stated | Partial | Yes | Contains a stale model-shape error; unrelated 98.34 MiB file needs review |
| Book Recommender Data Exploration | [notebook](../Python/projects/book_recommender/data_exploration.ipynb) | Unresolved project | EDA/NLP | Partial; no markdown | Dataset link present | Partial | Yes | Generated Chroma index is disposable; derived data is preserved |
| Eurovision 2016–2025 | [notebook](../Python/projects/eurovision_2016-25/eurovision.ipynb) | Scratch project | EDA | Incomplete | Source not documented in markdown | Partial | Yes | Four cells; date-range naming needs review |
| Heart Disease | [notebook](../Python/projects/heart_disease/heart_disease.ipynb) | Scratch project | Classification | Incomplete | UCI; helper provenance unclear | Partial | Yes | Helper URL now points to the tracked same-function implementation; notebook has three code cells and no markdown |
| Student Completion | [notebook](../Python/projects/student_completion/student_completion.ipynb) | Scratch project | EDA | Incomplete | Source not documented | Partial | Yes | No markdown; 21.29 MiB dataset |
| Wisconsin Breast Cancer Plots | [notebook](../Python/projects/wisconsin_breast_cancer_plots/wbc_plots.ipynb) | Visual experiment | Visualisation | Partial/template-like | scikit-learn dataset; prompt origin unclear | Likely | Yes | Obvious directory spelling error corrected during curation |
| Women’s E-Commerce Reviews | [notebook](../Python/projects/womens_ecommerce/womens.ipynb) | Exploratory project | EDA | Partial | External dataset; exact source not stated | Likely | Yes | Large dataset retained pending provenance review |

## Notes and course material

### Data manipulation

| Collection | Path | Category | Topic | Status | Provenance | Expected reproducibility | Outputs retained | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Cleaning Data | [notebook](../Python/data_manipulation/pandas/cleaning_data/cleaning_data.ipynb) | Course notes/exercises | Cleaning and validation | Substantial | DataCamp evidence | Likely | Yes | Repository URLs are current; local-path warning output was removed |
| Data Manipulation with Pandas | [notebook](../Python/data_manipulation/pandas/data_manipulation_with_pandas/data_manipulation_pandas.ipynb) | Course notes/exercises | Pandas foundations | Substantial | DataCamp | Likely | Yes | Colab badge and download links point to current paths |
| Dates and Times | [notebook](../Python/data_manipulation/pandas/dates_and_times/dates_and_times.ipynb) | Course notes/exercises | Datetime | Substantial | DataCamp evidence | Likely | Yes | Local Python data module is tracked |
| Joining Data with Pandas | [notebook](../Python/data_manipulation/pandas/joining_data_with_pandas/joining_data_pandas.ipynb) | Course notes/exercises | Joins and SQL | Substantial | DataCamp | Partial | Yes | Large supplied pickle collection and Chinook database |
| Categorical Data | [notebook](../Python/data_manipulation/pandas/working_with_categorical_data/categorical_data.ipynb) | Course notes/exercises | Categorical data | Substantial | DataCamp evidence | Partial | Yes | One retained error is scheduled for removal |
| Preprocessing for ML | [notebook](../Python/data_manipulation/preprocessing_for_machine_learning/pre_process_ml.ipynb) | Course notes/exercises | Preprocessing | Substantial | DataCamp evidence | Partial | Yes | Broad dependency set |

### Machine learning

| Collection | Path | Category | Topic | Status | Provenance | Expected reproducibility | Outputs retained | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Intro ML exercises | [directory](../Python/machine_learning/intro_ml_exercises/) | Exercises | ML foundations | Substantial | Source not stated | Likely | Yes | Module sequence retained |
| Intro to ML | [directory](../Python/machine_learning/intro_to_machine_learning/) | Lecture/conceptual notes | Probability, evaluation and linear regression | Substantial | Source not stated | Likely | Yes | Four-module sequence |
| Supervised learning | [notebook](../Python/machine_learning/supervised_learning/supervised_learning.ipynb) | Course notes/exercises | Classification and regression | Substantial | DataCamp evidence | Partial | Yes | Large music dataset; repository URLs are current |
| Decision-tree lectures and exercises | [directory](../Python/machine_learning/supervised_learning/decision_trees/) | Lectures/exercises | Trees and ensembles | Mixed; one near-empty and one interrupted | UCI datasets; task prompts; exact course source unclear | Partial | Yes | Sequence names preserved for navigation |
| Unsupervised learning, part 1 | [notebook](../Python/machine_learning/unsupervised_learning/unsupervised_1.ipynb) | Course notes/exercises | Clustering and dimension reduction | Substantial | DataCamp/UCI evidence | Partial | Yes | Local data modules and current repository URLs |
| Unsupervised learning, part 2 | [notebook](../Python/machine_learning/unsupervised_learning/unsupervised_2.ipynb) | Course notes/exercises | Cluster evaluation | Substantial | UCI and local helpers | Partial | Yes | Largest notebook output collection |
| K-Means primer | [notebook](../Python/machine_learning/unsupervised_learning/unsupervised_clustering.ipynb) | Lecture/conceptual notes | K-means | Substantial | Source boundary unclear | Likely | Yes | Raw-data path updated to the tracked dataset |

### Mathematics and NLP

| Collection | Path | Category | Topic | Status | Provenance | Expected reproducibility | Outputs retained | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Harmonograph | [notebook](../Python/mathematics/harmonograph.ipynb) | Mathematical experiment | Parametric visualisation | Complete experiment | Source not stated | Likely | Yes | Large plot is the main result |
| Mathematical foundations | [directory](../Python/mathematics/math_for_ds/) | Mathematics notes | Foundations | Substantial | Source not stated | Likely | Yes | Ordered modules retained |
| Mathematical reading | [directory](../Python/mathematics/reading_math/) | Mathematics notes | Notation and variance | Substantial | Source not stated | Likely | Yes | Provenance review |
| Mathematics study session | [directory](../Python/mathematics/study_sessions/) | Study session | Calculus | Small/coherent | Source not stated | Likely | Yes | Possible future archive move |
| NLP in Python | [notebook](../Python/nlp/nlp_in_python/nlp.ipynb) | Course notes/exercises | NLP | Partial | DataCamp evidence | Unlikely unchanged | Yes | Requires en_core_web_lg; historical error output was removed |
| NLP with spaCy | [notebook](../Python/nlp/nlp_with_spacy/nlp_spacy.ipynb) | Lecture note | spaCy | Near-empty | DataCamp evidence | Not assessed | No | Preserved for human review |

### Statistics and visualisation

| Collection | Path | Category | Topic | Status | Provenance | Expected reproducibility | Outputs retained | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Experimental Design | [notebook](../Python/statistics/experimental_design/experimental.ipynb) | Course notes/exercises | Experimental design | Substantial | DataCamp evidence | Partial | Yes | Multiple supplied datasets |
| Exploratory Data Analysis | [notebook](../Python/statistics/exploratory_data_analysis/datacamp/exploratory_data_analysis.ipynb) | Course notes/exercises | EDA | Substantial | DataCamp explicit | Partial | Yes | Sparse-checkout path and clone URL now match the current tree |
| EDA modules | [directory](../Python/statistics/exploratory_data_analysis/eda_modules/) | Lecture/conceptual notes | Python, graphics and EDA | Mixed | Source not stated; Olist data | Partial | Yes | Module 2 references missing olist_analyzer |
| Hypothesis Testing | [notebook](../Python/statistics/hypothesis_testing/hypothesis.ipynb) | Course notes/exercises | Inference | Substantial | DataCamp evidence | Likely | Yes | Attachments retained |
| Regression with Statsmodels | [directory](../Python/statistics/regression_with_statsmodels/) | Course notes | Regression | Part 1 substantial; part 2 near-empty | DataCamp evidence | Partial | Yes | Generic sequence names preserved |
| Sampling | [notebook](../Python/statistics/sampling/sampling.ipynb) | Course notes/exercises | Sampling | Substantial | DataCamp evidence | Partial | Yes | Feather and CSV data |
| Introduction to Matplotlib | [notebook](../Python/visualisation/matplotlib/intro_to_matplotlib/introduction_to_matplotlib.ipynb) | Course notes/exercises | Matplotlib | Substantial | DataCamp | Likely | Yes | Filename, badge and download links were corrected |
| Intermediate Seaborn | [notebook](../Python/visualisation/seaborn/intermediate/intermediate_seaborn.ipynb) | Course notes/exercises | Seaborn | Substantial | DataCamp evidence | Partial | Yes | Some example filenames are not local datasets |
| Introduction to Seaborn | [notebook](../Python/visualisation/seaborn/intro_to_seaborn/introduction_to_seaborn.ipynb) | Course notes/exercises | Seaborn | Substantial | DataCamp | Partial | Yes | Missing air-quality dataset remains unresolved |

## Helpers

| Path | Category | Topic | Status | Provenance | Expected reproducibility | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| [Python/helpers](../Python/helpers/) | Reusable helpers | Effect sizes, association, encoding, missingness and plotting | Mixed but parseable | Source not stated | Partial | Some notebooks assume Python is on sys.path |
| [Unsupervised helper](../Python/machine_learning/unsupervised_learning/helper/) | Notebook-local helpers | Cluster plots, reports, selection and stability | Used locally | Source not stated | Likely with scientific stack | Kept beside dependent notebooks |
| [EDA module helpers](../Python/statistics/exploratory_data_analysis/eda_modules/helpers/) | Notebook-local helpers | Categorical, missing and text analysis | Used locally | Source not stated | Partial | Referenced olist_analyzer is absent |
| [Vehicle utilities](../Python/projects/vehicles/utils.py) | Project helper | Custom transformers | Used by selected project | Personal/source boundary not stated | Likely | Required to load or rerun the vehicle pipeline |
| [Telco plotting utility](../Python/projects/telco_customer_churn/data/plot_cm_comparison.py) | Project helper | Confusion-matrix comparison | Used by selected project | Source not stated | Likely | Stored in data directory because notebook imports data.* |

## Study sessions and historical practice

These files are retained under archive/study_sessions. Their internal sequence
names are preserved where a more specific title would be speculative.

| Collection | Path | Category | Topic | Status | Provenance | Expected reproducibility | Outputs retained | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Machine Learning Reboot | [directory](../archive/study_sessions/data_reboot_airbnb_prices/) | Guided study session | ML workflow | Substantial | External challenge and nbresult | Unlikely unchanged | Yes | Tests and tiny pickle fixtures require review |
| Iris worksheet | [notebook](../archive/study_sessions/iris_sklearn/iris.ipynb) | Study exercise | Welch t-test | Coherent | scikit-learn Iris; worksheet source unclear | Likely | Yes | Archived from the former projects subtree |
| OOP refresher | [notebook](../archive/study_sessions/oop_refresher/oop-refresher.ipynb) | Study exercise | Python OOP | Near-empty | Source not stated | Likely | No | Preserved as historical practice |
| Path handling | [notebook](../archive/study_sessions/paths/paths.ipynb) | Study exercise | pathlib | Coherent | Source not stated | Likely | Yes | Uses a repository-relative path and resolves it at runtime |
| Random notebooks | [directory](../archive/study_sessions/random_notebooks/) | Scratch exercises | Heart disease and inference | Mixed | UCI for heart disease; otherwise unclear | Mixed | Yes | Ambiguous names preserved |

## Data and generated artefacts

Data remains beside dependent notebooks rather than being centralised. The
repository tracks many course datasets and external dataset snapshots. Their
presence does not imply original authorship or confirmed redistribution
permission.

Generated Chroma index files and saved models are documented separately from
source datasets. The [provenance document](provenance.md) records known sources,
the [reproducibility document](reproducibility.md) records data access and
environment caveats, and [human review](human_review.md) lists files whose
retention or redistribution needs a decision.
