# Repository audit

This document records the repository before curation. The snapshot was taken on
2026-07-15 from commit 7704069388218f6b0a6a90bb85033bcdc61a0c0d
(7704069, “concluded vehicles dataset”) after creating
chore/repository-curation. No history was rewritten.

The working tree already contained an unstaged edit to .gitignore. That edit is
directly related to this task and is deliberately excluded from this audit
commit. Counts below use the current tracked tree; discussion of the committed
.gitignore refers to the version at the recorded HEAD unless stated otherwise.

Methods used for this audit include git ls-files, working-tree byte counts,
extension and binary-signature heuristics, Git history and object inspection,
JSON parsing of every notebook, source/import/path extraction, output and
metadata inspection, exact-content hashing, and redacted credential-pattern
searches. “Dataset candidate” is a deliberately broad heuristic: a tracked file
with a common data extension or stored below a data directory. It is not a
claim about licensing or content.

## 1. Repository overview

### Top-level structure

| Path | Tracked files | Apparent role |
| --- | ---: | --- |
| .gitignore | 1 | Ignore policy |
| Python/data_manipulation | 90 | Pandas and preprocessing study material |
| Python/helpers | 6 | Reusable statistical and plotting helpers |
| Python/machine_learning | 43 | Supervised, unsupervised, and introductory ML material |
| Python/mathematics | 7 | Mathematical notes and experiments |
| Python/nlp | 3 | NLP notes |
| Python/projects | 96 | Projects mixed with guided work and study sessions |
| Python/statistics | 43 | Experimental design, EDA, inference, regression, and sampling |
| Python/visualisation | 14 | Matplotlib and Seaborn course notes |

There is no tracked root README, docs directory, dependency declaration, lock
file, environment file, or editor configuration. All substantive current
content is below the uppercase Python directory.

### Counts and size

| Measure | Pre-curation value |
| --- | ---: |
| Tracked files | 303 |
| Tracked working-tree size | 552,077,732 bytes (526.50 MiB) |
| Jupyter notebooks | 84 |
| Python files | 37 |
| Dataset candidates | 191 |
| Binary files (signature/extension heuristic) | 44 |
| Files larger than 1 MiB | 54 |
| Files larger than 5 MiB | 16 |
| Files larger than 25 MiB | 5 |
| Git commits | 308 |
| Packed Git object size | 279.80 MiB |

### Files by extension

| Extension | Count | Extension | Count |
| --- | ---: | --- | ---: |
| .csv | 123 | .ipynb | 84 |
| .py | 37 | .p | 19 |
| .txt | 11 | .jpg | 7 |
| .bin | 4 | .png | 3 |
| .pickle | 3 | .pkl | 2 |
| .tsv | 2 | .xlsx | 2 |
| no extension | 1 | .db | 1 |
| .joblib | 1 | .tgz | 1 |
| .data | 1 | .feather | 1 |

The 44 binary files comprise 19 .p files, seven JPEGs, four Chroma .bin
files, three PNGs, three .pickle files, two .pkl files, two Excel workbooks,
and one each of .db, .joblib, .tgz, and .feather.

### Largest tracked files

| MiB | Path |
| ---: | --- |
| 98.34 | Python/projects/bike_sharing/data/Most-Recent-Cohorts-Institution.csv |
| 58.44 | Python/statistics/exploratory_data_analysis/eda_modules/data/olist/olist_geolocation_dataset.csv |
| 50.86 | Python/projects/vehicles/vehicle_price_model_v1.pkl |
| 43.51 | Python/projects/online_retail_ii/data/online_retail_II.xlsx |
| 26.21 | Python/projects/analyzing_crime_in_los_angeles/data/crimes.csv |
| 21.29 | Python/projects/student_completion/data/Course_Completion_Prediction.csv |
| 19.19 | Python/machine_learning/supervised_learning/data/music.csv |
| 16.84 | Python/statistics/exploratory_data_analysis/eda_modules/data/olist/olist_orders_dataset.csv |
| 14.72 | Python/statistics/exploratory_data_analysis/eda_modules/data/olist/olist_order_items_dataset.csv |
| 13.68 | Python/statistics/exploratory_data_analysis/eda_modules/data/olist/olist_order_reviews_dataset.csv |
| 8.62 | Python/statistics/exploratory_data_analysis/eda_modules/data/olist/olist_customers_dataset.csv |
| 8.38 | Python/data_manipulation/pandas/joining_data_with_pandas/data/casts.p |
| 8.09 | Python/projects/womens_ecommerce/data/Womens Clothing E-Commerce Reviews.csv |
| 7.19 | Python/data_manipulation/pandas/working_with_categorical_data/data/cars.csv |
| 6.09 | Python/projects/book_recommender/data/books_cleaned.csv |
| 5.51 | Python/statistics/exploratory_data_analysis/eda_modules/data/olist/olist_order_payments_dataset.csv |

### Subject areas, sources, and repository condition

The current tree covers data manipulation, visualisation, statistics,
mathematics, supervised and unsupervised machine learning, NLP, and a range of
small or medium analyses. Evidence in paths, notebook prose, links, and history
shows extensive DataCamp-derived material; UCI datasets and ucimlrepo usage;
Kaggle datasets or competitions; TidyTuesday; a Machine Learning Reboot
challenge; OpenML; and guided project prompts. California housing material
downloads from the ageron/data repository and therefore appears
textbook-guided, but the exact source boundaries are not stated.

Evidence of generated or local state includes a 50.86 MiB trained vehicle
model, an Ames joblib model, a Chroma vector index, generated project CSVs,
test pickle fixtures, a downloaded housing archive, a Titanic submission, and
large embedded notebook outputs. The notebooks occupy 61.57 MiB, of which an
estimated 54.28 MiB is output JSON.

Evidence of bloat is material: five current files exceed 25 MiB, seventeen
notebooks contain more than 1 MiB of output JSON, and history retains multiple
large Chroma SQLite versions plus removed datasets and third-party book
material. Local-machine-specific state appears in notebook outputs as Windows
user-directory paths, in one source cell as an absolute home path, and in
Colab, VS Code, scheduling, and Data Wrangler metadata.

## 2. Content classification

Confidence means confidence in the classification, not confidence in
authorship. “Retain” means retain the analytical output unless a narrower
privacy, error, or transient-metadata cleanup is listed. Proposed paths are
only supplied where the benefit and intended name are clear.

### Notebook register

| Current path | Inferred title | Category | Topic | Apparent provenance | Confidence | Completeness | Output status | Recommended action | Proposed path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Python/data_manipulation/pandas/cleaning_data/cleaning_data.ipynb | Data Type Constraints | Course notes/exercises | Data cleaning | DataCamp evidence and asset links | High | Substantial | 270 KiB; non-sequential; local-path warning | Retain; clean transient/private output; update repository URLs | unchanged |
| Python/data_manipulation/pandas/data_manipulation_with_pandas/data_manipulation_pandas.ipynb | Introduction to Data Manipulation with Pandas | Course notes/exercises | Pandas | DataCamp and old Datacamp-Notebooks links | High | Substantial | 542 KiB; Colab metadata | Retain; clean metadata; repair links | unchanged |
| Python/data_manipulation/pandas/dates_and_times/dates_and_times.ipynb | Working with Dates in Python | Course notes/exercises | Dates and times | DataCamp evidence | High | Substantial | 166 KiB | Retain | unchanged |
| Python/data_manipulation/pandas/joining_data_with_pandas/joining_data_pandas.ipynb | Merging Data with Pandas | Course notes/exercises | Joins and SQL | DataCamp and predecessor-repository clone | High | Substantial | 1.12 MiB; local paths and email-bearing table outputs | Retain useful output; remove affected outputs; update links | unchanged |
| Python/data_manipulation/pandas/working_with_categorical_data/categorical_data.ipynb | Working with Categorical Data in Python | Course notes/exercises | Categorical data | DataCamp evidence | High | Substantial | 736 KiB; one retained error | Retain; remove error output | unchanged |
| Python/data_manipulation/preprocessing_for_machine_learning/pre_process_ml.ipynb | The Rationale for Data Preprocessing | Course notes/exercises | ML preprocessing | DataCamp evidence | High | Substantial | 1.30 MiB; Colab and local-path warning | Retain; clean metadata/private output | unchanged |
| Python/machine_learning/intro_ml_exercises/module_0.ipynb | Module 0 – Unit 1 | Exercises | ML foundations | Source not stated | Medium | Substantial | 8 KiB; VS Code metadata | Retain; document provenance uncertainty | unchanged |
| Python/machine_learning/intro_to_machine_learning/module_0.ipynb | The Statistical Bedrock | Lecture/conceptual notes | Probability and statistics | Source not stated; DataCamp references present | Medium | Substantial | 370 KiB; stale badge target | Retain; repair badge | unchanged |
| Python/machine_learning/intro_to_machine_learning/module_1.ipynb | Supervised Learning Framework | Lecture/conceptual notes | Supervised learning | Source not stated | Medium | Substantial | 195 KiB | Retain | unchanged |
| Python/machine_learning/intro_to_machine_learning/module_2.ipynb | Evaluating Regression | Lecture/conceptual notes | Regression metrics | Source not stated | Medium | Substantial | 2 KiB | Retain | unchanged |
| Python/machine_learning/intro_to_machine_learning/module_3.ipynb | Linear Regression | Lecture/conceptual notes | Linear regression | Source not stated | Medium | Substantial | 45 KiB | Retain | unchanged |
| Python/machine_learning/supervised_learning/decision_trees/module_1/exercises_1-0.ipynb | Decision Trees vs Logistic Regression | Exercises | Classification trees | UCI/ucimlrepo; task text included | High | Substantial | 472 KiB; non-sequential | Retain; provenance review | unchanged |
| Python/machine_learning/supervised_learning/decision_trees/module_1/exercises_1-1.ipynb | Entropy vs Gini | Exercises | Tree split criteria | UCI/ucimlrepo; task text included | High | Substantial | 1.18 MiB | Retain; provenance review | unchanged |
| Python/machine_learning/supervised_learning/decision_trees/module_1/exercises_1-2.ipynb | Regression Trees vs Linear Regression | Exercises | Regression trees | UCI/ucimlrepo; task text included | High | Substantial | 490 KiB; non-sequential | Retain; provenance review | unchanged |
| Python/machine_learning/supervised_learning/decision_trees/module_1/lecture_1-0.ipynb | Introduction to Classification Trees | Lecture notes | Classification trees | Source not stated | Medium | Substantial | 318 KiB | Retain | unchanged |
| Python/machine_learning/supervised_learning/decision_trees/module_1/lecture_1-1.ipynb | Building Blocks of Decision Trees | Lecture notes | Tree construction | Source not stated | Medium | Substantial | 453 KiB | Retain | unchanged |
| Python/machine_learning/supervised_learning/decision_trees/module_1/lecture_1-2.ipynb | Introduction to Regression Trees | Lecture notes | Regression trees | Source not stated | Medium | Substantial | 240 KiB | Retain | unchanged |
| Python/machine_learning/supervised_learning/decision_trees/module_2/exercises_2-1.ipynb | Diagnosing Bias and Variance | Exercises | Model evaluation | Task text included | Medium | Substantial | 2.77 MiB; non-sequential | Retain; provenance review | unchanged |
| Python/machine_learning/supervised_learning/decision_trees/module_2/exercises_2-2.ipynb | Ensemble Classifier for Heart Disease | Exercises | Ensembles | UCI/ucimlrepo | High | Substantial | 479 KiB; non-sequential | Retain; provenance review | unchanged |
| Python/machine_learning/supervised_learning/decision_trees/module_2/lecture_2-0.ipynb | Fundamentals of Supervised Learning | Lecture notes | Generalisation error | Source not stated | Medium | Substantial | 129 KiB | Retain | unchanged |
| Python/machine_learning/supervised_learning/decision_trees/module_2/lecture_2-1.ipynb | Estimating Generalisation Error | Lecture notes | Cross-validation | Source not stated | Medium | Substantial | 114 KiB | Retain | unchanged |
| Python/machine_learning/supervised_learning/decision_trees/module_2/lecture_2-2.ipynb | Introduction to Ensemble Learning | Lecture notes | Ensembles | Source not stated | Medium | Substantial | 49 KiB | Retain | unchanged |
| Python/machine_learning/supervised_learning/decision_trees/module_3/exercises_3-0.ipynb | Bagging Trees for Bank Marketing | Exercises | Bagging | UCI/ucimlrepo; task text included | High | Partial/interrupted | 54 KiB; KeyboardInterrupt and local path | Remove error output; retain source; human review | unchanged |
| Python/machine_learning/supervised_learning/decision_trees/module_3/lecture_3-0.ipynb | Distinguishing Ensemble Methods | Lecture notes | Bagging | Source not stated | Medium | Substantial | 208 KiB | Retain | unchanged |
| Python/machine_learning/supervised_learning/decision_trees/module_3/lecture_3-1.ipynb | Out-of-Bag Evaluation | Lecture notes | OOB evaluation | Source not stated | Medium | Substantial | 96 KiB | Retain | unchanged |
| Python/machine_learning/supervised_learning/decision_trees/module_3/lecture_3-2.ipynb | Introduction to Random Forests | Lecture notes | Random forests | Source not stated | Medium | Near-empty | No output | Preserve; human decision on completion/value | unchanged |
| Python/machine_learning/supervised_learning/supervised_learning.ipynb | The Two Paradigms of Machine Learning | Course notes/exercises | Supervised ML | DataCamp and predecessor-repository links | High | Substantial | 845 KiB; non-sequential; local-path warnings | Retain; clean affected output; update links | unchanged |
| Python/machine_learning/unsupervised_learning/unsupervised_1.ipynb | Supervised vs Unsupervised Learning | Course notes/exercises | Clustering and dimension reduction | DataCamp/UCI links | High | Substantial | 2.42 MiB | Retain; update data URLs | unchanged |
| Python/machine_learning/unsupervised_learning/unsupervised_2.ipynb | Cluster Visualisation and Evaluation | Course notes/exercises | Clustering | UCI and local helpers | High | Substantial | 4.27 MiB; local-path warnings | Retain; clean affected output | unchanged |
| Python/machine_learning/unsupervised_learning/unsupervised_clustering.ipynb | K-Means Clustering Primer | Lecture/conceptual notes | K-means | UCI; source boundary unclear | Medium | Substantial | 575 KiB | Retain; repair stale data path | unchanged |
| Python/mathematics/harmonograph.ipynb | Computer-Generated Harmonograph | Scratch experiment | Mathematical visualisation | Source not stated; DataCamp text signal | Medium | Complete as experiment | 2.01 MiB plot output | Preserve output | unchanged |
| Python/mathematics/math_for_ds/unit_0/module_01.ipynb | Mathematical Foundations, module 1 | Mathematics notes | Foundations | Source not stated | Medium | Substantial | Minimal output | Retain sequence | unchanged |
| Python/mathematics/math_for_ds/unit_0/module_02.ipynb | Mathematical Foundations, module 2 | Mathematics notes | Foundations | Source not stated | Medium | Substantial | 124 KiB | Retain sequence | unchanged |
| Python/mathematics/math_for_ds/unit_0/module_03.ipynb | Mathematical Foundations, module 3 | Mathematics notes | Foundations | Source not stated | Medium | Substantial | 807 KiB; non-sequential; local-path warning | Retain; clean affected output | unchanged |
| Python/mathematics/math_for_ds/unit_0/module_04.ipynb | Mathematical Foundations, module 4 | Mathematics notes | Foundations | Source not stated | Medium | Substantial | 171 KiB | Retain sequence | unchanged |
| Python/mathematics/reading_math/reading_1.ipynb | Mathematical Reading for Data Science | Mathematics notes | Variance and notation | Source not stated | Medium | Substantial | Minimal output; non-sequential | Preserve; provenance review | unchanged |
| Python/mathematics/study_sessions/study_1.ipynb | Removable Discontinuity | Study session | Calculus | Source not stated | Medium | Small but coherent | 197 KiB | Preserve; possible later archive move | unchanged |
| Python/nlp/nlp_in_python/nlp.ipynb | What is NLP? | Course notes/exercises | NLP | DataCamp evidence | High | Partial | 126 KiB; missing spaCy model error and local paths | Remove error/private output; document dependency | unchanged |
| Python/nlp/nlp_with_spacy/nlp_spacy.ipynb | What is NLP? | Lecture note | spaCy | DataCamp evidence | High | Near-empty | No output | Preserve; human review | unchanged |
| Python/projects/ames_house/ames_house.ipynb | Ames Housing Dataset | Selected modelling project | Regression | De Cock paper, OpenML, and Kaggle references | High | Substantial | 415 KiB; useful model results | Preserve output; remove regenerable joblib only | unchanged |
| Python/projects/analyzing_crime_in_los_angeles/crime_los_angeles.ipynb | Crime in Los Angeles | Guided project | EDA | Supplied project prompt; DataCamp evidence | High | Substantial | 145 KiB; scheduling/editor metadata | Retain; label as guided; clean metadata | unchanged |
| Python/projects/bike_sharing/bike.ipynb | Seoul Bike Sharing | Exploratory/guided project | Regression | UCI and Seoul data sources | High | Partial | 341 KiB; one model-shape error and local path | Retain; remove error output; review unrelated 98 MiB file | unchanged |
| Python/projects/book_recommender/data_exploration.ipynb | Book Recommender Data Exploration | Ambiguous project | EDA/NLP | Dataset links present; no notebook markdown | Medium | Partial | 388 KiB | Remove only generated Chroma index; human review | unchanged |
| Python/projects/california_house_pricing/california.ipynb | California Housing Prices | Guided/textbook-derived project | Regression | Downloads from ageron/data; exact source not stated | Medium | Substantial | 1.50 MiB; useful results | Preserve; label cautiously | unchanged |
| Python/projects/customer_analytics_preparing_data_for_modeling/notebook.ipynb | Customer Analytics Data Preparation | Guided project | Data types/memory | Supplied project instructions; DataCamp evidence | High | Complete exercise | 11 KiB; scheduling/editor metadata | Rename; clean metadata; label guided | Python/projects/customer_analytics_preparing_data_for_modeling/customer_analytics.ipynb |
| Python/projects/eurovision_2016-25/eurovision.ipynb | Eurovision 2016–2025 | Scratch/ambiguous project | EDA | Source URL not documented in markdown | Medium | Incomplete; four cells | 26 KiB | Preserve; human review | unchanged |
| Python/projects/exploring_airbnb_market_trends/notebook.ipynb | Exploring Airbnb Market Trends | Guided project | Data manipulation | Supplied prompt; DataCamp evidence | High | Complete exercise | 22 KiB; scheduling/Colab metadata | Rename; clean metadata; label guided | Python/projects/exploring_airbnb_market_trends/airbnb_market_trends.ipynb |
| Python/projects/heart_disease/heart_disease.ipynb | Heart Disease Scratch Notebook | Scratch/ambiguous project | Classification | UCI dataset; missing helper URL | Medium | Incomplete; three code cells | 6 KiB | Preserve; human review | unchanged |
| Python/projects/history_of_nobel_prize_winners/nobel_winnners.ipynb | Nobel Prize Winners Data Analysis | Guided project with extensions | EDA | Supplied/DataCamp project evidence; source boundary unclear | High | Substantial | 447 KiB; useful plots | Correct spelling; retain output; label guided | Python/projects/history_of_nobel_prize_winners/nobel_winners.ipynb |
| Python/projects/hypothesis_test_football_matches/hypothesis_football.ipynb | Hypothesis Test: Football Matches | Guided project | Hypothesis testing | Supplied prompt; DataCamp evidence | High | Complete exercise | 7 KiB | Retain; update predecessor-repository URLs | unchanged |
| Python/projects/magic_gamma/magic.ipynb | MAGIC Gamma Classification | Guided/exploratory project | Classification | UCI; old fcc path suggests external curriculum | Medium | Substantial | 257 KiB; stale badge/path | Retain; repair links; document provenance uncertainty | unchanged |
| Python/projects/modeling_agriculture/agriculture.ipynb | Sowing Success | Guided project | Feature selection | Supplied prompt; DataCamp evidence | High | Complete exercise | Minimal output; scheduling metadata | Retain; label guided; clean metadata | unchanged |
| Python/projects/modeling_car_insurance_claim_outcomes/car_insurance.ipynb | Car Insurance Claim Outcomes | Guided project | Logistic regression | Supplied prompt; DataCamp evidence | High | Complete exercise | 111 KiB; editor metadata | Retain; label guided; clean metadata | unchanged |
| Python/projects/online_retail_ii/online_retail_ii.ipynb | Online Retail II | Selected exploratory project | Customer segmentation | UCI DOI stated | High | Substantial | 4.16 MiB; analytically meaningful plots | Preserve output | unchanged |
| Python/projects/spotify/spotify_dataset.ipynb | Spotify Songs | Exploratory project | Regression/feature analysis | TidyTuesday source stated | High | Substantial but mixed notes | 3.15 MiB; useful plots; Colab metadata | Preserve output; clean metadata | unchanged |
| Python/projects/student_completion/student_completion.ipynb | Student Completion | Scratch/ambiguous project | EDA | Dataset source not documented | Medium | Incomplete; no markdown | 908 KiB | Preserve; human review | unchanged |
| Python/projects/study_sessions/data-reboot-airbnb-prices/airbnb.ipynb | Machine Learning Reboot Challenge | Guided study session | ML workflow | External challenge and nbresult tests | High | Substantial | 102 KiB | Move to archive; preserve fixtures pending review | archive/study_sessions/data_reboot_airbnb_prices/airbnb.ipynb |
| Python/projects/study_sessions/iris_sklearn/iris.ipynb | Welch Two-Sample t-Test Worksheet | Study exercise | Inference | scikit-learn Iris; worksheet source unclear | High | Coherent exercise | 9 KiB | Move to archive | archive/study_sessions/iris_sklearn/iris.ipynb |
| Python/projects/study_sessions/oop-refresher/oop-refresher.ipynb | OOP Refresher | Study exercise | Python OOP | Source not stated | High | Near-empty | No output | Move to archive; preserve name | archive/study_sessions/oop_refresher/oop-refresher.ipynb |
| Python/projects/study_sessions/paths/paths.ipynb | Paths | Study exercise | pathlib | Source not stated | High | Coherent exercise | Absolute source/output path | Move to archive; replace with repository-relative example | archive/study_sessions/paths/paths.ipynb |
| Python/projects/study_sessions/random_notebooks/heart_disease_ztm.ipynb | Heart Disease Dataset | Scratch experiment | Classification | UCI; “ztm” provenance unresolved | Medium | Near-empty | No output | Move to archive; human provenance review | archive/study_sessions/random_notebooks/heart_disease_ztm.ipynb |
| Python/projects/study_sessions/random_notebooks/study_1.ipynb | t-Test Exercises | Study exercise | Inference | Source not stated | High | Coherent | Minimal output | Move to archive | archive/study_sessions/random_notebooks/study_1.ipynb |
| Python/projects/study_sessions/random_notebooks/study_2.ipynb | z-Test Exercise | Study exercise | Inference | Source not stated | High | Coherent | 163 KiB; non-sequential | Move to archive; preserve plot | archive/study_sessions/random_notebooks/study_2.ipynb |
| Python/projects/study_sessions/random_notebooks/study_3.ipynb | Paired vs Independent Test | Study exercise | Inference | Source not stated | High | Coherent | Minimal output; non-sequential | Move to archive | archive/study_sessions/random_notebooks/study_3.ipynb |
| Python/projects/telcom_customer_churn/telcom_customer_churn.ipynb | Telco Customer Churn | Selected modelling project | Classification | Dataset widely circulated; exact source absent | Medium | Substantial | 986 KiB; useful plots | Correct path spelling; preserve output; provenance review | Python/projects/telco_customer_churn/telco_customer_churn.ipynb |
| Python/projects/titanic/titanic.ipynb | Titanic: Machine Learning from Disaster | Guided/exploratory project | Classification | Kaggle explicitly stated | High | Substantial | 245 KiB; useful plots | Preserve; label source; retain submission for review | unchanged |
| Python/projects/vehicles/vehicles.ipynb | Vehicle Dataset from CarDekho | Selected modelling project | Regression and text features | Kaggle source stated | High | Substantial | 842 KiB; useful results | Preserve output; remove regenerable 50.86 MiB model | unchanged |
| Python/projects/winsconsin_breast_cancer_plots/wbc_plots.ipynb | Wisconsin Breast Cancer Plots | Scratch experiment | Visualisation | scikit-learn dataset; prompt/template origin unclear | Medium | Partial/template-like | 1.79 MiB plots | Correct directory spelling; preserve; human review | Python/projects/wisconsin_breast_cancer_plots/wbc_plots.ipynb |
| Python/projects/womens_ecommerce/womens.ipynb | Women’s E-Commerce Clothing Reviews | Exploratory project | EDA | External dataset; exact source/terms not stated | Medium | Partial | 108 KiB | Preserve; provenance review | unchanged |
| Python/statistics/experimental_design/experimental.ipynb | The Goal of Experimental Design | Course notes/exercises | Experimental design | DataCamp evidence and supplied datasets | High | Substantial | 1.05 MiB; local-path warning | Retain; clean affected output; update URLs | unchanged |
| Python/statistics/exploratory_data_analysis/datacamp/exploratory_data_analysis.ipynb | Exploratory Data Analysis | Course notes/exercises | EDA | DataCamp explicit | High | Substantial | 3.09 MiB; non-sequential; Colab/VS Code metadata | Retain outputs; clean metadata; update links | unchanged |
| Python/statistics/exploratory_data_analysis/eda_modules/module_1.ipynb | Algorithmic Thinking | Lecture/conceptual notes | Python and text analysis | Source not stated | Medium | Substantial | Minimal output | Retain; repair stale data path | unchanged |
| Python/statistics/exploratory_data_analysis/eda_modules/module_2.ipynb | NumPy and Olist Analysis | Lecture/conceptual notes | NumPy/EDA | Olist source; source boundaries unclear | Medium | Substantial | 50 KiB; missing helpers.olist_analyzer | Preserve; human review dependency | unchanged |
| Python/statistics/exploratory_data_analysis/eda_modules/module_3-tobit-model.ipynb | Tobit Model Scenario | Scratch/lecture note | Censored regression | Source not stated | Medium | Small but coherent | 210 KiB | Preserve; review naming/import assumptions | unchanged |
| Python/statistics/exploratory_data_analysis/eda_modules/module_3.ipynb | Why Plotting Matters | Lecture/conceptual notes | Visual EDA | Source not stated | Medium | Substantial | 498 KiB; non-sequential | Retain | unchanged |
| Python/statistics/exploratory_data_analysis/eda_modules/module_4.ipynb | Theory of Analytical Graphics | Lecture/conceptual notes | Visualisation | Source not stated | Medium | Substantial | 736 KiB | Retain | unchanged |
| Python/statistics/exploratory_data_analysis/eda_modules/module_5.ipynb | EDA as Hypothesis Generation | Lecture/conceptual notes | EDA/inference | Source not stated | Medium | Substantial | 589 KiB | Retain | unchanged |
| Python/statistics/hypothesis_testing/hypothesis.ipynb | Framework of Hypothesis Testing | Course notes/exercises | Inference | DataCamp evidence | High | Substantial | 490 KiB | Retain | unchanged |
| Python/statistics/regression_with_statsmodels/part1.ipynb | The Core Idea of Regression | Course notes/exercises | Simple regression | DataCamp evidence | High | Substantial | 1.36 MiB | Retain; update dataset URL | unchanged |
| Python/statistics/regression_with_statsmodels/part2.ipynb | From Simple to Multiple Regression | Lecture note | Multiple regression | Source not stated | Medium | Near-empty | No output | Preserve; human decision on completion/name | unchanged |
| Python/statistics/sampling/sampling.ipynb | Why Sampling Matters | Course notes/exercises | Sampling | DataCamp evidence | High | Substantial | 1.58 MiB; non-sequential; local-path warning | Retain; clean affected output; update URLs | unchanged |
| Python/visualisation/matplotlib/intro_to_matplotlib/introduction_to_matplotlb.ipynb | Introduction to Matplotlib | Course notes/exercises | Matplotlib | DataCamp and old repository links | High | Substantial | 739 KiB; Colab metadata | Correct filename; update links; retain output | Python/visualisation/matplotlib/intro_to_matplotlib/introduction_to_matplotlib.ipynb |
| Python/visualisation/seaborn/intermediate/intermediate_seaborn.ipynb | Intermediate Data Visualisation with Seaborn | Course notes/exercises | Seaborn | DataCamp evidence | High | Substantial | 2.69 MiB; example paths not locally backed | Retain; document examples | unchanged |
| Python/visualisation/seaborn/intro_to_seaborn/introduction_to_seaborn.ipynb | Introduction to Seaborn | Course notes/exercises | Seaborn | DataCamp and old repository links | High | Substantial | 2.10 MiB; missing-data error and local path | Remove error output; update resolvable links; document missing dataset | unchanged |

### Supporting content

| Current path/group | Inferred category | Topic/provenance | Confidence | Recommended action |
| --- | --- | --- | --- | --- |
| Python/helpers/*.py | Reusable helpers | Statistical summaries, encoding, missingness, plotting | High | Preserve; compile; document imports and limits |
| Python/machine_learning/unsupervised_learning/helper/*.py | Project-local helpers | Clustering plots, reports, selection, stability | High | Preserve in place because notebooks import helper.* |
| Python/statistics/exploratory_data_analysis/eda_modules/helpers/*.py | Module-local helpers | EDA utilities | High | Preserve; missing olist_analyzer remains unresolved |
| Python/**/data and Python/**/datasets | Datasets and supplied exercise data | Mixed external sources and generated derivatives | Medium | Preserve by default; document source/licensing uncertainty |
| Python/projects/vehicles/vehicle_price_model_v1.pkl | Generated model artefact | Produced by vehicles.ipynb; not loaded | High | Remove from current tree; document regeneration |
| Python/projects/ames_house/ames_house_price_model.joblib | Generated model artefact | Produced by ames_house.ipynb; not loaded | High | Remove from current tree; document regeneration |
| Python/projects/book_recommender/data/chroma_books/** | Generated vector index | No current-tree reference; SQLite was already removed earlier | High | Remove current index files; ignore generated index paths |
| Python/projects/study_sessions/data-reboot-airbnb-prices/tests/*.pickle | Ambiguous test fixtures | External nbresult challenge; no direct current reference | Medium | Preserve and request human review |
| Python/data_manipulation/pandas/joining_data_with_pandas/data/chinook.db | Intentional dataset | Used in joining/SQL exercises | High | Preserve; use path-specific database ignore rules |
| Python/projects/titanic/titanic_submission.csv | Generated analysis output | Produced by notebook; small and potentially documentary | High | Preserve pending human choice |
| Python/projects/book_recommender/data/books_cleaned.csv and tagged_description.txt | Generated derived data | Produced by exploration notebook and likely useful to unfinished project | High | Preserve pending project decision |
| Images below project data directories | Supplied/documentary assets | Guided prompts and notebook rendering | Medium | Preserve; provenance review |

## 3. Naming audit

### Findings

- The top-level Python name is inconsistent with lowercase snake_case, but
  renaming the entire 302-file subtree would create high churn and link risk
  with little immediate navigation gain.
- Sequence names such as module_0, lecture_2-1, exercises_1-0, part1, part2,
  and study_1 are generic in isolation. Many retain useful course ordering,
  and several titles or source boundaries are too uncertain for safe renames.
- Two project notebooks are literally notebook.ipynb even though their
  subjects are unambiguous.
- Obvious spelling errors exist in nobel_winnners, winsconsin, telcom, and
  introduction_to_matplotlb.
- Hyphens and underscores are mixed in module names, data-reboot-airbnb-prices,
  oop-refresher, module_3-tobit-model, and eurovision_2016-25.
- Several vendor dataset names use spaces, uppercase letters, ampersands, or
  punctuation. Renaming them would require editing notebook URLs and could
  obscure their supplied filenames; preserve them unless a project move
  requires otherwise.
- eurovision_2016-25 encodes an ambiguous shortened year range and will become
  stale if the dataset expands. Preserve it for human review.
- No proposed target below currently collides with an existing tracked path.

### High-confidence path mapping

| Old path | Proposed path | Reason |
| --- | --- | --- |
| Python/projects/study_sessions/ | archive/study_sessions/ | Study sessions are not projects |
| Python/projects/customer_analytics_preparing_data_for_modeling/notebook.ipynb | Python/projects/customer_analytics_preparing_data_for_modeling/customer_analytics.ipynb | Subject is explicit |
| Python/projects/exploring_airbnb_market_trends/notebook.ipynb | Python/projects/exploring_airbnb_market_trends/airbnb_market_trends.ipynb | Subject is explicit |
| Python/projects/history_of_nobel_prize_winners/nobel_winnners.ipynb | Python/projects/history_of_nobel_prize_winners/nobel_winners.ipynb | Obvious spelling error |
| Python/projects/telcom_customer_churn/ | Python/projects/telco_customer_churn/ | Notebook title consistently uses “Telco” |
| Python/projects/telcom_customer_churn/telcom_customer_churn.ipynb | Python/projects/telco_customer_churn/telco_customer_churn.ipynb | Match corrected directory and title |
| Python/projects/winsconsin_breast_cancer_plots/ | Python/projects/wisconsin_breast_cancer_plots/ | Obvious spelling error |
| Python/visualisation/matplotlib/intro_to_matplotlib/introduction_to_matplotlb.ipynb | Python/visualisation/matplotlib/intro_to_matplotlib/introduction_to_matplotlib.ipynb | Obvious spelling error |

All other generic or inconsistent names remain unchanged unless later
validation establishes an unambiguous, useful title.

## 4. Navigation audit

At the snapshot, a new visitor cannot reliably determine what the repository
is because no README is tracked. The tree suggests topics, but it does not
explain:

- which project work is independent, guided, experimental, or incomplete;
- which notebooks are course exercises versus conceptual notes;
- why study sessions sit below projects;
- which materials are historical;
- where most datasets originated or whether redistribution has been reviewed;
- which notebooks are expected to execute unchanged;
- why outputs and generated artefacts are retained;
- what environment or Python version is expected.

The proposed navigation system is:

1. A restrained root README defining the repository as a learning archive.
2. A content index separating selected projects, guided projects, notes,
   exercises, helpers, archive material, and unresolved content.
3. Dedicated provenance, reproducibility, human-review, audit, and curation
   documents.
4. Minimal structural movement: retain the useful subject hierarchy below
   Python, but move the clearly labelled project study_sessions collection to
   archive/study_sessions.
5. Direct README links to a limited set of substantial, self-contained work:
   Ames housing, Online Retail II, Telco churn, vehicles, and Spotify, with
   honest provenance/status labels.

## 5. Git hygiene audit

The committed .gitignore is long and internally confused:

- .flake8 is duplicated.
- docs/ is ignored under an R pkgdown section although there is no current R
  content, preventing repository documentation from being tracked.
- .editorconfig and pyproject.toml are ignored even though they are normally
  intentional repository files.
- pyrightconfig.json is ignored as a stale project-specific rule.
- R, Celery, SageMath, PyInstaller, Pipenv, Poetry, PDM, and extensive
  operating-system rules are present without current-tree evidence that most
  are needed.
- Secret rules cover several named .env variants but not the clearer
  .env.* plus explicit example-file exception pattern.
- Chroma database/index paths are ignored, but four generated .bin files under
  chroma_books remain tracked.
- Generated .pkl, .pickle, and .joblib model artefacts are not addressed with
  a documented exception policy.
- Data is not globally ignored, which is correct for this archive.

The pre-existing working-tree edit already removes the .editorconfig and
pyproject.toml exclusions, removes several stale rules, and fixes a .lintr
trailing-space typo, but it still leaves docs/ ignored. Those intentions should
be preserved and the docs/ rule corrected in the dedicated .gitignore
replacement.

No .env, credential file, Python/R environment, cache, notebook checkpoint, or
IDE directory is currently tracked. Under the working .gitignore, the four
Chroma .bin files are tracked despite matching ignore rules.

Recommended policy: a concise repository-specific ignore file covering Python
caches/builds/environments, Jupyter checkpoints, secrets with example
exceptions, local editor/OS state, generated Chroma/vector indexes, generated
model extensions, logs/temporary files, and path-specific SQLite state. Small
source datasets remain trackable. Intentional binary datasets and fixtures
already tracked are not automatically removed.

## 6. Notebook hygiene audit

All 84 notebooks parse as JSON. Schema validation was not available at audit
time because nbformat is not installed.

| Signal | Count |
| --- | ---: |
| Notebooks with non-sequential execution counts | 17 |
| Notebooks retaining error outputs | 5 |
| Total retained error outputs | 5 |
| Notebooks with no markdown cells | 4 |
| Notebooks with three or fewer cells | 6 |
| Notebooks with more than 1 MiB of output JSON | 17 |
| Notebook output JSON (estimated) | 54.28 MiB |
| Notebooks with root Colab metadata | 17 |
| Notebooks with root accelerator metadata | 7 |
| Notebooks with root editor metadata | 2 |
| Notebooks with cell IDs | 16 |
| Data Wrangler custom MIME payloads | 711 across 56 notebooks |

Five error outputs are retained: an empty error in the categorical-data
notebook, a KeyboardInterrupt in the bank-marketing exercise, a missing spaCy
model, a regression feature-shape mismatch in bike sharing, and a missing
air_quality_no2_long.csv file in the introductory Seaborn notebook.

Transient metadata includes Colab output IDs, VS Code metadata, editor and
accelerator metadata, Data Wrangler payloads, execution timestamps,
scheduling IDs, kernel execution identifiers, and dataframe-visualiser state.
Standard language_info, kernelspec, semantic tags, cell IDs, and meaningful
layout metadata should be preserved.

One source cell in the paths study session contains an absolute home-directory
path and should be converted to a repository-relative exercise path. Other
home-directory matches occur only in output streams or tracebacks. Four
joining-data outputs render email-bearing tables. The addresses appear to come
from supplied datasets, but the outputs add no unique analytical value and
should be removed without printing the values.

Several notebooks contain predecessor-repository URLs
(Datacamp-Notebooks or data-science-practice-notebook) or old paths such as
supervised_learning_with_scikit-learn, unsupervised_learning_with_scikit-learn,
fcc_magic_gamma, exploratory_data_analysis/modules, and Intro to Matplotlib.
Most have obvious equivalents in the current tree and can be updated. The
missing Ames missing_summary.py and air_quality_no2_long.csv references have
no current equivalent and require human review.

Output policy:

- preserve outputs for selected projects, finished analyses, harmonograph
  rendering, and course notebooks where plots/tables are part of the notes;
- remove only outputs containing local paths or email-bearing tables, retained
  error outputs, redundant editor MIME payloads, and indisputably transient
  metadata;
- reset execution counts for cells whose substantive output is removed;
- do not clear notebooks merely because their output is large.

## 7. Artefact and size audit

### Working-tree candidates

| Artefact | Size/status | Recommendation |
| --- | --- | --- |
| vehicle_price_model_v1.pkl | 50.86 MiB; generated and not loaded | Remove; regeneration cell exists |
| ames_house_price_model.joblib | 23.52 KiB; generated and not loaded | Remove; regeneration cell exists |
| book_recommender/data/chroma_books/** | 1.19 MiB total; generated index; unreferenced | Remove; vector-search notebook is no longer in current tree |
| Most-Recent-Cohorts-Institution.csv | 98.34 MiB; unrelated to current bike notebook by reference scan | Preserve for human review because provenance/value is uncertain |
| Olist dataset collection | Multiple files totaling substantial size; helper dependency incomplete | Preserve; redistribution and missing-helper review |
| Online Retail II workbook | 43.51 MiB; used by substantial project | Preserve |
| crimes.csv | 26.21 MiB; used by guided project | Preserve |
| Course_Completion_Prediction.csv | 21.29 MiB; used by incomplete notebook | Preserve pending human decision |
| music.csv | 19.19 MiB; used by supervised-learning notes | Preserve |
| tests/*.pickle | Tiny, ambiguous external test fixtures | Preserve pending review |
| chinook.db | 1.02 MiB intentional exercise dataset | Preserve |
| housing.tgz | Download archive with extracted CSV and download code | Preserve; low benefit and source snapshot value |
| titanic_submission.csv | Small generated project output | Preserve pending review |

Exact hashing found three duplicate groups:

- gdp-ffill.csv and gdp-query.csv;
- pop-query.csv and pop.csv;
- experimental_design/data/salaries.csv and
  exploratory_data_analysis/datacamp/data/ds_salaries.csv.

The names suggest exercise-specific roles even where bytes match. Do not
deduplicate without checking notebook semantics and provenance.

### History-only concerns

Do not rewrite history in this task. The largest history-only concerns include
multiple versions of book_recommender/data/chroma_books/chroma.sqlite3
(approximately 63.04, 47.98, 31.50, 15.32, and 15.32 MiB blobs), a removed
27.64 MiB Smartphone.csv, removed book/course directories, R notebooks,
website files, local databases, archives, and earlier notebook layouts.
Current large datasets and the vehicle model also remain in history even after
future working-tree cleanup.

Optional future history cleanup would need a separate, explicitly authorised
plan, fresh backups, collaborator coordination, and force-push implications.

## 8. Security and privacy audit

A redacted current-tree scan checked common private-key, AWS, GitHub, OpenAI,
Google API, and generic credential-assignment patterns. It found no private
key blocks, AWS access-key IDs, GitHub tokens, Google API keys, or credential
assignments.

An OpenAI-shaped pattern matched 144 strings inside scikit-learn HTML estimator
representations. All matches occur only in execute_result text/html payloads,
in repeated groups, and are generated element identifiers rather than source
credentials.

Security/privacy findings:

- one absolute home-directory path appears in notebook source;
- local Windows user-directory paths appear in output streams/tracebacks in
  thirteen notebooks;
- email addresses appear in four rendered outputs in the joining-data
  notebook;
- email-like fields also occur in tracked public/supplied datasets including
  Olist sellers, customer/employee exercise data, and the large institutional
  cohorts file;
- no complete discovered value is reproduced in this report;
- the Chinook database was not semantically inspected row by row;
- external datasets may contain personal or contact fields even when publicly
  distributed.

The current-tree scan does not establish that Git history is secret-free.
History contains many deleted files and requires a dedicated historical secret
scan before making any such claim.

## 9. Provenance and attribution audit

| Source/evidence | Affected material | Current state | Recommendation |
| --- | --- | --- | --- |
| DataCamp | Data manipulation, visualisation, statistics, NLP, and several project prompts; 70 notebooks contain a DataCamp signal | Often implicit in links/prompts rather than a repository-level statement | Label course-derived/guided material and retain known links |
| Kaggle | Ames, Titanic, vehicles; incidental references elsewhere | Some notebook-level attribution exists | Preserve links; distinguish competition/guided work |
| UCI Machine Learning Repository | Decision-tree exercises, bike sharing, MAGIC Gamma, Online Retail II, heart disease, clustering | Several notebooks state UCI/DOI; others rely on ucimlrepo | Consolidate known dataset sources |
| TidyTuesday | Spotify notebook | Source link is present | Retain and index |
| Machine Learning Reboot / nbresult | Archived Airbnb challenge and tests | External challenge is evident; provider/terms not fully stated | Preserve and request redistribution/provenance review |
| ageron/data | California housing download | Strong textbook-guided signal; exact book/source not stated | Describe cautiously; do not claim originality |
| Supplied project prompts and images | LA crime, Airbnb trends, customer analytics, football, agriculture, car insurance, Nobel | Prompt language is embedded and sometimes dominates markdown | Label as guided; do not rewrite supplied text as original |
| External datasets | Olist, institutional cohorts, women’s e-commerce, Telco churn, student completion, and many course CSVs | Source/licensing often missing from current notebooks | Human review for attribution and redistribution |

No legal conclusion is made about copyright, permission, or licensing. The
presence of a source URL does not establish redistribution permission. The
repository should explicitly distinguish supplied prompts/data from personal
analysis where that distinction is evident and list unresolved cases rather
than invent attribution.

## 10. Reproducibility audit

Notebook metadata records Python 3.12.x in 40 notebooks and Python 3.13.7 in 36
notebooks; five notebooks have other 3.12 patch versions and five notebooks do
not record a version. Kernels are mainly named datacamp (69), DS (9), or
Python 3 (3). These names describe local environments, not portable
dependency specifications.

Frequently observed third-party imports include scikit-learn, NumPy, pandas,
Matplotlib, SciPy, Seaborn, statsmodels, pingouin, NLTK, spaCy, ucimlrepo,
TensorFlow, Polars, yfinance, geopandas, contextily, imbalanced-learn,
missingno, recordlinkage, and thefuzz. No current dependency declaration or
lock file exists, so a single precise environment cannot be inferred
honestly.

Important execution assumptions and failures:

- many notebooks fetch data from raw GitHub URLs, including predecessor
  repositories and old paths;
- selected projects use local tracked data, remote data, or both;
- the California notebook downloads and extracts housing.tgz;
- the NLP notebook requires the en_core_web_lg spaCy model;
- the archived Machine Learning Reboot notebook requires nbresult and external
  challenge infrastructure;
- eda_modules/module_2 imports helpers.olist_analyzer, which is absent from the
  current tree but present in history;
- helpers.plot_seaborn resolves only if the Python directory is on sys.path,
  an assumption not documented by the notebooks;
- local helper imports in dates/times, unsupervised learning, Telco churn, and
  vehicles have corresponding current files;
- not all notebooks specify random seeds, and retained outputs may reflect
  historical environments;
- notebooks with errors or near-empty content are not expected to run
  unchanged.

Likely or comparatively reproducible after path repair: vehicles (without the
saved model), Ames housing, Titanic, agriculture, customer analytics, football
hypothesis testing, modeling car insurance, Online Retail II, and many
course-note notebooks whose data is tracked.

Partially reproducible: Telco churn (heavy/optional TensorFlow dependency and
unclear dataset provenance), Spotify (mixed source and incomplete trailing
material), California housing (network and geospatial dependencies), MAGIC
Gamma (TensorFlow), and unsupervised-learning notebooks (network/local helper
assumptions).

Dependent on unavailable or ambiguous components: NLP with spaCy large model,
the archived Reboot challenge, EDA module 2, the incomplete heart-disease
notebook, introductory Seaborn’s missing air-quality file, and notebooks whose
prompts assume external platform state.

The least misleading environment policy is documentation-only guidance at the
root, with project-specific dependency notes for selected work. A generated
mega-requirements file would falsely imply one supported environment.

## 11. Conservative target structure

The existing subject hierarchy is already useful and should not be reorganised
for symmetry. The conservative target is:

    README.md
    docs/
        content_index.md
        curation_report.md
        human_review.md
        provenance.md
        reproducibility.md
        repository-audit.md
    Python/
        data_manipulation/
        helpers/
        machine_learning/
        mathematics/
        nlp/
        projects/
        statistics/
        visualisation/
    archive/
        study_sessions/

Projects, guided work, notes, and exercises will primarily be distinguished by
documentation rather than mass movement. Only the clearly misclassified
projects/study_sessions subtree moves to archive. Data stays near the notebooks
that depend on it; centralising all datasets would break context and paths.

## 12. Remediation map

| Category | Affected files | Change and rationale | Risk and validation | Rollback / commit boundary |
| --- | --- | --- | --- | --- |
| A | docs/repository-audit.md | Record pre-curation state | Risk: classification uncertainty; validate counts/paths | This standalone audit commit |
| A | README.md, docs/content_index.md | Define archive identity and navigation | Risk: overstatement; cross-check every linked path | docs: define repository scope and navigation |
| B, high confidence | Python/projects/study_sessions/** | Move clearly non-project study sessions to archive | Risk: broken paths/links; use git mv, validate notebook-relative files and searches | chore: normalise high-confidence paths |
| A/B, high confidence | Two notebook.ipynb files and four obvious misspellings listed in the naming map | Reduce ambiguity without inventing titles | Risk: broken badges/docs/imports; use git mv and link scan | Same path-normalisation commit |
| A | .gitignore | Replace confused rules while preserving pre-existing user intentions | Risk: newly ignored tracked fixtures/models; inspect git check-ignore and tracked ignored files | chore: replace repository gitignore |
| A | Chroma .bin index files | Remove unreferenced generated local index | Risk: unfinished recommender may expect it; current reference scan is empty, regeneration documented | chore: remove generated working-tree artefacts |
| B, high confidence | vehicle_price_model_v1.pkl, ames_house_price_model.joblib | Remove regenerable saved models that notebooks create but never load | Risk: external users may consume untracked artefacts; document regeneration and verify save cells | Same artefact-removal commit |
| A | Notebook root/cell editor metadata and Data Wrangler MIME payloads | Remove transient, redundant IDE state while preserving cells and standard outputs | Risk: broad notebook diffs; use parsed JSON, compare source/cell order/output counts, validate every notebook | chore: clean transient notebook metadata |
| A | Outputs containing local paths, rendered emails, or retained errors | Remove private/stale output only; reset affected execution counts | Risk: loss of diagnostic evidence; document exact notebook/cell counts, preserve source | Same notebook-cleanup commit |
| A | archive/study_sessions/paths/paths.ipynb | Replace absolute source path with repository-relative path | Risk: exercise semantics; validate referenced file exists | fix: update paths and imports after curation |
| A/B, high confidence | Notebooks with predecessor-repository or old-path URLs | Update only where a current tracked equivalent is unambiguous | Risk: branch/path mistakes; compare decoded URL path against git ls-files and spot-check network access | Same path/import-fix commit |
| C | Most-Recent-Cohorts-Institution.csv, Olist collection, Course Completion data, test pickles, generated derived CSVs, submission CSV, vendor images/data | Preserve because relevance, redistribution, or documentary value is uncertain | Record options and sizes in human-review document | docs: record provenance and reproducibility |
| C | Near-empty, no-markdown, template-like, or incomplete notebooks; missing helpers/data; generic sequence names | Preserve substantive content and request human decision | Validate that none are deleted or substantially rewritten | Same documentation commit |
| C | Exact duplicate datasets and near-duplicate missing-data helpers | Preserve because semantic roles/API names differ | Record hashes/paths and reference results | Same documentation commit |
| D | All Git history, force-push, ambiguous notebook deletion, analytical conclusions/methods, broad notebook reformatting | Explicitly prohibited | Confirm log ancestry and branch status | No commit |
| A | docs/curation_report.md and final indexes | Record exact changes, validation, remaining risks, before/after size, and commits | Risk: stale report; regenerate checks immediately before final commit | docs: record curation results and unresolved items |

The audit itself proposes no force-push, history rewrite, legal conclusion,
global notebook execution, statistical-method change, or deletion of ambiguous
educational content.
