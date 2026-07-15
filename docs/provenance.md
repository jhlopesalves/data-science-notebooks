# Provenance and attribution

This document records source evidence visible in the current repository. It
does not determine ownership, copyright status, redistribution permission or
licensing. A source URL is evidence of origin or access, not evidence that a
file may be redistributed.

The repository contains a mixture of supplied prompts, course exercises,
external datasets, personal notes, code written in response to exercises, and
analyses whose exact boundaries are not recorded. Where the distinction cannot
be established from the file itself, this document preserves the uncertainty.

## Evidence levels

- Explicit: the notebook names a source, links to it, or labels itself as a
  project/exercise from that source.
- Strong contextual evidence: path names, supplied prompt language, asset
  links and Git history agree, but the notebook does not contain a complete
  citation.
- Unresolved: the current tree does not contain enough information to assign a
  reliable source or distinguish starter material from later additions.

## Course and guided-project origins

| Material | Evidence | Supplied material apparent in tree | Apparent additions | Attribution status |
| --- | --- | --- | --- | --- |
| Data manipulation notebooks | DataCamp asset/repository links and strong repository-history context | Exercise datasets, examples and course sequence | Added markdown, executed code and retained outputs appear interleaved | DataCamp origin is strong; exact supplied/personal boundary is unresolved |
| Introductory Matplotlib and Seaborn | Old Datacamp-Notebooks links and course-style chapter flow | Datasets, prompts and example sequence | Explanatory markdown and executions | Label as DataCamp-derived; exact course names are not consistently stated |
| Statistics notebooks: EDA, experimental design, hypothesis testing, regression and sampling | DataCamp signals, supplied datasets and course-style progression | Prompts, examples and data | Explanations, implementations and outputs | DataCamp-derived; retain source references already present |
| NLP in Python and spaCy | DataCamp signals and course-style material | Examples and product-review data | Notes and executions | DataCamp-derived; spaCy notebook is near-empty |
| Decision-tree exercises | Explicit task instructions or learning targets; UCI/ucimlrepo datasets | Exercise briefs and dataset definitions | Implementations, diagnostics and plots | Course provider is not stated; do not assign one |
| Crime in Los Angeles | Supplied narrative and explicit questions; DataCamp contextual evidence | Prompt text, image and crimes dataset | Answer code and outputs | Guided project; original dataset/image source terms need review |
| Customer Analytics Data Preparation | “Project Instructions” and supplied business scenario; DataCamp evidence | Prompt, image and customer_train.csv | Transformation code and result | Guided project |
| Exploring Airbnb Market Trends | Supplied narrative and task list; DataCamp evidence | Prompt, image and three data files | Data preparation and answers | Guided project |
| History of Nobel Prize Winners | Guided-project context and DataCamp evidence | Nobel data, image and likely prompt structure | Expanded markdown, aggregation and visual analysis are present | Guided project with apparent extensions; exact boundary is unresolved |
| Football Match Hypothesis Test | Supplied journalist scenario and project instructions; DataCamp evidence | Prompt, image and match data | Test selection, code and result | Guided project |
| Sowing Success | Supplied project narrative and task; DataCamp evidence | Prompt, image and soil_measures.csv | Feature evaluation code | Guided project |
| Car Insurance Claim Outcomes | Supplied scenario/task; DataCamp evidence | Prompt, image and car_insurance.csv | Model comparison code | Guided project |
| Machine Learning Reboot | Notebook title, challenge instructions, nbresult imports and tests | Challenge scaffold, tests and small fixtures | Completed cells and retained results | External guided challenge; provider and redistribution terms unresolved |
| California Housing Prices | Notebook downloads housing.tgz from the ageron/data repository | Dataset acquisition and workflow strongly resemble textbook-guided material | Executed analysis and local modifications may be present | Describe as guided/textbook-derived; exact source edition and boundary not stated |
| MAGIC Gamma | UCI data citation and historical fcc path in the notebook | Dataset definition and curriculum-like progression | Multiple model implementations and notes | External curriculum origin likely but not proven; keep as unresolved |
| Titanic | Notebook identifies “Titanic - Machine Learning from Disaster” and discusses Kaggle submission | Competition data and submission format | EDA, models and predictions | Kaggle competition project; starter/personal boundary unresolved |

The earliest Git commit message is “Datacamp notebooks and exercises,” and the
history includes former repository names Datacamp-Notebooks and
data-science-practice-notebook. This supports a broad course-derived
classification but does not prove that every notebook containing the word
DataCamp originated there.

## Dataset sources recorded in notebooks

| Dataset/material | Repository location | Recorded source | Notes |
| --- | --- | --- | --- |
| Ames Housing | Python/projects/ames_house | [De Cock paper](https://jse.amstat.org/v19n3/decock.pdf), [data documentation](https://jse.amstat.org/v19n3/decock/DataDocumentation.txt), and OpenML access | Kaggle is also referenced in notebook discussion |
| Online Retail II | Python/projects/online_retail_ii | [UCI DOI 10.24432/C5CG6D](https://doi.org/10.24432/C5CG6D) | Large workbook is tracked locally |
| Vehicle Dataset from CarDekho | Python/projects/vehicles | [Kaggle dataset](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho) | Dataset filename is preserved as supplied |
| Spotify Songs | Python/projects/spotify | [TidyTuesday 2020-01-21](https://github.com/rfordatascience/tidytuesday/blob/master/data/2020/2020-01-21/readme.md) | Notebook says data came from the Spotify Web API via TidyTuesday |
| Seoul Bike Sharing | Python/projects/bike_sharing | UCI Machine Learning Repository and [Seoul Open Data](http://data.seoul.go.kr/) | The notebook also contains a large institutional-cohorts file that it does not reference |
| MAGIC Gamma Telescope | Python/projects/magic_gamma | UCI Machine Learning Repository | Specific current UCI landing URL is not embedded |
| Heart Disease exercises | Decision-tree exercises and heart-disease scratch notebooks | UCI/ucimlrepo | Dataset is fetched rather than redistributed in several notebooks |
| California Housing | Python/projects/california_house_pricing | [ageron/data housing archive](https://github.com/ageron/data/raw/main/housing.tgz) | Source strongly suggests textbook guidance |
| Olist | Python/statistics/exploratory_data_analysis/eda_modules/data/olist | Repository link and filenames only | Original marketplace source, licence and redistribution terms are not recorded |
| Telco Customer Churn | Python/projects/telco_customer_churn | No adequate original-source citation in current notebook | Common dataset title is not sufficient attribution |
| Women’s Clothing E-Commerce Reviews | Python/projects/womens_ecommerce | No adequate original-source citation in current notebook | Large third-party dataset; source/terms need review |
| Course Completion Prediction | Python/projects/student_completion | No adequate original-source citation | Incomplete notebook and 21.29 MiB dataset |
| Institutional cohorts | Python/projects/bike_sharing/data/Most-Recent-Cohorts-Institution.csv | No source statement near the bike project | Filename suggests an external education dataset; relevance is unclear |

Raw GitHub URLs that point back into this repository are access locations, not
original provenance. During curation, resolvable predecessor-repository URLs
were updated to current paths without changing the source claims.

## Supplied prompts, images and templates

Guided-project narratives and images are retained because they provide context
for the exercises. Their inclusion must not be interpreted as original
authorship. The following areas contain particularly visible supplied material:

- Python/projects/analyzing_crime_in_los_angeles;
- Python/projects/customer_analytics_preparing_data_for_modeling;
- Python/projects/exploring_airbnb_market_trends;
- Python/projects/history_of_nobel_prize_winners;
- Python/projects/hypothesis_test_football_matches;
- Python/projects/modeling_agriculture;
- Python/projects/modeling_car_insurance_claim_outcomes;
- course-note data and prompt cells throughout Python/data_manipulation,
  Python/statistics and Python/visualisation.

The Wisconsin breast-cancer plot notebook contains template-like objective text
whose source is not stated. The archive’s exercise worksheets may also include
supplied instructions. These are review items rather than candidates for
automatic rewriting.

## Apparent personal analysis

Many notebooks contain executed code, custom explanatory markdown, additional
plots, model comparisons and conclusions beyond the visible task statements.
Examples include the vehicle pipeline and comparison, Online Retail II customer
features and clustering, expanded Nobel analysis, Telco model comparisons, and
Ames model development. These are apparent additions within the notebooks, but
the repository does not preserve a reliable starter-code boundary. They are
therefore described as analysis present in the archive, not certified as
solely original work.

## Missing attribution and unresolved licensing

The main unresolved groups are:

- large external datasets without source or licence records;
- supplied project images and prompt text;
- DataCamp-derived datasets whose original upstream source is not repeated;
- Olist data and the missing Olist helper;
- the Machine Learning Reboot scaffold, tests and pickle fixtures;
- book-recommender source data and generated derivatives;
- Telco churn, student-completion and women’s e-commerce datasets;
- exact duplicate datasets stored under different exercises.

No root licence has been created because no single licence can be inferred for
this mixture of personal and third-party material. Decisions about removing,
privatising, replacing or adding attribution to particular supplied files are
listed in [human review](human_review.md).
