# Data science learning archive

This repository contains personal notes, exercises, experiments and selected
projects produced while studying data science, statistics, machine learning,
natural language processing, mathematics and related subjects. Materials vary
in completeness and polish. Some notebooks originate from or respond to
courses, books, tutorials and guided projects; source information is provided
where known.

The repository is maintained primarily as a learning archive rather than as a
polished software package or portfolio. Inclusion does not mean that every
method, conclusion or coding style reflects current practice. Historical,
incomplete and exploratory work is retained when it remains useful or when its
status is uncertain.

## How the archive is organised

| Area | Contents |
| --- | --- |
| [Selected analyses](docs/content_index.md#selected-analyses) | Substantial modelling or exploratory notebooks chosen for navigation, not as a claim of portfolio readiness |
| [Guided projects](docs/content_index.md#guided-projects) | Work based on supplied prompts, courses, competitions or tutorials |
| [Other project material](docs/content_index.md#exploratory-partial-and-unresolved-projects) | Early, partial, experimental or provenance-uncertain analyses |
| [Data manipulation](Python/data_manipulation/) | Pandas, cleaning, joins, dates, categorical data and preprocessing |
| [Machine learning](Python/machine_learning/) | Introductory, supervised, decision-tree and clustering material |
| [Mathematics](Python/mathematics/) | Mathematical foundations, reading notes and visual experiments |
| [NLP](Python/nlp/) | General NLP and spaCy study material |
| [Statistics](Python/statistics/) | EDA, experimental design, inference, regression and sampling |
| [Visualisation](Python/visualisation/) | Matplotlib and Seaborn notes and exercises |
| [Helpers](Python/helpers/) | Reusable statistical, encoding and plotting utilities |
| [Archive](archive/) | Study sessions, worksheets, refreshers and scratch practice retained for historical context |
| [Repository documentation](docs/) | Audit, provenance, reproducibility, review decisions and curation results |

The [content index](docs/content_index.md) is the main entry point. It gives
each listed item a category, status, provenance note, reproducibility
expectation and output policy. It intentionally gives more detail to selected
and guided projects than to individual supplied datasets.

## Selected work

These notebooks are relatively substantial or self-contained within this
archive:

| Notebook | Scope | Status |
| --- | --- | --- |
| [Ames Housing](Python/projects/ames_house/ames_house.ipynb) | Regression and model comparison using the Ames housing dataset | Substantial; saved model is regenerable |
| [Online Retail II](Python/projects/online_retail_ii/online_retail_ii.ipynb) | Customer-level feature construction and clustering | Substantial; large local workbook required |
| [Telco Customer Churn](Python/projects/telco_customer_churn/telco_customer_churn.ipynb) | Churn classification and model comparison | Substantial; provenance still needs review |
| [Vehicle Price Modelling](Python/projects/vehicles/vehicles.ipynb) | Used-vehicle price modelling with structured and text features | Substantial; Kaggle dataset |
| [Spotify Songs](Python/projects/spotify/spotify_dataset.ipynb) | Exploratory modelling with a TidyTuesday dataset | Substantial but mixed with study notes |

“Selected” means useful as a starting point for browsing. It does not assert
that a notebook is production-ready, fully reproducible or wholly independent.
The current repository does not provide enough evidence to label every
personal addition versus supplied scaffold precisely. No item is designated
an independent project unless that status can be documented.

## Projects, notes and exercises

- Projects apply analysis or modelling to a particular dataset or question.
  Some are guided and retain supplied instructions.
- Notes collect concepts, examples and code written while studying a topic.
- Exercises and study sessions are practice material. They may be short,
  course-ordered, interrupted or dependent on an external learning platform.
- Archived or unresolved material is preserved when deleting or relabelling it
  would require assumptions about provenance, value or intended use.

Known course and dataset origins are recorded in
[provenance](docs/provenance.md). Items needing a decision about attribution,
redistribution, naming or retention are listed in
[human review](docs/human_review.md).

## Environments, data and notebook outputs

The notebooks span multiple historical environments, mainly Python 3.12 and
3.13, and use a heterogeneous set of scientific libraries. There is no claim
that one root environment executes the entire archive. See
[reproducibility](docs/reproducibility.md) for observed dependencies,
project-specific caveats, missing components and suggested setup practices.

Small and context-specific datasets are generally stored beside the notebooks
that use them. Some notebooks instead fetch remote data, and some large local
datasets have unresolved redistribution or provenance questions. A tracked
dataset should not be assumed to be original to this repository.

Outputs are retained when they document a result, plot or model summary, or
when reproducing the environment may be difficult. Transient editor state,
local machine paths, error traces and generated caches are not treated as
analytical results. Not every notebook is expected to run unchanged.

## Repository status

The [repository audit](docs/repository-audit.md) records the pre-curation
state. The [curation report](docs/curation_report.md) records implemented
changes and validation. These documents distinguish current-tree cleanup from
optional future Git-history work; history is not rewritten as part of this
curation.

No repository-wide licence or permission statement is inferred from the
materials. Consult the recorded source information and unresolved provenance
notes before reusing third-party prompts, data or images.
