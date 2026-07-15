# Human review

These decisions were not automated because provenance, educational value,
redistribution permission, dependency assumptions or intended retention could
not be established with high confidence. Preserving a file here is not an
endorsement; it is a conservative choice.

## Priority decisions

| Path | Issue | Why unchanged | Options and risks | Recommended human decision |
| --- | --- | --- | --- | --- |
| Python/projects/bike_sharing/data/Most-Recent-Cohorts-Institution.csv | 98.34 MiB file is not referenced by bike.ipynb and appears unrelated to Seoul bike sharing | Size and apparent mismatch are not enough to prove it is disposable | Keep: retains possible future work but dominates repository size. Move to a matching project: may restore context but requires knowing intent. Remove/download externally: reduces 98.34 MiB but may lose an unrecorded dataset snapshot | Identify the dataset’s intended notebook and source; remove or relocate it only after that check |
| Python/statistics/exploratory_data_analysis/eda_modules/data/olist/ and module_2.ipynb | Large Olist collection; notebook imports missing helpers.olist_analyzer | Data may still support planned analysis; helper history and source terms need review | Restore helper from history: may revive work but reintroduces uncertain code. Rewrite import: risks changing analysis. Remove data/module: loses substantial material. Keep: retains 58.44 MiB geolocation file and broken dependency | Confirm whether module 2 should remain active; if yes, review and restore the historical helper with attribution and tests |
| Python/projects/student_completion/ | 21.29 MiB dataset with a nine-cell, no-markdown notebook and no adequate source record | Incomplete work may still be personally valuable | Keep and document source; complete/archive notebook; replace data with download instructions; remove entire project after backup | Establish dataset source and intended value before deciding whether to archive or remove |
| Python/projects/book_recommender/ | No notebook markdown; source data, 6.09 MiB books_cleaned.csv and 2.49 MiB tagged_description.txt; vector-search notebook was removed historically | Derived files may be needed to resume the project; current index cache was safely removed | Keep as incomplete project; archive; regenerate derivatives on demand; restore/rebuild vector search; remove after external backup | Decide whether the recommender will be resumed, then document upstream dataset and regeneration steps |
| archive/study_sessions/data_reboot_airbnb_prices/ | External challenge scaffold, nbresult imports, tests and tiny pickle fixtures | Provider, redistribution terms and fixture role are unclear | Keep privately/publicly with attribution; remove tests/fixtures; retain notebook only; delete after personal backup | Identify the challenge provider and terms; decide whether tests/fixtures belong in a public archive |
| Python/visualisation/seaborn/intro_to_seaborn/introduction_to_seaborn.ipynb | air_quality_no2_long.csv URL returns 404 and no local file exists | No trustworthy replacement was found in the current tree | Add the original dataset with source/terms; replace URL with an authoritative source; mark lesson non-runnable; remove only the affected exercise | Find the original upstream dataset and add attribution before changing the notebook |
| Python/projects/telco_customer_churn/ | Selected analysis uses a widely circulated dataset but current notebook does not record the original source | Guessing a canonical source could misattribute it | Add verified original citation; replace tracked CSV with download instructions; keep with unresolved note; privatise/remove dataset | Verify the exact dataset copy and record its source/licence |
| Python/projects/womens_ecommerce/ | 8.09 MiB third-party review dataset has no adequate source/terms in the notebook | Common filename alone is insufficient provenance | Add verified source; replace with download instructions; retain privately; remove after backup | Verify dataset origin and redistribution conditions |

## Incomplete, near-empty or ambiguous notebooks

| Path | Issue | Why unchanged | Options and risks | Recommended human decision |
| --- | --- | --- | --- | --- |
| Python/machine_learning/supervised_learning/decision_trees/module_3/lecture_3-2.ipynb | Two cells; introduction only | May be the start of a planned lecture | Complete, merge into prior lecture, or archive; merging may disrupt course sequence | Keep until the intended module sequence is confirmed |
| Python/nlp/nlp_with_spacy/nlp_spacy.ipynb | Two cells and no executable content | Historical placeholder may still indicate planned study | Complete, merge into NLP notes, archive or remove | Archive/remove only after confirming no planned continuation |
| Python/statistics/regression_with_statsmodels/part2.ipynb | Two cells; multiple-regression heading only | Sequence suggests planned continuation | Complete, merge with part1, archive or remove | Preserve sequence until intent is known |
| Python/projects/eurovision_2016-25/ | Four-cell exploratory notebook, no markdown, ambiguous shortened year range | Dataset may be an active personal collection | Complete/document source; rename range; archive; remove after backup | Confirm dataset provenance and whether the range ends in 2025 |
| Python/projects/heart_disease/heart_disease.ipynb | Three code cells, no markdown, dynamic helper fetch | Purpose and relation to archived heart-disease scratch work are unclear | Expand into project; merge with archive exercise; archive; remove | Decide whether it represents distinct work |
| Python/projects/wisconsin_breast_cancer_plots/wbc_plots.ipynb | Template-like plot objectives and 1.79 MiB retained figures | Prompt origin and intended completion are unclear | Keep as visual experiment; rewrite only with source attribution; archive; remove after backup | Identify template origin and whether visible figures have continuing value |
| Python/projects/bike_sharing/bike.ipynb | A univariate-regression cell ends at `y_train_temp =`, and previous output showed a model feature-shape mismatch | Completing the cell or changing feature selection could alter the analytical method and conclusions | Diagnose and correct in a separate analytical task; mark partial; archive | Treat as a future project-specific fix, not repository curation |
| Python/projects/spotify/spotify_dataset.ipynb | Substantial analysis ends with starter-code-style markdown | Main analysis and study notes may have been combined | Split notebook, remove redundant prompt, or retain historical sequence | Review manually before restructuring |
| Python/projects/womens_ecommerce/womens.ipynb | Only basic EDA despite project placement | Could be unfinished rather than disposable | Complete, relabel as exploration, archive or remove after backup | Relabel/archive if no continuation is planned |

## Naming and structure

| Path | Issue | Why unchanged | Options and risks | Recommended human decision |
| --- | --- | --- | --- | --- |
| Python/ | Uppercase top-level name conflicts with lowercase snake_case policy | Moving 302 pre-curation files would create broad link/history churn for limited gain | Keep; rename to python; distribute material across notes/exercises/projects | Keep unless a larger repository migration is planned |
| module_*, lecture_*, exercises_*, part1/part2 and study_* paths | Generic names are numerous | Module order is useful and many standalone titles/provenance boundaries are uncertain | Rename from headings; keep ordered names; add per-directory indexes | Prefer indexes over mass renaming; rename only when a stable descriptive title is useful |
| Python/projects/eurovision_2016-25 | Mixed punctuation and ambiguous year abbreviation | Intended range/refresh policy is unknown | eurovision_2016_2025, eurovision_analysis, or archive | Decide after confirming dataset scope |
| Python/statistics/exploratory_data_analysis/eda_modules/module_3-tobit-model.ipynb | Hyphens and module numbering are inconsistent | The notebook may be a side study rather than module 3 replacement | Rename with snake_case, move beside regression notes, or preserve | Review relation to module_3.ipynb before moving |
| archive/study_sessions/oop_refresher/oop-refresher.ipynb | Directory was normalised but filename retains a hyphen | Renaming the file adds little clarity and may preserve the original exercise label | Rename to oop_refresher.ipynb or keep | Optional low-priority rename |

## Datasets, duplicates and generated outputs

| Path | Issue | Why unchanged | Options and risks | Recommended human decision |
| --- | --- | --- | --- | --- |
| joining_data_with_pandas/data/gdp-ffill.csv and gdp-query.csv | Exact duplicate bytes with distinct exercise names | Names may encode separate pedagogical steps | Deduplicate and update code; retain both; generate one from the other | Retain unless course semantics are reviewed |
| joining_data_with_pandas/data/pop-query.csv and pop.csv | Exact duplicate bytes with distinct exercise names | Same as above | Same as above | Retain unless course semantics are reviewed |
| experimental_design/data/salaries.csv and exploratory_data_analysis/datacamp/data/ds_salaries.csv | Exact duplicate data in different course contexts | Deduplication would couple unrelated exercises | Centralise, keep both, or document shared upstream source | Prefer source attribution over structural deduplication |
| Python/helpers/missing.py and eda_modules/helpers/missing.py | Near-identical implementations with different public function names | Merging could break imports and examples | Consolidate with compatibility alias; retain both; choose one API | Review usage and add tests before consolidation |
| Python/projects/titanic/titanic_submission.csv | Small generated submission is reproducible | It may be a documentary project result | Keep; remove and regenerate; archive outputs separately | Keep unless generated outputs are being standardised project by project |
| Python/projects/book_recommender/data/books_cleaned.csv and tagged_description.txt | Generated derivatives are reproducible from current exploration code | They may be the only convenient input for future recommender work | Keep; remove and regenerate; publish checksums/download recipe | Decide with the project’s future status |
| archive/study_sessions/paths/data/documents/output.txt and playground outputs | May be exercise-generated | Files are part of the path-handling lesson context | Keep as fixtures; regenerate; remove | Keep unless the exercise is redesigned |

## Provenance, privacy and redistribution

| Path/group | Issue | Why unchanged | Options and risks | Recommended human decision |
| --- | --- | --- | --- | --- |
| Guided-project images and prompt text | Likely supplied third-party material | Removing it would reduce context; legal status cannot be inferred | Add attribution; replace images; link rather than track; privatise | Review source-by-source, starting with prominent guided projects |
| Course datasets under Python/data_manipulation, Python/statistics and Python/visualisation | Upstream sources and redistribution terms are often absent | Course context is strong but exact origin varies | Add original sources; replace with download instructions; keep privately | Build a dataset manifest before public redistribution decisions |
| Olist sellers, joining customer/employee data and institutional cohorts | Files contain email-like contact fields | They appear to be public/supplied datasets, but privacy was not assessed row by row | Remove contact columns, replace with downloads, retain with source terms, or remove data | Verify source terms and whether contact fields are necessary |
| Repository history | Deleted books, databases, archives and multiple Chroma SQLite versions remain | This task prohibits history rewriting | Leave history; run historical secret/licence audit; perform separate coordinated cleanup | Audit history before any future public-release or size-reduction effort |

## Optional history cleanup

Current-tree deletion does not shrink existing Git objects. History still
contains multiple Chroma SQLite blobs, removed third-party book material,
earlier large datasets and prior model/data versions. A future cleanup may be
worthwhile for clone size, but only as a separate authorised project with:

- a full backup and object-size report;
- a historical secret/provenance scan;
- a documented keep/remove list;
- collaborator coordination;
- explicit acceptance that commit hashes change and a force-push may be
  required.

No history cleanup or force-push is part of this curation branch.
