---
number: 23
title: Analytical SQL + ORM + Feature Delivery
phase: Data Engineering for DS
bundles:
- bundle_scaling_data
- bundle_orchestration_tracking
project:
  title: “Pagila Analytics & Feature Store”
  dataset: Pagila (PostgreSQL sample); optionally join IMDB basics for enrichment
    exercises.
  dataset_links:
  - Pagila official repositories and Postgres mirrors; IMDB Non-Commercial Datasets
    (TSV). (https://github.com/devrimgunduz/pagila)
  metrics:
  - SQL correctness via unit tests (expectations on row counts and invariants); query
    latency; feature freshness.
  nuances:
  - Idempotent ETL; schema evolution; how to backfill without breaking model expectations.
code_focus:
- Write performant **SQL** with analytic/window functions; CTEs; materialised views
  for features.
- Use **SQLAlchemy 2.0** to manage connections and transactions; parameterised queries;
  ORM vs Core.
- Spin up **PostgreSQL** locally (Docker) and load a sample schema; index selection
  and EXPLAIN/ANALYZE basics.
- 'First contact with a **feature store** (Feast): define entities, feature views,
  and online/offline stores.'
math_stats:
- Correct computation of rolling metrics and leakage-free feature windows; pitfalls
  of late materialisation.
- Cardinality and join selectivity; understanding skew and its impact on group-by
  estimates.
docs:
- '[PostgreSQL Window Functions](https://www.postgresql.org/docs/current/functions-window.html)'
- '[SQLAlchemy 2.0 Tutorial](https://docs.sqlalchemy.org/en/20/tutorial/index.html)'
- '[dbt Documentation](https://docs.getdbt.com/docs/introduction)'
- '[DuckDB & Pandas Integration](https://duckdb.org/docs/guides/python/pandas.html)'
- '[Feast Documentation](https://docs.feast.dev/)'
bibliography:
- Vassilakis — *Mastering PostgreSQL in Application Development* (2e).
- Karau & Warren — *High Performance Spark* (for contrastive reading on joins and
  partitioning).
- Feast team whitepapers/blog posts on offline/online consistency.
---

## Summary

This week makes you fluent in the lingua franca of data. You will craft features where they belong—close to the data—then deliver them consistently to training and serving. This saves entire classes of bugs and makes downstream modelling honest.
## Project Description

Load Pagila; write SQL to compute customer recency/frequency/monetary (RFM), churn proxies, and rolling spend features with leakage-safe windows. Register 3–5 features in Feast, materialise to offline store, and run an online inference demo from Python.
## Code Focus

- Write performant **SQL** with analytic/window functions; CTEs; materialised views for features.
- Use **SQLAlchemy 2.0** to manage connections and transactions; parameterised queries; ORM vs Core.
- Spin up **PostgreSQL** locally (Docker) and load a sample schema; index selection and EXPLAIN/ANALYZE basics.
- First contact with a **feature store** (Feast): define entities, feature views, and online/offline stores.
## Math & Stats

- Correct computation of rolling metrics and leakage-free feature windows; pitfalls of late materialisation.
- Cardinality and join selectivity; understanding skew and its impact on group-by estimates.
## Docs

- [PostgreSQL Window Functions](https://www.postgresql.org/docs/current/functions-window.html)
- [SQLAlchemy 2.0 Tutorial](https://docs.sqlalchemy.org/en/20/tutorial/index.html)
- [dbt Documentation](https://docs.getdbt.com/docs/introduction)
- [DuckDB & Pandas Integration](https://duckdb.org/docs/guides/python/pandas.html)
- [Feast Documentation](https://docs.feast.dev/)
## Bibliography

- Vassilakis — *Mastering PostgreSQL in Application Development* (2e).
- Karau & Warren — *High Performance Spark* (for contrastive reading on joins and partitioning).
- Feast team whitepapers/blog posts on offline/online consistency.
