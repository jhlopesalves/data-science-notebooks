---
number: 22
title: Out-of-Core Analytics and Columnar Workflows
phase: Data Engineering for DS
bundles:
- bundle_scaling_data
project:
  title: '“NYC Taxi: From Raw to Features at Scale”'
  dataset: NYC TLC Trip Records (Parquet/CSV).
  dataset_links:
  - NYC TLC official portal; AWS Open Data registry; Google Cloud Marketplace listing
    (if you prefer BigQuery). (https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
  metrics:
  - Wall-clock, peak memory, and cost (if executed in the cloud); verify aggregates
    against small-n Pandas baseline.
  nuances:
  - Partitioning strategy by time and zone; avoiding task graph blow-ups; data type
    discipline (categoricals vs strings).
code_focus:
- Process tens of millions of rows with **Dask DataFrame**; map-partitions patterns;
  `persist()` vs `compute()`.
- Query large Parquet partitions with **DuckDB** from Python; pushdown filters; `read_parquet`,
  `read_csv_auto`, `FROM read_parquet(...)`.
- 'Optional Spark: reproduce one query in PySpark to understand API differences.'
- Ray or Dask comparison for parallel task graphs; when to stay single-node versus
  cluster.
math_stats:
- Throughput vs latency; cost models for IO-bound vs CPU-bound operators; effect of
  columnar storage on scan cost.
- 'Sampling error in big data: why “n is large” does not remove bias; design a robust
  sub-sampling strategy.'
docs:
- '[Dask DataFrame](https://docs.dask.org/en/stable/dataframe.html)'
- '[DuckDB Docs](https://duckdb.org/docs/)'
- '[PySpark DataFrame Guide](https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/dataframe.html)'
- '[Ray Data](https://docs.ray.io/en/latest/data/data.html)'
bibliography:
- Kleppmann — *Designing Data-Intensive Applications* (O’Reilly).
- McKinney — *Python for Data Analysis* (3e), chapters on performance and out-of-core
  strategies.
- Lakshmanan — *Data Science on the Google Cloud Platform* (O’Reilly), for mental
  models of columnar storage and Parquet.
---

## Summary

You will learn to stop pretending that everything fits in memory. The goal is pragmatic: build a pattern you can re-use for any “medium-big” dataset on a laptop or Colab, reserving clusters for when you truly need them. Mastery here pays off every time you meet a CSV with eight digits of rows.
## Project Description

Build an end-to-end notebook that: (1) lazily loads a year of trips; (2) computes rolling hourly aggregates by pickup zone; (3) materialises a features table in DuckDB; (4) benchmarks Dask vs DuckDB for a representative group-by; (5) exports training features to Parquet for downstream modelling.
## Code Focus

- Process tens of millions of rows with **Dask DataFrame**; map-partitions patterns; `persist()` vs `compute()`.
- Query large Parquet partitions with **DuckDB** from Python; pushdown filters; `read_parquet`, `read_csv_auto`, `FROM read_parquet(...)`.
- Optional Spark: reproduce one query in PySpark to understand API differences.
- Ray or Dask comparison for parallel task graphs; when to stay single-node versus cluster.
## Math & Stats

- Throughput vs latency; cost models for IO-bound vs CPU-bound operators; effect of columnar storage on scan cost.
- Sampling error in big data: why “n is large” does not remove bias; design a robust sub-sampling strategy.
## Docs

- [Dask DataFrame](https://docs.dask.org/en/stable/dataframe.html)
- [DuckDB Docs](https://duckdb.org/docs/)
- [PySpark DataFrame Guide](https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/dataframe.html)
- [Ray Data](https://docs.ray.io/en/latest/data/data.html)
## Bibliography

- Kleppmann — *Designing Data-Intensive Applications* (O’Reilly).
- McKinney — *Python for Data Analysis* (3e), chapters on performance and out-of-core strategies.
- Lakshmanan — *Data Science on the Google Cloud Platform* (O’Reilly), for mental models of columnar storage and Parquet.
