# Spark SQL versus DuckDB benchmark

Each `queries/*.sql` file is a workload. It runs **unchanged** on Spark SQL and on DuckDB
against a view named `posts` over `data/processed/askreddit_clean/`. The SQL deliberately
uses only constructs shared by both dialects. For example, UTC hour and weekday come
from integer arithmetic on `created_utc`, not from engine-specific date functions.

| Workload | Exercises |
| --- | --- |
| `q1_scan_summary` | Full scan with global aggregates |
| `q2_monthly_volume` | GROUP BY a partition column |
| `q3_weekday_hour` | Derived expressions, 168 groups |
| `q4_label_distribution` | Chronological split + frozen eight-class CASE labels |
| `q5_monthly_median` | Exact median per group |
| `q6_title_openers` | String functions over every title |
| `q7_repeated_titles` | High-cardinality string GROUP BY + top-k |
| `q8_join_monthly_median` | Aggregate, join back to every row, aggregate again |

The runner is [`src/askreddit/benchmark.py`](../src/askreddit/benchmark.py):

```bash
JAVA_HOME=/usr/lib/jvm/java-21-openjdk SPARK_LOCAL_IP=127.0.0.1 PYTHONPATH=src \
  .venv/bin/python -m askreddit.benchmark --threads 2 4 --run-id spark_vs_duckdb_YYYYMMDD
```

Method:

- Thread counts match in each configuration: Spark `local[N]` and DuckDB `threads=N`.
- Memory is capped at 2 GB on both engines.
- Neither engine caches data or results, so every run re-reads Parquet.
- One warm-up run comes before 5 timed runs. Reports give the median, min and max wall-clock time.
- Timings cover planning, execution and fetching the small result into Python. Engine startup is reported separately.
- The two engines' results must agree, with floats compared at a relative tolerance of 1e-9.
- `q4` is also checked against the experiment's independently computed class counts.

Outputs are written once to `reports/benchmark/RUN/`: `results.json`, `timings.csv`,
`figures/median_times.png` and `summary_vi.md`. All runs happen on one laptop with Spark
in local mode. The numbers describe single-machine behavior at this data size, not
cluster performance.

## Equivalence lesson

The first full run disagreed on `q6_title_openers`. Spark types the literal `1.0` as
`DECIMAL(2,1)`, so `avg(CASE … THEN 1.0 ELSE 0.0 END)` returned 5-decimal values
(0.20792). DuckDB returned a double (0.2079248…). Counts and `length(title)` agreed
exactly: both count Unicode code points, 128,059,879 in total. The workload now
averages integers, which yields a double on both engines. The fixture test uses 503 rows,
so group shares need more than five decimals, and the test would catch such a
regression. That superseded run was discarded and the benchmark re-run.
