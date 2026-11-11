# Data source and output specification

## Source

Use the **submissions** in [open-index/arctic on Hugging Face](https://huggingface.co/datasets/open-index/arctic),
which distributes [Arctic Shift](https://github.com/ArthurHeitmann/arctic_shift)
Reddit archives as Parquet. The dataset card documents the submissions schema
and January–December 2025 partitions. Only r/AskReddit posts are in project scope.

The source configuration is in [configs/project.toml](../configs/project.toml).
It currently refers to the `main` revision. DuckDB 1.5.4 with `httpfs` performs
the extraction. The full-month manifest records the source pattern and a commit
ID only when the Hugging Face API verifies the same commit before and after the
read. If verification is unavailable, the revision is `null`; such a file is
not eligible for automatic reuse. A detected revision change aborts the extract.

## Monthly submissions paths

Dataset-relative paths follow `data/submissions/2025/MM/*.parquet`. Each monthly
partition contains multiple Parquet shards; all matching shards belong to that
month's input. The local outputs are:

| Month | Source pattern in `open-index/arctic` | Local output |
| --- | --- | --- |
| 2025-01 | `data/submissions/2025/01/*.parquet` | `data/extracted/askreddit-2025-01.parquet` |
| 2025-02 | `data/submissions/2025/02/*.parquet` | `data/extracted/askreddit-2025-02.parquet` |
| 2025-03 | `data/submissions/2025/03/*.parquet` | `data/extracted/askreddit-2025-03.parquet` |
| 2025-04 | `data/submissions/2025/04/*.parquet` | `data/extracted/askreddit-2025-04.parquet` |
| 2025-05 | `data/submissions/2025/05/*.parquet` | `data/extracted/askreddit-2025-05.parquet` |
| 2025-06 | `data/submissions/2025/06/*.parquet` | `data/extracted/askreddit-2025-06.parquet` |
| 2025-07 | `data/submissions/2025/07/*.parquet` | `data/extracted/askreddit-2025-07.parquet` |
| 2025-08 | `data/submissions/2025/08/*.parquet` | `data/extracted/askreddit-2025-08.parquet` |
| 2025-09 | `data/submissions/2025/09/*.parquet` | `data/extracted/askreddit-2025-09.parquet` |
| 2025-10 | `data/submissions/2025/10/*.parquet` | `data/extracted/askreddit-2025-10.parquet` |
| 2025-11 | `data/submissions/2025/11/*.parquet` | `data/extracted/askreddit-2025-11.parquet` |
| 2025-12 | `data/submissions/2025/12/*.parquet` | `data/extracted/askreddit-2025-12.parquet` |

Pass the monthly `hf://` pattern directly to DuckDB's `read_parquet`; `httpfs`
resolves the matching shards. The source pattern is:

```text
hf://datasets/open-index/arctic/data/submissions/YYYY/MM/*.parquet
```

DuckDB's [Hugging Face support](https://duckdb.org/docs/1.5/core_extensions/httpfs/hugging_face)
supports these globs. DuckDB reads remote shards, filters rows and selects
columns, then writes **one ZSTD-compressed local Parquet file per month**, for
12 outputs when the year command completes. Full source shards are not saved
locally. Sample mode replaces `*.parquet` with `000.parquet` and writes
`data/samples/askreddit-YYYY-MM-sample.parquet`; this is only one shard, not a
complete month or a random sample.

## UTC date range and selection

Select `lower(subreddit) = 'askreddit'`, retaining the source casing, and use the
half-open creation-time interval:

- Start, inclusive: **2025-01-01 00:00:00 UTC** (`1735689600` Unix seconds).
- End, exclusive: **2026-01-01 00:00:00 UTC** (`1767225600` Unix seconds).

Thus the year filter is `1735689600 <= created_utc < 1767225600`.
Each monthly output must additionally use its own UTC month boundaries, starting
at the first day of that month and ending just before the next month begins.
Apply the timestamp filter even when reading a matching source partition.
Posting-time features will also use UTC, independent of the WSL host timezone.

## Six output columns

Retain these columns in this order. The expected types follow the
[published submissions schema](https://huggingface.co/datasets/open-index/arctic#submissions-datasubmissionsyyyymmnnnparquet).

| Column | Expected DuckDB type | Meaning and intended use |
| --- | --- | --- |
| `id` | `VARCHAR` | Submission identifier; supports record identity and later deduplication. |
| `subreddit` | `VARCHAR` | Community name; matches `askreddit` case-insensitively, with original casing retained. |
| `title` | `VARCHAR` | Post title; source of text features. |
| `selftext` | `VARCHAR` | Post body; retained for inspection, outside the planned title/time feature set. |
| `created_utc` | `BIGINT` | Creation time as Unix seconds; source of UTC time features. |
| `score` | `BIGINT` | Archived Reddit score; used to derive the popularity target. |

The extractor keeps source values, including null scores, null bodies, and
deletion/removal markers, in the monthly extracts. Cleaning and missing-value
handling belong to the planned PySpark
stage. The high/low cutoff and split strategy remain undecided. Any cutoff
estimated from the score distribution must be fitted using training data only.

Recorded scores describe the archive snapshot, not a live or final lifetime
score. The model's intended inputs are titles and posting times; `score` is a
label source and must not enter the feature vector. Generated data belongs in
the Git-ignored `data/` directory.

## Validation and resume

Before promotion, each temporary output is read back to verify the exact schema,
a positive row count equal to DuckDB's copy count, the subreddit predicate, the
selected month's UTC interval, and ZSTD compression. Distinct IDs are reported;
duplicates are retained. Files are atomically renamed within the output directory.

Full-month metadata is stored in `artifacts/ingest_manifest.json` after each
successful month. Resume rechecks source identity, schema, row count, UTC bounds,
file size, and checksum. Samples do not create full-month manifest entries.
See the [README commands](../README.md#ingestion-commands) for usage.
