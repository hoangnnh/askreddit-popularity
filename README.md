    # AskReddit Popularity

IE212 project to predict **high or low popularity** for r/AskReddit posts from
their titles and posting times. Popularity labels will be derived from recorded
Reddit scores in the January–December 2025 submissions archive. The score
threshold and evaluation protocol are still to be defined; `score` will be used
to construct the target, never as an input feature.

## Current scope

DuckDB ingestion is implemented in [src/askreddit/ingest.py](src/askreddit/ingest.py).
It supports a single-shard sample, complete monthly extracts, and resumable
sequential extraction of a year. PySpark cleaning is implemented in
[src/askreddit/clean.py](src/askreddit/clean.py). Feature engineering, chronological
splitting, labeling, modeling, and the demo remain planned.

| Stage | Intended responsibility | Status |
| --- | --- | --- |
| DuckDB ingestion | Read remote Parquet shards, select AskReddit posts, and write one filtered local Parquet file per month | Implemented |
| PySpark cleaning | Validate community membership, remove invalid text records, deduplicate globally, and report descriptive statistics | Implemented |
| PySpark feature engineering | Derive title and UTC posting-time features | Planned |
| PySpark modeling | Define score-based labels, train classifiers, and evaluate high/low popularity predictions | Planned |
| DuckDB analysis | Query local Parquet files for exploratory analysis and reports | Planned |
| Streamlit demo | Present analysis and model predictions using the saved outputs | Planned |

The source is [open-index/arctic on Hugging Face](https://huggingface.co/datasets/open-index/arctic),
a Parquet distribution of the Arctic Shift archive. The pipeline design is remote
submissions → DuckDB monthly filtering → local Parquet → PySpark processing and
training → analysis and demo. See [the data specification](docs/data-source.md)
and [the D2 pipeline diagram](docs/pipeline.d2).

## Directory layout

```text
askreddit-popularity/
├── README.md
├── .gitignore
├── pyproject.toml              # Python package metadata and dependency groups
├── configs/
│   └── project.toml            # Source, UTC boundaries, columns, and local paths
├── docs/
│   ├── data-source.md          # Source and monthly output specification
│   └── pipeline.d2            # Planned pipeline diagram
├── src/
│   └── askreddit/
│       ├── __init__.py         # Importable package
│       ├── ingest.py           # DuckDB extraction, validation, and resume CLI
│       └── clean.py            # PySpark cleaning of explicit local inputs
├── notebooks/                 # Future exploration notebooks
│   └── .gitkeep
├── benchmarks/                # Future ingestion and processing benchmarks
│   └── .gitkeep
├── app/                       # Future Streamlit application
│   └── .gitkeep
├── tests/
│   ├── test_ingest.py          # Local ingestion integration and CLI tests
│   └── test_clean.py           # Small local fixtures for cleaning and statistics
└── reports/                   # Reports, figures, and evaluation summaries
    └── .gitkeep
```

`data/` and `artifacts/` contain generated local files and are ignored by Git.
Samples use `data/samples/askreddit-2025-MM-sample.parquet`; complete months use
`data/extracted/askreddit-2025-MM.parquet`, with `MM` from `01` to `12`.
The full-month manifest is `artifacts/ingest_manifest.json`. DuckDB extensions
are cached in `artifacts/duckdb_extensions/`. Cleaned data uses
`data/samples/cleaned-2025-01/` for the pilot and `data/processed/cleaned-2025/`
for the full year. Future saved models will live under `artifacts/`.
The `.gitkeep` files retain currently empty project directories on GitHub.

## Python package setup

The package requires Python 3.11 or newer and uses a `src/` package layout with
setuptools. From the repository root in WSL:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pytest -q
```

The base dependency is pinned to **DuckDB 1.5.4**. Ingestion installs and loads
the `httpfs` extension; first use needs network access to the DuckDB extension
server, and extraction needs access to Hugging Face and its file storage.
The connection uses the `curl` HTTP client, four DuckDB threads, a 2 GB
DuckDB memory limit, a 120-second HTTP timeout, and five HTTP retries. These
settings allow slower transfers more time after intermittent remote-read failures
were observed; validation and atomic output promotion remain mandatory.
Optional dependency groups are `spark` (PySpark), `demo` (Streamlit), and `dev`
(pytest). The Spark group pins PySpark 4.3.0. Cleaning runs locally with two
Spark worker threads, a 1 GB driver heap, and UTC session time. Java 21 is
available in this WSL environment at `/usr/lib/jvm/java-21-openjdk`; the cleaning
commands below select it explicitly without changing the system Java default.

[configs/project.toml](configs/project.toml) is loaded by the ingestion CLI.
Paths are relative to the repository root; the configured scope is 2025 in UTC.
The virtual environment, Python caches, build outputs, `data/`, and `artifacts/`
are excluded from Git.

## Ingestion commands

Run these commands from the repository root in WSL. Start with the January sample:

```bash
.venv/bin/python -m askreddit.ingest --month 2025-01 --sample
```

This reads only `hf://datasets/open-index/arctic/data/submissions/2025/01/000.parquet`
and writes `data/samples/askreddit-2025-01-sample.parquet`. It is a shard sample,
not a random sample or a complete month. A sample never marks a month complete
in the manifest, and running the sample command again refreshes it.

To extract all shards for one month:

```bash
.venv/bin/python -m askreddit.ingest --month 2025-01
```

This uses `read_parquet` on
`hf://datasets/open-index/arctic/data/submissions/2025/01/*.parquet` and writes
`data/extracted/askreddit-2025-01.parquet`.

To process January through December in order:

```bash
.venv/bin/python -m askreddit.ingest --year 2025
```

The year command stops on failure. Rerun it to reuse completed months whose
files and source identity still validate. Each file uses ZSTD compression and
contains `id`, `subreddit`, `title`, `selftext`, `created_utc`, and `score`, in
that order. Selection uses `lower(subreddit) = 'askreddit'` and each month's
half-open UTC timestamp interval. Source `score` and `selftext` values, including
nulls and deletion/removal markers, are preserved; records are not deduplicated.

Files are written to unique temporary paths in the output directory and read
back to validate column names/order/types, a positive row count matching the
copy count, subreddit, UTC bounds, and compression. Only validated files are
atomically moved to their final names. Ordinary failures and Ctrl-C clean up the
current temporary file; a hard-killed run may leave a temporary file, which is
never treated as a completed month. An existing final file survives a failed
replacement. A process lock prevents concurrent CLI runs from editing the
manifest together.

For every completed full month, `artifacts/ingest_manifest.json` records the
month, source path, verified source revision (or `null`), output path, row count,
distinct ID count, file size, UTC extrema, SHA-256 checksum, and elapsed seconds.
Revision verification checks the Hugging Face commit before and after the read;
a detected change aborts promotion. Resume requires a matching source path and
verified current commit, plus a fresh validation of the local file against its
manifest statistics and checksum. If the revision cannot be verified, extraction
can proceed, but the month cannot be reused on that basis. The manifest itself
is replaced atomically after each successful full month.

Each command prints a JSON result containing the observed counts and metadata.
The test suite uses local Parquet fixtures and does not scan remote full months.

## PySpark cleaning

Install the cleaning and test dependencies into the existing environment:

```bash
.venv/bin/python -m pip install -e '.[spark,dev]'
```

Run the **January pilot** yourself from the repository root:

```bash
JAVA_HOME=/usr/lib/jvm/java-21-openjdk \
.venv/bin/python -m askreddit.clean \
  --input data/extracted/askreddit-2025-01.parquet \
  --output data/samples/cleaned-2025-01/
```

Once all 12 monthly extracts are complete, run the **full year in one command**
so deduplication covers IDs across monthly files:

```bash
JAVA_HOME=/usr/lib/jvm/java-21-openjdk \
.venv/bin/python -m askreddit.clean \
  --input data/extracted/askreddit-2025-{01..12}.parquet \
  --output data/processed/cleaned-2025/
```

WSL Bash and Zsh expand the braces into exactly 12 file paths. A missing file
fails before Spark starts; this command cannot silently clean only the months
downloaded so far. The CLI also accepts one or more explicit local file paths
and quoted local file globs. Inputs must be Parquet files; remote URLs are
rejected. Paths are resolved from the working directory, and repeated references
to the same local file are read once.

Cleaning applies these rules in order:

1. Check every input's six-column schema. Reject the run if any row has a null
   subreddit or a community other than `askreddit` (case-insensitive). Null or
   whitespace-only IDs also fail verification rather than merging unrelated rows.
2. Drop null or whitespace-only titles. Whitespace includes tabs, newlines, and
   Unicode whitespace. Titles retained in the output keep their original text.
3. Drop rows whose whitespace-trimmed `selftext` is exactly `[removed]` or
   `[deleted]`. Empty, whitespace-only, and null bodies remain valid when the
   title is valid. A body merely mentioning either marker remains valid.
4. Deduplicate by `id` across the complete combined input, after text filtering.
   Keep the row with the earliest `created_utc`, with null timestamps last.
   For tied timestamps, keep the lexicographically smallest SHA-256 hash of
   the JSON struct containing all six original columns in column order, including
   nulls. This provides a stable choice independent of input order and Spark
   partitioning. Identical rows are interchangeable; a hash tie is resolved by
   the original column values in ascending order, nulls last. The rule selects
   one original row; it does not infer which conflicting record is authoritative.

The result retains exactly `id`, `subreddit`, `title`, `selftext`, `created_utc`,
and `score`, in that order and with their original types and values. In particular,
scores are never filled, averaged, clipped, relabeled, or normalized. Missing,
negative, and zero scores remain in retained rows.

The output is a **Spark Parquet directory**, with ZSTD-compressed part files and
`_cleaning_report.json`. The report is also printed as JSON. It includes total
input/output counts, sequential removal counts (each dropped row is counted at
its first removal step), the deduplication rule, input paths, and score statistics
both before and after cleaning:

- Missing and non-null score counts.
- Number and share of scores equal to 1; the share's denominator is the number
  of non-null scores in that stage.
- Exact median and 75th percentile, using linear interpolation over non-null
  scores. Shares and percentiles are `null` when no scores are present.

**January statistics are exploratory.** Neither the pilot nor the full-year
cleaning report defines popularity labels or thresholds. Those decisions belong
to the later chronological split stage; any data-derived cutoff must be fitted
using the training period only.

The cleaner refuses an existing output path. It writes to a temporary sibling
directory, checks the written schema and row count, and promotes the result only
after success. For a rerun, choose a new output directory or explicitly remove an
unneeded previous result yourself. Cleaning does not modify ingestion files or
the ingestion manifest.

Run the small local fixture tests with:

```bash
JAVA_HOME=/usr/lib/jvm/java-21-openjdk .venv/bin/python -m pytest -q
```
