# AskReddit Popularity

IE212 project to predict **eight popularity levels (0–7)** for r/AskReddit posts.
The project now includes data validation/cleaning and a chronological eight-class
experiment using the upper-remainder median labeling procedure of Fang et al.
(2016), reused by Chua (2021). `score` constructs the target and must never be a
predictive input. Features use only title and UTC posting time.

## Current scope

DuckDB ingestion is implemented in [src/askreddit/ingest.py](src/askreddit/ingest.py).
It supports a single-shard sample, complete monthly extracts, and resumable
sequential extraction of a year. PySpark validation and cleaning are implemented in
[src/askreddit/clean.py](src/askreddit/clean.py) and
[src/askreddit/quality.py](src/askreddit/quality.py). The reusable experiment entry
point is [src/askreddit/experiment.py](src/askreddit/experiment.py), with separate
labeling, feature, training and evaluation modules. Saved receipts distinguish
completed full-data stages from partial runs. The demo remains planned.

| Stage | Intended responsibility | Status |
| --- | --- | --- |
| DuckDB ingestion | Read remote Parquet shards, select AskReddit posts, and write one filtered local Parquet file per month | Implemented |
| PySpark validation and cleaning | Profile all local extracts, normalize valid records, quarantine rejected/duplicate rows, and verify read-back invariants | Implemented; execution results are recorded under `reports/data_quality/` |
| PySpark feature engineering | Train-fitted unigram/bigram TF-IDF and title/UTC posting-time features | Executed on all 1,686,985 cleaned posts; run `fang_chua_20261121_v1` |
| PySpark modeling | Train-only median labels, chronological LR/RF comparisons, validation selection and held-out test evaluation | All ten candidates completed; see the [Vietnamese results report](reports/experiments/fang_chua_20261121_v1/results_vi.md) |
| DuckDB analysis | Inspect individual January post scores and their distribution in a local notebook | January notebook implemented; full-year analysis planned |
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
│   ├── data-validation-cleaning.md # Execution, cleaning, and audit policies
│   └── pipeline.d2            # Planned pipeline diagram
├── src/
│   └── askreddit/
│       ├── __init__.py         # Importable package
│       ├── ingest.py           # DuckDB extraction, validation, and resume CLI
│       ├── clean.py            # Reproducible validation/cleaning CLI
│       └── quality.py          # Distributed profiling and output invariants
├── notebooks/
│   ├── 01_data_validation_cleaning.ipynb
│   └── 01_january_2025_scores.ipynb # Existing historical exploration
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
are cached in `artifacts/duckdb_extensions/`. The older pilot remains at
`data/samples/cleaned-2025-01/`. The current cleaner
writes `data/processed/askreddit_clean/`, with rejected and duplicate rows in
`data/processed/askreddit_quarantine/` and JSON reports in `reports/data_quality/`.
Future saved models will live under `artifacts/`.
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
Optional dependency groups are `spark` (PySpark), `demo` (Streamlit), `dev`
(pytest), and `notebook` (JupyterLab, IPython kernel, pandas, and Matplotlib).
The Spark group pins PySpark 4.3.0. Cleaning runs locally with two
Spark worker threads, a 2 GB driver heap, and UTC session time. Java 21 is
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

## PySpark validation and cleaning

Use the existing local extracts and installed environment. The cleaner does not
install packages, download data, create labels, or run feature engineering.
See [the execution and policy guide](docs/data-validation-cleaning.md) for the
full schema, quarantine format, duplicate policy, tests, failure behavior, and
manual dependency commands for another machine.

Run from the repository root, choosing three **new** destinations:

```bash
JAVA_HOME=/usr/lib/jvm/java-21-openjdk SPARK_LOCAL_IP=127.0.0.1 PYTHONPATH=src \
  .venv/bin/python -m askreddit.clean \
  --input 'data/extracted/askreddit-2025-??.parquet' \
  --output data/processed/askreddit_clean_rerun \
  --quarantine data/processed/askreddit_quarantine_rerun \
  --reports-dir reports/data_quality_rerun \
  --timestamp-unit seconds \
  --min-created-utc 2005-01-01T00:00:00Z \
  --max-created-utc 2026-11-19T00:00:00Z
```

The inspected 12 finalized extracts contain 1,686,985 rows from
2025-01-01 00:00:07 through 2025-12-31 23:59:25 UTC. All six-field schemas agree.
The completed full run retained all 1,686,985 rows, with zero rejected or duplicate
records and all read-back invariants passing. It normalized 36,534 titles and
cleared 30,601 removed/deleted body markers while retaining those posts. See
[the completed run](reports/data_quality/run.json) and
[execution notes](reports/data_quality/execution_notes.json).
The filename glob intentionally excludes samples, the previous cleaned pilot,
and two invalid zero-byte temporary Parquet files; these are documented in
[the repository inspection](reports/repository_inspection.json). No original
Parquet file is modified or deleted.

Cleaning trims valid base-36 IDs, canonicalizes AskReddit membership, normalizes
title whitespace, and converts missing/removed/deleted selftext to empty strings.
Missing or invalid IDs, titles, timestamps, scores, and foreign communities go
to quarantine with original values and explicit reasons. Zero and negative scores
remain valid. Timestamps use a declared unit, explicit sanity bounds, and UTC.
Valid duplicates are resolved deterministically using earliest timestamp and a
stable content-hash tie-breaker; every losing record is audited. Scores are never
ranked highest-first, clipped, filled, or labeled.

The output preserves six logical fields and adds integer `year`/`month` partitions.
Read-back validation checks all invariants and reconciles input with cleaned plus
quarantined records. Before/after JSON profiles and a successful `run.json` certify
a completed run. Existing destinations and overlapping paths are rejected. Add
`--validate-only` to produce only the before profile in a fresh report directory.

Run tests and open the new validation notebook with the existing environment:

```bash
JAVA_HOME=/usr/lib/jvm/java-21-openjdk SPARK_LOCAL_IP=127.0.0.1 PYTHONPATH=src \
  .venv/bin/python -m pytest -q
JAVA_HOME=/usr/lib/jvm/java-21-openjdk SPARK_LOCAL_IP=127.0.0.1 \
  .venv/bin/python -m jupyterlab notebooks/01_data_validation_cleaning.ipynb
```

The notebook defaults to reviewing saved reports and verifying output read-back.
Its optional execution cell calls the reusable pipeline with configurable paths.
The older January score notebook below is historical exploration and is not part
of the current validation/cleaning workflow.

## Chronological eight-class experiment

See [the experimental design and execution guide](docs/fang-chua-experiment.md)
for scientific attribution, exact median semantics, leakage safeguards, the fixed
candidate search and output layout. The versioned configuration is
[configs/fang_chua_v1.json](configs/fang_chua_v1.json).

Run `fang_chua_20261121_v1` completed on the full dataset in about 15 minutes
37 seconds. Exact train-derived score boundaries are `[2, 3, 4, 6, 11, 46, 271]`.
Validation selected `combined_a1_r003` (inverse-frequency weighting, L2 penalty
0.03): validation macro-F1 0.1198; test macro-F1 0.1135, balanced accuracy
0.1889 and accuracy 35.79%. The majority baseline has test macro-F1 0.1033 and
accuracy 70.46%. All six combined LR candidates reached the 40-iteration cap;
their convergence is not established, and eight-class discrimination remains weak.

See the [full Vietnamese report](reports/experiments/fang_chua_20261121_v1/results_vi.md),
[model comparison CSV](reports/experiments/fang_chua_20261121_v1/model_comparison.csv)
and [final verification](reports/experiments/fang_chua_20261121_v1/verification.json).
All 16 focused tests passed; all 23 stage receipts and unchanged input hashes
were verified. Repeat with a new run ID:

```bash
JAVA_HOME=/usr/lib/jvm/java-21-openjdk SPARK_LOCAL_IP=127.0.0.1 PYTHONPATH=src \
  .venv/bin/python -m askreddit.experiment \
  --config configs/fang_chua_v1.json --run-id fang_chua_reproduction

PYTHONPATH=src .venv/bin/python -m askreddit.report_results \
  --run-id fang_chua_reproduction
```

A fresh run writes under `data/experiments/`, `artifacts/experiments/`, and
`reports/experiments/`, each in its own run directory. Existing cleaned Parquet
remains unchanged. Use a new run ID to repeat the experiment, or `--resume` to
reuse only artifacts whose input/configuration/code and saved file hashes verify.
Use `--stage prepare|features|train|evaluate` for individual stages; test prediction
is gated on completed validation and a frozen selection record.

The target is archived score, without an assumed 24-hour observation horizon.
Ordinary eight-class `macro_f1` and Fang-style `mean_threshold_f1` are distinct
metrics. Training weights are never used for primary evaluation metrics.

## Inspect January 2025 post scores

Open [the January score notebook](notebooks/01_january_2025_scores.ipynb) to read
individual post IDs, titles, archived scores, and UTC posting times. It defaults
to the complete January extract at `data/extracted/askreddit-2025-01.parquet`.
Set `DATASET = "cleaned"` and rerun all cells to inspect the cleaning pilot at
`data/samples/cleaned-2025-01/` instead. The selected source must already exist;
the notebook reads local files without starting Spark or downloading data.

From the repository root in WSL:

```bash
.venv/bin/python -m pip install -e '.[notebook]'
.venv/bin/python -m jupyterlab notebooks/01_january_2025_scores.ipynb
```

Open the local URL printed by JupyterLab if WSL does not open a browser. Choose
the Python kernel from the project environment, then **Run → Run All Cells**.
To stop the server, press Ctrl-C in its terminal and confirm shutdown if asked.

Alternatively, open the repository in VS Code connected to WSL. Install/enable
its Python and Jupyter extensions, open the notebook, and use **Select Kernel →
Python Environments** to choose this repository's `.venv/bin/python`. Click
**Run All**. The notebook locates the repository from either the root directory
or `notebooks/`; a global kernel may not have the required packages.

The browsing controls support exact post IDs, literal title search, inclusive
minimum/maximum scores, missing scores, chronological or score sorting, and
pagination. For example, set `MIN_SCORE = 100` for scores of at least 100, or
set both `MIN_SCORE` and `MAX_SCORE` to `1` to inspect score-1 posts. Increase
`PAGE` to continue reading. Only rerun the browsing controls cell after editing
those filters; rerun all cells after changing `DATASET`.

The notebook also displays descriptive statistics, exact score frequencies,
and plots with a logarithmic count axis. It saves a plot to
`artifacts/exploration/2025-01/{raw,cleaned}-score-frequency.png`. Set
`EXPORT_CSV = True` in the optional export cell to save **all** posts from the
selected source, independently of browsing filters, to
`artifacts/exploration/2025-01/{raw,cleaned}-post-scores.csv`. Repeated runs
replace generated files for that source. Keep the source notebook's outputs
cleared before committing it; large outputs belong in Git-ignored `artifacts/`.

These are archived scores, not live Reddit scores or raw upvote counts. January
statistics are exploratory; they do not set the final popularity thresholds,
which must be estimated from the chronological training partition only.
