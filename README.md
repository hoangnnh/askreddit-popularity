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
[src/askreddit/quality.py](src/askreddit/quality.py). The full-year EDA is
[src/askreddit/eda.py](src/askreddit/eda.py). The experiment entry point is
[src/askreddit/experiment.py](src/askreddit/experiment.py), with separate labeling,
feature, training and evaluation modules. Saved receipts distinguish completed
full-data stages from partial runs.

| Stage | Responsibility | Status |
| --- | --- | --- |
| DuckDB ingestion | Read remote Parquet shards, select AskReddit posts, write one local Parquet file per month | Completed |
| PySpark validation and cleaning | Profile extracts, normalize valid records, quarantine rejects/duplicates, verify read-back | Completed; see `reports/data_quality/` |
| PySpark EDA | Profile, volume, score distribution, title/time associations (train only), label readiness | Completed; see [docs/eda.md](docs/eda.md) and [reports/eda/summary_vi.md](reports/eda/summary_vi.md) |
| Labels and features | Train-only upper-median labels, chronological splits, train-fitted TF-IDF + title/UTC features | Completed in runs v1 and v2 |
| PySpark modeling | Weighted multinomial LR / RF comparisons, validation selection, held-out evaluation | v1 and v2 completed; see [docs/fang-chua-experiment.md](docs/fang-chua-experiment.md) |
| Spark vs DuckDB benchmark | Identical SQL workloads on both engines | Completed; see [benchmarks/README.md](benchmarks/README.md) |
| Streamlit demo | Analysis, evaluation and Spark-free prediction from saved outputs | Implemented in [app/streamlit_app.py](app/streamlit_app.py) |

The source is [open-index/arctic on Hugging Face](https://huggingface.co/datasets/open-index/arctic),
a Parquet distribution of the Arctic Shift archive. The pipeline is remote
submissions → DuckDB monthly filtering → local Parquet → PySpark cleaning, EDA,
labeling, features and training → exported model → Streamlit demo, with DuckDB
used for the engine benchmark. See [the data specification](docs/data-source.md)
and [the D2 pipeline diagram](docs/pipeline.d2).

## Directory layout

```text
askreddit-popularity/
├── README.md
├── pyproject.toml              # Package metadata and dependency groups
├── configs/
│   ├── project.toml            # Ingestion source, UTC boundaries, columns, paths
│   ├── fang_chua_v1.json       # Experiment v1 (historical)
│   ├── fang_chua_v2.json       # Experiment v2 (current) + disclosure
│   └── demo.json               # Report/model paths read by the Streamlit app
├── docs/                       # Data source, cleaning, EDA, experiment, pipeline diagram
├── src/askreddit/
│   ├── ingest.py               # DuckDB extraction, validation, resume CLI
│   ├── clean.py, quality.py    # PySpark validation/cleaning and invariants
│   ├── eda.py, eda_report.py   # Full-year EDA aggregates; figures + Vietnamese summary
│   ├── labeling.py             # Chronological splits, exact train-only median labels
│   ├── features.py             # Train-fitted TF-IDF and title/UTC features
│   ├── training.py, evaluation.py, experiment.py, report_results.py
│   ├── benchmark.py            # Spark SQL vs DuckDB runner
│   ├── export_model.py         # Spark LR → NumPy export with parity verification
│   └── inference.py            # Spark-free preprocessing + LR inference
├── benchmarks/queries/         # Shared SQL workloads for both engines
├── app/streamlit_app.py        # Demo
├── notebooks/                  # Validation notebook; historical January exploration
├── tests/                      # Focused tests with small local fixtures
└── reports/                    # Data quality, EDA, experiments, benchmark (JSON/CSV/PNG/MD)
```

`data/` and `artifacts/` contain generated local files and are ignored by Git.
Complete months use `data/extracted/askreddit-2025-MM.parquet`; the cleaned dataset
is `data/processed/askreddit_clean/`; experiment data, fitted preprocessing, models
and the exported demo model live under `data/experiments/`, `artifacts/experiments/`
and `artifacts/demo/`.

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

## Exploratory data analysis

The full-year EDA reads the cleaned dataset with PySpark (about 2.5 minutes). Popularity
associations use the train months only. See [docs/eda.md](docs/eda.md) for the method and
findings, and [reports/eda/summary_vi.md](reports/eda/summary_vi.md) for the Vietnamese
report with figures.

```bash
JAVA_HOME=/usr/lib/jvm/java-21-openjdk SPARK_LOCAL_IP=127.0.0.1 PYTHONPATH=src \
  .venv/bin/python -m askreddit.eda --config configs/fang_chua_v2.json --output reports/eda_rerun
PYTHONPATH=src .venv/bin/python -m askreddit.eda --render-only --output reports/eda   # figures only
```

## Spark SQL versus DuckDB benchmark

Eight shared SQL workloads run on both engines, with matched threads and no caching, and
the engines' results must agree. See [benchmarks/README.md](benchmarks/README.md) and
[the Vietnamese results](reports/benchmark/spark_vs_duckdb_20261125/summary_vi.md).
On this laptop (Spark local mode, 1.7 M rows), DuckDB was 4–25× faster per query.
Spark's fixed planning and scheduling costs dominate at this size. Both engines returned
identical results.

```bash
JAVA_HOME=/usr/lib/jvm/java-21-openjdk SPARK_LOCAL_IP=127.0.0.1 PYTHONPATH=src \
  .venv/bin/python -m askreddit.benchmark --threads 2 4 --run-id spark_vs_duckdb_new
```

## Streamlit demo

The app reads the saved EDA, experiment and benchmark reports listed in
[configs/demo.json](configs/demo.json). It predicts with a NumPy export of the selected LR,
so it never starts Spark. The export checks the NumPy pipeline against Spark on 3,000
validation/test posts plus Unicode edge cases, and refuses to finish on any mismatch. The
current export agrees on every prediction; the largest probability difference is 4.4e-16.

```bash
.venv/bin/python -m pip install -e '.[demo]'          # once: Streamlit
JAVA_HOME=/usr/lib/jvm/java-21-openjdk SPARK_LOCAL_IP=127.0.0.1 PYTHONPATH=src \
  .venv/bin/python -m askreddit.export_model --run-id fang_chua_20261125_v2   # once per selected model
.venv/bin/streamlit run app/streamlit_app.py
```

## Chronological eight-class experiment

See [the experimental design and execution guide](docs/fang-chua-experiment.md)
for scientific attribution, exact median semantics, leakage safeguards, the fixed
candidate search and output layout. The versioned configuration is
[configs/fang_chua_v1.json](configs/fang_chua_v1.json).

**Current run: `fang_chua_20261125_v2`** ([configs/fang_chua_v2.json](configs/fang_chua_v2.json)).
It uses balanced class weights everywhere, `regParam` 1e-2 to 1e-4, and up to 200
iterations, and it adds a structured-only ablation. Validation selected `combined_lr_r1e4`,
which converged in 158 iterations. Validation macro-F1 is 0.1433. On test, macro-F1 is
**0.1327**, balanced accuracy 0.2161 and accuracy 33.29%, against the majority baseline's
0.1033 and 70.46%. Text-only LR is tied (test 0.1333). Time-only (0.0304) and
structured-only (0.0827) models fall below the baseline. v2 was designed after v1's
test results were seen; this is disclosed in the config and in the
[v2 Vietnamese report](reports/experiments/fang_chua_20261125_v2/results_vi.md).

Historical run `fang_chua_20261121_v1` completed on the full dataset in about 15 minutes
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

### Post-hoc prior adjustment

[src/askreddit/prior_adjustment.py](src/askreddit/prior_adjustment.py) re-scores the
saved LR/RF candidates of a completed run without retraining them. It changes each
class score to `s_k(x) - tau * log(pi_k)`, where `pi` is the training label
distribution ([Menon et al., 2021](https://arxiv.org/abs/2007.07314)). LR uses its
margins and RF uses log probabilities. Each candidate chooses one `tau` from a grid
(default −1 to 2, step 0.05) by validation macro-F1. Test is scored only at `tau = 0`
and at the chosen value. At `tau = 0`, the validation and test confusion matrices must
equal the saved ones exactly. Every model and feature artifact is hash-checked against
its receipt first.

```bash
JAVA_HOME=/usr/lib/jvm/java-21-openjdk SPARK_LOCAL_IP=127.0.0.1 PYTHONPATH=src \
  .venv/bin/python -m askreddit.prior_adjustment --run-id fang_chua_20261121_v1
```

Results go to `reports/experiments/<run-id>/prior_adjustment/`. That folder holds
`plan.json`, which is written before scoring, plus `validation_sweep.json`,
`results.json`, `comparison.csv`, `results_vi.md` and confusion figures. The command
refuses to overwrite an existing output folder; use `--output-name` to pick a new one.
For v1, this analysis was designed after the v1 test metrics were seen, so its test
numbers are not a strict hold-out.

On v1, the best `tau` tracked each candidate's class weighting:

| v1 candidates | Class weighting | Best `tau` |
| --- | --- | ---: |
| unweighted | none | 0.85–1.0 |
| `combined_a05_*` | square-root | 0.5 |
| `combined_a1_*` | inverse-frequency | 0 |

The adjustment recovers the candidates that had predicted only class 0, without
retraining them. Validation then selected `combined_a05_r003` at `tau = 0.5`, with
test macro-F1 0.1245. That is above v1's original 0.1135 but below v2's retrained
`combined_lr_r1e4` (0.1327). Details:
[v1 prior-adjustment report](reports/experiments/fang_chua_20261121_v1/prior_adjustment/results_vi.md).

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
