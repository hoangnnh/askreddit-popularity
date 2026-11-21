# IE212 Group 14 — AskReddit Popularity Prediction

> Current design: the eight-class experiment in [fang-chua-experiment.md](fang-chua-experiment.md) supersedes the binary-model plans in this historical snapshot. See run receipts for executed results.

Status snapshot taken on 2026-11-12 in the Asia/Saigon time zone. The active repository is `/home/hoangnnh/Coding/IE212/project/askreddit-popularity` in the Arch Linux WSL distribution. Run project commands from that repository root.

## Working with the user

- The user now wants implementation guidance and code execution in the same Codex turn. For an implementation request, inspect the current repository, briefly explain the approach, make the requested changes, run the relevant checks, and report what actually ran. Answer explanation-only requests directly.
- Use Vietnamese for explanations unless the user asks for English. Use English for reusable Codex prompts and repository documentation.
- Run bounded local verification as part of the work. For a long extraction or training job the user chooses to run, provide the exact WSL command and how to verify its result. If the user asks Codex to run a job through completion, follow through. Check for an existing run before launching another.
- Treat the status below as a dated snapshot. Inspect the current manifest, outputs, and source before claiming a stage is complete or rerunning it.

## Research question

Analyze how title characteristics and posting time relate to popularity on r/AskReddit, then predict high versus low popularity using the title and posting time available at publication. The outcome is the Reddit `score` recorded in the archive. The unit of analysis is one submission. Use `score` to construct the target and `selftext` for cleaning. Model inputs are derived from `title` and `created_utc`.

## Dataset and ingestion

- The corpus is Arctic Shift Reddit **submissions** published from January through December 2025. Filter by `created_utc` in the half-open UTC interval `[2025-01-01, 2026-01-01)` and by subreddit, case-insensitively. The archive was originally distributed as monthly dumps through Academic Torrents. Ingestion reads the corresponding `open-index/arctic` Parquet mirror directly with DuckDB `httpfs`. [Mirror dataset card](https://huggingface.co/datasets/open-index/arctic), [Arctic Shift dump catalog](https://github.com/ArthurHeitmann/arctic_shift/blob/master/download_links.md)
- The source URI pattern is `hf://datasets/open-index/arctic/data/submissions/2025/MM/*.parquet`. Keep `id`, `subreddit`, `title`, `selftext`, `created_utc` (Unix epoch seconds), and `score`. Store only the filtered AskReddit rows locally in `data/extracted/askreddit-2025-MM.parquet`. Measure corpus row count and size from the actual outputs after all 12 months finish.
- Ingestion supports a sample from `000.parquet`, a complete month, and a sequential full year. The sample lives under `data/samples/` and does not complete a month. `artifacts/ingest_manifest.json` records finalized months, source revisions, counts, file sizes, timestamps, checksums, and elapsed time. The CLI validates temporary output before promoting it to its final path. A rerun reuses a complete month only if the recorded source revision and the local validation, statistics, and checksum match. An interrupted month restarts as a month.

```bash
.venv/bin/python -m askreddit.ingest --month 2025-01 --sample
.venv/bin/python -m askreddit.ingest --month 2025-01
.venv/bin/python -m askreddit.ingest --year 2025
```

## Current repository and evidence

- The local Git repository is on `main` and had no commits at the time of inspection. `README.md`, `pyproject.toml`, `.gitignore`, `configs/project.toml`, `docs/data-source.md`, and `docs/pipeline.d2` are present. Python uses a `src/` layout and `.venv`. `pyproject.toml` requires Python 3.11 or newer, pins DuckDB 1.5.4, and defines `spark` (PySpark 4.3.0), `demo` (Streamlit), and `dev` (pytest) extras. Dependency declarations and actual runtime availability should be checked separately.
- `src/askreddit/ingest.py` and `tests/test_ingest.py` are present. The ingestion preflight previously ran `--help` successfully. The local one-shard January sample exists at `data/samples/askreddit-2025-01-sample.parquet` (67,853 bytes).
- The full-year CLI was running at the time of inspection. January and February were the only finalized monthly files observed. Their six column names, Parquet metadata row counts, and filesystem sizes were checked locally and matched the manifest. The figures below are before cleaning. Remaining months and annual totals are pending.

| UTC month | Rows | Bytes |
| --- | ---: | ---: |
| 2025-01 | 147,226 | 4,872,672 |
| 2025-02 | 134,902 | 4,569,196 |
| Total observed so far | 282,128 | 9,441,868 |

- `src/askreddit/clean.py` is already present. Source inspection confirmed a PySpark CLI with `--input` and `--output`, schema checks, filters, deterministic deduplication, and a JSON cleaning report. A fresh Spark execution and a finalized January cleaned Parquet directory were not observed during this status check. The README still describes PySpark processing as planned. Verify the current Spark environment and any tests or results added since this snapshot before editing the implementation.
- Chronological splitting, labels, features, model training, evaluation, benchmarking, and Streamlit are planned. The inspected `src/askreddit/` contained `ingest.py`, `clean.py`, and `__init__.py`.

## Next milestone — January cleaning pilot

1. Check that the WSL `.venv` can import PySpark and that a compatible JVM is available. Inspect any in-progress changes and local checks before running the pilot.
2. Use the finalized January file as an explicit input. The current cleaning CLI requires a new output directory. Read `_cleaning_report.json` in the output directory and verify the result with Spark or DuckDB.

```bash
.venv/bin/python -m askreddit.clean \
  --input data/extracted/askreddit-2025-01.parquet \
  --output data/samples/cleaned-2025-01
```

3. Verify the output schema, input/output counts, per-step exclusions, and distinct IDs. Review the share of non-null scores equal to 1, the median, and the 75th percentile. The pilot is for verification and exploratory analysis. Once all 12 extracts are finalized, pass all monthly files to one cleaning job so deduplication can span months. Put full-year cleaned data under `data/processed/`.

The current cleaner rejects unexpected subreddits and null or blank IDs. It drops null or whitespace-only titles and rows with trimmed `selftext` equal to `[removed]` or `[deleted]`. It keeps empty or null bodies for title-only posts. For conflicting duplicate IDs, it keeps the row with the earliest `created_utc`, then uses a stable hash and source-column values to break ties. It preserves the selected source row and original `score`. Missing scores are retained and reported. Median, p75, and the share with `score = 1` use non-null scores.

## Remaining pipeline

1. **Explore the clean data.** Inspect score quantiles, mass at `score = 1`, a score-frequency plot with a logarithmic count axis, and the title and posting-time distributions. Keep the January pilot separate from the final analysis.
2. **Split and label chronologically.** Assign older posts to train, later posts to validation, and the newest posts to test by `created_utc`. Record actual cutoffs in the configuration. Calculate the median and p75 on train only. The proposed rule is high at or above training p75 and low below the training median. Exclude the middle group and report that evaluating the extremes makes the task easier. Inspect ties and class counts before committing the rule. If quantile ties make the labels impractical, consider justified fixed cutoffs or a regression formulation with a score transformation valid for the observed range. Handle missing scores explicitly before creating labels.
3. **Build structured features.** Use title length in characters and words, UTC hour and day of week, and interpretable question types such as `what`, `why`, `how`, and `if you could`. Consider one-hot or cyclical hour/day encoding. U.S. Eastern time is optional and needs an explicit time zone if included.
4. **Build text features.** Normalize title case and punctuation, tokenize with `RegexTokenizer`, and generate unigrams, bigrams, and trigrams. Keep stop words in the default version to preserve phrases such as `if you could`. Compare with an English `StopWordsRemover` variant. Use `CountVectorizer` with `vocabSize` and `minDF`, then IDF. Keep the vocabulary so model terms can be mapped back to n-grams. Fit vocabulary, IDF, and feature selectors on train only.
5. **Train and evaluate.** Fit L2 Logistic Regression on TF-IDF and structured features. Compare it with GBT or Random Forest using top-k n-gram features selected with `ChiSqSelector` plus structured features. Tune on chronological validation and evaluate on held-out test. Use a majority-class baseline. Report ROC-AUC, precision, recall, F1, and the confusion matrix with high popularity as the positive class. Interpret LR coefficients, tree feature importance, and posting-time effects.
6. **Benchmark and serve.** Compare Spark with single-machine DuckDB using the same inputs and equivalent operations. Build Streamlit views for trends and predictions. Export the LR vocabulary, IDF weights, coefficients, intercept, feature order, preprocessing rules, and decision threshold so the app can reproduce LR predictions without starting Spark. Compare exported probabilities with Spark on the same examples. Use the tree model for offline comparison. Save aggregates as Parquet and model inference metadata as an appropriate local artifact. Delta Lake is an optional extension.

## Project layout and expected outputs

`src/askreddit/` owns reusable pipeline code. `configs/` holds shared project settings. `docs/` holds dataset and pipeline specifications. `notebooks/` is for exploratory analysis, `benchmarks/` for comparative workloads, `app/` for Streamlit, `tests/` for focused checks, and `reports/` for figures and evaluation summaries. Keep large and generated files under Git-ignored `data/` and `artifacts/`.

The final deliverables are the batch ingestion and cleaning pipeline, cleaned Parquet, labeled chronological datasets, trained Logistic Regression and tree models, evaluation and feature analysis, a reproducible Spark versus DuckDB benchmark, and a Streamlit demo. Report these as completed only when their actual outputs and checks exist.
