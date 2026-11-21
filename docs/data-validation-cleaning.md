# Data validation and cleaning

This phase reads existing local extracts with PySpark. It does not invoke ingestion,
install dependencies, access remote data, create labels, engineer predictive features,
or train models. The future task is classification into eight popularity levels (0–7),
using Recursive Median Binning later. `score` is reserved for target construction and
must never become a predictive input.

## Verified production results — 2026-11-19

The complete Spark run retained **1,686,985 of 1,686,985 records (100%)**.
There were **zero rejected records, duplicate IDs, or duplicate losers**. All
required null checks and read-back invariants passed. Actual coverage is
2025-01-01 00:00:07 through 2025-12-31 23:59:25 UTC; all 12 monthly extracts share
the same four-string/two-long schema. Scores range from 0 to 114,515, including
311,166 zero scores. No negative scores occur in this source; synthetic tests
verify that the cleaner preserves them.

Whitespace was normalized in 36,534 titles. Exactly 22,662 `[removed]` and 7,939
`[deleted]` bodies became empty strings, preserving their posts. Twelve cleaned
Parquet part files were written, one per month, totaling about 54 MiB. The empty
quarantine has one readable schema-bearing Parquet file. SHA-256 checks verified
all 19 original Parquet paths unchanged, including samples and temporary files.

The completed run took about 138 seconds. Two preliminary runs exposed JVM heap
limits; their failure reports remain under `reports/data_quality_failed_*`.
Conflict analysis now narrows to repeated IDs first, and wide caches live on disk
with smaller input partitions and batches. The successful run uses a 2 GB JVM heap.
See `reports/data_quality/execution_notes.json` for the resolved issues and runtime
warnings, and `reports/test_results/` for test and notebook execution evidence.

## Environment and execution

Inspection found Arch Linux, Python 3.14.8 in `.venv`, PySpark 4.3.0, pytest 9.2.0,
and Java 21 alongside the system Java 25 default. Use the existing environment and
select Java 21 per command. No required dependency is missing. PyArrow and pandas
are not required by this pipeline. PySpark warns about the installed pandas 3.1.1;
the pipeline does not use pandas, pandas UDFs, or Arrow conversion.

From the repository root, run the local tests:

```bash
JAVA_HOME=/usr/lib/jvm/java-21-openjdk SPARK_LOCAL_IP=127.0.0.1 PYTHONPATH=src \
  .venv/bin/python -m pytest -q
```

The completed production run uses the following command (the three destinations
must not already exist):

```bash
JAVA_HOME=/usr/lib/jvm/java-21-openjdk SPARK_LOCAL_IP=127.0.0.1 PYTHONPATH=src \
  .venv/bin/python -m askreddit.clean \
  --input 'data/extracted/askreddit-2025-??.parquet' \
  --output data/processed/askreddit_clean \
  --quarantine data/processed/askreddit_quarantine \
  --reports-dir reports/data_quality \
  --timestamp-unit seconds \
  --min-created-utc 2005-01-01T00:00:00Z \
  --max-created-utc 2026-11-19T00:00:00Z
```

For an exact logical rerun after those outputs exist, use fresh destinations:

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

The input glob deliberately selects finalized monthly extracts. Initial inspection
found 19 Parquet paths: 12 finalized extracts, one sample, four old pilot part files,
and two zero-byte temporary files. The sample and pilot overlap the full extracts;
the temporary files cannot be read as Parquet. They are listed in
`reports/repository_inspection.json` and left untouched. They are not silently
skipped if explicitly supplied: the run fails and writes a per-file error report.
Input months are reported from data timestamps, not inferred from the filenames.

Use `--validate-only` to write only the before profile and schema report. Choose
a fresh `--reports-dir`; the output and quarantine arguments must still pass path
guards. Use `--help` for all arguments. CLI defaults use two local worker threads,
a 2 GB JVM driver heap, eight shuffle partitions, adaptive execution, and four write
partitions. Audit and output frames are cached on disk, with small input partitions
and cache batches to limit JVM heap pressure. `--driver-memory`,
`--shuffle-partitions`, and `--write-partitions` are configurable. These conservative
defaults also fit this host, which exposes about 8 GB RAM rather than 16 GB.

If preparing a different machine, manually install only missing dependencies;
the application never installs anything. For Arch Linux, these are manual commands,
not commands run by the cleaner:

```bash
# Only if Java 21 is absent:
sudo pacman -S jdk21-openjdk
# Only if a Python environment is absent:
python3 -m venv .venv
# Only if the pipeline/test dependencies are absent:
.venv/bin/python -m pip install 'pyspark==4.3.0' pytest
# Only if running the existing ingestion tests as well:
.venv/bin/python -m pip install 'duckdb==1.5.4'
```

## Schema and normalization policy

Every supplied file is inspected independently before full reading. Exactly the
six logical fields must exist, in any order. The four text fields must be strings.
Timestamp and score may be integral physical types or strings containing signed
whole numbers that fit a 64-bit integer. Unsupported schemas, corrupt files, extra
fields, and missing fields abort the run with file-level diagnostics. Supported
schema differences are reported and read in separate groups before unioning.
This avoids Spark silently choosing one incompatible Parquet physical schema.
Floating point, decimal, and native timestamp fields require an explicit reviewed
extension; the cleaner never rounds them or assumes their meaning.

- Subreddit: trim Unicode whitespace, match case-insensitively, accept an optional
  `r/` prefix, and standardize retained values to `AskReddit`. Quarantine other
  communities and missing values.
- ID: trim Unicode whitespace and require a nonempty lowercase base-36 string
  (`[a-z0-9]+`), matching the inspected archive IDs. Preserve the string; do not
  generate replacements, lowercase IDs, or strip `t3_` prefixes automatically.
- Title: quarantine null/empty/whitespace-only values; trim and collapse Unicode
  whitespace to one space. Preserve punctuation, emojis, and text content.
- Selftext: null, empty, and standalone `[removed]`/`[deleted]` markers become
  empty strings. Marker matching ignores surrounding whitespace and case. Trim
  other bodies while preserving internal whitespace and meaningful text. An empty
  body never causes rejection.
- Timestamp: the inspected archive uses Unix seconds. Default to seconds; a
  numeric value resembling milliseconds is quarantined under the configured sanity
  bounds, never automatically divided. `--timestamp-unit milliseconds` explicitly
  enables conversion only for values divisible by 1,000; fractional seconds are
  quarantined. Invalid syntax, overflow, missing values, and values outside the
  inclusive bounds are quarantined with original values. UTC is mandatory.
- Bounds: 2005-01-01 is a configurable conservative sanity floor, not an inferred
  dataset start. The default upper bound is run-start UTC. Supply the same upper
  bound on reruns for reproducibility. The executed run fixes it at
  2026-11-19T00:00:00Z. Actual coverage is measured independently.
- Score: preserve every valid signed 64-bit integer, including zero and negatives.
  Quarantine nulls, blank/nonnumeric/fractional representations, overflow, NaN, and
  infinity. No clipping, filling, transformation, score-based ranking, or labels.

The clean schema is `id`, `subreddit`, `title`, `selftext` as strings;
`created_utc`, `score` as long; and `year`, `month` as integers derived in UTC.

## Deterministic duplicate handling

Profile raw duplicate IDs before filtering. Compare all six original logical
fields, including nulls, and report conflicts in subreddit, title, selftext,
timestamp, and score. Null IDs are excluded from duplicate grouping and quarantined;
invalid non-null IDs remain visible in raw duplicate statistics.

Only otherwise valid rows compete for retention. Group by trimmed ID, then select
the earliest UTC timestamp, followed by the lexicographically smallest SHA-256 of
the normalized six-field JSON. Resolve hash ties using normalized JSON, original
JSON, and source path. This extends the earlier cleaner's timestamp/hash policy.
It is independent of input row ordering and Spark partitioning, and never directly
prefers the highest score. The choice is reproducible but does not claim that a
particular archive snapshot is authoritative.

All losing rows are quarantined. `duplicate_exact` denotes a group with identical
original six-field content; `duplicate_conflict` denotes a group with any differing
field. Mixed groups use the latter reason, while `exact_duplicate_excess` separately
counts repeated identical rows even within conflicting groups. Each duplicate audit
row carries conflict field names, rank, group size, and the SHA-256 of the retained
original JSON. Byte-identical copies from the same file are interchangeable.

## Outputs, audit, and failure behavior

- `data/processed/askreddit_clean/`: ZSTD Parquet partitioned by `year` and `month`.
  Rows are distributed by partition keys and sorted by ID within each partition;
  at most one million rows are written per file. Empty results retain a readable
  unpartitioned schema file with both partition columns.
- `data/processed/askreddit_quarantine/`: separate Parquet with all original fields
  as `raw_*` strings, `_raw_json` preserving original values/types, source filename,
  normalized fields, all reasons, exclusive primary reason, and duplicate audit
  information. `before_cleaning/schema.json` records original source schemas.
- `reports/data_quality/before_cleaning/schema.json`: per-file schema, size,
  SHA-256, conversion policy, and any errors.
- `reports/data_quality/before_cleaning/profile.json`: total and monthly/file counts,
  field null counts/percentages, invalid values, score signs/ranges, body markers,
  timestamp coverage, normalization counts, and duplicate/conflict statistics.
- `reports/data_quality/before_cleaning/duplicate_ids/`: distributed Parquet of
  every raw duplicate ID and field-variant counts, avoiding driver collection.
- `reports/data_quality/after_cleaning/profile.json`: read-back totals, monthly counts,
  rejection reasons, retained percentage, ranges, null/duplicate counts, and invariants.
- `reports/data_quality/run.json`: successful run metadata, fixed policy, paths,
  input integrity verification, warnings, runtime, and post-validation results.

Input files are never written. Symlinks are resolved before checking path overlap;
all output/report destinations must be absent and disjoint. Spark writes with
`errorifexists`, never overwrite. Reports are reserved with an exclusive directory
creation, and each JSON file is created exclusively. The same inputs and policy
produce the same logical records; Parquet filenames and run times may differ.

Read back both datasets and require unique IDs, valid community/title/time/score,
no required nulls, correct UTC partition values, audited quarantine rows, and count
reconciliation. Original file hashes are verified again before certifying success.
The successful `run.json` is the completion marker. A failed run records
`failure.json` when possible; any partial outputs remain uncertified for inspection.
Choose fresh paths after failure. Outputs are not promoted as a cross-directory
atomic transaction, and consumers should require the completion marker.

The count equation is `input = cleaned + quarantined`, where
`quarantined = rejected_invalid + duplicate_losers`. Invalid rows do not compete in
deduplication. A row can have multiple reasons, so all-reason counts can overlap.
Primary reasons are exclusive in this order: community, ID, title, timestamp, score.
Duplicate losers have one duplicate reason. Zero-input percentages and extrema
are JSON null rather than fabricated zeros.

## Notebook

`notebooks/01_data_validation_cleaning.ipynb` loads the saved JSON reports and reads
the resulting Parquet schema/counts without Pandas. It defaults to reviewing an
existing completed run. Set `RUN_PIPELINE = True` and select fresh paths to execute
the same reusable pipeline from the notebook. Start Jupyter with the same Java
selection as above and use the project Python kernel. It contains no EDA, labels,
features, or modeling. The older January exploration notebook is left unchanged.
