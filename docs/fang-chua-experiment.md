# Eight class popularity experiment

The target is the submission score recorded in the archive. Score observation times
are not available in the six-field extracts: this is not a 24-hour popularity
target or a fully reconstructed historical forecast. Predictors use only title
and UTC posting time. Associations do not establish causal effects of wording
or posting time.

## Attribution and scope

The eight-level rule comes from [Fang, Cheng and Ostendorf (2016), section 3](https://aclanthology.org/W16-6209.pdf),
and is reused by [Chua (2021), section 4.2](https://aclanthology.org/2021.acl-srw.10.pdf).
Put scores at most 1 in level 0. Repeatedly take the median of the remaining upper
scores, put values strictly below it in levels 1 through 6, and retain the final
remainder as level 7. Chua's appendix loop bound conflicts with its prose; this
implementation follows the prose and the requested six adaptive splits.
Fang studies comments and uses discussion context, including later responses.
Those response features are not available at submission time and are excluded here.

The cumulative-threshold metric also follows those papers: compute binary F1 at
each threshold `label >= j`, `j=1..7`, and average seven scores. This is reported
as `mean_threshold_f1`, separately from ordinary eight-class macro-F1.

The following are **our project choices**, not claims about the papers: calendar
splits, train-only threshold estimation, exact even-sample median convention,
Unicode/text preprocessing, feature dimensions, train-only fitted transformations,
logistic regression, Spark Random Forest, weight smoothing, validation search,
the fixed eight-label evaluation universe, and the unseen-title evaluation slice.

## Fixed experimental design

All intervals are UTC and half open:

| Split | Inclusive start | Exclusive end |
| --- | --- | --- |
| Train | 2025-01-01 | 2025-10-01 |
| Validation | 2025-10-01 | 2025-12-01 |
| Test | 2025-12-01 | 2026-01-01 |

Validate physical file schemas, required fields, timestamp coverage and global ID
uniqueness before modeling. Global ID uniqueness also proves within/across-split
ID disjointness. Repeated normalized titles are counted for each split pair, kept
in all splits, and flagged in test for evaluation on titles unseen in training.

Thresholds use TRAIN scores only. Spark first aggregates exact score frequencies.
An ordered cumulative-frequency window locates the two central observations;
only their values and a count reach Python. The ordinary median is their exact
rational average. Integer score cutoffs are `ceil(median)`; tied scores always
receive the same label. Empty children and non-increasing cutoffs fail the run.
Freeze the mapping for validation/test. Independently recompute the medians and
label counts with the already-installed DuckDB, without extensions or downloads.
The historical boundary list is diagnostic only, never the fitting procedure.

Preserve original titles. Stateless preprocessing uses NFKC, Unicode whitespace
collapse, and lowercase. The word tokenizer retains Unicode letters/numbers and
internal apostrophes, including negation; no stopword removal or NLP downloads.
Word unigrams and bigrams share a train-fitted CountVectorizer vocabulary capped
at 12,000 terms, with train document frequency at least 10. Fit IDF on train and
L2-normalize each TF-IDF vector. Out-of-vocabulary held-out terms are ignored.
Character/word counts, question-mark count/presence, uppercase-to-letter ratio,
and cyclic UTC hour/weekday form nine structured features. Empty letter
denominators produce a zero uppercase ratio. Fit structured standard deviations
on train without centering. Time-only LR uses four bounded cyclic features.

The feature builder explicitly projects to ID, title and timestamp before deriving
any input. Labels are joined only after feature transformation. Neither score,
selftext, ID values, split membership nor title-overlap flags enter predictor vectors.
No StringIndexer is used: numeric labels retain their order from 0 to 7.

`configs/fang_chua_v1.json` declares ten candidates: majority; time-only LR;
text-only LR; six combined LRs crossing `alpha={0,0.5,1}` with
`regParam={0.03,0.1}`; and a compact RF. Multinomial L2 LR uses at most 40
iterations with tolerance 1e-5; iteration-limit status is reported. RF uses the
128 most frequent train vocabulary entries plus structured features, 16 trees,
depth 6, 16 bins, square-root feature subsampling and internal bagging rate 0.7.
Every estimator receives all training rows; internal RF bagging is not a reduced
experimental dataset. RF is an optional resource-dependent comparison; a failure
is recorded explicitly and never replaced by fabricated or smoke-run metrics.

Training weights are proportional to `(N_train / (8*n_k))**alpha`, normalized to
mean training-row weight 1. Evaluation uses natural frequencies and no training
weights. Choose highest validation macro-F1 rounded to 12 decimals, breaking ties
by candidate ID ascending. Freeze selection before predicting test. Score every
completed predefined candidate on test without revising the selected model.

Metrics use a fixed universe 0–7: macro-F1, mean class recall/balanced accuracy,
accuracy, support-weighted F1, all per-class metrics, raw and row-normalized
confusion matrices, and seven cumulative binary F1 scores. Zero denominators
contribute zero. Absent classes remain in the macro average. No Spark weighted
F1 metric is mislabeled as macro-F1. Only at most 64 confusion counts are collected.

## Run v2 (current): weighting, regularization and convergence fixes

Run v1 exposed three problems:

- Every candidate with `alpha` 0 or 0.5 predicted class 0 on all held-out rows. In this
  config `alpha` is the class-weight exponent, not an elastic-net mix.
- The time-only and text-only ablations were unweighted, so they were uninformative.
- The selected LR's objective fell only from 2.0794 (ln 8) to 2.0765 before the
  40-iteration cap. L2 `regParam` 0.03 is strong for unstandardized, L2-normalized
  TF-IDF inputs.

[`configs/fang_chua_v2.json`](../configs/fang_chua_v2.json) changes only these points:

- `alpha = 1` for every candidate.
- `regParam` ∈ {1e-2, 1e-3, 1e-4} for text-only and combined LR.
- `maxIter = 200`.
- A structured-only ablation (nine title/time features), exposed as `structured_features`.
- A weighted RF.

Labels, splits, vocabulary and feature definitions are identical. The thresholds again
fit to `[2, 3, 4, 6, 11, 46, 271]`, and DuckDB verification passed.

**Disclosure.** v2 was designed on 2026-11-25 after v1's validation *and* test metrics
had been viewed. Selection still uses validation macro-F1 only, but v2's test split is
not an untouched hold-out in the strict sense. The disclosure is stored in the config
and printed in the report.

Run `fang_chua_20261125_v2` completed all ten candidates in 35 min 52 s.

| Candidate | Val macro-F1 | Test macro-F1 | Test bal. acc. | Test accuracy | LR iterations |
| --- | ---: | ---: | ---: | ---: | --- |
| majority | 0.1037 | 0.1033 | 0.1250 | 70.46% | — |
| time_lr | 0.0319 | 0.0304 | 0.1374 | 2.12% | 3 |
| structured_lr | 0.0889 | 0.0827 | 0.1598 | 20.88% | 31 |
| text_lr_r1e2 / r1e3 / r1e4 | 0.1369 / 0.1389 / 0.1431 | 0.1285 / 0.1309 / 0.1333 | 0.2122 (r1e4) | 34.90% (r1e4) | 61 / 72 / 70 |
| combined_lr_r1e2 / r1e3 | 0.0793 / 0.1126 | 0.0741 / 0.1034 | — | — | 200 (cap) / 200 (cap) |
| **combined_lr_r1e4 (selected)** | **0.1433** | **0.1327** | **0.2161** | 33.29% | 158 (converged) |
| rf_weighted | 0.1048 | 0.0987 | 0.1769 | 26.20% | — |

The selected model's objective fell from 2.0794 to 1.9567 and converged before the cap.
On test it reached:

- weighted F1 0.4307 and mean threshold F1 0.1972;
- macro-F1 0.1337 on the 125,662 test posts whose titles never appear in train.

Per-class test recall ranges from 0.09 (class 3) to 0.47 (class 7), and precision on
classes 4–7 stays below 0.05.

Interpretation:

- **The title wording carries the predictive signal.** Text-only and combined LR are
  tied at about 0.143 validation macro-F1.
- **Title statistics and UTC posting time add essentially nothing beyond the words.**
  Time alone converged in three iterations and is worse than the majority baseline on
  macro-F1. Balanced class weights make a signal-free model scatter predictions into
  rare classes.
- **The gain is real but modest:** test macro-F1 0.1033 → 0.1327 (v1: 0.1135). The
  model trades accuracy (70.5% → 33.3%) for recall on rare classes. Eight-class
  discrimination of archived scores from titles alone remains hard.

A separate post-hoc analysis adjusts v1's saved models toward the training class prior
(logit adjustment, no retraining). See
[the v1 prior-adjustment report](../reports/experiments/fang_chua_20261121_v1/prior_adjustment/results_vi.md).
Its best validation choice reaches test macro-F1 0.1245, below v2's retrained model.

Full results: [Vietnamese report](../reports/experiments/fang_chua_20261125_v2/results_vi.md),
[model comparison](../reports/experiments/fang_chua_20261125_v2/model_comparison.csv).
Changing `features.py` (the new `structured_features` column) changes the stage
source hashes, so `--resume` of the completed v1 run is refused by design. Its saved
outputs are unaffected.

## Execution and reproducibility

Run `fang_chua_20261121_v1` completed all ten candidates on the full dataset in
937.23 seconds. Train/validation/test contain 1,299,855 / 253,025 / 134,105 posts.
Exact fitted medians are 3, 4, 6, 11, 46 and 271; the integer score boundaries
match the historical diagnostic. DuckDB independently verified the medians and
every split's label counts.

Validation selected `combined_a1_r003`, with macro-F1 0.1198. Its test macro-F1
is 0.1135, balanced accuracy 0.1889, accuracy 0.3579, weighted F1 0.4416 and
mean threshold F1 0.1851. Eight candidates predicted class 0 throughout both
held-out splits: majority, time-only LR, text-only LR, all four combined LRs
with alpha 0 or 0.5, and RF. Their test macro-F1 is 0.1033 and accuracy 0.7046.
Inverse-frequency weighting modestly improved macro-F1 at a large accuracy
cost. All six combined LRs reached the 40-iteration cap; convergence is not
established. The 125,662 test posts with titles unseen in train yielded selected
model macro-F1 0.1151. No further tuning was performed after inspecting test.

The [Vietnamese report](../reports/experiments/fang_chua_20261121_v1/results_vi.md)
contains full distributions, per-class results, figures and timings. The
[verification record](../reports/experiments/fang_chua_20261121_v1/verification.json)
confirms all 23 stage receipts, all 12 cleaned Parquet hashes, the 19 original
Parquet paths from the prior inventory, and all 16 focused tests. The complete
runtime log is saved as `execution.log` in the same report directory.

Use the existing environment and a fresh run ID; no installation or ingestion is
performed:

```bash
JAVA_HOME=/usr/lib/jvm/java-21-openjdk SPARK_LOCAL_IP=127.0.0.1 PYTHONPATH=src \
  .venv/bin/python -m pytest -q tests/test_experiment.py tests/test_experiment_resume.py

JAVA_HOME=/usr/lib/jvm/java-21-openjdk SPARK_LOCAL_IP=127.0.0.1 PYTHONPATH=src \
  .venv/bin/python -m askreddit.experiment \
  --config configs/fang_chua_v1.json --run-id fang_chua_reproduction

PYTHONPATH=src .venv/bin/python -m askreddit.report_results \
  --run-id fang_chua_reproduction
```

Choose a different run ID for a new experiment. The cleaner's source and output
Parquet are read-only. Each new run gets separate data, artifact and report
directories. Existing destinations are refused without explicit `--resume`.
Reuse verifies input SHA-256 fingerprints, complete configuration, Python version,
stage source-code hashes and every completed artifact's file hashes. An incomplete
artifact has no completion receipt and is never treated as a successful stage.
If a partial output directory prevents retry, choose a new run ID; do not overwrite it.

For staged execution, use `--stage prepare`, then `--resume --stage features`,
`--resume --stage train`, and `--resume --stage evaluate`. To train one registered
candidate use `--resume --stage train --candidate combined_a05_r003`. Evaluation
refuses to open test for prediction until all required candidates have completed
validation. The input count/label distribution checks necessarily inspect test
targets for data validation, never for fitting or selection.

The existing local Spark configuration uses two threads and a 2 GB heap. Persist
large frames on disk; spill into the run's artifact directory instead of the
host's RAM-backed `/tmp`. Fixed seed 212 controls RF. Floating-point reductions
and vocabulary tie ordering can vary between Spark environments; saved models,
the vocabulary, configuration and provenance define the executed experiment.

No required dependency was missing during inspection. SciPy is absent but is not
needed for these Spark JVM estimators. Python, PySpark, NumPy, DuckDB, pytest and
Matplotlib are already installed. If a future environment lacks them, stop that
stage and install manually; the pipeline contains no installer/download code.

The runtime reported that native `libblas.so.3` could not be loaded; Spark
continued fitting LR using its available fallback. Native BLAS is optional
acceleration, not a requirement for this completed path. On this Arch Linux host,
the optional manual command is `sudo pacman -S blas`; it was not executed.

## Outputs

For run ID `RUN`:

- `data/experiments/RUN/labeled/`: original logical fields, frozen labels, split,
  normalized title and train-title overlap flag, partitioned by split/year/month.
- `data/experiments/RUN/features/{train,validation,test}/`: fitted feature vectors
  with numeric labels and evaluation metadata, without score as a feature.
- `artifacts/experiments/RUN/preprocessing/`: fitted Spark PipelineModel containing
  vocabulary, IDF, scaling and vector assembly. Use `base_features` before applying it.
- `artifacts/experiments/RUN/models/`: saved candidate classifiers.
- `artifacts/experiments/RUN/selected/manifest.json`: selected classifier and fitted
  preprocessing references, feature metadata and frozen label mapping.
- `reports/experiments/RUN/`: input checks, class distributions, exact rational
  medians, independent verification, title overlap, preregistered search, timings,
  candidate metrics, frozen selection, test metrics, comparison CSV/JSON and
  confusion-matrix PNGs. `results_vi.md` reports actual results in Vietnamese.
- `receipts/`, `events.jsonl`, and `failures/`: verified stage completion and precise
  failure diagnostics. `completed.json` certifies a completed full-data experiment.

Tests exercise exact/weighted/even medians including large integers, degenerate
bins, half-open boundaries, train-only fitting, frozen tied-score labels, leakage
guards, train-only preprocessing, saved-model read-back, class-weight normalization,
metric definitions, deterministic selection and actual small-fixture LR/RF fitting.
Additional resume tests reject changed input/configuration, modified or missing
artifacts, and accidental reuse of an existing directory as a fresh run.

Timing definitions: `training_seconds` measures `estimator.fit` after loading and
checking the training frame. Feature transformation times include writing and
counting the persisted output. Prediction times include materializing the primary
confusion counts; the optional unseen-title slice is an additional aggregation.
