# Binary popularity experiment (registered proposal)

This experiment implements the task as written in the group's registered proposal:
predict whether an r/AskReddit question belongs to the **high** or **low**
popularity group from its title and UTC posting time. It runs alongside the
eight-class Fang/Chua experiment ([fang-chua-experiment.md](fang-chua-experiment.md))
and reuses its cleaned input, chronological splits and feature code. The target is
the archived `score`; it is never a predictive input.

## Runs

| Run | Config | Population | Low | High | Posting time in models |
| --- | --- | --- | --- | --- | --- |
| `binary_20261125_v3` | [binary_v3.json](../configs/binary_v3.json) | all posts | `score <= 1` | `score >= 3` | yes |
| `binary_20261125_v4` | [binary_v4.json](../configs/binary_v4.json) | posts with `score >= 2` | `score = 2` | `score >= 4` | no (time-only ablation only) |
| **`binary_20261125_v5`** (current) | [binary_v5.json](../configs/binary_v5.json) | v4 on the title-deduplicated data ([score-prediction.md](score-prediction.md)) | `score = 2` | `score >= 4` | no (time-only ablation only) |

**Why v4.** v3's strongest chi-square terms were `i`, `my`, `me`, `why` and `do i`:
personal-advice and non-question wording that r/AskReddit's rules forbid. Posts with
score ≤ 1 (71% of train) therefore mix posts that were removed by moderators with
posts readers ignored, and v3 largely learns "will this post be removed". v4 keeps
only posts with `score >= 2`, the upper remainder on which Fang et al. (2016) and
Chua (2021) compute their medians (they keep score ≤ 1 as their own level 0; here it
is excluded). v3 also showed that posting time alone is weak (validation AUC 0.546),
so v4 removes hour/weekday from every predictive model and keeps only a time-only
ablation to measure that signal. The v4 design used v3 validation results, not v3
test results.

## Labels (train-only quantiles)

Both quantiles are computed on the training split only, as the exact inverse
empirical CDF (the smallest train score `s` with `F(s) >= q`).

**v4** (train posts with score ≥ 2: 371,389 records; median 3, p75 4). Strict
`< median` gives 49.1% (closest to the nominal 50%) and `>= p75` gives 28.7%
(closest to the nominal 25%), the literal "below the median" / "top quartile".

| Group | Rule | Train records |
| --- | --- | ---: |
| Excluded | `score <= 1` | 928,466 |
| Low | `score < 3` (score = 2) | 182,480 (49.1% of population) |
| Middle (excluded) | `score = 3` | 82,444 (22.2%) |
| High | `score >= 4` | 106,465 (28.7%) |

**v3** (all train posts):

| Group | Rule | Train records |
| --- | --- | ---: |
| Low | `score <= median` (median = 1) | 928,466 (71.4%) |
| Middle (excluded) | `median < score <= p75` (score = 2) | 182,480 (14.0%) |
| High | `score > p75` (p75 = 2) | 188,909 (14.5%) |

For v3, the proposal asks for "top 25% vs below the median, middle dropped". Scores are so
heavily tied (71% of train posts have score ≤ 1) that an exact top-25% group cannot
be cut without splitting posts with equal scores. The strict `> p75` rule keeps the
high group above the quartile boundary. Dropping the middle group makes the task
easier than real use and is reported as a limitation. DuckDB recomputes both
quantiles and every split's group counts independently from the cleaned Parquet.

## Features

The same train-fitted features as the eight-class run v2, refitted on the kept
(low/high) training rows:

- NFKC-normalized lowercase title, `RegexTokenizer`, unigrams and bigrams **with stop
  words kept** (`what is the best`, `if you could`), `CountVectorizer`
  (vocabSize 12,000, minDF 10) + train IDF + L2 normalization.
- Title characters, words, question-mark count/presence, uppercase ratio, and UTC
  hour/weekday as sin/cos, scaled by train standard deviation.
- **Chi-square n-gram selection** for the tree models: MLlib `ChiSquareTest` on binary
  train term presence, top 128 terms by statistic. `ChiSqSelector` uses the same test
  but ranks by p-value; on more than a million rows many p-values underflow to 0 and
  tie, so ranking by the statistic (one degree of freedom per term) keeps the order
  meaningful. GBT input = the 128 selected TF-IDF columns + the 9 structured features.

## Models, selection and metrics

Predeclared v3 candidates ([configs/binary_v3.json](../configs/binary_v3.json)), all
trained on every low/high training row with balanced class weights. v4 uses the same
families without posting time: `title_structured_lr` (length, question marks,
uppercase), `text_lr_r1e5`, `combined_title_lr_r1e4/r1e5/r1e6` (TF-IDF + title
structure), and `gbt_d5`/`gbt_d7` on chi-square top-128 + title structure, plus the
`time_lr` ablation and the majority baseline.

| Candidate | Model | Features |
| --- | --- | --- |
| `majority` | always "low" | — |
| `time_lr` | L2 logistic regression | hour/weekday sin/cos |
| `structured_lr` | L2 logistic regression | length, question marks, uppercase, time |
| `text_lr_r1e4` | L2 logistic regression | TF-IDF only |
| `combined_lr_r1e2/r1e3/r1e4` | L2 logistic regression, regParam 1e-2/1e-3/1e-4 | TF-IDF + structured |
| `gbt_d5`, `gbt_d7` | Gradient-boosted trees, 60 iterations, depth 5/7 | chi-square top-128 + structured |

- **Selection:** highest validation AUC-ROC (October–November 2025), tie broken by ID.
- **Decision threshold:** per candidate, the probability threshold that maximizes
  validation F1 of the high class; frozen with the selection before test is scored.
- **Metrics (positive class = high):** exact AUC-ROC (Mann–Whitney with tied ranks),
  PR-AUC (average precision), precision, recall, F1, accuracy, balanced accuracy and
  the 2×2 confusion matrix, on validation and on the December 2025 test month. The
  report also gives the unseen-title slice, the majority baseline and the
  always-"high" F1 reference.
- **Analysis:** logistic-regression coefficients mapped back to n-grams, GBT feature
  importances, and the hour/weekday log-odds curves implied by the sin/cos
  coefficients.

**Disclosure.** The test month had already been scored by the eight-class runs v1
and v2, so v3 test numbers are not a strictly untouched hold-out. Selection and
thresholds use validation only.

## Commands

```bash
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk
.venv/bin/python -m askreddit.binary_experiment --config configs/binary_v4.json --run-id binary_20261125_v4
.venv/bin/python -m askreddit.binary_report --run-id binary_20261125_v4
```

Stages (`--stage prepare|features|train|evaluate|all`) write receipts and are
resumable with `--resume`. Each stage's receipt hashes only the modules it depends
on, so a later fix to evaluation code does not invalidate completed labels,
features or trained candidates. Run one heavy Spark job at a time on this laptop.

## Results

| Run | Selected | Test AUC-ROC | Test PR-AUC (base rate) | Balanced acc. @0.5 | P / R @0.5 | Best GBT AUC | Time-only AUC |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| v3 | `combined_lr_r1e4` | 0.724 | 0.323 (0.183) | 0.670 | 0.283 / 0.781 | 0.709 (`gbt_d7`) | 0.540 |
| v4 | `combined_title_lr_r1e5` | 0.594 | 0.479 (0.394) | 0.566 | 0.461 / 0.545 | 0.567 (`gbt_d5`) | 0.517 |
| **v5** (deduplicated) | `combined_title_lr_r1e5` | **0.600** | 0.492 (0.401) | 0.572 | 0.474 / 0.556 | 0.578 (`gbt_d5`) | 0.516 |

- v4's strongest positive terms are topical (`trump`, `nsfw`, `epstein`, `sex`,
  `compliment`, `instantly`, `depression`); negative ones include `or`, `vs`, `ai`,
  `prank`, `shoes`. v3's are rule-violation markers (`i`, `my`, `why`, `do i`).
- In v4 the validation max-F1 threshold (0.346) predicts "high" for 87% of low posts;
  F1 0.568 is barely above always-high (0.565). Classification is therefore read at
  probability 0.5, which the evaluation code records for every candidate. Choosing to
  emphasize 0.5 was decided after viewing v4 results.
- Linear models beat GBT in both runs: the title signal is spread across many sparse
  n-grams that the top-128 chi-square subset cannot capture.

Full Vietnamese reports: [v4](../reports/experiments/binary_20261125_v4/results_vi.md),
[v3](../reports/experiments/binary_20261125_v3/results_vi.md).

## Demo export

```bash
JAVA_HOME=/usr/lib/jvm/java-21-openjdk SPARK_LOCAL_IP=127.0.0.1 PYTHONPATH=src \
  .venv/bin/python -m askreddit.export_binary_model --run-id binary_20261125_v4 --threshold-policy probability_0_5
```

The export writes the best validation LR (`combined_title_lr_r1e5`) to
`artifacts/demo/binary_20261125_v4/`. It agrees with Spark on 3,010 posts (maximum
probability difference 2.2e-16, no label flips). The Streamlit app predicts from the
title alone with this export.

## Not yet done (proposal extensions)

Trigrams, the stop-word-removal ablation and US-Eastern posting time.
