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
| `binary_20261125_v5` | [binary_v5.json](../configs/binary_v5.json) | v4 on the title-deduplicated data ([score-prediction.md](score-prediction.md)) | `score = 2` | `score >= 4` | no (time-only ablation only) |

| **`viral_20261126_v1`** (current) | [viral_v1.json](../configs/viral_v1.json) | posts with `score >= 2`, deduplicated | `2 <= score <= 270` | `score >= 271` | no (time-only ablation only) |

**Why viral.** In v4/v5 "high" means score ≥ 4: about three quarters of those posts
have only 4–9 points, and telling 2 points from 4 is mostly luck. The viral run makes
"high" mean genuinely popular: score ≥ 271, the train-fitted level-7 boundary of the
eight-level run. Every other engaged post is "low", so no middle group is dropped. The
proposal allows a fixed threshold when scores are concentrated. See
[Viral run](#viral-run-current) below.

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

## Viral run (current)

[configs/viral_v1.json](../configs/viral_v1.json): posts with score ≥ 2 of the
title-deduplicated data. High = score ≥ 271 (about 2% of train), low = score 2–270,
title-only features.

**Imbalance handling, compared on validation.** Every candidate trains on TRAIN rows
only; validation and test keep the natural ~2% viral rate.

- *Balanced class weights*, n/(2 n_k), on all training rows.
- *Random undersampling of the low class* to 1:5, 1:10 or 1:20 (low : high). Every
  viral post is kept, and low posts are sampled with a fixed seed, stratified by
  label × month so every month keeps its share. Implemented in
  `binary_training.undersample` with Spark `sampleBy`.
- Both inflate the predicted odds of "viral". Each candidate stores the correction
  `prior_logit_offset = log(w_low · kept_low_fraction / w_high)`, and the evaluation
  reports raw and corrected mean probability and Brier score.

Selection uses **validation PR-AUC**, because ROC-AUC is insensitive to how rare the
viral class is. Each candidate's threshold maximizes validation F1 and is frozen
before test.

**Results** ([Vietnamese report](../reports/experiments/viral_20261126_v1/results_vi.md)). Test = December
2025, 36,679 engaged posts, 2.31% viral.

| Candidate | Imbalance handling | Train rows | Val PR-AUC | Test AUC | Test PR-AUC | Test P / R / F1 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| **`lr_us10_r1e5`** (selected) | undersample 1:10 | 78,747 | **0.144** | **0.761** | **0.086** | 0.137 / 0.172 / 0.153 |
| `lr_weighted_r1e5` | balanced weights | 345,675 | 0.137 | 0.760 | 0.086 | 0.118 / 0.251 / 0.161 |
| `lr_us5_r1e4` | undersample 1:5 | 42,942 | 0.138 | 0.743 | 0.077 | 0.106 / 0.202 / 0.139 |
| `lr_us10_r1e4` | undersample 1:10 | 78,747 | 0.130 | 0.730 | 0.070 | 0.120 / 0.126 / 0.123 |
| `lr_us20_r1e4` | undersample 1:20 | 150,387 | 0.115 | 0.709 | 0.060 | 0.098 / 0.086 / 0.092 |
| `gbt_us10` | undersample 1:10 | 78,747 | 0.112 | 0.651 | 0.058 | 0.132 / 0.092 / 0.108 |
| `gbt_weighted` | balanced weights | 345,675 | 0.104 | 0.645 | 0.056 | 0.097 / 0.116 / 0.105 |
| `title_structured_lr` | balanced weights | 345,675 | 0.045 | 0.555 | 0.028 | — |
| `time_lr` (ablation) | balanced weights | 345,675 | 0.029 | 0.541 | 0.025 | — |
| `majority` | — | — | 0.024 | 0.500 | 0.023 | 0 |

- **Undersampling 1:10 matches class weights** (test AUC 0.761 vs 0.760, PR-AUC 0.086 vs
  0.086) while training on 23% of the rows; GBT also trains faster (81 s vs 111 s).
- With the stronger penalty (1e-4), more negatives hurt (1:5 > 1:10 > 1:20). In an
  unweighted mean loss the rare class's share shrinks as negatives are added, so the
  same penalty suppresses its signal more. The weaker penalty (1e-5) removes the gap.
- **Prior correction works**: mean corrected probability 0.021 vs the observed 0.023
  on test (raw 0.082).
- PR-AUC drops from validation (0.144) to test (0.086): December's most popular posts
  react to new events (see `notebooks/02_top_posts_by_score.ipynb`) that train never saw.
- The title separates **truly viral** posts (AUC 0.76) far better than score 4 from
  score 2 (v5, AUC 0.60). LR beats GBT by about 0.11 AUC.
- The demo uses `lr_us10_r1e5` (best LR by validation PR-AUC, frozen max-F1 threshold
  0.318). Viral probability = P(engaged) × P(viral | engaged), both prior-corrected.
  The export matches Spark on 3,010 posts (maximum difference 5.6e-16).

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
