# Title deduplication and two-stage score prediction

This document covers three additions made on 2026-11-25: title-level deduplication,
a title-only binary run on the deduplicated data (v5), and a two-stage score estimate
used by the demo. Every model here uses **the title only**; posting time is not an
input (a time-only model remains as an analysis ablation in the binary runs).

## What the score measures

Arctic Shift archives each post when it appears and re-fetches it about **36 hours
later**; the dumps carry the re-fetched score
([Arctic Shift](https://github.com/ArthurHeitmann/arctic_shift)). The open-index
Parquet mirror has no `retrieved_on` column, so this horizon comes from the archive's
documentation rather than from the data. The target is therefore "score about 1.5
days after posting". AskReddit posts receive almost all of their votes within the
first day, so this is close to the final score. A one-week horizon would need new data
collected from the Reddit API.

## Why the same question gets 1 point or thousands

[`title_ceiling.py`](../src/askreddit/title_ceiling.py) measures this on the cleaned
data before deduplication
([title_ceiling.json](../reports/data_quality_dedup/title_ceiling.json)):

- 53,331 normalized titles were posted more than once (181,052 postings).
- 2,808 of them produced at least one posting with score ≥ 271. For **78.5%** of
  those, the median posting of the same title scored ≤ 2.
- Within titles posted at least 5 times, knowing the exact title explains at most
  **18.9%** of the variance of log(1 + score). This is an in-sample, optimistic bound.

Timing, the first few votes, who saw the post and moderator removal decide the rest.
This matches the literature on social influence and the limits of popularity
prediction (Salganik et al. 2006; Muchnik et al. 2013; Martin et al. 2016;
Lakkaraju et al. 2013). A title-only model can rank titles by their *chances*. It
cannot tell which repost of a question will go viral.

## Deduplication

[`dedup.py`](../src/askreddit/dedup.py) writes `data/processed/askreddit_dedup/`
with the cleaned schema
([report](../reports/data_quality_dedup/run.json)):

- The key is the NFKC + whitespace-collapsed + lowercased title, with curly quotes
  mapped to straight ones.
- Placeholder titles such as `[ Removed by moderator ]` are dropped (497).
- For each key it keeps the posting with the **highest score**. Ties go to the earliest
  `created_utc`, then the smallest `id`.
- The result is 1,686,985 → **1,558,760** posts, after dropping 127,728 non-best repeats.

The share of posts with score ≤ 1 is almost unchanged (71.3% → 71.1%). The label
becomes "the best score this question reached in 2025". Deduplication is global, so
no title spans two splits. Choosing the kept posting uses every posting's score,
including validation/test months; the configs disclose this.

## Binary run v5 (registered proposal task)

[configs/binary_v5.json](../configs/binary_v5.json) is v4 on the deduplicated data:
posts with score ≥ 2, low = score 2, high = score ≥ 4 (train median 3, p75 4), title
only. Validation selected `combined_title_lr_r1e5`. Test results:

- AUC-ROC **0.600** and PR-AUC 0.492, against a 0.401 positive rate.
- At probability 0.5: balanced accuracy 0.572, precision 0.474, recall 0.556.
- The best GBT (`gbt_d5`) reaches AUC 0.578. The time-only ablation reaches 0.516.

[Report](../reports/experiments/binary_20261125_v5/results_vi.md).

## Two-stage score estimate

**Stage 1: will the post get any engagement?**
[configs/engagement_v1.json](../configs/engagement_v1.json) classifies score ≥ 2 vs
score ≤ 1 on all deduplicated posts with fixed cuts (no middle group).

- Validation selected `combined_title_lr_r1e5`.
- Test AUC-ROC **0.727**, PR-AUC 0.470 (base rate 0.299), and balanced accuracy 0.675
  at probability 0.5.
- The score ≤ 1 group includes moderator-removed posts, so this stage largely learns
  r/AskReddit's rules. Personal-advice titles ("How do I ask my boss for a raise?")
  get ≈ 0%.
- Training used balanced class weights. The export adds the prior correction
  `log(w_low / w_high)` to the logit: on a 3,000-post sample the corrected mean
  probability is 0.292 against an observed rate of 0.287 (0.466 before correction).

[Report](../reports/experiments/engagement_20261125_v1/results_vi.md).

**Stage 2: if engaged, how high?**
[`score_regression.py`](../src/askreddit/score_regression.py)
([configs/score_v1.json](../configs/score_v1.json)) fits ridge regression of
log(1 + score) from the title on posts with score ≥ 2 (train 345,675, validation
68,097, test 36,679). Validation RMSE selected `combined_title_ridge_r1e5`. Test results:

| Metric (test) | Ridge | Mean baseline |
| --- | ---: | ---: |
| RMSE, log(1 + score) | 1.062 | 1.078 |
| R², log scale | 0.029 (validation 0.073) | 0.000 |
| Spearman (ranking) | 0.132 | 0 |
| Median absolute error, points | 1.52 | 1.88 |
| 80% interval coverage | 0.800 | — |
| AUC for score ≥ 4 vs score 2 | 0.582 | 0.5 |

The absolute fit is weak, as the ceiling analysis predicts. The **relative** signal is
useful. In the test decile with the highest predicted scores, 10.1% of posts reached
≥ 100 points, against 0.7% in the lowest decile (14×). The 80% interval comes from the
10th–90th percentiles of validation residuals (−0.67, +0.75 on the log scale) and
covers 80.0% of test posts.

## Demo

The prediction tab takes a title only and shows three outputs:

- the proposal's high/low group (v5, threshold 0.5);
- the calibrated probability of engagement (stage 1);
- if engaged, the expected score with an 80% interval and a relative rank among
  engaged posts (stage 2).

It also lists the terms that push the score up or down. All three exports are NumPy
reimplementations checked against Spark on 3,010 posts: the maximum difference is
≤ 9e-16 and no labels flip.

```bash
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk SPARK_LOCAL_IP=127.0.0.1 PYTHONPATH=src
.venv/bin/python -m askreddit.dedup
.venv/bin/python -m askreddit.title_ceiling
.venv/bin/python -m askreddit.binary_experiment --config configs/binary_v5.json --run-id binary_20261125_v5
.venv/bin/python -m askreddit.binary_experiment --config configs/engagement_v1.json --run-id engagement_20261125_v1
.venv/bin/python -m askreddit.score_regression --config configs/score_v1.json --run-id score_20261125_v1
.venv/bin/python -m askreddit.binary_report --run-id binary_20261125_v5
.venv/bin/python -m askreddit.binary_report --run-id engagement_20261125_v1
.venv/bin/python -m askreddit.export_binary_model --run-id binary_20261125_v5 --threshold-policy probability_0_5
.venv/bin/python -m askreddit.export_binary_model --run-id engagement_20261125_v1 --threshold-policy probability_0_5
.venv/bin/python -m askreddit.export_score_model --run-id score_20261125_v1
.venv/bin/streamlit run app/streamlit_app.py
```

**Disclosure.** These runs were designed after earlier runs' results, including
their December test scores, had been seen. Selection and thresholds use validation
only.
