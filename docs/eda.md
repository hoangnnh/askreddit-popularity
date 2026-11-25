# Exploratory data analysis

[`src/askreddit/eda.py`](../src/askreddit/eda.py) computes the full-year EDA with PySpark
from `data/processed/askreddit_clean/`. Only small aggregate tables reach Python.
[`src/askreddit/eda_report.py`](../src/askreddit/eda_report.py) renders figures and a
Vietnamese summary from the saved JSON without starting Spark.

## Scope and leakage rules

- **Whole year (descriptive):** schema, counts, coverage, uniqueness, missing values,
  repeated titles, daily and monthly volume, score distribution, label counts per split.
- **Train only (2025-01-01 ≤ t < 2025-10-01 UTC):** every *association* between
  popularity and title characteristics, question patterns, posting time or correlations.
  Validation and test months therefore never inform modeling decisions.
- Label thresholds are fitted on train with the experiment's exact procedure
  (`labeling.fit_thresholds`). Monthly refits on train are a stability diagnostic only.
- Title statistics reuse the model feature definitions (`features.base_features`):
  NFKC plus whitespace collapse, the model tokenizer, question marks and uppercase ratio.
  Hour and weekday use UTC with Monday = 0.
- Quantiles are exact: the inverse empirical CDF computed from distributed frequency tables.
- Associations are descriptive. They do not control for topic, author or audience size,
  and they do not establish causal effects.

Popularity measures used throughout: share with score ≥ 2 (label ≥ 1), share in labels 6–7
(score ≥ 46, about the top 1.1%), mean label, and exact median score.

## Outputs

`reports/eda/` is written once (a new run needs a new directory):

| File | Content |
| --- | --- |
| `run.json` | Input SHA-256 fingerprints, configuration, versions, per-section timings |
| `profile.json` | Schema, counts, coverage, nulls, title-only posts, repeated normalized titles |
| `monthly_volume.json` | Monthly and daily volume, monthly score summaries |
| `score_distribution.json`, `score_frequencies.json` | Quantiles, zero/one concentration, upper tail, exact frequency tables per split |
| `title_characteristics.json` | Character/word length distributions, question marks, uppercase ratio, pattern shares |
| `posting_time.json` | UTC hour and weekday volume (year) and popularity (train), weekday × hour grid |
| `associations.json` | Popularity by length, question marks, body presence, uppercase ratio, 15 question patterns; Spearman/Pearson correlations |
| `label_feasibility.json` | Train thresholds, per-split class counts, non-empty checks, boundary ties, monthly stability, analytical majority baseline |
| `figures/*.png`, `summary_vi.md` | Eight figures and the Vietnamese summary |

## Commands

```bash
JAVA_HOME=/usr/lib/jvm/java-21-openjdk SPARK_LOCAL_IP=127.0.0.1 PYTHONPATH=src \
  .venv/bin/python -m askreddit.eda --config configs/fang_chua_v2.json --output reports/eda

# Re-render figures and summary from saved JSON (no Spark):
PYTHONPATH=src .venv/bin/python -m askreddit.eda --render-only --output reports/eda
```

## Findings (run of 2026-11-25, 1,686,985 posts, about 2.5 minutes)

The data:

- **Coverage.** There are 365 UTC days, averaging 4,622 posts per day (min 3,564, max 5,423).
  Volume peaks in June–July (about 5,000 per day), dips in October–November (about 4,150)
  and recovers in December. All IDs are unique, and no field has nulls.
- **Bodies.** Only 3 posts have a body: r/AskReddit does not allow body text. `selftext`
  carries no information, which supports the title-only design.
- **Repeated titles.** 173,707 posts (10.3%) share a normalized title with another post.
  The most frequent is "what should i do?" (520 posts).
- **Cleaning gap.** 486 posts have the title `[ Removed by moderator ]`, and 12 more are
  `[ Removed by Reddit ]`. These placeholders replace the original title and should be
  excluded by a future cleaning version. At 0.03% of posts they do not change these
  results.

Scores:

- **Concentration and tail.** 18.4% of posts have score 0, 52.8% have score 1, and 71.3%
  have score ≤ 1. Median 1, p90 3, p99 69, p99.9 4,093, maximum 114,515, mean 16.5.
  Accuracy is therefore uninformative, and thresholds must come from quantiles.
- **Stability.** The share of posts with score ≤ 1 is stable across splits: train 71.4%,
  validation 70.9%, test 70.5%. The monthly share with score ≥ 2 ranges from 25.4%
  (February) to 30.3% (June–July).

Titles (train associations; overall, 28.6% of train posts have score ≥ 2 and 1.13% fall
in labels 6–7):

- **Shape.** Median length is 66 characters and 12 words; 99.9% of titles contain "?".
- **Question marks.** Titles without "?" almost never reach score ≥ 2 (1.0% of 1,541
  posts). Two or more "?" give 17.9%, against 29.7% for exactly one.
- **Length.** Titles of 20–79 characters do best (31–33%). Titles of 200 or more
  characters reach 7.5%, and titles under 20 characters reach 12.8%.
- **Uppercase.** A higher uppercase ratio goes with lower popularity: 31.2% below 0.05,
  17.5% at 0.05–0.10, 1.7% at 0.50 or above.
- **Question patterns** (share with score ≥ 2, then lift over the train overall):

  | Pattern | Score ≥ 2 | Lift |
  | --- | ---: | ---: |
  | contains "if you could" | 42.7% | 1.49 |
  | starts with "what" | 37.2% | 1.30 (1.27 for labels 6–7) |
  | addresses the reader (you/your) | 34.8% | 1.22 |
  | starts with a yes/no verb | 12.7% | 0.44 |
  | starts with "why" | 6.6% | 0.23 |
  | first person (I/my/me) | 1.9% | 0.07 |

  NSFW mentions are rarely upvoted at all (13.4%), but they reach labels 6–7 3.1 times
  as often as average: they are polarizing.
- **Correlations.** Univariate correlations with score are weak and all negative:
  Spearman −0.066 for question marks to −0.097 for uppercase ratio.

**Interpretation (a hypothesis, not tested).** The most negative patterns are first person,
missing question marks, and "why" or yes/no openers. They match r/AskReddit rules against
personal-advice, non-question and loaded questions. Posts removed by automated or human
moderation keep scores of 0–1, so the archived score probably measures rule compliance as
well as audience interest. Removal status is not among the six retained fields, so this
cannot be verified here.

Posting time (UTC):

- **Volume** peaks at 16–20 UTC (87,649 posts at 17:00) and is lowest at 08:00 (51,110).
- **Popularity runs opposite to volume.** The share with score ≥ 2 is highest at 03–04 UTC
  (about 33%) and lowest at 12–13 UTC (25.6%). This fits less competition at quiet hours
  and is not a causal claim.
- **Weekdays are nearly flat:** 28.1–29.6%, with Saturday highest. This agrees with the
  experiment: a time-only model has no usable eight-class signal.

Label readiness:

- **Thresholds.** The train-fitted boundaries are `[2, 3, 4, 6, 11, 46, 271]`. All eight
  classes are non-empty in every split. Equal scores never split; 182,480 train posts sit
  exactly at boundary 2.
- **Fragility.** Label 1 exists only because 49.1% of train posts with score ≥ 2 have
  score exactly 2, just under 50%. In each of January–April that share is 50.1–52.8%, so
  a refit on one of those months alone makes label 1 empty, and the procedure correctly
  refuses. Refits on May–September give `[2, 3, 4, 6, 10–11, 34–46, 194–333]`. The lower
  boundaries are stable; the two upper boundaries drift with the fitting period.
- **Baseline.** Always predicting class 0 gives a macro-F1 of about 0.103–0.104 in every
  split (accuracy about 71%).
