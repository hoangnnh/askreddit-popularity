"""Deduplication key, regression metrics and Spark-free score/engagement inference."""

import math
import os
import sys

import numpy as np
import pytest

from askreddit.binary_inference import BinaryPopularityModel
from askreddit.features import STRUCTURED_COLUMNS
from askreddit.score_inference import ScoreModel
from askreddit.score_regression import average_ranks, pairwise_auc, regression_metrics


def test_regression_metrics_extremes():
    y = np.log1p(np.array([2, 3, 5, 10, 40, 300, 2, 4, 7, 9], dtype=float))
    perfect = regression_metrics(y, y.copy())
    assert perfect["rmse_log"] == 0 and perfect["r2_log"] == 1 and perfect["spearman"] == pytest.approx(1)
    constant = regression_metrics(y, np.full_like(y, y.mean()), residual_quantiles=(-10, 10))
    assert constant["spearman"] == 0 and constant["r2_log"] == pytest.approx(0) and constant["interval_coverage"] == 1
    assert sum(d["records"] for d in constant["deciles"]) == y.size


def test_ranks_and_auc_with_ties():
    assert average_ranks(np.array([3.0, 1.0, 3.0, 2.0])).tolist() == [3.5, 1.0, 3.5, 2.0]
    assert pairwise_auc([True, False, True, False], [0.2, 0.2, 0.2, 0.2]) == 0.5
    assert pairwise_auc([True, False], [0.9, 0.1]) == 1.0 and pairwise_auc([True, True], [0.1, 0.2]) is None


def toy_metadata(**extra):
    return {"features": "combined_title_features", "vocabulary": ["u:what", "u:you"], "structured_columns": STRUCTURED_COLUMNS,
            "feature_dimension": 7, "threshold": 0.5, **extra}


def toy_arrays(intercept):
    return {"coefficients": np.array([0.5, 1.0, 0, 0, 0, 0, 0]), "intercept": np.array(intercept),
            "idf": np.ones(2), "std": np.ones(9)}


def test_score_model_prediction_interval_and_rank():
    metadata = toy_metadata(residual_quantiles={"q10": -0.5, "q90": 1.5}, prediction_percentiles=list(np.linspace(0.5, 3.0, 101)),
                            population="posts with score >= 2")
    model = ScoreModel(metadata, toy_arrays(1.0))
    result = model.predict_score("What do you do?")
    margin = 1.0 + (0.5 + 1.0) / math.sqrt(2)  # L2-normalized counts of "what" and "you"
    assert result["log_prediction"] == pytest.approx(margin)
    assert result["score"] == pytest.approx(math.expm1(margin))
    low, high = result["interval"]
    assert low < result["score"] < high and 0 <= result["relative_percentile"] <= 100
    assert not model.uses_time


def test_prior_correction_undoes_balanced_weights():
    weights = [0.7, 1.75]  # n/(2 n_k) for a 71/29 split
    model = BinaryPopularityModel(toy_metadata(prior_logit_offset=math.log(weights[0] / weights[1])), toy_arrays(0.0))
    raw, calibrated = model.predict_proba("Hello there"), model.calibrated_probability("Hello there")
    assert raw == pytest.approx(0.5)
    assert calibrated == pytest.approx(weights[0] / (weights[0] + weights[1]))


def test_dedup_keeps_best_posting_per_title(tmp_path):
    from askreddit.clean import create_spark
    from askreddit.dedup import deduplicate
    os.environ["PYSPARK_PYTHON"] = sys.executable
    spark = create_spark("1g", 2)
    try:
        rows = [("a", "AskReddit", "Why are you single?", "", 10, 5, 2025, 1),
                ("b", "AskReddit", "why are  you single?", "", 20, 900, 2025, 1),
                ("c", "AskReddit", "Why are you single?", "", 5, 900, 2025, 1),
                ("d", "AskReddit", "What’s up?", "", 1, 3, 2025, 1),
                ("e", "AskReddit", "What's up?", "", 2, 1, 2025, 1),
                ("f", "AskReddit", "[ Removed by moderator ]", "", 3, 50, 2025, 1)]
        frame = spark.createDataFrame(rows, "id string, subreddit string, title string, selftext string, created_utc long, "
                                            "score long, year int, month int")
        ranked, keyed = deduplicate(frame)
        kept = {r.id for r in ranked.filter("title_rank = 1").collect()}
        assert kept == {"c", "d"}  # tie on 900 -> earliest created_utc; curly/straight apostrophes merge
        assert keyed.filter("is_placeholder").count() == 1
    finally:
        spark.stop()
