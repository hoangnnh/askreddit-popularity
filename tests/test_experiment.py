"""Exact labels, chronological boundaries, leakage prevention and metric definitions."""

from copy import deepcopy
from fractions import Fraction
import json
import os
from pathlib import Path
import sys

import pytest
from pyspark.ml import PipelineModel
from pyspark.sql import functions as F

from askreddit.clean import create_spark
from askreddit.evaluation import evaluate_predictions, metrics_from_confusion
from askreddit.features import base_features, fit_preprocessing, normalize_unicode, validate_feature_sources
from askreddit.labeling import (apply_labels, assign_splits, class_distributions, class_weights,
                               exact_median, fit_thresholds, split_intervals)
from askreddit.training import choose_candidate, fit_candidate, timed_evaluation

CONFIG = json.loads((Path(__file__).resolve().parents[1] / "configs/fang_chua_v1.json").read_text())


@pytest.fixture(scope="module")
def spark():
    os.environ["PYSPARK_PYTHON"] = sys.executable
    session = create_spark("1g", 2)
    yield session
    session.stop()


def test_split_half_open_boundaries_and_disjoint_ids(spark):
    intervals = split_intervals(CONFIG["split_dates"])
    t0, t1 = intervals["train"]
    v1 = intervals["validation"][1]
    end = intervals["test"][1]
    frame = spark.createDataFrame([(str(i), t) for i, t in enumerate([t0 - 1, t0, t1 - 1, t1, v1 - 1, v1, end - 1, end])], "id string, created_utc long")
    rows = assign_splits(frame, CONFIG["split_dates"]).orderBy("id").collect()
    assert [r.split for r in rows] == [None, "train", "train", "validation", "validation", "test", "test", None]
    sets = {name: {r.id for r in rows if r.split == name} for name in intervals}
    assert not (sets["train"] & sets["validation"] or sets["train"] & sets["test"] or sets["validation"] & sets["test"])
    bad = deepcopy(CONFIG["split_dates"])
    bad["validation"][0] = bad["train"][0]
    with pytest.raises(ValueError):
        split_intervals(bad)


@pytest.mark.parametrize("rows,expected", [([(2, 1), (3, 1), (7, 2)], Fraction(5)),
    ([(2, 9), (100, 2)], Fraction(2)), ([(2**63 - 2, 1), (2**63 - 1, 1)], Fraction(2**64 - 3, 2))])
def test_exact_frequency_weighted_median(spark, rows, expected):
    histogram = spark.createDataFrame(rows, "score long, count long")
    median, count = exact_median(histogram)
    assert median == expected and count == sum(n for _, n in rows)


def test_train_only_fitting_frozen_mapping_and_equal_scores(spark):
    train = spark.createDataFrame([(v, "train") for v in range(130)], "score long, split string")
    model = fit_thresholds(train)
    assert model["effective_boundaries"] == [2, 66, 98, 114, 122, 126, 128]
    assert model["steps"][0]["median_exact"] == "131/2"
    rows = [(v, split) for split in ("train", "validation", "test") for v in [-1, 0, 1, 2, 65, 66, 97, 98, 127, 128, 129, 100000]]
    data = spark.createDataFrame(rows, "score long, split string")
    with pytest.raises(ValueError, match="TRAIN"):
        fit_thresholds(data)
    labeled = apply_labels(data, model)
    assert labeled.groupBy("score").agg(F.countDistinct("label").alias("n")).filter("n != 1").count() == 0
    actual = {r.score: r.label for r in labeled.collect()}
    assert actual == {-1: 0, 0: 0, 1: 0, 2: 1, 65: 1, 66: 2, 97: 2, 98: 3, 127: 6, 128: 7, 129: 7, 100000: 7}
    distribution = class_distributions(labeled)
    assert sum(d["total"] for d in distribution.values()) == len(rows)
    assert model["effective_boundaries"] == [2, 66, 98, 114, 122, 126, 128]


@pytest.mark.parametrize("scores", [[], [0, 1], [2] * 100, [2, 3]])
def test_degenerate_label_steps_fail(spark, scores):
    frame = spark.createDataFrame([(v, "train") for v in scores], "score long, split string")
    with pytest.raises(ValueError):
        fit_thresholds(frame)


def test_class_weights_train_mean_one():
    counts = [1000, 100, 50, 25, 12, 6, 3, 2]
    for alpha in (0, 0.5, 1):
        weights = class_weights(counts, alpha)
        assert sum(c * w for c, w in zip(counts, weights)) / sum(counts) == pytest.approx(1)
        assert weights[7] / weights[0] == pytest.approx((counts[0] / counts[7]) ** alpha)
    assert class_weights(counts, 0) == [1.0] * 8


def test_macro_and_threshold_f1_have_distinct_definitions(spark):
    matrix = [[0] * 8 for _ in range(8)]
    matrix[0][0], matrix[1][0], matrix[1][1], matrix[7][6], matrix[7][7] = 8, 1, 1, 1, 1
    metrics = metrics_from_confusion(matrix)
    assert metrics["accuracy"] == pytest.approx(10 / 12)
    assert metrics["macro_f1"] == pytest.approx((16/17 + 2/3 + 2/3) / 8)
    assert metrics["weighted_f1"] == pytest.approx(((16/17)*8 + (2/3)*2 + (2/3)*2) / 12)
    assert metrics["balanced_accuracy"] == pytest.approx((1 + 0.5 + 0.5) / 8)
    assert metrics["mean_threshold_f1"] == pytest.approx((6/7 + 5 + 2/3) / 7)
    predictions = [(i, j) for i in range(8) for j in range(8) for _ in range(matrix[i][j])]
    assert evaluate_predictions(spark.createDataFrame(predictions, "label int, prediction int")) == metrics
    assert metrics_from_confusion([[0] * 8 for _ in range(8)])["macro_f1"] == 0


def test_selection_uses_only_validation_and_deterministic_ties():
    results = [{"candidate": {"id": "b"}, "validation": {"macro_f1": 0.2}, "test": {"macro_f1": 1}},
               {"candidate": {"id": "a"}, "validation": {"macro_f1": 0.2}, "test": {"macro_f1": 0}}]
    assert choose_candidate(results) == "a"
    results[0]["validation"]["macro_f1"] = 0.21
    assert choose_candidate(results) == "b"


def test_unicode_features_and_leakage_guard(spark, tmp_path):
    assert normalize_unicode("  Ａ  cafe\u0301\n 😀? ") == "A café 😀?"
    with pytest.raises(ValueError, match="Only"):
        validate_feature_sources(["title", "created_utc", "score"])
    config = {**CONFIG["features"], "vocab_size": 64, "min_df": 1, "rf_text_features": 8}
    data = spark.createDataFrame([
        ("a", "DON'T stop now?", 1735689600, 0, "train", 0),
        ("b", "Why don't we stop?", 1735693200, 999, "train", 7),
        ("c", "??? 😀", 1735693200, 4, "train", 3),
    ], "id string, title string, created_utc long, score long, split string, label int")
    base = base_features(data, config)
    assert "score" not in base.columns and "label" not in base.columns
    rows = {r.id: r for r in base.collect()}
    assert "don't" in rows["a"].tokens and rows["c"].uppercase_ratio == 0
    assert rows["c"].question_count == 3 and rows["a"].question_present == 1
    model, metadata = fit_preprocessing(data, config)
    assert "u:don't" in model.stages[0].vocabulary
    unseen = data.withColumn("title", F.lit("heldoutonlytoken")).withColumn("split", F.lit("test"))
    assert "u:heldoutonlytoken" not in model.stages[0].vocabulary
    with pytest.raises(ValueError, match="TRAIN"):
        fit_preprocessing(unseen, config)
    one = model.transform(base_features(data, config)).select("id", "combined_features").orderBy("id").collect()
    two = model.transform(base_features(data.withColumn("score", F.lit(-500)), config)).select("id", "combined_features").orderBy("id").collect()
    assert one == two
    output = tmp_path / "preprocessing"
    model.write().save(str(output))
    restored = PipelineModel.load(str(output))
    assert restored.transform(base_features(data, config)).select("id", "combined_features").orderBy("id").collect() == one
    assert metadata["dimensions"]["combined_features"] == len(model.stages[0].vocabulary) + 9


def test_eight_class_lr_and_rf_execute_with_numeric_labels(spark):
    from pyspark.ml.linalg import Vectors
    rows = [(str(i), i % 8, Vectors.dense(float(i % 8), float(i % 3)), True) for i in range(80)]
    data = spark.createDataFrame(rows, ["id", "label", "time_features", "title_seen_in_train"])
    for spec in [{"kind": "lr", "features": "time_features", "alpha": 0.5, "reg_param": 0.1, "max_iter": 3},
                 {"kind": "rf", "features": "time_features", "alpha": 0.5, "num_trees": 2, "max_depth": 2, "max_bins": 8}]:
        model, info = fit_candidate(data, spec, [10] * 8, 212)
        result = timed_evaluation(data, spec, model, info)
        assert model.numClasses == 8 and result["records"] == 80
        assert len(result["per_class"]) == 8
        assert info["training_records"] == 80
