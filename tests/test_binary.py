"""Binary train-quantile labels, exact metrics, chi-square selection and candidate execution."""

from fractions import Fraction
from itertools import product
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import pytest
from pyspark.ml.linalg import Vectors
from pyspark.sql import functions as F

from askreddit.binary_evaluation import (average_precision, best_f1_threshold, evaluate_scores,
                                         metrics_at_threshold, roc_auc)
from askreddit.binary_experiment import choose_candidate, cyclic_effects, display_term
from askreddit.binary_labeling import (apply_binary_labels, binary_class_weights, fit_binary_thresholds,
                                       group_distributions, group_sql, quantile_from_histogram,
                                       thresholds_from_histogram)
from askreddit.binary_prepare import chi_square_selection
from askreddit.binary_training import collect_scores, fit_binary_candidate
from askreddit.clean import create_spark

CONFIG = json.loads((Path(__file__).resolve().parents[1] / "configs/binary_v3.json").read_text())


@pytest.fixture(scope="module")
def spark():
    os.environ["PYSPARK_PYTHON"] = sys.executable
    session = create_spark("1g", 2)
    yield session
    session.stop()


def test_inverse_ecdf_quantiles_and_tie_rule():
    histogram = [(0, 2), (1, 5), (2, 2), (3, 1)]
    assert quantile_from_histogram(histogram, Fraction(1, 2)) == 1
    assert quantile_from_histogram(histogram, Fraction(3, 4)) == 2
    assert quantile_from_histogram([(1, 1), (2, 1)], Fraction(1, 2)) == 1
    model = thresholds_from_histogram(histogram)
    assert (model["median"], model["p75"]) == (1, 2)
    assert model["train_counts"] == {"excluded": 0, "low": 7, "middle": 2, "high": 1}
    with pytest.raises(ValueError):
        thresholds_from_histogram([(1, 10)])


V4_RULES = {"population_min_score": 2, "low_rule": "lt_median", "high_rule": "ge_p75"}
V4_HISTOGRAM = [(0, 50), (1, 100), (2, 49), (3, 22), (4, 8), (5, 21)]


def test_v4_population_and_literal_rules():
    model = thresholds_from_histogram(V4_HISTOGRAM, V4_RULES)
    assert (model["median"], model["p75"], model["population_records"]) == (3, 4, 100)
    assert model["train_counts"] == {"excluded": 150, "low": 49, "middle": 22, "high": 29}
    with pytest.raises(ValueError):  # median == p75 with <= / >= would put score 2 in both groups
        thresholds_from_histogram([(2, 10), (3, 1)], {"low_rule": "le_median", "high_rule": "ge_p75"})
    with pytest.raises(ValueError):
        thresholds_from_histogram(V4_HISTOGRAM, {"low_rule": "below"})


def test_fixed_engagement_rules():
    model = thresholds_from_histogram(V4_HISTOGRAM, {"low_rule": "le_value", "low_value": 1, "high_rule": "ge_value", "high_value": 2})
    assert (model["low_cut"], model["high_cut"]) == (1, 2)
    assert model["train_counts"] == {"excluded": 0, "low": 150, "middle": 0, "high": 100}
    assert "THEN 'low'" in group_sql(model) and "score >= 2" in group_sql(model)
    with pytest.raises(ValueError):
        thresholds_from_histogram(V4_HISTOGRAM, {"low_rule": "le_value", "high_rule": "ge_p75"})


def test_v4_group_expression_matches_sql(spark):
    model = thresholds_from_histogram(V4_HISTOGRAM, V4_RULES)
    frame = spark.createDataFrame([(str(s), "train", s) for s in range(8)], "id string, split string, score long")
    expression = {r.id: (r.popularity_group, r.label) for r in apply_binary_labels(frame, model).collect()}
    frame.createOrReplaceTempView("v4_scores")
    sql = {r.id: r.g for r in spark.sql(f"SELECT id, {group_sql(model)} AS g FROM v4_scores").collect()}
    expected = {"0": "excluded", "1": "excluded", "2": "low", "3": "middle", "4": "high", "5": "high", "6": "high", "7": "high"}
    assert {k: v[0] for k, v in expression.items()} == sql == expected
    assert expression["2"][1] == 0.0 and expression["4"][1] == 1.0 and expression["0"][1] is None


def test_train_only_fit_and_frozen_application(spark):
    train = spark.createDataFrame([(s, "train") for s in [0, 0, 1, 1, 1, 1, 1, 2, 2, 3]], "score long, split string")
    model = fit_binary_thresholds(train)
    assert (model["median"], model["p75"]) == (1, 2)
    with pytest.raises(ValueError):
        fit_binary_thresholds(train.withColumn("split", train.split.substr(1, 0)))
    frame = spark.createDataFrame([("a", "train", 0), ("b", "train", 2), ("c", "validation", 3), ("d", "test", 1000)],
                                  "id string, split string, score long")
    rows = {r.id: (r.popularity_group, r.label) for r in apply_binary_labels(frame, model).collect()}
    assert rows == {"a": ("low", 0.0), "b": ("middle", None), "c": ("high", 1.0), "d": ("high", 1.0)}
    distributions = group_distributions(apply_binary_labels(frame, model))
    assert distributions["train"]["kept"] == 1 and distributions["train"]["groups"]["middle"]["count"] == 1


def test_balanced_weights_have_mean_one():
    counts = [9, 1]
    weights = binary_class_weights(counts)
    assert sum(c * w for c, w in zip(counts, weights)) / sum(counts) == pytest.approx(1.0)
    assert weights[1] / weights[0] == pytest.approx(9.0)


def brute_auc(y, p):
    pos = [s for s, t in zip(p, y) if t == 1]
    neg = [s for s, t in zip(p, y) if t == 0]
    return sum(1.0 if a > b else 0.5 if a == b else 0.0 for a, b in product(pos, neg)) / (len(pos) * len(neg))


def test_auc_with_ties_matches_pairwise_definition():
    rng = np.random.default_rng(212)
    for _ in range(20):
        y = rng.integers(0, 2, 40)
        y[:2] = [0, 1]
        p = np.round(rng.random(40), 1)
        assert roc_auc(y, p) == pytest.approx(brute_auc(y, p))
    assert roc_auc([0, 1, 0, 1], [0.3] * 4) == 0.5


def test_threshold_metrics_average_precision_and_best_f1():
    y = np.array([1, 0, 1, 0, 0, 1])
    p = np.array([0.9, 0.8, 0.7, 0.4, 0.2, 0.1])
    m = metrics_at_threshold(y, p, 0.7)
    assert m["confusion_matrix"] == [[2, 1], [1, 2]]
    assert (m["precision"], m["recall"]) == (pytest.approx(2 / 3), pytest.approx(2 / 3))
    assert average_precision(y, p) == pytest.approx((1 / 3) * 1 + (1 / 3) * (2 / 3) + (1 / 3) * 0.5)
    brute = max(np.unique(p), key=lambda t: (metrics_at_threshold(y, p, t)["f1"], t))
    assert metrics_at_threshold(y, p, best_f1_threshold(y, p))["f1"] == pytest.approx(metrics_at_threshold(y, p, brute)["f1"])
    result = evaluate_scores(y, p, 0.7, seen=[True, True, False, False, False, False])
    assert result["unseen_train_titles"]["records"] == 4
    assert result["roc_curve"]["fpr"][0] == 0.0 and result["roc_curve"]["tpr"][-1] == 1.0
    with pytest.raises(ValueError):
        roc_auc([1, 1], [0.1, 0.2])


def test_selection_helpers():
    results = [{"candidate": {"id": "b", "kind": "lr"}, "validation": {"auc_roc": 0.7}},
               {"candidate": {"id": "a", "kind": "gbt"}, "validation": {"auc_roc": 0.7}},
               {"candidate": {"id": "c", "kind": "lr"}, "validation": {"auc_roc": 0.6}}]
    assert choose_candidate(results) == "a" and choose_candidate(results, "lr") == "b"
    assert display_term("b:if you") == "if you" and display_term("hour_sin") == "hour_sin"
    effects = cyclic_effects({"hour_sin": 1.0, "hour_cos": 0.0, "weekday_sin": 0.0, "weekday_cos": 2.0},
                             {"hour_sin": 0.5, "hour_cos": 1.0, "weekday_sin": 1.0, "weekday_cos": 1.0})
    assert sum(effects["hour_utc"]) == pytest.approx(0.0) and effects["hour_utc"][6] == pytest.approx(2.0)
    assert effects["weekday_monday0"][0] == pytest.approx(2.0 - sum(2 * math.cos(2 * math.pi * d / 7) for d in range(7)) / 7)


def test_chi_square_selection_ranks_informative_terms(spark):
    rows = [(f"h{i}", "If you could fly?", 1735689600 + i, 1.0, "train") for i in range(30)] + \
           [(f"l{i}", "Why is my code slow?", 1735689600 + i, 0.0, "train") for i in range(30)] + \
           [(f"m{i}", "What is the best food?", 1735689600 + i, float(i % 2), "train") for i in range(30)]
    train = spark.createDataFrame(rows, "id string, title string, created_utc long, label double, split string")
    vocabulary = ["u:what", "u:if", "u:why", "u:the", "u:you"]
    result = chi_square_selection(train, vocabulary, CONFIG["features"], 2)
    assert {t["term"] for t in result["selected_terms"]} <= {"u:if", "u:why", "u:you"}
    assert all(t["degrees_of_freedom"] == 1 for t in result["selected_terms"])
    top = result["selected_terms"][0]
    assert top["high_rate_when_present"] in (0.0, 1.0) and top["train_documents"] == 30


def test_binary_lr_and_gbt_execute(spark):
    data = [(float(i % 2), Vectors.dense([float(i % 2) + 0.1 * (i % 3), float(i % 5)]), i % 4 == 0) for i in range(80)]
    frame = spark.createDataFrame(data, ["label", "selected_features", "title_seen_in_train"]) \
        .withColumn("combined_features", F.col("selected_features"))
    counts = [40, 40]
    for spec in ({"id": "lr", "kind": "lr", "features": "combined_features", "reg_param": 0.01, "max_iter": 20},
                 {"id": "gbt", "kind": "gbt", "features": "selected_features", "max_iter": 5, "max_depth": 2,
                  "max_bins": 8, "step_size": 0.1, "subsampling_rate": 1.0}):
        model, info = fit_binary_candidate(frame, spec, counts, 212)
        y, p, seen = collect_scores(frame, spec, model, info)
        assert info["num_classes"] == 2 and len(y) == 80 and seen.sum() == 20
        assert roc_auc(y, p) > 0.9
    _, info = fit_binary_candidate(frame, {"id": "majority", "kind": "majority"}, [50, 30], 212)
    y, p, _ = collect_scores(frame, {"kind": "majority"}, None, info)
    assert info["majority_label"] == 0 and np.all(p == pytest.approx(30 / 80)) and roc_auc(y, p) == 0.5


def test_numpy_binary_inference_matches_spark(spark, tmp_path):
    from pyspark.ml import PipelineModel
    from pyspark.ml.feature import VectorAssembler, VectorSlicer
    from pyspark.ml.functions import vector_to_array
    from askreddit.binary_inference import BinaryPopularityModel
    from askreddit.binary_prepare import TITLE_INDICES
    from askreddit.export_model import EDGE_TITLES
    from askreddit.features import STRUCTURED_COLUMNS, base_features, fit_preprocessing
    titles = ["What is your favorite food?", "Why do cats purr so loudly?", "If you could fly, where would you go?",
              "I just moved, what now", "What's the best advice you've ever received?", "WHAT IS THE WORST JOB YOU HAD?"]
    rows = [(f"r{i}", titles[i % len(titles)] + ("" if i % 3 else " really?"), 1735689600 + 3571 * i, float(i % 2 == 0 or i % 5 == 0), "train")
            for i in range(120)]
    train = spark.createDataFrame(rows, "id string, title string, created_utc long, label double, split string")
    features = {**CONFIG["features"], "min_df": 1}
    preprocessing, _ = fit_preprocessing(train, features)
    title_pipeline = PipelineModel(stages=[
        VectorSlicer(inputCol="structured_features", outputCol="title_structured_features", indices=TITLE_INDICES),
        VectorAssembler(inputCols=["text_features", "title_structured_features"], outputCol="combined_title_features")])
    frame = title_pipeline.transform(preprocessing.transform(base_features(train, features))).join(train.select("id", "label"), "id")
    spec = {"id": "lr", "kind": "lr", "features": "combined_title_features", "reg_param": 0.01, "max_iter": 50}
    model, _ = fit_binary_candidate(frame, spec, [int(frame.filter("label = 0").count()), int(frame.filter("label = 1").count())], 212)
    vocabulary = list(preprocessing.stages[0].vocabulary)
    metadata = {"features": spec["features"], "vocabulary": vocabulary, "structured_columns": STRUCTURED_COLUMNS,
                "feature_dimension": model.numFeatures, "threshold": 0.5}
    local = BinaryPopularityModel(metadata, {"coefficients": model.coefficients.toArray(), "intercept": np.array(model.intercept),
                                             "idf": preprocessing.stages[1].idf.toArray(), "std": preprocessing.stages[4].std.toArray()})
    probe = spark.createDataFrame([(f"p{i}", t, 1764547200 + 7000 * i) for i, t in enumerate(titles + EDGE_TITLES)],
                                  "id string, title string, created_utc long")
    scored = model.setFeaturesCol(spec["features"]).transform(
        title_pipeline.transform(preprocessing.transform(base_features(probe, features)))) \
        .select("title", vector_to_array("probability")[1].alias("p")).collect()
    assert not local.uses_time
    assert max(abs(local.predict_proba(r.title) - r.p) for r in scored) < 1e-9
    assert local.contributions(titles[0]) and local.top_terms(3)["high"]
