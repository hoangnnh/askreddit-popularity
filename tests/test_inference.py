"""Spark-free inference must reproduce Spark preprocessing + LR exactly."""

import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest
from pyspark.sql import functions as F

from askreddit.clean import create_spark, utc_seconds
from askreddit.export_model import EDGE_TITLES, export
from askreddit.features import fit_preprocessing, normalize_unicode as spark_normalize, transform_features
from askreddit.inference import PopularityModel, normalize_unicode, structured_values, tokenize
from askreddit.training import fit_candidate

CONFIG = json.loads((Path(__file__).resolve().parents[1] / "configs/fang_chua_v2.json").read_text())
TITLES = ["What is your favorite food?", "Why do cats purr so loudly?", "If you could fly, where would you go?",
          "[Serious] How do you cope with stress?", "Would you rather have money or time?", "I just moved, what now",
          "Redditors, which movie is overrated?", "What's the best advice you've ever received?",
          "Don’t you think coffee is overrated?", "WHAT IS THE WORST JOB YOU HAD?"] + EDGE_TITLES


@pytest.fixture(scope="module")
def spark():
    os.environ["PYSPARK_PYTHON"] = sys.executable
    session = create_spark("1g", 2)
    yield session
    session.stop()


def test_pure_python_text_helpers_match_spark_definitions():
    for text in ["  Ａ  café\n 😀? ", "tab\tand nbsp", "İstanbul"] + TITLES:
        assert normalize_unicode(text) == spark_normalize(text)
    assert tokenize("don't stop snake_case 12½ ² naïve ’tis") == ["don't", "stop", "snake", "case", "12½", "²", "naïve", "tis"]
    monday = utc_seconds("2025-01-06T13:00:00Z")
    values = structured_values("Why NOT?? ok", monday)
    assert values[:5] == [12.0, 3.0, 2.0, 1.0, 4 / 8]
    assert values[5] == pytest.approx(np.sin(13 * 2 * np.pi / 24)) and values[7] == pytest.approx(0.0)


def write_clean(spark, path):
    start, end = utc_seconds("2025-01-01T00:00:00Z"), utc_seconds("2026-01-01T00:00:00Z")
    rows = [(f"t{i:04d}", "AskReddit", TITLES[i % len(TITLES)] + ("" if i % 3 else f" number {i % 7}"), "",
             start + (end - start - 1) * i // 399, (i * 37) % 300) for i in range(400)]
    frame = spark.createDataFrame(rows, "id string, subreddit string, title string, selftext string, created_utc long, score long")
    frame.withColumns({"year": F.year(F.timestamp_seconds("created_utc")), "month": F.month(F.timestamp_seconds("created_utc"))}) \
        .write.partitionBy("year", "month").parquet(str(path))
    return frame


@pytest.mark.parametrize("features", ["combined_features", "text_features", "structured_features", "time_features"])
def test_export_parity_with_spark(spark, tmp_path, features):
    clean = write_clean(spark, tmp_path / "clean")
    config = {**CONFIG, "input": str(tmp_path / "clean"),
              "features": {**CONFIG["features"], "vocab_size": 300, "min_df": 1, "rf_text_features": 16}}
    data = clean.withColumns({"split": F.lit("train"), "label": (F.col("score") % 8).cast("int"), "title_seen_in_train": F.lit(True)})
    preprocessing, _ = fit_preprocessing(data, config["features"])
    spec = {"id": "candidate", "kind": "lr", "features": features, "alpha": 1.0, "reg_param": 0.001, "max_iter": 30}
    model, _ = fit_candidate(transform_features(data, preprocessing, config["features"]), spec, [50] * 8, 212)
    artifacts, reports = tmp_path / "artifacts", tmp_path / "reports"
    run = artifacts / "run"
    preprocessing.write().save(str(run / "preprocessing"))
    model.write().save(str(run / "models" / "candidate"))
    (reports / "run").mkdir(parents=True)
    bounds = [2, 3, 4, 6, 11, 46, 271]
    mapping = [{"label": k, "lower_inclusive": None if k == 0 else bounds[k - 1], "upper_exclusive": bounds[k] if k < 7 else None}
               for k in range(8)]
    (reports / "run" / "label_boundaries.json").write_text(json.dumps({"mapping": mapping}))
    (reports / "run" / "config.json").write_text(json.dumps(config))
    (reports / "run" / "model_comparison.json").write_text(json.dumps({"rows": [
        {"candidate": "candidate", "split": "validation", "macro_f1": 0.1, "balanced_accuracy": 0.1, "accuracy": 0.1,
         "weighted_f1": 0.1, "mean_threshold_f1": 0.1}]}))
    (run / "selected").mkdir(parents=True)
    (run / "selected" / "manifest.json").write_text(json.dumps({
        "candidate": spec, "preprocessing": str(run / "preprocessing"), "model": str(run / "models" / "candidate"),
        "label_mapping": str(reports / "run" / "label_boundaries.json")}))
    verification = export(spark, "run", tmp_path / "out", artifacts, reports, sample_size=80, seed=1)
    assert verification["passed"] and verification["rows_compared"] > len(EDGE_TITLES)
    local = PopularityModel.load(tmp_path / "out")
    result = local.predict("What is your favorite food?", utc_seconds("2025-12-24T18:00:00Z"))
    assert 0 <= result["label"] <= 7 and sum(result["probabilities"]) == pytest.approx(1)
    assert result["score_range"].startswith("score")
    assert local.contributions("What is your favorite food?", utc_seconds("2025-12-24T18:00:00Z"), result["label"])
    assert bool(local.top_terms(7)) == (features in ("combined_features", "text_features"))
    with pytest.raises(FileExistsError):
        export(spark, "run", tmp_path / "out", artifacts, reports, sample_size=10, seed=1)
