"""Stage 2 of the two-stage score estimate: ridge regression of log(1 + score) from the title.

Population: posts with ``score >= min_score`` (default 2) of the title-deduplicated
dataset, i.e. posts that received at least one net upvote. The archived score is
the value Arctic Shift re-fetched about 36 hours after posting. Features are the
train-fitted TF-IDF (uni+bigrams) and title-structure features; posting time is not
an input. Candidates are predeclared in the config; selection uses validation RMSE.

Outputs (fresh directories per run ID; no resume): labeled data and features under
``data/experiments/RUN``, fitted preprocessing and models under
``artifacts/experiments/RUN``, and JSON/CSV/PNG reports under ``reports/experiments/RUN``.
"""

from __future__ import annotations

import argparse
from contextlib import suppress
import csv
from datetime import datetime, timezone
import json
import logging
import math
import os
from pathlib import Path
import platform
import re
import sys
from time import perf_counter

import numpy as np
from pyspark.ml import PipelineModel
from pyspark.ml.feature import VectorAssembler, VectorSlicer
from pyspark.ml.regression import LinearRegression, LinearRegressionModel
from pyspark.sql import functions as F

from askreddit.binary_prepare import TITLE_COLUMNS, TITLE_INDICES
from askreddit.clean import CLEAN_SCHEMA, create_spark, file_fingerprint, guard_paths, write_json
from askreddit.experiment import digest
from askreddit.features import base_features, fit_preprocessing, validate_feature_sources
from askreddit.labeling import SPLITS, assign_splits, split_intervals

LOG = logging.getLogger(__name__)
VERSION = "score-regression-v1"
FEATURE_COLUMNS = ("text_features", "title_structured_features", "combined_title_features")
INTERVAL = (0.1, 0.9)


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    _, first, counts = np.unique(values[order], return_index=True, return_counts=True)
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.repeat(first + (counts + 1) / 2.0, counts)
    return ranks


def regression_metrics(y, p, *, residual_quantiles=None, score=None) -> dict:
    """Metrics on the log(1 + score) scale plus score-scale summaries."""
    y, p = np.asarray(y, dtype=np.float64), np.asarray(p, dtype=np.float64)
    if y.shape != p.shape or y.ndim != 1 or y.size < 2 or not (np.isfinite(y).all() and np.isfinite(p).all()):
        raise ValueError("Labels and predictions must be finite 1-D arrays of equal length")
    error = p - y
    sst = float(np.sum((y - y.mean()) ** 2))
    ry, rp = average_ranks(y), average_ranks(p)
    spearman = float(np.corrcoef(ry, rp)[0, 1]) if np.std(rp) > 0 else 0.0
    pearson = float(np.corrcoef(y, p)[0, 1]) if np.std(p) > 0 else 0.0
    result = {"records": int(y.size), "rmse_log": float(np.sqrt(np.mean(error ** 2))), "mae_log": float(np.mean(np.abs(error))),
              "r2_log": 1 - float(np.sum(error ** 2)) / sst if sst else 0.0, "spearman": spearman, "pearson": pearson,
              "mean_label": float(y.mean()), "mean_prediction": float(p.mean())}
    actual = np.expm1(y) if score is None else np.asarray(score, dtype=np.float64)
    result["median_abs_error_score"] = float(np.median(np.abs(np.expm1(p) - actual)))
    if residual_quantiles is not None:
        low, high = residual_quantiles
        result["interval_coverage"] = float(np.mean((y >= p + low) & (y <= p + high)))
    edges = np.quantile(p, np.linspace(0, 1, 11))
    bins = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, 9)
    result["deciles"] = [{"decile": int(d + 1), "records": int(np.sum(bins == d)),
                          "mean_predicted_score": float(np.expm1(p[bins == d]).mean()),
                          "median_actual_score": float(np.median(actual[bins == d])),
                          "mean_actual_log": float(y[bins == d].mean()), "mean_predicted_log": float(p[bins == d].mean()),
                          "share_actual_ge_10": float(np.mean(actual[bins == d] >= 10)),
                          "share_actual_ge_100": float(np.mean(actual[bins == d] >= 100))}
                         for d in range(10) if np.any(bins == d)]
    return result


def pairwise_auc(y_high, p) -> float | None:
    """AUC of the regression output for a binary target (Mann-Whitney with average ranks)."""
    y_high, p = np.asarray(y_high, dtype=bool), np.asarray(p, dtype=np.float64)
    positives, negatives = int(y_high.sum()), int((~y_high).sum())
    if not positives or not negatives:
        return None
    ranks = average_ranks(p)
    return float((ranks[y_high].sum() - positives * (positives + 1) / 2) / (positives * negatives))


class ScoreRun:
    def __init__(self, config: dict, run_id: str):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id) or config.get("algorithm_version") != VERSION:
            raise ValueError("Bad run ID or unsupported algorithm version")
        split_intervals(config["split_dates"])
        validate_feature_sources(config["features"]["feature_sources"])
        ids = [c["id"] for c in config["candidates"]]
        if len(ids) != len(set(ids)) or any(c["kind"] not in ("mean", "ridge") for c in config["candidates"]) or \
                any(c["kind"] == "ridge" and c["features"] not in FEATURE_COLUMNS for c in config["candidates"]):
            raise ValueError("Candidates must be unique mean/ridge specs over known feature columns")
        self.config = config
        self.input = Path(config["input"]).resolve()
        self.data, self.artifacts, self.reports = [(Path(config[k]) / run_id).resolve() for k in ("data_root", "artifact_root", "report_root")]
        self.input_files = sorted(self.input.rglob("*.parquet"))
        if not self.input_files or not (self.input / "_SUCCESS").is_file():
            raise ValueError("A completed local Parquet dataset is required")
        guard_paths(self.input_files, [self.data, self.artifacts, self.reports])
        for path in (self.data, self.artifacts, self.reports):
            path.mkdir(parents=True, exist_ok=False)
        identity = {"config": config, "input_files": [file_fingerprint(p) for p in self.input_files],
                    "python_version": platform.python_version()}
        write_json(self.reports / "identity.json", {**identity, "identity_sha256": digest(identity)})
        write_json(self.reports / "config.json", config)
        write_json(self.reports / "search_plan.json", {"registered_at_utc": datetime.now(timezone.utc).isoformat(),
                   "candidates": config["candidates"], "selection": config["selection"], "test_used_for_tuning": False})

    def events(self, phase, **details):
        LOG.info("%s %s", phase, details)
        with (self.reports / "events.jsonl").open("a") as handle:
            handle.write(json.dumps({"at_utc": datetime.now(timezone.utc).isoformat(), "phase": phase, **details}) + "\n")


def prepare(spark, run):
    raw = spark.read.parquet(str(run.input))
    if dict(raw.dtypes) != dict(CLEAN_SCHEMA):
        raise ValueError(f"Unexpected input schema: {raw.dtypes}")
    minimum = int(run.config["min_score"])
    data = assign_splits(raw, run.config["split_dates"])
    if data.filter(F.col("split").isNull()).limit(1).count():
        raise ValueError("Every record must fall inside a split interval")
    population = data.filter(F.col("score") >= minimum).withColumn("label", F.log1p(F.col("score").cast("double")))
    target = run.data / "labeled"
    population.repartition(run.config["spark"]["write_partitions"], "split", "month").write.mode("errorifexists") \
        .option("compression", "zstd").partitionBy("split", "year", "month").parquet(str(target))
    written = spark.read.parquet(str(target))
    counts = {r.split: r["count"] for r in written.groupBy("split").count().collect()}
    totals = {r.split: r["count"] for r in data.groupBy("split").count().collect()}
    train = written.filter(F.col("split") == "train")
    stats = train.agg(F.avg("label").alias("mean_log"), F.stddev("label").alias("std_log"),
                      *[F.expr(f"percentile(score, {q})").alias(f"score_p{int(q * 100)}") for q in (0.25, 0.5, 0.75, 0.9, 0.99)]).first().asDict()
    summary = {"min_score": minimum, "population_records": counts, "all_records": totals,
               "population_share": {s: counts.get(s, 0) / totals[s] for s in SPLITS}, "train_label_stats": stats,
               "label": "log(1 + score)", "score_observation": "Archived score re-fetched by Arctic Shift about 36 hours after posting."}
    if any(counts.get(s, 0) == 0 for s in SPLITS):
        raise ValueError(f"Empty split in the regression population: {counts}")
    write_json(run.reports / "population.json", summary)
    run.events("prepare_completed", **{k: summary[k] for k in ("population_records", "train_label_stats")})
    return written


def build_features(spark, run, labeled):
    config = run.config["features"]
    train = labeled.filter(F.col("split") == "train")
    started = perf_counter()
    preprocessing, metadata = fit_preprocessing(train, config)
    preprocessing.write().save(str(run.artifacts / "preprocessing"))
    preprocessing = PipelineModel.load(str(run.artifacts / "preprocessing"))
    title = PipelineModel(stages=[
        VectorSlicer(inputCol="structured_features", outputCol="title_structured_features", indices=TITLE_INDICES),
        VectorAssembler(inputCols=["text_features", "title_structured_features"], outputCol="combined_title_features")])
    title.write().save(str(run.artifacts / "title_features"))
    title = PipelineModel.load(str(run.artifacts / "title_features"))
    for split in SPLITS:
        frame = labeled.filter(F.col("split") == split)
        predictors = title.transform(preprocessing.transform(base_features(frame, config))).select("id", *FEATURE_COLUMNS)
        predictors.join(frame.select("id", "label", "score"), "id").repartition(run.config["spark"]["shuffle_partitions"], "id") \
            .write.mode("errorifexists").option("compression", "zstd").parquet(str(run.data / "features" / split))
    width = len(preprocessing.stages[0].vocabulary)
    metadata["dimensions"] = {"text_features": width, "title_structured_features": len(TITLE_COLUMNS),
                              "combined_title_features": width + len(TITLE_COLUMNS)}
    metadata["title_columns"] = TITLE_COLUMNS
    metadata["fitting_seconds"] = perf_counter() - started
    write_json(run.reports / "features.json", metadata)
    run.events("features_completed", seconds=metadata["fitting_seconds"], vocabulary=width)
    return preprocessing


def collect(frame, spec, model, train_mean):
    if spec["kind"] == "mean":
        rows = frame.select("label", "score").collect()
        predictions = np.full(len(rows), train_mean)
    else:
        rows = model.transform(frame.select("label", "score", F.col(spec["features"]).alias("features"))) \
            .select("label", "score", "prediction").collect()
        predictions = np.fromiter((r.prediction for r in rows), dtype=np.float64, count=len(rows))
    labels = np.fromiter((r.label for r in rows), dtype=np.float64, count=len(rows))
    scores = np.fromiter((r.score for r in rows), dtype=np.float64, count=len(rows))
    return labels, predictions, scores


def evaluate_split(labels, predictions, scores, high_cut, low_cut, residual_quantiles=None):
    metrics = regression_metrics(labels, predictions, residual_quantiles=residual_quantiles, score=scores)
    binary = (scores >= high_cut) | (scores < low_cut)
    metrics["auc_high_vs_low"] = pairwise_auc(scores[binary] >= high_cut, predictions[binary])
    metrics["auc_definition"] = f"score >= {high_cut} vs score < {low_cut}, middle excluded (binary task's rule)"
    return metrics


def train_and_evaluate(spark, run):
    features = {s: spark.read.parquet(str(run.data / "features" / s)) for s in SPLITS}
    train_mean = features["train"].agg(F.avg("label")).first()[0]
    cuts = run.config["binary_comparison"]
    results = {}
    for spec in run.config["candidates"]:
        started = perf_counter()
        model = None
        info = {"training_records": features["train"].count()}
        if spec["kind"] == "ridge":
            estimator = LinearRegression(featuresCol="features", labelCol="label", regParam=spec["reg_param"], elasticNetParam=0.0,
                                         maxIter=spec["max_iter"], tol=1e-6, standardization=False, solver="l-bfgs")
            model = estimator.fit(features["train"].select("label", F.col(spec["features"]).alias("features")))
            # The training summary exists only on the fitted (not the reloaded) model.
            info.update(iterations=model.summary.totalIterations, feature_dimension=model.numFeatures,
                        reached_iteration_limit=model.summary.totalIterations >= spec["max_iter"])
            path = run.artifacts / "models" / spec["id"]
            model.write().save(str(path))
            model = LinearRegressionModel.load(str(path))
        else:
            info.update(train_mean_log=train_mean, feature_dimension=0)
        info["training_seconds"] = perf_counter() - started
        labels, predictions, scores = collect(features["validation"], spec, model, train_mean)
        residuals = labels - predictions
        quantiles = [float(np.quantile(residuals, q)) for q in INTERVAL]
        validation = evaluate_split(labels, predictions, scores, cuts["high_min"], cuts["low_below"], quantiles)
        validation["residual_quantiles"] = dict(zip(("q10", "q90"), quantiles))
        validation["prediction_percentiles"] = [float(v) for v in np.quantile(predictions, np.linspace(0, 1, 101))]
        results[spec["id"]] = {"candidate": spec, "training": info, "validation": validation, "model": model}
        write_json(run.reports / "candidates" / f"{spec['id']}.json", {k: v for k, v in results[spec["id"]].items() if k != "model"})
        run.events("candidate_completed", candidate=spec["id"], validation_rmse=validation["rmse_log"],
                   validation_spearman=validation["spearman"], seconds=info["training_seconds"])
    ranked = sorted(results.values(), key=lambda r: (round(r["validation"]["rmse_log"], 12), r["candidate"]["id"]))
    selected = ranked[0]["candidate"]["id"]
    write_json(run.reports / "selection.json", {"selected_candidate": selected, "criterion": run.config["selection"],
               "frozen_before_test": True, "frozen_at_utc": datetime.now(timezone.utc).isoformat()})
    rows = []
    for cid, result in results.items():
        q = result["validation"]["residual_quantiles"]
        labels, predictions, scores = collect(features["test"], result["candidate"], result["model"], train_mean)
        test = evaluate_split(labels, predictions, scores, cuts["high_min"], cuts["low_below"], (q["q10"], q["q90"]))
        write_json(run.reports / "test" / f"{cid}.json", test)
        result["test"] = test
        for split in ("validation", "test"):
            m = result[split]
            rows.append({"candidate": cid, "kind": result["candidate"]["kind"], "split": split, "selected": cid == selected,
                         **{k: m[k] for k in ("rmse_log", "mae_log", "r2_log", "spearman", "pearson", "median_abs_error_score",
                                              "interval_coverage", "auc_high_vs_low", "records")},
                         "training_seconds": result["training"]["training_seconds"],
                         "feature_dimension": result["training"].get("feature_dimension", 0)})
        run.events("test_candidate_completed", candidate=cid, rmse=test["rmse_log"], spearman=test["spearman"])
    with (run.reports / "model_comparison.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(run.reports / "model_comparison.json", {"selected_candidate": selected, "rows": rows})
    return results, selected


def analysis(run, results, selected, preprocessing, top=30):
    vocabulary = list(preprocessing.stages[0].vocabulary)
    names = {"text_features": vocabulary, "title_structured_features": TITLE_COLUMNS,
             "combined_title_features": vocabulary + TITLE_COLUMNS}
    out = {}
    for cid, result in results.items():
        spec = result["candidate"]
        if spec["kind"] != "ridge":
            continue
        coefficients = dict(zip(names[spec["features"]], result["model"].coefficients.toArray().tolist()))
        text = sorted(((t, c) for t, c in coefficients.items() if t[:2] in ("u:", "b:")), key=lambda x: x[1])
        out[cid] = {"intercept": result["model"].intercept,
                    "structured": {c: coefficients[c] for c in TITLE_COLUMNS if c in coefficients},
                    "top_positive": [{"term": t[2:], "kind": t[0], "coefficient": c} for t, c in reversed(text[-top:])],
                    "top_negative": [{"term": t[2:], "kind": t[0], "coefficient": c} for t, c in text[:top]],
                    "unit": "Change in log(1 + score) per unit of L2-normalized TF-IDF (text) or per train SD (structure)"}
    write_json(run.reports / "analysis.json", {"selected_candidate": selected, "ridge": out})
    return out


def figures(run, results, selected, coefficients):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from askreddit.binary_report import MUTED, NEGATIVE, POSITIVE, SERIES, TEXT, coefficient_figure, style
    target = run.reports / "figures"
    target.mkdir(exist_ok=True)
    deciles = results[selected]["test"]["deciles"]
    fig, ax = plt.subplots(figsize=(7, 4.6), constrained_layout=True)
    x = [d["decile"] for d in deciles]
    ax.plot(x, [math.expm1(d["mean_predicted_log"]) for d in deciles], color=SERIES[0], marker="o", linewidth=2,
            label="Dự đoán (trung bình log, đổi về score)")
    ax.plot(x, [math.expm1(d["mean_actual_log"]) for d in deciles], color=SERIES[1], marker="o", linewidth=2,
            label="Thực tế (trung bình log, đổi về score)")
    ax.plot(x, [d["median_actual_score"] for d in deciles], color=MUTED, linestyle="--", linewidth=1.5, label="Thực tế (trung vị)")
    ax.set(xticks=x, xlabel="Nhóm thập phân theo score dự đoán (1 = thấp nhất)", ylabel="Score",
           title=f"Hiệu chỉnh trên test — {selected}")
    style(ax)
    ax.legend(frameon=False, labelcolor=TEXT)
    fig.savefig(target / "deciles_test.png", dpi=160)
    plt.close(fig)
    if selected in coefficients:
        coefficient_figure(plt, coefficients[selected], f"N-gram ảnh hưởng tới log(1 + score) — {selected}",
                           target / "top_ngrams.png")
    _ = (POSITIVE, NEGATIVE)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/score_v1.json")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    spark = run = None
    try:
        config = json.loads(Path(args.config).read_text())
        run = ScoreRun(config, args.run_id)
        os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
        scratch = run.artifacts / "spark_tmp"
        scratch.mkdir()
        os.environ["SPARK_LOCAL_DIRS"] = str(scratch)
        spark = create_spark(config["spark"]["driver_memory"], config["spark"]["shuffle_partitions"])
        labeled = prepare(spark, run)
        preprocessing = build_features(spark, run, labeled)
        results, selected = train_and_evaluate(spark, run)
        coefficients = analysis(run, results, selected, preprocessing)
        figures(run, results, selected, coefficients)
        if [file_fingerprint(p) for p in run.input_files] != json.loads((run.reports / "identity.json").read_text())["input_files"]:
            raise ValueError("Input files changed during the run")
        write_json(run.reports / "completed.json", {"status": "completed", "selected_candidate": selected,
                   "test_used_for_tuning": False, "completed_at_utc": datetime.now(timezone.utc).isoformat()})
        run.events("completed", selected=selected)
    except BaseException as error:
        if run is not None:
            with suppress(Exception):
                write_json(run.reports / "failures" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f") + ".json"),
                           {"status": "failed", "error": str(error)})
        LOG.exception("Score regression failed")
        return 130 if isinstance(error, KeyboardInterrupt) else 1
    finally:
        if spark is not None:
            with suppress(Exception):
                spark.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
