"""Binary candidate fitting (L2 logistic regression, gradient-boosted trees) and validation scoring."""

import json
from time import perf_counter

import numpy as np
from pyspark import StorageLevel
from pyspark.ml.classification import (GBTClassificationModel, GBTClassifier, LogisticRegression,
                                        LogisticRegressionModel)
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F

from askreddit.binary_evaluation import best_f1_threshold, evaluate_scores
from askreddit.binary_labeling import binary_class_weights
from askreddit.binary_prepare import FEATURE_MODULES
from askreddit.clean import write_json

CANDIDATE_MODULES = ["binary_training", "binary_evaluation", "binary_labeling"]
THRESHOLD_POLICY = "Probability threshold maximizing validation F1 of the high class; frozen before test. Majority uses 0.5."


def fit_binary_candidate(train, spec, counts, seed):
    if spec["kind"] == "majority":
        return None, {"majority_label": int(counts[1] > counts[0]), "train_positive_rate": counts[1] / sum(counts),
                      "training_records": sum(counts), "training_seconds": 0.0, "class_weights": [1.0, 1.0]}
    weights = binary_class_weights(counts) if spec.get("class_weighting", "balanced") == "balanced" else [1.0, 1.0]
    dataset = train.select("label", F.col(spec["features"]).alias("features")).withColumn(
        "class_weight", F.when(F.col("label") == 1.0, weights[1]).otherwise(weights[0])).persist(StorageLevel.DISK_ONLY)
    try:
        actual = dataset.count()
        if actual != sum(counts):
            raise ValueError("Candidate must train on every low/high training record")
        if spec["kind"] == "lr":
            estimator = LogisticRegression(family="binomial", featuresCol="features", labelCol="label",
                weightCol="class_weight", regParam=spec["reg_param"], elasticNetParam=0.0,
                maxIter=spec["max_iter"], tol=1e-6, standardization=False, maxBlockSizeInMB=1.0)
        elif spec["kind"] == "gbt":
            estimator = GBTClassifier(featuresCol="features", labelCol="label", weightCol="class_weight",
                maxIter=spec["max_iter"], maxDepth=spec["max_depth"], maxBins=spec["max_bins"],
                stepSize=spec["step_size"], subsamplingRate=spec["subsampling_rate"], featureSubsetStrategy="all",
                seed=seed, maxMemoryInMB=256, cacheNodeIds=False)
        else:
            raise ValueError(f"Unknown candidate type: {spec['kind']}")
        started = perf_counter()
        model = estimator.fit(dataset)
        info = {"training_records": actual, "training_seconds": perf_counter() - started, "class_weights": weights,
                "weight_normalization": "Balanced n/(2 n_k); mean training-row weight equals 1",
                "num_classes": model.numClasses, "feature_dimension": model.numFeatures}
        if model.numClasses != 2:
            raise ValueError("Estimator did not produce a binary model")
        if spec["kind"] == "lr":
            info.update(iterations=model.summary.totalIterations,
                        objective_history=[float(v) for v in model.summary.objectiveHistory],
                        reached_iteration_limit=model.summary.totalIterations >= spec["max_iter"])
        else:
            info.update(num_trees=model.getNumTrees, total_num_nodes=model.totalNumNodes)
        return model, info
    finally:
        dataset.unpersist()


def load_binary_model(spec, path):
    if spec["kind"] == "majority":
        return None
    return (LogisticRegressionModel if spec["kind"] == "lr" else GBTClassificationModel).load(str(path))


def collect_scores(frame, spec, model, info):
    """Collect (label, P(high), title_seen_in_train); a few hundred thousand rows at most."""
    columns = ["label", "title_seen_in_train"]
    if spec["kind"] == "majority":
        rows = frame.select(*columns).collect()
        scores = np.full(len(rows), info["train_positive_rate"], dtype=np.float64)
    else:
        scored = model.transform(frame.select(*columns, F.col(spec["features"]).alias("features")))
        rows = scored.select(*columns, vector_to_array("probability")[1].alias("p")).collect()
        scores = np.fromiter((r.p for r in rows), dtype=np.float64, count=len(rows))
    labels = np.fromiter((int(r.label) for r in rows), dtype=np.int64, count=len(rows))
    seen = np.fromiter((bool(r.title_seen_in_train) for r in rows), dtype=bool, count=len(rows))
    return labels, scores, seen


def score_split(frame, spec, model, info, threshold=None):
    started = perf_counter()
    labels, scores, seen = collect_scores(frame, spec, model, info)
    if threshold is None:
        threshold = 0.5 if spec["kind"] == "majority" else best_f1_threshold(labels, scores)
    metrics = evaluate_scores(labels, scores, threshold, seen=seen)
    metrics["prediction_seconds"] = perf_counter() - started
    return threshold, metrics


def train_stage(spark, run, candidate_id=None):
    if not run.completed("features", FEATURE_MODULES):
        raise ValueError("Complete features before training")
    train_counts = json.loads((run.reports / "class_distributions.json").read_text())["train"]["groups"]
    counts = [train_counts["low"]["count"], train_counts["high"]["count"]]
    metadata = json.loads((run.reports / "features.json").read_text())
    train = spark.read.parquet(str(run.data / "features/train"))
    validation = spark.read.parquet(str(run.data / "features/validation"))
    candidates = [c for c in run.config["candidates"] if candidate_id is None or c["id"] == candidate_id]
    if not candidates:
        raise ValueError(f"Unknown candidate: {candidate_id}")
    for spec in candidates:
        name = spec["id"]
        if run.completed("candidate_" + name, CANDIDATE_MODULES):
            continue
        run.events("candidate_started", candidate=name, configuration=spec)
        try:
            model, info = fit_binary_candidate(train, spec, counts, run.config["seed"])
        except Exception as error:
            if spec["kind"] != "gbt":
                raise
            # Tree models are resource-conditional on this laptop. Keep the exact failure;
            # never substitute a sample run or invented metrics.
            path = run.reports / "candidates" / (name + ".json")
            write_json(path, {"candidate": spec, "status": "failed_optional_comparison", "error": str(error)})
            run.finish("candidate_" + name, [path], CANDIDATE_MODULES)
            run.events("optional_candidate_failed", candidate=name, error=str(error))
            continue
        info.setdefault("feature_dimension", metadata["dimensions"].get(spec.get("features"), 0))
        saved = []
        if model is not None:
            path = run.artifacts / "models" / name
            model.write().save(str(path))
            model = load_binary_model(spec, path)
            saved.append(path)
        threshold, metrics = score_split(validation, spec, model, info)
        result = {"candidate": spec, "status": "completed", "training": info,
                  "threshold": {"value": threshold, "policy": THRESHOLD_POLICY}, "validation": metrics}
        path = run.reports / "candidates" / (name + ".json")
        write_json(path, result)
        saved.append(path)
        run.finish("candidate_" + name, saved, CANDIDATE_MODULES)
        run.events("candidate_completed", candidate=name, training_seconds=info["training_seconds"],
                   validation_auc=metrics["auc_roc"], validation_f1=metrics["f1"], threshold=threshold)
        del model
        spark.catalog.clearCache()
