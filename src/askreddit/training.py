"""Predeclared full-data candidate fitting and validation-only model selection."""

from time import perf_counter

from pyspark import StorageLevel
from pyspark.ml.classification import (LogisticRegression, LogisticRegressionModel,
                                        RandomForestClassifier, RandomForestClassificationModel)
from pyspark.sql import functions as F

from askreddit.evaluation import evaluate_predictions
from askreddit.labeling import class_weights


def weight_expression(weights):
    mapping = F.create_map(*[value for i, w in enumerate(weights) for value in (F.lit(i), F.lit(w))])
    return F.element_at(mapping, F.col("label").cast("int"))


def fit_candidate(train, spec, counts, seed):
    if spec["kind"] == "majority":
        return None, {"majority_label": max(range(8), key=lambda k: (counts[k], -k)),
                      "training_records": sum(counts), "training_seconds": 0.0, "class_weights": [1.0] * 8}
    weights = class_weights(counts, spec["alpha"])
    dataset = train.select("label", F.col(spec["features"]).alias("features")).withColumn(
        "class_weight", weight_expression(weights)).persist(StorageLevel.DISK_ONLY)
    try:
        actual = dataset.count()
        if actual != sum(counts):
            raise ValueError("Candidate must train on every training record")
        if spec["kind"] == "lr":
            estimator = LogisticRegression(family="multinomial", featuresCol="features", labelCol="label",
                weightCol="class_weight", regParam=spec["reg_param"], elasticNetParam=0.0,
                maxIter=spec["max_iter"], tol=1e-5, standardization=False, maxBlockSizeInMB=1.0)
        elif spec["kind"] == "rf":
            estimator = RandomForestClassifier(featuresCol="features", labelCol="label", weightCol="class_weight",
                numTrees=spec["num_trees"], maxDepth=spec["max_depth"], maxBins=spec["max_bins"],
                featureSubsetStrategy="sqrt", seed=seed, subsamplingRate=0.7, maxMemoryInMB=128)
        else:
            raise ValueError(f"Unknown candidate type: {spec['kind']}")
        started = perf_counter()
        model = estimator.fit(dataset)
        info = {"training_records": actual, "training_seconds": perf_counter() - started,
                "class_weights": weights, "weight_normalization": "Mean training-row weight equals 1",
                "num_classes": model.numClasses, "feature_dimension": model.numFeatures}
        if model.numClasses != 8:
            raise ValueError("Estimator did not preserve the fixed eight-label universe")
        if spec["kind"] == "lr":
            info.update(iterations=model.summary.totalIterations,
                        objective_history=[float(v) for v in model.summary.objectiveHistory],
                        reached_iteration_limit=model.summary.totalIterations >= spec["max_iter"])
        return model, info
    finally:
        dataset.unpersist()


def predictions_for(frame, spec, model, info):
    if spec["kind"] == "majority":
        return frame.select("id", "label", "title_seen_in_train").withColumn("prediction", F.lit(info["majority_label"]))
    data = frame.select("id", "label", "title_seen_in_train", F.col(spec["features"]).alias("features"))
    return model.transform(data).select("id", "label", "prediction", "title_seen_in_train")


def timed_evaluation(frame, spec, model, info, *, unseen_slice=False):
    started = perf_counter()
    predictions = predictions_for(frame, spec, model, info).persist(StorageLevel.DISK_ONLY)
    try:
        metrics = evaluate_predictions(predictions)
        metrics["prediction_seconds"] = perf_counter() - started
        if unseen_slice:
            metrics["unseen_train_titles"] = evaluate_predictions(predictions.filter(~F.col("title_seen_in_train")))
        return metrics
    finally:
        predictions.unpersist()


def choose_candidate(results):
    if not results:
        raise ValueError("No completed candidates available")
    return sorted(results, key=lambda r: (-round(r["validation"]["macro_f1"], 12), r["candidate"]["id"]))[0]["candidate"]["id"]


def load_model(spec, path):
    if spec["kind"] == "majority":
        return None
    cls = LogisticRegressionModel if spec["kind"] == "lr" else RandomForestClassificationModel
    return cls.load(str(path))
