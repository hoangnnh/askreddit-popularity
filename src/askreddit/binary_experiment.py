"""Offline, resumable binary (high vs low popularity) experiment from the registered proposal.

Stages: prepare (train-quantile labels) -> features (train-fitted TF-IDF + chi-square
top-k n-grams) -> train (LR / GBT, validation threshold) -> evaluate (frozen selection,
test metrics, coefficient and importance analysis).
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
import sys

from pyspark.ml import PipelineModel

from askreddit.binary_prepare import TITLE_COLUMNS, BinaryRun, features_stage, prepare_stage
from askreddit.binary_training import (CANDIDATE_MODULES, load_binary_model, score_split, train_stage)
from askreddit.clean import create_spark, file_fingerprint, write_json
from askreddit.experiment import digest
from askreddit.features import STRUCTURED_COLUMNS, TIME_COLUMNS

LOG = logging.getLogger(__name__)
EVALUATION_MODULES = ["binary_experiment", "binary_training", "binary_evaluation"]
COMPARISON_FIELDS = ["candidate", "kind", "split", "selected", "auc_roc", "pr_auc", "threshold", "precision", "recall", "f1",
                     "accuracy", "balanced_accuracy", "predicted_positive_rate", "records", "positives",
                     "training_seconds", "feature_dimension"]


def choose_candidate(results, kind=None):
    pool = [r for r in results if kind is None or r["candidate"]["kind"] == kind]
    if not pool:
        return None
    return sorted(pool, key=lambda r: (-round(r["validation"]["auc_roc"], 12), r["candidate"]["id"]))[0]["candidate"]["id"]


def display_term(term: str) -> str:
    return term[2:] if term[:2] in ("u:", "b:") else term


def cyclic_effects(coefficients: dict, scale: dict) -> dict:
    """Logit contribution of hour/weekday (sin, cos) terms, centred on their mean over the cycle."""
    def curve(prefix, period):
        a = coefficients[prefix + "_sin"] / scale[prefix + "_sin"]
        b = coefficients[prefix + "_cos"] / scale[prefix + "_cos"]
        values = [a * math.sin(2 * math.pi * h / period) + b * math.cos(2 * math.pi * h / period) for h in range(period)]
        mean = sum(values) / period
        return [v - mean for v in values]
    return {"hour_utc": curve("hour", 24), "weekday_monday0": curve("weekday", 7),
            "unit": "Change in log-odds of high popularity relative to the cycle mean"}


def model_analysis(run, results, top=30):
    preprocessing = PipelineModel.load(str(run.artifacts / "preprocessing"))
    vocabulary = list(preprocessing.stages[0].vocabulary)
    std = dict(zip(STRUCTURED_COLUMNS, preprocessing.stages[4].std.toArray().tolist()))
    features = json.loads((run.reports / "features.json").read_text())
    selected_terms = [vocabulary[i] for i in features["chi_square"]["selected_indices"]]
    names = {"time_features": TIME_COLUMNS, "structured_features": STRUCTURED_COLUMNS, "text_features": vocabulary,
             "combined_features": vocabulary + STRUCTURED_COLUMNS, "selected_features": selected_terms + STRUCTURED_COLUMNS,
             "title_structured_features": TITLE_COLUMNS, "combined_title_features": vocabulary + TITLE_COLUMNS,
             "selected_title_features": selected_terms + TITLE_COLUMNS}
    unscaled = {c: 1.0 for c in TIME_COLUMNS}
    analysis = {"logistic_regression": {}, "gradient_boosted_trees": {},
                "coefficient_scale": "Text inputs are L2-normalized TF-IDF; structured inputs are divided by their train standard deviation (coefficients are per 1 SD); time-only inputs are raw sin/cos."}
    for result in results:
        spec = result["candidate"]
        if spec["kind"] == "majority":
            continue
        model = load_binary_model(spec, run.artifacts / "models" / spec["id"])
        labels = names[spec["features"]]
        if spec["kind"] == "lr":
            values = model.coefficients.toArray().tolist()
            if len(values) != len(labels):
                raise ValueError("Coefficient/feature-name length mismatch")
            entries = dict(zip(labels, values))
            text = sorted(((t, c) for t, c in entries.items() if t[:2] in ("u:", "b:")), key=lambda x: x[1])
            entry = {"intercept": model.intercept,
                     "structured": {c: entries[c] for c in STRUCTURED_COLUMNS if c in entries},
                     "time": {c: entries[c] for c in TIME_COLUMNS if c in entries}}
            if text:
                entry["top_positive"] = [{"term": display_term(t), "kind": t[0], "coefficient": c} for t, c in reversed(text[-top:])]
                entry["top_negative"] = [{"term": display_term(t), "kind": t[0], "coefficient": c} for t, c in text[:top]]
            if all(c in entries for c in TIME_COLUMNS):
                entry["time_effects"] = cyclic_effects(entries, unscaled if spec["features"] == "time_features" else std)
            analysis["logistic_regression"][spec["id"]] = entry
        else:
            importances = model.featureImportances.toArray().tolist()
            ranked = sorted(zip(labels, importances), key=lambda x: -x[1])
            analysis["gradient_boosted_trees"][spec["id"]] = {
                "top_features": [{"feature": display_term(n), "kind": "structured" if n in STRUCTURED_COLUMNS else n[0],
                                  "importance": v} for n, v in ranked[:top]],
                "structured_importance_share": sum(v for n, v in ranked if n in STRUCTURED_COLUMNS),
                "time_importance_share": sum(v for n, v in ranked if n in TIME_COLUMNS),
            }
        del model
    return analysis


def evaluate_stage(spark, run):
    if run.completed("evaluation", EVALUATION_MODULES):
        return
    results = []
    for spec in run.config["candidates"]:
        if not run.completed("candidate_" + spec["id"], CANDIDATE_MODULES):
            raise ValueError(f"Candidate incomplete; test is still sealed: {spec['id']}")
        result = json.loads((run.reports / "candidates" / (spec["id"] + ".json")).read_text())
        if result["status"] == "completed":
            results.append(result)
        elif spec["kind"] != "gbt":
            raise ValueError(f"Required candidate failed: {spec['id']}")
    selected = choose_candidate(results)
    best = {kind: choose_candidate(results, kind) for kind in ("lr", "gbt")}
    thresholds = {r["candidate"]["id"]: r["threshold"]["value"] for r in results}
    selection_path = run.reports / "selection.json"
    frozen = {"selected_candidate": selected, "best_by_family": best, "thresholds": thresholds,
              "criterion": run.config["selection"], "validation_results_sha256": digest(results)}
    if selection_path.exists():
        prior = json.loads(selection_path.read_text())
        if any(prior[k] != v for k, v in frozen.items()):
            raise ValueError("Frozen validation selection changed")
    else:
        write_json(selection_path, {**frozen, "frozen_before_test": True, "frozen_at_utc": datetime.now(timezone.utc).isoformat()})
    run.events("selection_frozen", candidate=selected, best_by_family=best)
    test = run.data / "features/test"
    test_frame = spark.read.parquet(str(test))
    rows = []
    for result in results:
        spec, info = result["candidate"], result["training"]
        path = run.reports / "test" / (spec["id"] + ".json")
        if run.completed("test_" + spec["id"], EVALUATION_MODULES):
            metrics = json.loads(path.read_text())
        else:
            model = load_binary_model(spec, run.artifacts / "models" / spec["id"])
            _, metrics = score_split(test_frame, spec, model, info, threshold=thresholds[spec["id"]])
            write_json(path, metrics)
            run.finish("test_" + spec["id"], [path], EVALUATION_MODULES)
            del model
        for split, scores in (("validation", result["validation"]), ("test", metrics)):
            rows.append({"candidate": spec["id"], "kind": spec["kind"], "split": split, "selected": spec["id"] == selected,
                         **{k: v for k, v in scores.items() if k in COMPARISON_FIELDS},
                         "training_seconds": info["training_seconds"], "feature_dimension": info["feature_dimension"]})
        run.events("test_candidate_completed", candidate=spec["id"], auc_roc=metrics["auc_roc"], f1=metrics["f1"])
    analysis = model_analysis(run, results)
    derived = ("analysis.json", "model_comparison.csv", "model_comparison.json", "completed.json")
    for name in derived:  # Derived from frozen receipts; an interrupted evaluation may have left a partial set.
        (run.reports / name).unlink(missing_ok=True)
    write_json(run.reports / "analysis.json", analysis)
    with (run.reports / "model_comparison.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMPARISON_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    write_json(run.reports / "model_comparison.json", {"selected_candidate": selected, "best_by_family": best, "rows": rows})
    current = [file_fingerprint(p) for p in run.input_files]
    if current != json.loads((run.reports / "identity.json").read_text())["input_files"]:
        raise ValueError("Cleaned inputs changed during the experiment")
    write_json(run.reports / "completed.json", {"status": "completed", "selected_candidate": selected, "best_by_family": best,
        "full_training_rows_used": True, "input_files_unchanged": True, "test_used_for_tuning": False,
        "completed_at_utc": datetime.now(timezone.utc).isoformat()})
    run.finish("evaluation", [selection_path, run.reports / "analysis.json", run.reports / "model_comparison.csv",
                              run.reports / "model_comparison.json", run.reports / "completed.json"], EVALUATION_MODULES)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/binary_v3.json")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", choices=["prepare", "features", "train", "evaluate", "all"], default="all")
    parser.add_argument("--candidate", help="Train one predefined candidate, only with --stage train")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run = spark = None
    try:
        if args.candidate and args.stage != "train":
            raise ValueError("--candidate requires --stage train")
        config = json.loads(Path(args.config).read_text())
        run = BinaryRun(config, args.run_id, args.resume)
        os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
        scratch = run.artifacts / "spark_tmp"
        scratch.mkdir(exist_ok=True)
        os.environ["SPARK_LOCAL_DIRS"] = str(scratch)
        spark = create_spark(config["spark"]["driver_memory"], config["spark"]["shuffle_partitions"])
        run.events("stage_started", stage=args.stage, spark_version=spark.version)
        if args.stage in ("prepare", "all"):
            prepare_stage(spark, run)
        if args.stage in ("features", "all"):
            features_stage(spark, run)
        if args.stage in ("train", "all"):
            train_stage(spark, run, args.candidate)
        if args.stage in ("evaluate", "all"):
            evaluate_stage(spark, run)
        run.events("stage_completed", stage=args.stage)
    except BaseException as error:
        if run is not None:
            path = run.reports / "failures" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f") + ".json")
            with suppress(Exception):
                write_json(path, {"stage": args.stage, "candidate": args.candidate, "error": str(error),
                                  "status": "failed", "partial_artifacts_are_not_complete": True})
        LOG.exception("Experiment stage failed")
        return 130 if isinstance(error, KeyboardInterrupt) else 1
    finally:
        if spark is not None:
            with suppress(Exception):
                spark.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
