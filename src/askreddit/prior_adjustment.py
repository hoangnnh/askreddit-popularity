"""Post-hoc prior (logit) adjustment of a completed experiment's saved candidates.

Each saved LR/RF candidate keeps its weights. Its per-class scores s_k(x) are
shifted to s_k(x) - tau * log(pi_k), where pi is the TRAINING label distribution
(Menon et al., 2021, "Long-tail learning via logit adjustment"). One tau per
candidate is chosen on validation eight-class macro-F1; test is scored only at
tau = 0 and at that chosen tau. tau = 0 must reproduce the saved validation and
test confusion matrices exactly, and every saved model/feature artifact is
verified against its receipt before use. Nothing is retrained.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import logging
import math
import os
from pathlib import Path
import re
import sys
from time import perf_counter

from pyspark import StorageLevel
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F

from askreddit.clean import create_spark, write_json
from askreddit.evaluation import LABELS, confusion_figure, metrics_from_confusion
from askreddit.experiment import inventory
from askreddit.training import load_model

LOG = logging.getLogger(__name__)
EPSILON = 1e-12
REFERENCE = "https://arxiv.org/abs/2007.07314"
SUMMARY_KEYS = ("macro_f1", "balanced_accuracy", "accuracy", "weighted_f1", "mean_threshold_f1")
DISCLOSURE = ("Designed on 2026-11-25 after the validation AND test metrics of this run were viewed. "
              "tau is chosen on validation macro-F1 only, but test is not an untouched hold-out in the strict sense.")
DISCLOSURE_VI = ("Phân tích này được thiết kế ngày 2026-11-25 sau khi đã xem kết quả validation VÀ test của run này. "
                 "τ chỉ được chọn bằng macro-F1 trên validation, nhưng test không còn là hold-out hoàn toàn chưa nhìn thấy "
                 "theo nghĩa chặt.")


def tau_grid(minimum: float, maximum: float, step: float) -> list[float]:
    if step <= 0 or minimum > maximum:
        raise ValueError("The tau grid needs a positive step and minimum <= maximum")
    count = round((maximum - minimum) / step)
    if not math.isclose(minimum + count * step, maximum, abs_tol=1e-9):
        raise ValueError("The tau grid step must divide the range exactly")
    # Adding 0.0 turns a rounded -0.0 into 0.0 for clean JSON.
    grid = [round(minimum + i * step, 10) + 0.0 for i in range(count + 1)]
    if 0.0 not in grid:
        raise ValueError("The tau grid must contain 0 so the saved predictions can be reproduced")
    return grid


def log_priors(counts: list[int]) -> list[float]:
    if len(counts) != 8 or min(counts) <= 0:
        raise ValueError("Priors require eight nonempty training classes")
    total = sum(counts)
    return [math.log(count / total) for count in counts]


def class_scores(kind: str):
    """Per-class scores whose first argmax is the model's own prediction."""
    if kind == "lr":
        # Multinomial margins equal log-softmax plus a per-row constant.
        return vector_to_array("rawPrediction")
    if kind == "rf":
        return F.transform(vector_to_array("probability"), lambda p: F.log(F.greatest(p, F.lit(EPSILON))))
    raise ValueError(f"Prior adjustment needs per-class scores; unsupported candidate kind: {kind}")


def scored_frame(frame, spec, model):
    data = frame.select(F.col("label").cast("int").alias("label"), F.col(spec["features"]).alias("features"))
    return model.transform(data).select("label", class_scores(spec["kind"]).alias("scores"))


def adjusted_confusions(scored, log_prior: list[float], taus: list[float]) -> list[list[list[int]]]:
    """One 8x8 confusion matrix per tau from integer labels and per-class score arrays."""
    if len(log_prior) != 8:
        raise ValueError("Expected eight log priors")
    grid = scored.sparkSession.createDataFrame(list(enumerate(taus)), "tau_index int, tau double")
    prior = F.array(*[F.lit(float(v)) for v in log_prior])
    adjusted = F.zip_with("scores", prior, lambda score, log_pi: score - F.col("tau") * log_pi)
    counts = (scored.crossJoin(F.broadcast(grid)).withColumn("adjusted", adjusted)
              .withColumn("prediction", (F.array_position("adjusted", F.array_max("adjusted")) - 1).cast("int"))
              .groupBy("tau_index", "label", "prediction").count().collect())
    matrices = [[[0] * 8 for _ in LABELS] for _ in taus]
    for row in counts:
        if row.label not in LABELS or row.prediction not in LABELS:
            raise ValueError("Labels and adjusted predictions must be integers in 0–7")
        matrices[row.tau_index][row.label][row.prediction] += row["count"]
    return matrices


def sweep(frame, spec, model, log_prior, taus):
    scored = scored_frame(frame, spec, model).persist(StorageLevel.DISK_ONLY)
    try:
        if scored.filter(F.size("scores") != 8).limit(1).count():
            raise ValueError(f"Candidate {spec['id']} did not produce eight class scores")
        return adjusted_confusions(scored, log_prior, taus)
    finally:
        scored.unpersist()


def summary(metrics: dict) -> dict:
    predicted = [sum(row[k] for row in metrics["confusion_matrix"]) for k in LABELS]
    return {**{k: metrics[k] for k in SUMMARY_KEYS}, "predicted_per_class": predicted}


def choose_tau(rows: list[dict]) -> float:
    """Highest validation macro-F1; ties prefer the smallest |tau|, then the smaller tau."""
    if not rows:
        raise ValueError("Empty tau sweep")
    return max(rows, key=lambda r: (round(r["macro_f1"], 12), -abs(r["tau"]), -r["tau"]))["tau"]


def verify_receipt(reports: Path, stage: str, identity: str) -> None:
    """Hash-check a stage's artifacts. Source hashes are not compared: nothing is refit."""
    receipt = json.loads((reports / "receipts" / f"{stage}.json").read_text())
    if receipt["identity_sha256"] != identity:
        raise ValueError(f"Receipt {stage} belongs to a different run identity")
    for artifact in receipt["artifacts"]:
        if inventory(Path(artifact["path"])) != artifact["files"]:
            raise ValueError(f"Artifact changed or is incomplete: {artifact['path']}")


def run_analysis(spark, run_id: str, report_root: str | Path, taus: list[float],
                 candidates: list[str] | None = None, output_name: str = "prior_adjustment") -> dict:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id) or not re.fullmatch(r"[A-Za-z0-9_-]+", output_name):
        raise ValueError("Run ID and output name must contain only letters, numbers, underscores and hyphens")
    reports = Path(report_root).resolve() / run_id

    def read(name):
        return json.loads((reports / name).read_text())

    config, completed = read("config.json"), read("completed.json")
    if completed.get("status") != "completed":
        raise ValueError("Prior adjustment needs a completed experiment")
    identity = read("identity.json")["identity_sha256"]
    counts = [r["count"] for r in read("class_distributions.json")["train"]["labels"]]
    log_prior = log_priors(counts)
    known = {c["id"]: c for c in config["candidates"]}
    unknown = set(candidates or []) - set(known)
    if unknown:
        raise ValueError(f"Unknown candidates: {sorted(unknown)}")
    specs = [c for c in config["candidates"] if c["kind"] != "majority" and (not candidates or c["id"] in candidates)]
    specs = [c for c in specs if read(f"candidates/{c['id']}.json")["status"] == "completed"]
    if not specs:
        raise ValueError("No completed LR/RF candidates to adjust")
    verify_receipt(reports, "features", identity)
    for spec in specs:
        verify_receipt(reports, "candidate_" + spec["id"], identity)
        verify_receipt(reports, "test_" + spec["id"], identity)
    out = reports / output_name
    out.mkdir(exist_ok=False)
    write_json(out / "plan.json", {
        "registered_at_utc": datetime.now(timezone.utc).isoformat(), "run_id": run_id,
        "method": "Post-hoc logit adjustment: argmax_k s_k(x) - tau * log(pi_k); s = LR margins or log RF probabilities.",
        "reference": REFERENCE, "taus": taus, "prior_source": "train split label counts",
        "training_counts": counts, "log_priors": log_prior, "probability_floor": EPSILON,
        "candidates": [s["id"] for s in specs],
        "tau_selection": "Per candidate: largest validation macro-F1 (rounded to 12 decimals), then smallest |tau|, then smaller tau.",
        "candidate_selection": "Largest adjusted validation macro-F1 (rounded to 12 decimals), then candidate ID ascending.",
        "test_policy": "Test scored only at tau=0 (reproduction check) and at the validation-chosen tau.",
        "reproduction_check": "tau=0 must equal the saved validation and test confusion matrices exactly.",
        "source_hash_policy": "Saved models and features are hash-verified against receipts; source hashes are not compared because nothing is refit.",
        "disclosure": DISCLOSURE, "disclosure_vi": DISCLOSURE_VI,
    })
    data = Path(config["data_root"]).resolve() / run_id / "features"
    validation, test = spark.read.parquet(str(data / "validation")), spark.read.parquet(str(data / "test"))
    models = Path(config["artifact_root"]).resolve() / run_id / "models"
    results, sweeps = [], {}
    for spec in specs:
        name = spec["id"]
        started = perf_counter()
        LOG.info("Adjusting %s over %d tau values", name, len(taus))
        model = load_model(spec, models / name)
        saved_validation = read(f"candidates/{name}.json")["validation"]
        saved_test = read(f"test/{name}.json")
        matrices = sweep(validation, spec, model, log_prior, taus)
        if matrices[taus.index(0.0)] != saved_validation["confusion_matrix"]:
            raise ValueError(f"tau=0 does not reproduce the saved validation predictions of {name}")
        validation_metrics = [metrics_from_confusion(m) for m in matrices]
        sweeps[name] = [{"tau": t, **summary(m)} for t, m in zip(taus, validation_metrics)]
        tau = choose_tau(sweeps[name])
        test_matrices = sweep(test, spec, model, log_prior, [0.0, tau])
        if test_matrices[0] != saved_test["confusion_matrix"]:
            raise ValueError(f"tau=0 does not reproduce the saved test predictions of {name}")
        result = {"candidate": spec, "chosen_tau": tau, "tau_on_grid_edge": tau in (taus[0], taus[-1]),
                  "validation": {"tau_0": validation_metrics[taus.index(0.0)], "chosen": validation_metrics[taus.index(tau)]},
                  "test": {"tau_0": metrics_from_confusion(test_matrices[0]), "chosen": metrics_from_confusion(test_matrices[1])},
                  "seconds": perf_counter() - started}
        write_json(out / "candidates" / f"{name}.json", result)
        results.append(result)
        LOG.info("%s: tau=%s validation macro-F1 %.4f -> %.4f", name, tau,
                 result["validation"]["tau_0"]["macro_f1"], result["validation"]["chosen"]["macro_f1"])
        del model
        spark.catalog.clearCache()
    selected = sorted(results, key=lambda r: (-round(r["validation"]["chosen"]["macro_f1"], 12), r["candidate"]["id"]))[0]
    rows = [{"candidate": r["candidate"]["id"], "split": split, "tau": tau, "selected": r is selected,
             **{k: r[split][key][k] for k in SUMMARY_KEYS}, "records": r[split][key]["records"]}
            for r in results for split in ("validation", "test")
            for key, tau in (("tau_0", 0.0), ("chosen", r["chosen_tau"]))]
    with (out / "comparison.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["candidate", "split", "tau", "selected", *SUMMARY_KEYS, "records"])
        writer.writeheader()
        writer.writerows(rows)
    for split in ("validation", "test"):
        confusion_figure(selected[split]["chosen"], out / "figures" / f"{selected['candidate']['id']}_{split}_adjusted.png",
                         f"{selected['candidate']['id']} — {split}, tau={selected['chosen_tau']}")
    write_json(out / "validation_sweep.json", sweeps)
    baseline = next((r for r in read("model_comparison.json")["rows"]
                     if r["candidate"] == "majority" and r["split"] == "test"), None)
    outcome = {"run_id": run_id, "selected_candidate": selected["candidate"]["id"], "selected_tau": selected["chosen_tau"],
               "original_selection": completed["selected_candidate"],
               "majority_test_macro_f1": baseline["macro_f1"] if baseline else None,
               "candidates": {r["candidate"]["id"]: {"chosen_tau": r["chosen_tau"], "tau_on_grid_edge": r["tau_on_grid_edge"],
                              **{f"{split}_{key}": summary(r[split][key]) for split in ("validation", "test") for key in ("tau_0", "chosen")}}
                              for r in results},
               "test_used_for_tuning": False, "disclosure": DISCLOSURE,
               "completed_at_utc": datetime.now(timezone.utc).isoformat()}
    write_json(out / "results.json", outcome)
    (out / "results_vi.md").write_text(render_vi(outcome, taus))
    return outcome


def render_vi(outcome: dict, taus: list[float]) -> str:
    chosen = outcome["candidates"][outcome["selected_candidate"]]
    baseline = outcome["majority_test_macro_f1"]
    lines = [
        f"# Điều chỉnh prior sau huấn luyện — run `{outcome['run_id']}`", "",
        f"> **Lưu ý:** {DISCLOSURE_VI}", "",
        "Mô hình đã lưu không được huấn luyện lại. Điểm lớp s_k(x) (margin của LR, log xác suất của RF) được trừ "
        "τ·log π_k, với π là phân phối nhãn của tập train "
        f"([Menon và cộng sự, 2021]({REFERENCE})). Mỗi ứng viên chọn một τ theo macro-F1 tám lớp trên validation, "
        f"lưới τ từ {taus[0]} đến {taus[-1]} ({len(taus)} giá trị). τ = 0 tái tạo chính xác ma trận nhầm lẫn đã lưu "
        "của validation và test (đã kiểm tra). Test chỉ được chấm tại τ = 0 và tại τ đã chọn.", "",
        f"Ứng viên tốt nhất sau điều chỉnh (theo validation): **{outcome['selected_candidate']}**, τ = **{outcome['selected_tau']}**. "
        f"Macro-F1 trên test: {chosen['test_tau_0']['macro_f1']:.4f} → **{chosen['test_chosen']['macro_f1']:.4f}**"
        + (f" (baseline majority {baseline:.4f})." if baseline is not None else "."),
        f"Lựa chọn gốc của run: `{outcome['original_selection']}`.", "",
        "| Ứng viên | τ* | Val macro-F1 (τ=0 → τ*) | Test macro-F1 (τ=0 → τ*) | Test balanced acc. (τ=0 → τ*) | Test accuracy (τ=0 → τ*) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, r in outcome["candidates"].items():
        cells = [f"{r[f'{s}_tau_0'][m]:.4f} → {r[f'{s}_chosen'][m]:.4f}" for s, m in (
            ("validation", "macro_f1"), ("test", "macro_f1"), ("test", "balanced_accuracy"), ("test", "accuracy"))]
        edge = " (biên lưới)" if r["tau_on_grid_edge"] else ""
        lines.append(f"| {name} | {r['chosen_tau']}{edge} | " + " | ".join(cells) + " |")
    lines += ["", "Chi tiết: [results.json](results.json), [validation_sweep.json](validation_sweep.json), "
              "[comparison.csv](comparison.csv), [plan.json](plan.json).", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--report-root", default="reports/experiments")
    parser.add_argument("--candidate", action="append", help="Restrict to this candidate ID; repeatable")
    parser.add_argument("--tau-min", type=float, default=-1.0)
    parser.add_argument("--tau-max", type=float, default=2.0)
    parser.add_argument("--tau-step", type=float, default=0.05)
    parser.add_argument("--output-name", default="prior_adjustment")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    spark = None
    try:
        taus = tau_grid(args.tau_min, args.tau_max, args.tau_step)
        config = json.loads((Path(args.report_root) / args.run_id / "config.json").read_text())
        os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
        scratch = Path(config["artifact_root"]).resolve() / args.run_id / "spark_tmp"
        scratch.mkdir(parents=True, exist_ok=True)
        os.environ["SPARK_LOCAL_DIRS"] = str(scratch)
        spark = create_spark(config["spark"]["driver_memory"], config["spark"]["shuffle_partitions"])
        outcome = run_analysis(spark, args.run_id, args.report_root, taus, args.candidate, args.output_name)
        LOG.info("Selected %s at tau=%s", outcome["selected_candidate"], outcome["selected_tau"])
    except BaseException as error:
        LOG.exception("Prior adjustment failed")
        return 130 if isinstance(error, KeyboardInterrupt) else 1
    finally:
        if spark is not None:
            spark.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
