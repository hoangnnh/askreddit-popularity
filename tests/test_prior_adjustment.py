"""Post-hoc prior adjustment: exact tau=0 reproduction, direction, selection and artifact checks."""

from copy import deepcopy
import json
import math
import os
from pathlib import Path
import sys

import duckdb
import pytest

from askreddit.clean import create_spark, write_json
from askreddit.evaluation import evaluate_predictions
from askreddit.experiment import Run
from askreddit.prior_adjustment import (adjusted_confusions, choose_tau, log_priors, run_analysis,
                                        sweep, tau_grid)
from askreddit.training import fit_candidate, load_model, predictions_for, timed_evaluation

CONFIG = json.loads((Path(__file__).resolve().parents[1] / "configs/fang_chua_v1.json").read_text())
COUNTS = [200, 80, 40, 30, 20, 15, 10, 5]
LR = {"id": "lr_tiny", "kind": "lr", "features": "combined_features", "alpha": 0.0, "reg_param": 0.01, "max_iter": 30}
RF = {"id": "rf_tiny", "kind": "rf", "features": "rf_features", "alpha": 0.0, "num_trees": 4, "max_depth": 4, "max_bins": 16}


@pytest.fixture(scope="module")
def spark():
    os.environ["PYSPARK_PYTHON"] = sys.executable
    session = create_spark("1g", 2)
    yield session
    session.stop()


def synthetic(spark, offset=0):
    """Imbalanced, overlapping eight-class data so an unweighted model favours class 0."""
    from pyspark.ml.linalg import Vectors
    labels = [k for k, n in enumerate(COUNTS) for _ in range(n)]
    rows = []
    for i, label in enumerate(labels):
        a = ((i + offset) * 7919 % 101) / 101 - 0.5
        b = ((i + offset) * 104729 % 97) / 97 - 0.5
        vector = Vectors.dense(0.25 * label + a, b)
        rows.append((f"r{offset}_{i}", label, False, vector, vector))
    return spark.createDataFrame(rows, ["id", "label", "title_seen_in_train", "combined_features", "rf_features"])


def test_tau_grid_and_log_priors():
    grid = tau_grid(-1, 2, 0.05)
    assert len(grid) == 61 and grid[0] == -1 and grid[-1] == 2 and 0.0 in grid
    assert all(math.copysign(1, t) == 1 for t in grid if t == 0)
    for bad in ((0.1, 1, 0.1), (0, 1, 0), (1, 0, 0.1), (0, 1, 0.3)):
        with pytest.raises(ValueError):
            tau_grid(*bad)
    assert log_priors([5] * 8) == pytest.approx([math.log(1 / 8)] * 8)
    with pytest.raises(ValueError):
        log_priors([5] * 7 + [0])


def test_choose_tau_uses_validation_macro_f1_with_deterministic_ties():
    rows = [{"tau": t, "macro_f1": f} for t, f in ((-0.5, 0.3), (0.0, 0.2), (0.5, 0.3), (1.0, 0.1))]
    assert choose_tau(rows) == -0.5
    rows[2]["macro_f1"] = 0.31
    assert choose_tau(rows) == 0.5
    assert choose_tau([{"tau": 0.0, "macro_f1": 0.2}, {"tau": 0.25, "macro_f1": 0.2}]) == 0.0


def test_adjusted_confusions_hand_computed(spark):
    rest = [-100.0] * 6
    scored = spark.createDataFrame([(1, [0.0, -0.6, *rest]), (0, [0.0, -2.0, *rest])], "label int, scores array<double>")
    log_prior = [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    matrices = adjusted_confusions(scored, log_prior, [0.0, 0.5, 1.0, 2.0])
    predicted = [{label: next(j for j in range(8) if m[label][j]) for label in (0, 1)} for m in matrices]
    # Row 1: -0.6 + tau beats 0 only once tau > 0.6. Row 0: -2 + 2 ties 0; first argmax keeps class 0.
    assert predicted == [{0: 0, 1: 0}, {0: 0, 1: 0}, {0: 0, 1: 1}, {0: 0, 1: 1}]


@pytest.mark.parametrize("spec", [LR, RF], ids=["lr", "rf"])
def test_tau_zero_reproduces_model_and_larger_tau_leaves_majority(spark, spec):
    data = synthetic(spark)
    model, info = fit_candidate(data, spec, COUNTS, 212)
    expected = evaluate_predictions(predictions_for(data, spec, model, info))["confusion_matrix"]
    matrices = sweep(data, spec, model, log_priors(COUNTS), [0.0, 0.5, 1.0, 2.0])
    assert matrices[0] == expected
    majority = [sum(row[0] for row in m) for m in matrices]
    assert majority == sorted(majority, reverse=True)
    if spec["kind"] == "lr":
        assert majority[-1] < majority[0]


def saved_run(spark, tmp_path):
    """A miniature completed run with real receipts, saved models and feature Parquet."""
    source = tmp_path / "source"
    source.mkdir()
    (source / "_SUCCESS").touch()
    with duckdb.connect(config={"autoinstall_known_extensions": False, "autoload_known_extensions": False}) as connection:
        connection.execute("COPY (SELECT 1 AS fixture) TO ? (FORMAT PARQUET)", [str(source / "fixture.parquet")])
    config = deepcopy(CONFIG)
    config.update(input=str(source), data_root=str(tmp_path / "data"), artifact_root=str(tmp_path / "artifacts"),
                  report_root=str(tmp_path / "reports"), candidates=[{"id": "majority", "kind": "majority"}, LR, RF])
    run = Run(config, "fixture", False)
    frames = {split: synthetic(spark, offset) for split, offset in (("train", 0), ("validation", 11), ("test", 23))}
    for split, frame in frames.items():
        frame.write.parquet(str(run.data / "features" / split))
    write_json(run.reports / "class_distributions.json",
               {"train": {"labels": [{"label": k, "count": n} for k, n in enumerate(COUNTS)]}})
    write_json(run.reports / "features.json", {"fixture": True})
    run.finish("features", [run.data / "features", run.reports / "features.json"], ["features"])
    rows = []
    for spec in config["candidates"]:
        model, info = fit_candidate(frames["train"], spec, COUNTS, 212)
        saved = []
        if model is not None:
            path = run.artifacts / "models" / spec["id"]
            model.write().save(str(path))
            model = load_model(spec, path)
            saved.append(path)
        validation = timed_evaluation(frames["validation"], spec, model, info)
        candidate = run.reports / "candidates" / f"{spec['id']}.json"
        write_json(candidate, {"candidate": spec, "status": "completed", "training": info, "validation": validation})
        run.finish("candidate_" + spec["id"], [*saved, candidate], ["training"])
        test = timed_evaluation(frames["test"], spec, model, info)
        write_json(run.reports / "test" / f"{spec['id']}.json", test)
        run.finish("test_" + spec["id"], [run.reports / "test" / f"{spec['id']}.json"], ["training"])
        rows.append({"candidate": spec["id"], "split": "test", "macro_f1": test["macro_f1"]})
    write_json(run.reports / "model_comparison.json", {"rows": rows})
    write_json(run.reports / "completed.json", {"status": "completed", "selected_candidate": "lr_tiny"})
    return run


def test_end_to_end_on_saved_run_and_tamper_detection(spark, tmp_path):
    run = saved_run(spark, tmp_path)
    taus = tau_grid(-1, 2, 0.5)
    outcome = run_analysis(spark, "fixture", tmp_path / "reports", taus)
    out = run.reports / "prior_adjustment"
    assert set(outcome["candidates"]) == {"lr_tiny", "rf_tiny"} and outcome["test_used_for_tuning"] is False
    for name in ("plan.json", "results.json", "validation_sweep.json", "comparison.csv", "results_vi.md"):
        assert (out / name).is_file()
    assert len(json.loads((out / "validation_sweep.json").read_text())["lr_tiny"]) == len(taus)
    for result in outcome["candidates"].values():
        assert result["chosen_tau"] in taus
        assert result["validation_chosen"]["macro_f1"] >= result["validation_tau_0"]["macro_f1"]
    with pytest.raises(FileExistsError):
        run_analysis(spark, "fixture", tmp_path / "reports", taus)
    victim = next(p for p in sorted((run.artifacts / "models" / "lr_tiny").rglob("*")) if p.is_file())
    victim.write_bytes(victim.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="changed or is incomplete"):
        run_analysis(spark, "fixture", tmp_path / "reports", taus, output_name="again")
    assert not (run.reports / "again").exists()
    with pytest.raises(ValueError, match="Unknown candidates"):
        run_analysis(spark, "fixture", tmp_path / "reports", taus, candidates=["nope"], output_name="other")
