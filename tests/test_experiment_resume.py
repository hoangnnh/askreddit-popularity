"""Resume must reject changed input/configuration/artifacts and existing fresh paths."""

from copy import deepcopy
import json
from pathlib import Path

import duckdb
import pytest

from askreddit.experiment import Run


@pytest.fixture
def config(tmp_path):
    value = json.loads((Path(__file__).resolve().parents[1] / "configs/fang_chua_v1.json").read_text())
    source = tmp_path / "source"
    source.mkdir()
    (source / "_SUCCESS").touch()
    with duckdb.connect(config={"autoinstall_known_extensions": False, "autoload_known_extensions": False}) as connection:
        connection.execute("COPY (SELECT 1 AS fixture) TO ? (FORMAT PARQUET)", [str(source / "fixture.parquet")])
    value.update(input=str(source), data_root=str(tmp_path / "data"),
                 artifact_root=str(tmp_path / "artifacts"), report_root=str(tmp_path / "reports"))
    return value


def test_resume_checks_input_and_configuration(config):
    original = Run(config, "test", False)
    assert Run(config, "test", True).identity_sha == original.identity_sha
    with pytest.raises(ValueError, match="already exists"):
        Run(config, "test", False)
    changed = deepcopy(config)
    changed["features"]["min_df"] += 1
    with pytest.raises(ValueError, match="Cannot resume"):
        Run(changed, "test", True)
    source = original.input_files[0]
    source.write_bytes(source.read_bytes() + b"fixture mutation")
    with pytest.raises(ValueError, match="Cannot resume"):
        Run(config, "test", True)


def test_receipts_reject_missing_or_changed_artifacts(config):
    run = Run(config, "test", False)
    artifact = run.artifacts / "example.json"
    artifact.write_text('{"fixture": 1}')
    assert not run.completed("fixture", ["labeling"])
    run.finish("fixture", [artifact], ["labeling"])
    assert run.completed("fixture", ["labeling"])
    artifact.write_text('{"fixture": 2}')
    with pytest.raises(ValueError, match="changed or is incomplete"):
        run.completed("fixture", ["labeling"])
    artifact.unlink()
    with pytest.raises(ValueError, match="changed or is incomplete"):
        run.completed("fixture", ["labeling"])
