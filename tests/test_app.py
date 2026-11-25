"""Headless smoke test of the Streamlit demo against the generated reports and exported model."""

import json
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/demo.json").read_text())
REQUIRED = [ROOT / CONFIG["eda_dir"] / "profile.json", ROOT / CONFIG["experiment_dir"] / "completed.json",
            ROOT / CONFIG["model_dir"] / "model.json", ROOT / CONFIG["benchmark_dir"] / "results.json"]


@pytest.mark.skipif(not all(p.exists() for p in REQUIRED), reason="Generated reports or exported model are not present")
def test_app_renders_every_tab_and_predicts():
    app = AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=120)
    app.run()
    assert not app.exception, app.exception
    assert not app.warning, [w.value for w in app.warning]  # every data source was found
    app.text_input[0].input("What's the most useless talent you have?")
    next(b for b in app.button if b.label == "Dự đoán").click()
    app.run()
    assert not app.exception, app.exception
    assert any("Nhãn dự đoán" in s.value for s in app.success)
    app.radio[0].set_value("test").run()
    assert not app.exception
