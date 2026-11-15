"""Execute the example notebooks from top to bottom. Slow and needs the real table: run with ``pytest -m notebooks``."""
import json
import os
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
NOTEBOOKS = sorted(EXAMPLES.glob("*.ipynb"))


@pytest.mark.notebooks
@pytest.mark.parametrize("path", NOTEBOOKS, ids=[p.stem for p in NOTEBOOKS])
def test_example_notebook_runs(path, tmp_path, monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    monkeypatch.chdir(tmp_path)                       # the score log of the example is written here, not in the repository
    notebook = json.loads(path.read_text(encoding="utf-8"))
    namespace = {"__name__": "__main__"}
    for number, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        exec(compile(source, f"{path.name} cell {number}", "exec"), namespace)
        plt.close("all")
    assert os.path.exists("score_log_examples.csv")
    assert len(NOTEBOOKS) >= 2
