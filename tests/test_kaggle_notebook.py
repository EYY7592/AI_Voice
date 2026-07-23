import json
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "02_chifraud_dual_script_experiment.ipynb"


def test_kaggle_notebook_pins_code_revision_containing_data_cli() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    match = re.search(r"SCAMLENS_CODE_REVISION = ['\"]([0-9a-f]{40})['\"]", source)

    assert match is not None, "Kaggle notebook 必須固定可重現的專案程式 revision"
    completed = subprocess.run(
        ["git", "cat-file", "-e", f"{match.group(1)}:src/chifraud_data.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, "固定的專案 revision 必須包含 src.chifraud_data CLI"
