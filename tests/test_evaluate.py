import subprocess
import sys
from pathlib import Path

def run(cmd: str):
    """Run a CLI command and fail if it errors."""
    result = subprocess.run(f"{sys.executable} -m {cmd}", shell=True, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return result.stdout + result.stderr

def test_evaluate_metadata_command():
    out = run("imagerec evaluate --type metadata --data-dir tests/test_data/train --model-path tests/test_data/test_model.pt")
    assert "Architecture: shufflenet_v2_x0_5" in out

def test_evaluate_metrics_command():
    out = run("imagerec evaluate --type metrics --data-dir tests/test_data/train --model-path tests/test_data/test_model.pt")
    assert "Accuracy Score" in out or "Confusion Matrix" in out or "Classification Report" in out