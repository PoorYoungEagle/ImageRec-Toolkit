import subprocess
import sys
from pathlib import Path

def run(cmd: str):
    """Run a CLI command and fail if it errors."""
    result = subprocess.run(f"{sys.executable} -m {cmd}", shell=True, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return result.stdout

def test_train_command(tmp_path: Path):
    run(f"imagerec train --data-dir tests/test_data --epochs 1 --output-dir {str(tmp_path)} --model-name model --architecture shufflenet_v2_x0_5 --no-timestamp --no-pretrained --no-augments")
    assert (tmp_path / "model" / "best.pt").exists()

def test_retrain_command(tmp_path: Path):
    run(f"imagerec retrain --data-dir tests/test_data --epochs 1 --model-path tests/test_data/test_model.pt --output-dir {str(tmp_path)} --model-name model --no-augments")
    assert (tmp_path / "model.pt").exists()