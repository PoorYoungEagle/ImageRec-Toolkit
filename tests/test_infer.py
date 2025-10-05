import subprocess
import sys
from pathlib import Path

def run(cmd: str):
    """Run a CLI command and fail if it errors."""
    result = subprocess.run(f"{sys.executable} -m {cmd}", shell=True, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return result.stdout + result.stderr

def test_infer_command():
    out = run("imagerec infer --input-path tests/test_data/train/Colosseum/colo1.jpg --model-path tests/test_data/test_model.pt")
    print(out)
    assert "Predicted" in out or "Rank" in out

def test_classify_command(tmp_path: Path):
    out = run(f"imagerec classify --input-dir tests/test_data/train/Colosseum --model-path tests/test_data/test_model.pt --output-dir {str(tmp_path)}")
    assert (tmp_path / "Colosseum").exists() or (tmp_path / "Pyramid of Giza").exists()