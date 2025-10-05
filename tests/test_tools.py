import subprocess
import sys
from pathlib import Path

def run(cmd: str):
    """Run a CLI command and fail if it errors."""
    result = subprocess.run(f"{sys.executable} -m {cmd}", shell=True, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return result.stdout + result.stderr

def test_tools_compress_command(tmp_path: Path):
    run(f"imagerec tools compress --model-path tests/test_data/test_model.pt --output-path {str(tmp_path)}")
    assert (tmp_path / "test_model_compressed.pt").exists()

def test_tools_split_command(tmp_path: Path):
    run(f"imagerec tools split --input-dir tests/test_data/train --output-dir {str(tmp_path)} --split 0.5")
    assert (tmp_path / "train" / "Colosseum").exists() or (tmp_path / "val" / "Pyramid of Giza").exists()

def test_tools_limit_command(tmp_path: Path):
    out = run(f"imagerec tools limit --input-dir tests/test_data/train --output-dir {str(tmp_path)} --dry-run")
    assert (tmp_path / "train_backup").exists() or "Started limiting class" in out

def test_tools_export_onnx_command(tmp_path: Path):
    run(f"imagerec tools export --format onnx --model-path tests/test_data/test_model.pt --output-path {str(tmp_path)}")
    assert (tmp_path / "test_model.onnx").exists()

def test_tools_export_torchlite_command(tmp_path: Path):
    run(f"imagerec tools export --format torchscript --model-path tests/test_data/test_model.pt --output-path {str(tmp_path)}")
    assert (tmp_path / "test_model_torchscript.pt").exists()