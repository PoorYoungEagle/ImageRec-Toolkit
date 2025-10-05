import subprocess
import sys

def run(cmd: str):
    """Run a CLI command and fail if it errors."""
    result = subprocess.run(f"{sys.executable} -m {cmd}", shell=True, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return result.stdout

def test_help_command():
    out = run("imagerec --help")
    assert "usage" in out

def test_train_help_command():
    out = run("imagerec train --help")
    assert "usage" in out

def test_retrain_help_command():
    out = run("imagerec retrain --help")
    assert "usage" in out

def test_infer_help_command():
    out = run("imagerec infer --help")
    assert "usage" in out

def test_classify_help_command():
    out = run("imagerec classify --help")
    assert "usage" in out

def test_evaluate_help_command():
    out = run("imagerec evaluate --help")
    assert "usage" in out

def test_plot_help_command():
    out = run("imagerec plot --help")
    assert "usage" in out

def test_tools_help_command():
    out = run("imagerec tools --help")
    assert "usage" in out

def test_tools_compress_help_command():
    out = run("imagerec tools compress --help")
    assert "usage" in out

def test_tools_export_help_command():
    out = run("imagerec tools export --help")
    assert "usage" in out

def test_tools_lr_help_command():
    out = run("imagerec tools find-lr --help")
    assert "usage" in out

def test_tools_limit_help_command():
    out = run("imagerec tools limit --help")
    assert "usage" in out

def test_tools_split_help_command():
    out = run("imagerec tools split --help")
    assert "usage" in out