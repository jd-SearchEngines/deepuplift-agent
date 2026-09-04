import importlib.util
import subprocess
import sys

import pytest


@pytest.mark.skipif(importlib.util.find_spec("build") is None, reason="python-build not installed")
def test_wheel_and_sdist_build_smoke(tmp_path):
    result = subprocess.run([sys.executable, "-m", "build", "--no-isolation", "--outdir", str(tmp_path)], check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert list(tmp_path.glob("*.whl"))
    assert list(tmp_path.glob("*.tar.gz"))
