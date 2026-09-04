from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from pathlib import Path


def _run(command: list[str], *, cwd: Path, timeout: int = 900) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, text=True, capture_output=True, timeout=timeout, check=False)


def run_packaging_validation(*, root: str | Path | None = None, python_executable: str | None = None) -> dict:
    """Build artifacts and install the wheel in a fresh venv.

    The result is evidence, not a default assumption.  Callers should keep the
    returned failure details in the release bundle.
    """
    root_path = Path(root or Path(__file__).resolve().parents[3]).resolve()
    python = python_executable or sys.executable
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="deepuplift-package-") as temp:
        temp_path = Path(temp)
        dist = temp_path / "dist"
        build = _run([python, "-m", "build", "--wheel", "--sdist", "--outdir", str(dist)], cwd=root_path)
        if build.returncode != 0 and "No module named build" in (build.stderr or build.stdout):
            bootstrap = _run([python, "-m", "pip", "install", "--upgrade", "build", "setuptools", "wheel"], cwd=root_path, timeout=600)
            if bootstrap.returncode == 0:
                build = _run([python, "-m", "build", "--wheel", "--sdist", "--outdir", str(dist)], cwd=root_path)
        wheels = sorted(dist.glob("*.whl"))
        sdists = sorted(dist.glob("*.tar.gz"))
        result = {
            "status": "FAIL",
            "python": python,
            "wheel_exists": bool(wheels),
            "sdist_exists": bool(sdists),
            "wheel": wheels[0].name if wheels else None,
            "sdist": sdists[0].name if sdists else None,
            "build_returncode": build.returncode,
            "fresh_venv_install": False,
            "import": False,
            "quickstart": False,
            "runtime_seconds": None,
        }
        if build.returncode != 0 or not wheels or not sdists:
            result["error"] = (build.stderr or build.stdout)[-4000:]
            result["runtime_seconds"] = time.perf_counter() - started
            return result
        venv = temp_path / "venv"
        created = _run([python, "-m", "venv", str(venv)], cwd=root_path)
        venv_python = venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        install = _run([str(venv_python), "-m", "pip", "install", str(wheels[0])], cwd=root_path, timeout=1200)
        result["fresh_venv_install"] = install.returncode == 0
        if install.returncode != 0:
            result["error"] = (install.stderr or install.stdout)[-4000:]
            result["runtime_seconds"] = time.perf_counter() - started
            return result
        imported = _run([str(venv_python), "-c", "import deepuplift; assert deepuplift.__version__"], cwd=root_path)
        quickstart = _run([str(venv_python), "-c", "from deepuplift.data import synthetic_ground_truth; from deepuplift.models import build_model; d=synthetic_ground_truth(80, seed=7); m=build_model('S-Learner'); m.fit(d); p=m.predict(d); assert len(p.unit_id)==80"], cwd=root_path, timeout=300)
        result["import"] = imported.returncode == 0
        result["quickstart"] = quickstart.returncode == 0
        result["status"] = "PASS" if result["import"] and result["quickstart"] else "FAIL"
        if result["status"] != "PASS":
            result["error"] = ((imported.stderr or imported.stdout) + (quickstart.stderr or quickstart.stdout))[-4000:]
        result["runtime_seconds"] = time.perf_counter() - started
        return result


__all__ = ["run_packaging_validation"]
