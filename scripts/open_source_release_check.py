from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def tracked_files() -> list[Path]:
    output = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
    return [ROOT / item for item in output.decode().split("\0") if item]


def main() -> int:
    files = tracked_files()
    findings: list[str] = []
    forbidden_content = re.compile("|".join(["/Users" + "/", "/home" + "/", r"[A-Za-z0-9._%+-]+" + "@" + r"[A-Za-z0-9.-]+" + r"\." + r"[A-Za-z]{2,}", r"BEGIN (RSA|OPENSSH|EC|DSA) PRIVATE KEY", "ghp" + "_" + r"[A-Za-z0-9]+"]))
    for path in files:
        if path.suffix.lower() in {".pyc", ".png", ".jpg", ".jpeg", ".pt", ".pth", ".ckpt", ".bin", ".safetensors"}:
            findings.append(f"binary_or_weight:{path.relative_to(ROOT)}")
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            findings.append(f"non_text_file:{path.relative_to(ROOT)}")
            continue
        if forbidden_content.search(text):
            findings.append(f"personal_path_or_credential:{path.relative_to(ROOT)}")
    for name in [".env", ".env.local"]:
        if (ROOT / name).exists():
            findings.append(f"environment_file:{name}")
    if any(str(path).startswith(str(ROOT / "runs")) or str(path).startswith(str(ROOT / "release_runs")) for path in files):
        findings.append("generated_runs_tracked")
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    version_match = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.MULTILINE)
    if not version_match or version_match.group(1) not in readme:
        findings.append("README_version_mismatch")
    from deepuplift.models.registry import MODEL_REGISTRY
    for spec in MODEL_REGISTRY.values():
        if spec.maturity in {"INTERFACE_ONLY", "OPTIONAL"} and spec.runnable is True and spec.maturity == "INTERFACE_ONLY":
            findings.append(f"interface_marked_runnable:{spec.name}")
    result = {"status": "PASS" if not findings else "FAIL", "tracked_files": len(files), "findings": findings}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not findings else 1


if __name__ == "__main__":
    raise SystemExit(main())
