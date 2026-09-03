from __future__ import annotations

import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepuplift.core.open_source_frameworks import GITHUB_REPOS, open_source_framework_rows  # noqa: E402


def _github_request(repo: str, token: str | None) -> dict[str, object]:
    if repo.startswith("local/") or repo.startswith("docs/"):
        return {
            "repo": repo,
            "html_url": repo,
            "api_status": "local_source",
            "api_error": "",
        }
    url = f"https://api.github.com/repos/{repo.strip()}"
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "DeepUplift-Agent-GitHub-Framework-Audit",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return {
            "repo": repo,
            "html_url": f"https://github.com/{repo}",
            "api_status": f"http_{exc.code}",
            "api_error": "GitHub API unavailable or rate-limited; rerun with GITHUB_TOKEN for live metadata.",
        }
    except Exception as exc:
        return {
            "repo": repo,
            "html_url": f"https://github.com/{repo}",
            "api_status": "error",
            "api_error": type(exc).__name__,
        }
    return {
        "repo": payload.get("full_name", repo),
        "html_url": payload.get("html_url", f"https://github.com/{repo}"),
        "description": payload.get("description", ""),
        "stargazers_count": payload.get("stargazers_count"),
        "forks_count": payload.get("forks_count"),
        "open_issues_count": payload.get("open_issues_count"),
        "updated_at": payload.get("updated_at"),
        "pushed_at": payload.get("pushed_at"),
        "archived": payload.get("archived"),
        "default_branch": payload.get("default_branch"),
        "license": (payload.get("license") or {}).get("spdx_id"),
        "api_status": "ok",
    }


def _activity(meta: dict[str, object]) -> str:
    if meta.get("api_status") != "ok":
        return str(meta.get("api_status"))
    pushed = str(meta.get("pushed_at") or meta.get("updated_at") or "")
    if pushed >= "2025":
        return "active_recent"
    if pushed >= "2023":
        return "warm"
    if pushed:
        return "stale_check_before_install"
    return "unknown"


def main() -> None:
    token = os.environ.get("GITHUB_TOKEN")
    rows = []
    frameworks = open_source_framework_rows()
    for framework in frameworks:
        name = framework["framework"]
        repo_spec = GITHUB_REPOS.get(name, "")
        repo_entries = [item.strip() for item in repo_spec.split(";") if item.strip()]
        if not repo_entries:
            rows.append(
                {
                    "framework": name,
                    "repo": "",
                    "api_status": "no_repo",
                    "adoption_status": framework.get("local_status", ""),
                    "platform_layer": framework.get("scenario_fit", ""),
                    "integration_path": framework.get("integration_path", ""),
                    "interview_line": framework.get("interview_line", ""),
                }
            )
            continue
        for repo in repo_entries:
            meta = _github_request(repo, token)
            row = {
                "framework": name,
                **meta,
                "activity": _activity(meta),
                "adoption_status": framework.get("local_status", ""),
                "platform_layer": framework.get("scenario_fit", ""),
                "integration_path": framework.get("integration_path", ""),
                "limitation": framework.get("limitations", ""),
                "next_step": framework.get("next_step", ""),
                "interview_line": framework.get("interview_line", ""),
            }
            rows.append(row)

    status_counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get("api_status", "unknown"))
        status_counts[key] = status_counts.get(key, 0) + 1

    payload = {
        "status": "ok",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "uses_token": bool(token),
        "note": "If GitHub API returns http_403, rerun with GITHUB_TOKEN. The UI still keeps source links and guarded adoption decisions.",
        "status_counts": status_counts,
        "rows": rows,
    }
    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    json_path = reports / "github_framework_audit_latest.json"
    csv_path = reports / "github_framework_audit_latest.csv"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    columns = [
        "framework",
        "repo",
        "html_url",
        "stargazers_count",
        "updated_at",
        "pushed_at",
        "archived",
        "activity",
        "api_status",
        "adoption_status",
        "integration_path",
        "next_step",
        "interview_line",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"status": "ok", "json": str(json_path), "csv": str(csv_path), "status_counts": status_counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
