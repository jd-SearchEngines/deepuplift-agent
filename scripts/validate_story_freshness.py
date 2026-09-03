from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_REFERENCE_PATHS = [
    "docs/DEEPUplift_AGENT_ONEPAGER_CN.md",
    "docs/DEEPUplift_AGENT_INTERVIEW_EVIDENCE_PACK.md",
    "docs/DEEPUplift_AGENT_EVIDENCE_INDEX.md",
    "reports/deepuplift_agent_interview_pack_latest.zip",
]


def latest_story_screenshot(reports_dir: Path) -> Path | None:
    candidates = sorted(
        reports_dir.glob("ui_screenshots_*/story.png"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate that the Story screenshot is fresh relative to generated interview docs.")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--reference", action="append", default=[])
    args = parser.parse_args()

    references = [Path(path) for path in (args.reference or DEFAULT_REFERENCE_PATHS)]
    existing_refs = [path for path in references if path.exists()]
    story = latest_story_screenshot(Path(args.reports_dir))
    failures: list[str] = []
    if story is None:
        failures.append("No Story screenshot found")
    if not existing_refs:
        failures.append("No reference artifacts found")

    newest_ref = max(existing_refs, key=lambda path: path.stat().st_mtime) if existing_refs else None
    if story is not None and newest_ref is not None and story.stat().st_mtime < newest_ref.stat().st_mtime:
        failures.append(f"Latest Story screenshot {story} is older than {newest_ref}")

    payload = {
        "status": "fail" if failures else "ok",
        "story": str(story) if story else "",
        "story_mtime": story.stat().st_mtime if story else None,
        "newest_reference": str(newest_ref) if newest_ref else "",
        "newest_reference_mtime": newest_ref.stat().st_mtime if newest_ref else None,
        "references": [str(path) for path in existing_refs],
        "failures": failures,
    }
    print(json.dumps(payload, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
