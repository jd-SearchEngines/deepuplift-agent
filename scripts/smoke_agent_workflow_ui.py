from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from playwright.sync_api import expect, sync_playwright


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_latest_artifacts import validate_ui_compare_bundle


def latest_agent_bundle(reports_dir: Path, since: float) -> Path:
    bundles = sorted(
        [
            path
            for path in reports_dir.glob("agent_compare_bundle_*.json")
            if path.stat().st_mtime >= since - 1
        ],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not bundles:
        raise AssertionError("Agent workflow did not generate an agent_compare_bundle artifact")
    return bundles[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a browser smoke test for the Agent causal workflow.")
    parser.add_argument("--url", default="http://localhost:8501")
    parser.add_argument("--rows", type=int, default=220)
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--timeout-ms", type=int, default=180_000)
    args = parser.parse_args()

    reports_dir = Path(args.reports_dir)
    started_at = time.time()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1800, "height": 1300}, device_scale_factor=1)
        page.goto(args.url, wait_until="domcontentloaded", timeout=60_000)
        page.get_by_text("DeepUplift Agent Workbench").first.wait_for(timeout=60_000)

        load_rows = page.get_by_label("加载行数").first
        load_rows.fill(str(args.rows))
        load_rows.press("Enter")
        page.wait_for_timeout(1500)

        page.get_by_role("tab", name="Agent").click(timeout=15_000)
        page.get_by_text("Run Causal Workflow", exact=True).last.click(timeout=15_000)
        body = page.locator("body").inner_text(timeout=15_000)
        if "Use model-specific recommended presets" not in body:
            raise AssertionError("Agent workflow preset controls are not visible")
        page.get_by_role("button", name="Run workflow").click(timeout=30_000)
        expect(page.get_by_text("Workflow finished.").first).to_be_visible(timeout=args.timeout_ms)
        expect(page.get_by_text("Download workflow bundle manifest").first).to_be_visible(timeout=30_000)
        browser.close()

    bundle_path = latest_agent_bundle(reports_dir, since=started_at)
    validation = validate_ui_compare_bundle(reports_dir, require=True)
    print(
        json.dumps(
            {
                "status": "ok",
                "bundle": str(bundle_path),
                "validation": validation,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
