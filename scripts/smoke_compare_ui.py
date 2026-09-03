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

from scripts.validate_latest_artifacts import read_json


def latest_compare_bundle(reports_dir: Path, since: float) -> Path:
    bundles = sorted(
        [
            path
            for path in reports_dir.glob("compare_bundle_*.json")
            if path.stat().st_mtime >= since - 1
        ],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not bundles:
        raise AssertionError("Compare UI did not generate a compare_bundle artifact")
    return bundles[0]


def validate_compare_bundle(path: Path) -> dict:
    payload = read_json(path)
    ranking = payload.get("ranking") or []
    runs = payload.get("runs") or []
    model_cards = payload.get("model_cards") or []
    if int(payload.get("schema_version") or 1) < 2:
        raise AssertionError(f"Expected schema v2 compare bundle: {path}")
    if len(ranking) < 2 or len(runs) < 2 or len(model_cards) < 2:
        raise AssertionError(f"Compare bundle is incomplete: {path}")
    return {
        "path": str(path),
        "ranking": len(ranking),
        "runs": len(runs),
        "model_cards": len(model_cards),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a browser smoke test for the Compare page.")
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

        page.get_by_role("tab", name="模型对比").click(timeout=15_000)
        page.get_by_role("button", name="Train comparison").click(timeout=30_000)
        expect(page.get_by_text("Comparison finished.").first).to_be_visible(timeout=args.timeout_ms)
        body = page.locator("body").inner_text(timeout=30_000)
        required = [
            "Download compare report",
            "Download compare readiness JSON",
            "Download compare bundle manifest",
        ]
        missing = [text for text in required if text not in body]
        if missing:
            raise AssertionError(f"Compare download controls missing: {missing}")
        browser.close()

    bundle_path = latest_compare_bundle(reports_dir, since=started_at)
    validation = validate_compare_bundle(bundle_path)
    print(json.dumps({"status": "ok", "bundle": validation}, ensure_ascii=False))


if __name__ == "__main__":
    main()
