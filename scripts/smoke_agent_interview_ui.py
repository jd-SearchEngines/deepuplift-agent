from __future__ import annotations

import argparse

import time

from playwright.sync_api import sync_playwright


def wait_for_terms(page, terms: list[str], timeout_ms: int) -> str:
    deadline = time.time() + timeout_ms / 1000
    body = ""
    while time.time() < deadline:
        body = page.locator("body").inner_text(timeout=15_000)
        missing = [term for term in terms if term.lower() not in body.lower()]
        if not missing:
            return body
        page.wait_for_timeout(500)
    raise AssertionError(f"Page body missing expected terms: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test interview-coach prompts in the Agent tab.")
    parser.add_argument("--url", default="http://localhost:8501")
    parser.add_argument("--timeout-ms", type=int, default=60_000)
    args = parser.parse_args()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1800, "height": 1200}, device_scale_factor=1)
        page.goto(args.url, wait_until="domcontentloaded", timeout=args.timeout_ms)
        page.get_by_text("DeepUplift Agent Workbench").first.wait_for(timeout=args.timeout_ms)
        page.get_by_role("tab", name="Agent", exact=True).click(timeout=15_000)

        page.get_by_role("button", name="为什么不用普通转化率模型？").click(timeout=15_000)
        wait_for_terms(page, ["难点 I01", "P(Y=1|X)", "uplift", "policy value"], args.timeout_ms)

        page.get_by_role("button", name="观测数据有偏怎么办？").click(timeout=15_000)
        wait_for_terms(page, ["难点 I03", "selection bias", "SSB", "overlap"], args.timeout_ms)

        page.get_by_role("button", name="5分钟怎么演示？").click(timeout=15_000)
        wait_for_terms(page, ["5 分钟面试演示路径", "Scenario Model Matrix", "Scenario ROI Helper", "evidence pack"], args.timeout_ms)
        browser.close()

    print("Agent interview UI smoke passed.")


if __name__ == "__main__":
    main()
