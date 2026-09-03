from __future__ import annotations

import argparse
import time
from pathlib import Path

from playwright.sync_api import sync_playwright


EXPECTED_TEXT = {
    "Workflow": [
        "Industrial Uplift Workflow",
        "Business Problem / Treatment Design",
        "Data & Diagnostics",
        "Model Selection",
        "Evaluation Metrics",
        "Policy / ROI Decision",
        "Predict / Scoring",
        "Evidence / Reproducibility",
        "Current Session Evidence Status",
        "Optimization Closure Board",
        "Repo Change Intake Board",
        "Industrial Gate Checklist",
        "ML Training Workbench Overview",
        "Growth Decision Control Plane",
    ],
    "场景工作台": [
        "履历业务场景工作台",
        "场景内 Model Compare",
        "综合分",
        "Top-K policy value / ROI / 风险",
        "从 0-1 到相对最优的优化路径",
        "当前模型是否业内先进",
        "模型限制与依赖",
        "工业扩展场景雷达",
    ],
    "Data": ["Dataset", "Treatment"],
    "Agent": [
        "Agent Chat",
        "Quick Prompt Groups",
        "模型选择",
        "评估与 Policy",
        "工业场景",
        "LLM / Agent",
        "前端设计",
        "推荐哪个模型？",
        "解释当前模型卡？",
        "uplift 模型到底怎么评估？公式怎么讲？",
        "为什么 QINI/AUUC 不等于增量利润？",
        "OPE / IPS / DR 怎么评估新策略？",
        "LLM routing 为什么是 uplift 问题？",
        "什么时候从 uplift 升级到 bandit？",
        "参考了哪些 GitHub 前端框架？",
        "为什么当前阶段还是 Streamlit，而不是直接 React？",
        "哪些设计模式被吸收到当前页面？",
        "训练平台设计",
        "参考了哪些开源模型训练平台？",
        "哪些训练平台设计被吸收到当前页面？",
        "公开产品参考",
        "参考了哪些字节/火山引擎公开数据产品？",
        "哪些公开产品设计被吸收到当前页面？",
        "Agent Evidence Card",
        "Download Agent transcript",
    ],
    "Models": ["Model Catalog", "Frontend Framework Audit / UI Inspiration Matrix", "ML Training Platform Inspiration Matrix", "Training Platform Pattern Adoption", "MLflow", "ByteDance-style Public Product Inspiration Matrix", "Public Product Pattern Adoption", "Public Products", "Model Source & Code Provenance", "Industrial Model Layer Architecture", "Frontier Plugin Backlog", "Scenario-to-Model Decision Ladder", "P0 baseline", "P1 plugin", "P2 extension", "UTBoost", "ECUP / full-funnel uplift", "Treatment Type Matrix", "Scenario Model Matrix", "Business Scenario Model Recommendation", "Treatment Type", "Scenario Tag", "Scenario Tags", "Model Card", "Preset Coverage", "Ready Backends", "Guarded Backend Runtime Guide"],
    "Model-Deconstruction": [
        "Model Deconstruction Workbench",
        "Deconstruction Coverage",
        "Deep Model Training Evidence",
        "Deep Model Objective Diagnostics",
        "Algorithm Claim Ledger",
        "Paper Reproduction Gap Ledger",
        "Model Evidence Promotion Matrix",
        "Model Upgrade Recipe Cards",
        "Deep Model Telemetry Smoke",
        "Deep Model Forward Hook Contracts",
        "Deep Model Forward Hook Smoke",
        "Deep Model Hook Training Bridge",
        "Deep Model Trainer Collector Smoke",
        "Deep Model Telemetry-Benchmark Linkage",
        "Deep Model Telemetry Ablation Gate",
        "Deep Model Ablation Interpretation",
        "Deep Model Ablation Promotion Matrix",
        "Deep Model Ablation Next Experiment Plan",
        "Deep Model Ablation Command Contract",
        "Deep Model Ablation Command Contract Smoke",
        "Deep Model Ablation Recommended Command Smoke",
        "Deep Model Telemetry Gap Matrix",
        "Deep Model Battle Report",
        "Battle Interpretation / 面试讲法",
        "Ablation Interpretation / 面试讲法",
        "Ablation Promotion / 面试讲法",
        "Ablation Next Experiments / 面试追问",
        "Paper-Level Benchmark Evidence",
        "Benchmark Dashboard",
        "Benchmark Interpretation / 面试讲法",
        "术语解释 / Glossary",
        "Source Call Chain",
        "Code / Math / Flow",
        "External Sources And License Gate",
        "Golden Question Regression",
        "Model Deconstruction Agent",
        "Download model deconstruction catalog CSV",
        "Model Review Card",
        "Selected Model Battle Cards",
    ],
    "Evaluation": ["Evaluation Metrics", "术语解释 / Glossary", "Metric Risk Legend", "QINI", "AUUC", "uplift@K / Top-K observed uplift", "policy value", "incremental profit / cost-aware gain", "delayed feedback uplift", "full-funnel ECUP metric", "calibration", "bootstrap CI", "overlap / propensity", "oracle top-k recall"],
    "Knowledge": [
        "Causal Knowledge Base",
        "术语速查 / Glossary",
        "Industrial Knowledge Map",
        "Source Adoption Drilldown",
        "LLM + Causal Frontier Map",
        "LLM Interview Q&A",
        "Latest Regression Health",
        "Ready Backends",
        "Source Readiness Trend",
        "Latest Regression Runs",
        "Model Availability",
    ],
    "Story": ["Project Story", "Optimization Closure Board", "Visual Upgrade Before / After", "UI Inspiration Matrix", "ML Training Platform Inspiration Matrix", "ByteDance-style Public Product Inspiration Matrix", "Industrial Demo Path", "Resume / Interview Evidence", "Production Migration Snapshot", "Models", "Policy", "Evidence", "Show full evidence library", "Download Resume project brief", "Download Evidence ZIP"],
    "Evidence": ["Evidence Dashboard", "Optimization Closure Board", "Repo Change Intake Board", "Regression Gate Artifacts", "Regression Warning / Launch Guardrail Audit", "Promotion / Launch Review Cards", "Regression Warning Triage", "Pilot Readiness Scorecard", "Pilot Experiment Plan", "Pilot Experiment Command Smoke", "Code / Model Provenance", "Frontier Model Source Gate", "Scenario Model Decision Ladder", "Source Quality Gate", "Latest Training Smoke", "LLM Routing Smoke", "Frontier Synthetic Dataset Smoke", "Frontier Training Evidence", "Deep Model Training Evidence", "Deep Model Objective Diagnostics", "Algorithm Claim Ledger", "Paper Reproduction Gap Ledger", "Model Evidence Promotion Matrix", "Model Upgrade Recipe Cards", "Deep Model Telemetry Smoke", "Deep Model Forward Hook Contracts", "Deep Model Forward Hook Smoke", "Deep Model Trainer Collector Smoke", "Deep Model Telemetry-Benchmark Linkage", "Deep Model Telemetry Ablation Gate", "Deep Model Ablation Interpretation", "Ablation Interpretation / 面试讲法", "Deep Model Ablation Promotion Matrix", "Ablation Promotion / 面试讲法", "Deep Model Ablation Next Experiment Plan", "Ablation Next Experiments / 面试追问", "Deep Model Ablation Command Contract", "Command Contract / 命令可执行性证据", "Deep Model Ablation Command Contract Smoke", "Deep Model Ablation Recommended Command Smoke", "Deep Model Ablation Command Smoke Coverage", "Deep Model Telemetry Gap Matrix", "Deep Model Battle Report", "Battle Interpretation / 面试讲法", "Paper-Level Benchmark Evidence", "Benchmark Dashboard", "Benchmark Interpretation / 面试讲法", "术语解释 / Glossary", "Model Deconstruction Evidence", "UI Screenshot Gallery / Visual Regression Evidence", "ML Training Platform Source Gate", "Public Product Source Gate", "Download Evidence Pack", "LLM interview Q&A"],
    "Policy": ["Policy Value", "Industrial Threshold Rule", "Coupon / Ads / Growth ROI Templates", "Scenario ROI Helper", "Scenario ROI Lab", "Uplift Application Expansion Matrix", "Launch Guardrail Board", "Metric Disagreement Lab", "incremental profit", "LLM Routing ROI Simulator", "Business scenario", "Offline-To-Online Validation Playbook"],
    "History": [
        "Run History",
        "Readiness level",
        "Minimum readiness score",
        "Tags",
        "Run Tags / Notes",
        "Mark candidate",
        "Evidence Summary",
        "Download selected run evidence ZIP",
    ],
    "Compare": ["Model Compare", "Compare Candidate Presets", "Use model-specific recommended presets"],
    "Predict": ["Predict"],
}


def visible_page_text(page) -> str:
    parts = [page.locator("body").inner_text(timeout=15_000)]
    tab_containers = page.locator('[data-testid="stTabs"]')
    for index in range(min(tab_containers.count(), 4)):
        try:
            parts.append(tab_containers.nth(index).inner_text(timeout=5_000))
        except Exception:
            continue
    return "\n".join(parts)


def click_streamlit_tab(page, tab_label: str) -> None:
    try:
        page.get_by_role("tab", name=tab_label, exact=True).click(timeout=5_000, force=True)
        return
    except Exception:
        tab = page.locator('[data-testid="stTab"]').filter(has_text=tab_label).first
        tab.click(timeout=15_000, force=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture UI smoke screenshots from a running Streamlit app.")
    parser.add_argument("--url", default="http://localhost:8501")
    parser.add_argument("--tabs", nargs="+", default=["Workflow", "Data", "Models", "Model-Deconstruction", "Evaluation", "Evidence", "Agent"])
    parser.add_argument("--width", type=int, default=1800)
    parser.add_argument("--height", type=int, default=1100)
    parser.add_argument("--min-bytes", type=int, default=20_000)
    args = parser.parse_args()

    out_dir = Path("reports") / f"ui_screenshots_{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": args.width, "height": args.height}, device_scale_factor=1)
        page.goto(args.url, wait_until="domcontentloaded", timeout=60_000)
        page.get_by_text("DeepUplift Agent Workbench").first.wait_for(timeout=60_000)
        captured = []
        tab_aliases = {
            "Workflow": "流程总览",
            "Scenario": "场景工作台",
            "Resume": "场景工作台",
            "场景": "场景工作台",
            "Data": "数据",
            "Diagnostics": "诊断",
            "Models": "模型",
            "Model-Deconstruction": "模型拆解",
            "Train": "训练",
            "Compare": "模型对比",
            "Evaluation": "指标评估",
            "Policy": "Policy/ROI",
            "Predict": "打分导出",
            "Evidence": "证据",
            "Knowledge": "知识库",
            "Multi-Treatment": "多 Treatment",
            "Dose-Response": "剂量响应",
            "History": "历史",
            "Story": "履历 Story",
        }
        for tab in args.tabs:
            tab_label = tab_aliases.get(tab, tab)
            click_streamlit_tab(page, tab_label)
            if tab == "History":
                page.get_by_text("Run Tags / Notes").first.click(timeout=15_000)
            expected = EXPECTED_TEXT.get(tab_label, EXPECTED_TEXT.get(tab, []))
            body = ""
            missing = expected
            deadline = time.time() + 15
            while time.time() < deadline:
                page.wait_for_timeout(500)
                body = visible_page_text(page)
                missing = [text for text in expected if text not in body]
                if not missing:
                    break
            if missing:
                raise AssertionError(f"Tab {tab} is missing expected text: {missing}")
            path = out_dir / f"{tab.lower().replace('-', '_')}.png"
            page.screenshot(path=str(path), full_page=False)
            if path.stat().st_size < args.min_bytes:
                raise AssertionError(f"Screenshot is too small and may be blank: {path} ({path.stat().st_size} bytes)")
            captured.append(str(path))
        browser.close()

    print("\n".join(captured))


if __name__ == "__main__":
    main()
