from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepuplift.core.failure_modes import (
    diagnostic_case_rows,
    diagnostic_case_summary_rows,
    model_failure_mode_rows,
    scenario_pitfall_rows,
)


DOCS = ROOT / "docs"


def clean(value: object) -> str:
    return str(value).replace("\n", "<br>").replace("|", "\\|")


def md_table(rows: list[dict], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(clean(row.get(col, "")) for col in columns) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def build_model_failure_doc() -> None:
    rows = model_failure_mode_rows()
    write(
        DOCS / "UPLIFT_MODEL_FAILURE_MODES.md",
        f"""
# Uplift 模型 Failure Modes

这份文档对应 UI：`Models -> Model Risk Matrix` 和 `Evaluation -> Failure Modes / 指标误导案例`。

## 模型风险矩阵

{md_table(rows, ["family", "risk", "symptom", "root_cause", "diagnostic", "misleading_metric", "fix", "ui_evidence", "interview_line"])}

## 面试讲法

不要只说“我支持很多模型”。更专家的讲法是：每类模型都有适用条件和失效模式。
我会先用 P0 baseline 建立可解释基线，再用 overlap、bootstrap CI、calibration、policy value 和 scenario-specific risk guardrail 判断模型是否可信。
复杂模型只有在 treatment feature、样本量、反馈窗口和业务价值函数都支持时才值得升级。
""",
    )


def build_scenario_pitfall_doc() -> None:
    rows = scenario_pitfall_rows()
    write(
        DOCS / "UPLIFT_SCENARIO_PITFALLS.md",
        f"""
# Uplift 业务场景 Pitfall Cards

这份文档对应 UI：`场景工作台 -> Scenario Pitfall Cards`。

## 场景问题库

{md_table(rows, ["scenario", "pitfall", "symptom", "diagnostic", "fix", "interview_line"])}

## 面试讲法

工业 uplift 的难点往往不是“能不能训练模型”，而是业务目标是否定义对了。
发券要扣成本和套利，广告要看增量而不是归因，CRM 要看疲劳和延迟，Marketplace 要看干扰，LLM routing 要看增量质量减成本。
""",
    )


def build_diagnostic_cases_doc() -> None:
    rows = diagnostic_case_rows()
    summary = diagnostic_case_summary_rows()
    summary_table = md_table(summary, list(summary[0].keys())) if summary else "_Run `scripts/generate_uplift_diagnostic_cases.py` to refresh mock summaries._"
    write(
        DOCS / "UPLIFT_DIAGNOSTIC_CASES.md",
        f"""
# Uplift Mock Diagnostic Cases

这份文档对应 UI：`Evaluation -> Mock Diagnostic Cases` 和 `Evidence -> Diagnostic Cases`。

## Case Card 设计

{md_table(rows, ["case_id", "case_name", "problem_type", "mock_signal", "affected_models", "misleading_metric", "diagnostic_method", "fix_path", "interview_line"])}

## 最新 Mock Summary

{summary_table}

## 面试讲法

我用 mock cases 不是为了替代真实业务数据，而是为了把失败模式讲清楚：
同一套 trainer/evaluator/UI 可以展示弱 uplift、poor overlap、QINI 高但 ROI 低、D1/D30 标签成熟差异、full-funnel click trap、geo holdout 冲突和 LLM routing 成本陷阱。
""",
    )


def build_metric_misleading_doc() -> None:
    rows = [
        {
            "metric": "QINI / AUUC",
            "misleading_case": "high_qini_low_roi",
            "why": "排序能找到响应人群，但没有扣 treatment cost、风险惩罚和预算约束。",
            "fix": "同时报告 incremental profit、policy value、ROI/iROAS 和 cost-aware threshold。",
        },
        {
            "metric": "uplift@K",
            "misleading_case": "imbalanced_treatment / extreme_propensity",
            "why": "Top-K 桶里 treatment/control 样本少或 common support 差时，observed uplift 高方差。",
            "fix": "加 bootstrap CI、minimum support gate、overlap trimming。",
        },
        {
            "metric": "click uplift",
            "misleading_case": "full_funnel_click_trap",
            "why": "点击增量可能不带来转化或利润增量。",
            "fix": "使用 full-funnel ECUP 和 conversion/profit stage policy value。",
        },
        {
            "metric": "D1 conversion",
            "misleading_case": "delayed_feedback_trap",
            "why": "短期标签会低估慢响应用户和长期留存价值。",
            "fix": "看 D1/D7/D14/D30 uplift、censored rate 和成熟窗口。",
        },
        {
            "metric": "offline policy value",
            "misleading_case": "geo_holdout_conflict",
            "why": "离线模型可能受 selection bias 或时空干扰影响，线上 geo holdout 不稳定。",
            "fix": "geo holdout、switchback、shadow rollout 和预算小流量验证。",
        },
        {
            "metric": "LLM strong model win-rate",
            "misleading_case": "llm_routing_cost_trap",
            "why": "强模型绝对质量提升不一定覆盖模型成本和延迟。",
            "fix": "quality uplift * value - model_cost - latency_penalty。",
        },
    ]
    write(
        DOCS / "UPLIFT_METRIC_MISLEADING_CASES.md",
        f"""
# Uplift 指标误导案例

这份文档对应 UI：`Evaluation -> Failure Modes / 指标误导案例`。

{md_table(rows, ["metric", "misleading_case", "why", "fix"])}

## 面试讲法

我不会把 QINI/AUUC 当作最终上线结论。QINI/AUUC 是 ranking 证据，uplift@K 是投放桶证据，policy value/ROI 是业务价值证据，calibration/bootstrap/overlap 是可信度证据，geo holdout/A-B 是线上增量证据。
""",
    )


def main() -> None:
    build_model_failure_doc()
    build_scenario_pitfall_doc()
    build_diagnostic_cases_doc()
    build_metric_misleading_doc()
    print("generated failure mode docs")


if __name__ == "__main__":
    main()
