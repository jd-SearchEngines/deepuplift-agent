from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepuplift.core.evaluation_formulas import (
    evaluation_formula_rows,
    optimization_path_rows,
    policy_derivation_rows,
)
from deepuplift.core.open_source_frameworks import (
    open_source_framework_counts,
    open_source_framework_rows,
    framework_status_summary,
)


DOCS = Path("docs")
REPORTS = Path("reports")


def md_table(rows: list[dict[str, str]], columns: list[str]) -> str:
    def clean(value: object) -> str:
        return str(value).replace("\n", "<br>").replace("|", "\\|")

    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(clean(row.get(col, "")) for col in columns) + " |")
    return "\n".join([header, sep, *body])


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def build_metric_doc() -> None:
    rows = evaluation_formula_rows()
    write(
        DOCS / "UPLIFT_EVALUATION_METRICS_FORMULAS.md",
        f"""
# Uplift 评估指标公式与推导

这份文档对应 UI：`Evaluation -> Metric Formula Cards`。它把 uplift 评估拆成四层：

- 排序：QINI、AUUC、uplift@K。
- 业务价值：policy value、incremental profit、ROI / iROAS。
- 可信度：calibration、bootstrap CI、overlap / trimming。
- 工业扩展：delayed feedback、full-funnel ECUP、LLM routing cost-quality。

## 公式总表

{md_table(rows, ["metric", "formula", "derivation_intuition", "business_meaning", "when_to_use", "misleading_when", "code_source", "artifact_field"])}

## 面试讲法

1. 先定义 potential outcome：同一个用户只有一个事实结果，uplift 要估计未观察到的反事实差值。
2. 再讲 ranking：QINI/AUUC 只判断排序，不直接等于上线收益。
3. 然后讲业务决策：发券、广告、LLM routing 都要把 uplift 乘以价值并扣成本。
4. 最后讲可信度：overlap、bootstrap、calibration、sensitivity 是上线前的闸门。
""",
    )


def build_policy_doc() -> None:
    rows = policy_derivation_rows()
    write(
        DOCS / "UPLIFT_POLICY_VALUE_AND_ROI_DERIVATION.md",
        f"""
# Policy Value 与 ROI 推导

这份文档对应 UI：`Policy -> Scenario ROI Lab` 和 `Evaluation -> Policy Value Formula Cards`。

## 推导卡片

{md_table(rows, ["topic", "formula", "steps", "business_example", "ui_location", "code_source"])}

## 核心结论

- uplift score 是排序分数，policy value 是部署策略价值。
- 发券：`incremental_profit = uplift * gross_margin - coupon_cost - abuse/fatigue_penalty`。
- 广告：`iROAS = incremental_revenue / media_cost`，分子必须是增量收入，不是归因收入。
- LLM routing：`quality_uplift * business_value - model_cost - latency_penalty`。
- 预算约束下的最优阈值不是固定 Top 10%，而是满足预算和风险闸门的最大净收益点。
""",
    )


def build_optimization_path_doc() -> None:
    rows = optimization_path_rows()
    write(
        DOCS / "UPLIFT_MODEL_OPTIMIZATION_PATH.md",
        f"""
# Uplift 项目从 0-1 到相对最优的优化路径

这份文档对应 UI：`Story / 场景工作台 / Evaluation`。它用于说明这个项目不是一次性堆模型，而是按照工业落地链路持续优化。

{md_table(rows, ["stage", "why", "what_changes", "metric_effect", "risk", "evidence"])}

## 面试讲法

可以按下面逻辑展开：

1. V0 用业务规则或转化率模型建立反例，指出“高转化不等于高增量”。
2. V1-V2 用 S/T/X/DR/R learner 把建模目标从 CVR 切到 CATE，并处理不均衡和观测偏差。
3. V3-V5 加入 CausalForest、uplift tree、深度 interaction/contrastive 模型，增强异质性和复杂交互。
4. V6-V7 把离线 ranking 接到 ROI、预算、calibration、bootstrap CI、overlap gate。
5. V8 用 holdout/geo/ghost ads/shadow rollout 把离线结论对齐线上增量。
""",
    )


def build_framework_doc() -> None:
    rows = open_source_framework_rows()
    counts = open_source_framework_counts()
    status_rows = framework_status_summary()
    write(
        DOCS / "UPLIFT_OPEN_SOURCE_FRAMEWORK_AUDIT.md",
        f"""
# Uplift / CATE 开源框架审计

这份文档对应 UI：`Models -> Open Source Framework Audit` 和 `Knowledge -> Open Source Framework Audit`。

## 状态摘要

- total: `{counts["total"]}`
- ready: `{counts["ready"]}`
- guarded: `{counts["guarded"]}`
- optional: `{counts["optional"]}`
- backlog: `{counts["backlog"]}`

{md_table(status_rows, ["status", "frameworks", "examples", "how_to_read"])}

## 框架总表

{md_table(rows, ["framework", "owner", "source_type", "github_repo", "github_stars", "github_activity", "supported_models", "treatment_support", "scenario_fit", "local_status", "integration_path", "limitations", "next_step", "url"])}

## 工程原则

- P0 先用 `LightGBM + S/T/X/R/DR`、`EconML` 和 `scikit-uplift` 打稳定基线。
- CausalML、UTBoost、CatBoost、grf、DoubleML、DoWhy 先通过 guarded/optional gate，不影响主流程。
- 不把“有论文/有库”直接等价为“当前平台 fully runnable”。
- 每个框架都要能追到 source、adapter、model card、metrics、UI 和 evidence。
""",
    )
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "open_source_framework_audit_latest.json").write_text(
        json.dumps({"counts": counts, "rows": rows, "status_rows": status_rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_industrial_expansion_doc() -> None:
    write(
        DOCS / "UPLIFT_INDUSTRIAL_CASE_EXPANSION.md",
        """
# 工业案例扩充与场景库接入说明

本轮平台补强的原则是：工业案例不能只写成资料列表，必须映射到可跑数据、模型对比、评估指标和 Policy 决策。

## 已落地到工作台的场景

- 百度外卖天降红包：红包成本、毛利、套利风险、sure thing / sleeping dog、Top-K 增量利润。
- 滴滴 C 端乘客补贴：乘客补贴金额、发单/完单、延迟留存、预算约束。
- 滴滴 B 端司机补贴：供给上线、接单/完单、等待时间下降、spillover / SUTVA 风险。
- 滴滴城市/时段预算分配：城市供需缺口、边际 ROI、预算分配、跨市场干扰。
- Shopee 广告投放：impression-click-conversion 全链路、iROAS、holdout/audience targeting。
- DoorDash ghost ads、Spotify 站内消息、Airbnb incremental LTV、marketplace subsidy、LLM routing 等扩展场景。
- 电商多券面额/多门槛：coupon face value、min spend、套利风险、成本敏感 ROI。
- CRM 多 journey 重叠触达：overlap index、疲劳、退订、蚕食风险和长期价值。
- 连续出价/折扣剂量响应：bid multiplier、discount rate、saturation、边际成本和 dose-response。
- Geo holdout / 地理增量实验：geo/day holdout cell、switchback block、synthetic-control gap、weak overlap 和 spillover risk。

## 每个案例进入平台的映射

1. `examples/datasets/manifest.json`：数据集入口。
2. `deepuplift/core/industrial_scenario_lab.py`：场景 schema、policy 目标、风险字段。
3. `deepuplift/core/resume_scenario_workbench.py`：履历场景的 compare、policy、优化路径。
4. `deepuplift/core/industry_playbook.py`：工业资料和 Agent 问答证据。
5. `Evaluation / Policy / Models / Evidence / Agent`：可视化体现。

## 下一步扩充方向

- 电商多券面额和多门槛。
- CRM 多 journey 重叠触达。
- merchant cofund / marketplace subsidy。
- creative / text treatment uplift。
- continuous treatment：补贴金额、出价、折扣率、触达频次。
- delayed feedback 和 full-funnel 神经网络 adapter。
- geo experiment / synthetic control artifact import，把离线 uplift 排序接到线上增量验证。
""",
    )


def main() -> None:
    build_metric_doc()
    build_policy_doc()
    build_optimization_path_doc()
    build_framework_doc()
    build_industrial_expansion_doc()
    print("generated evaluation/framework docs")


if __name__ == "__main__":
    main()
