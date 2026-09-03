from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def latest(root: Path, pattern: str) -> Path | None:
    matches = sorted(root.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def read_json(path: Path | None) -> dict[str, Any]:
    if not path or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def best_regression(regression: dict[str, Any]) -> dict[str, Any]:
    rows = regression.get("results") or []
    if not rows:
        return {}
    return max(rows, key=lambda row: row.get("qini") if row.get("qini") is not None else float("-inf"))


def build_onepager(reports_dir: Path) -> str:
    generated_at = time.strftime("%Y-%m-%d %H:%M:%S %z")
    audit_path = latest(reports_dir, "model_catalog_audit_*.json")
    regression_path = latest(reports_dir, "agent_regression_*.json")
    compare_path = latest(reports_dir, "no_ui_compare_manifest_*.json")
    screenshots_path = latest(reports_dir, "ui_screenshots_*")
    zip_path = reports_dir / "deepuplift_agent_interview_pack_latest.zip"
    audit = read_json(audit_path)
    regression = read_json(regression_path)
    compare = read_json(compare_path)
    best = best_regression(regression)
    ready_sources = audit.get("ready_source_counts") or {}
    ready_sources_text = "、".join(f"{source} {count}" for source, count in sorted(ready_sources.items())) or "暂无"
    source_total = len(audit.get("source_counts") or {})
    family_total = len(audit.get("ready_family_counts") or {})
    compare_count = len(compare.get("ranking") or [])

    return f"""# DeepUplift Agent 中文项目一页纸

生成时间：`{generated_at}`

## 项目定位

面向增长、营销、CRM、推荐和广告投放场景的 **Uplift Modeling & Causal Decision Workbench**。项目目标不是预测“谁会转化”，而是判断“谁因为触达才会产生增量转化”，并把这个判断落到可诊断、可训练、可评估、可投放、可复现的完整工作流里。

## 我做了什么

- 将 DeepUplift 从模型脚本库改造成可视化 causal decision workbench。
- 设计统一 model registry、adapter、model card、preset 和 dependency guard，覆盖 **{audit.get("registered_models", "NA")} 个注册模型**，其中 **{audit.get("ready_models", "NA")} 个本地 ready**。
- 打通 Data、Diagnostics、Train、Compare、Predict、Policy、History、Models、Knowledge、Story、Agent 页面。
- 增加 overlap、selection bias / SSB proxy、feature balance、calibration、bootstrap CI、sensitivity、policy value、evidence bundle 等决策级能力。
- 构建 full regression gate：编译、模型审计、训练 smoke、no-UI compare、UI 截图、artifact 校验、resume snapshot、evidence index、interview ZIP 全链路验证。

## 当前可量化结果

| 指标 | 数值 |
| --- | ---: |
| 注册模型 | {audit.get("registered_models", "NA")} |
| 本地 ready 模型 | {audit.get("ready_models", "NA")} |
| 后端来源数 | {source_total} |
| ready 模型族数 | {family_total} |
| ready 后端 | {ready_sources_text} |
| 回归 smoke runs | {len(regression.get("results") or [])} |
| no-UI compare candidates | {compare_count} |
| 最新 smoke 最优模型 | {best.get("model", "NA")} |
| 最新 smoke QINI | {best.get("qini", "NA")} |
| 最新 smoke AUUC | {best.get("auuc", "NA")} |

## 面试可讲难点

1. Uplift modeling 和普通转化率预测的区别：从 `P(Y|X)` 转向 `E[Y|T=1,X]-E[Y|T=0,X]`。
2. Treatment/control 设计：先诊断随机化、样本比例、特征平衡和 outcome 分布，再训练模型。
3. Selection bias / SSB / confounding：用 propensity、weak overlap、feature balance 和推荐模型族做风险控制。
4. 多模型统一工程：用 registry、adapter、model card、preset 管住 {audit.get("registered_models", "NA")} 个模型的扩展复杂度。
5. 输出 contract 统一：所有模型归一成 `y0_pred / y1_pred / uplift_score` 后再评估。
6. 决策级评估：QINI/AUUC 之外增加 Top-K、policy value、calibration、bootstrap CI 和 sensitivity。
7. 可复现治理：每个 run 保存 metrics、predictions、model、run note、evidence ZIP，并通过 full regression gate 验证。

## 推荐简历写法

> 设计并实现 DeepUplift Agent，一个面向增长营销场景的 uplift modeling & causal decision workbench，支持 treatment/control 数据诊断、{audit.get("registered_models", "NA")} 个 uplift/causal 模型目录、{audit.get("ready_models", "NA")} 个本地可运行模型、多模型自动对比、QINI/AUUC/Top-K/policy value/calibration/bootstrap CI 评估、预测名单导出和 evidence bundle 回归验证。

## 现场演示路径

```bash
scripts/deepuplift_agent.sh start
scripts/deepuplift_agent.sh demo-lite
scripts/deepuplift_agent.sh demo
scripts/deepuplift_agent.sh pack
```

页面顺序：`Story -> Data -> Diagnostics -> Models -> Agent -> Compare -> Policy -> History`。

## 证据入口

- 模型审计：`{audit_path or "missing"}`
- 回归报告：`{regression_path or "missing"}`
- Compare manifest：`{compare_path or "missing"}`
- UI 截图目录：`{screenshots_path or "missing"}`
- 面试 ZIP：`{zip_path}`
- Evidence index：`docs/DEEPUplift_AGENT_EVIDENCE_INDEX.md`
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a one-page Chinese DeepUplift Agent project summary.")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--output", default="docs/DEEPUplift_AGENT_ONEPAGER_CN.md")
    args = parser.parse_args()

    markdown = build_onepager(Path(args.reports_dir))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown, encoding="utf-8")
    print(json.dumps({"status": "ok", "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
