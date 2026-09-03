# DeepUplift Agent 中文项目一页纸

生成时间：`2026-05-21 10:01:13 +0800`

## 项目定位

面向增长、营销、CRM、推荐和广告投放场景的 **Uplift Modeling & Causal Decision Workbench**。项目目标不是预测“谁会转化”，而是判断“谁因为触达才会产生增量转化”，并把这个判断落到可诊断、可训练、可评估、可投放、可复现的完整工作流里。

## 我做了什么

- 将 DeepUplift 从模型脚本库改造成可视化 causal decision workbench。
- 设计统一 model registry、adapter、model card、preset 和 dependency guard，覆盖 **103 个注册模型**，其中 **73 个本地 ready**。
- 打通 Data、Diagnostics、Train、Compare、Predict、Policy、History、Models、Knowledge、Story、Agent 页面。
- 增加 overlap、selection bias / SSB proxy、feature balance、calibration、bootstrap CI、sensitivity、policy value、evidence bundle 等决策级能力。
- 构建 full regression gate：编译、模型审计、训练 smoke、no-UI compare、UI 截图、artifact 校验、resume snapshot、evidence index、interview ZIP 全链路验证。

## 当前可量化结果

| 指标 | 数值 |
| --- | ---: |
| 注册模型 | 103 |
| 本地 ready 模型 | 73 |
| 后端来源数 | 8 |
| ready 模型族数 | 7 |
| ready 后端 | DeepUplift 36、EconML 12、LightGBM 11、scikit-uplift 14 |
| 回归 smoke runs | 2 |
| no-UI compare candidates | 2 |
| 最新 smoke 最优模型 | DRLearnerGBM |
| 最新 smoke QINI | -7.318814518814519 |
| 最新 smoke AUUC | -17.90794622044622 |

## 面试可讲难点

1. Uplift modeling 和普通转化率预测的区别：从 `P(Y|X)` 转向 `E[Y|T=1,X]-E[Y|T=0,X]`。
2. Treatment/control 设计：先诊断随机化、样本比例、特征平衡和 outcome 分布，再训练模型。
3. Selection bias / SSB / confounding：用 propensity、weak overlap、feature balance 和推荐模型族做风险控制。
4. 多模型统一工程：用 registry、adapter、model card、preset 管住 103 个模型的扩展复杂度。
5. 输出 contract 统一：所有模型归一成 `y0_pred / y1_pred / uplift_score` 后再评估。
6. 决策级评估：QINI/AUUC 之外增加 Top-K、policy value、calibration、bootstrap CI 和 sensitivity。
7. 可复现治理：每个 run 保存 metrics、predictions、model、run note、evidence ZIP，并通过 full regression gate 验证。

## 推荐简历写法

> 设计并实现 DeepUplift Agent，一个面向增长营销场景的 uplift modeling & causal decision workbench，支持 treatment/control 数据诊断、103 个 uplift/causal 模型目录、73 个本地可运行模型、多模型自动对比、QINI/AUUC/Top-K/policy value/calibration/bootstrap CI 评估、预测名单导出和 evidence bundle 回归验证。

## 现场演示路径

```bash
scripts/deepuplift_agent.sh start
scripts/deepuplift_agent.sh demo-lite
scripts/deepuplift_agent.sh demo
scripts/deepuplift_agent.sh pack
```

页面顺序：`Story -> Data -> Diagnostics -> Models -> Agent -> Compare -> Policy -> History`。

## 证据入口

- 模型审计：`reports/model_catalog_audit_20260521-100105.json`
- 回归报告：`reports/agent_regression_20260521-100109.json`
- Compare manifest：`reports/no_ui_compare_manifest_20260521-100112.json`
- UI 截图目录：`reports/ui_screenshots_20260521-095603`
- 面试 ZIP：`reports/deepuplift_agent_interview_pack_latest.zip`
- Evidence index：`docs/DEEPUplift_AGENT_EVIDENCE_INDEX.md`
