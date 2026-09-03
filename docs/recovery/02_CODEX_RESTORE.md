# Project Context Recovery

给后续 GPT / Codex 的项目上下文摘要。完整对话不进入仓库；需要继续工作时，先读本文件、`03_LIGHTWEIGHT_CONVERSATION.md` 和 [`docs/architecture/THREE_LAYER_PLATFORM.md`](../architecture/THREE_LAYER_PLATFORM.md)。

## Project identity

```text
DeepUplift Agent｜面向增长、营销与资源分配的因果决策平台
```

平台把 uplift / CATE 从单一模型训练扩展为：数据设计诊断、模型比较、策略价值评估、人群筛选、证据生成和上线前 readiness 检查。

当前版本是 `v0.2.0` framework skeleton：binary coupon allocation 已有一键 E2E；multi-discrete、continuous、optional backend 和大数据 backend 已建立 contract/extension boundary。

## Current boundaries

- 数据层：`deepuplift.data`，负责 schema、预处理、数据 profile、manifest 和 causal diagnostics。
- 模型层：`deepuplift.models` 加 `deepuplift.core.registry`，负责模型工厂、adapter、训练和依赖声明。
- 决策层：`deepuplift.decision`，负责指标、校准、Top-K、policy value、OPE、readiness 和证据。
- 应用入口：`app.py` 和 `scripts/`，负责 Streamlit、CLI smoke、回归和文档生成。
- `deepuplift.core` 当前是兼容层与应用服务层，不能被误解为第四个业务层。

## Evidence discipline

必须区分：框架/语法通过、真实模型离线证据、线上实验结果和生产结论。没有随机化实验或可信 holdout，不要声称 online lift、ROI 提升或生产可用。

## Next discussion topics

1. 继续冻结三层 API 和 config/artifact schema；
2. 增加公开数据集和可复现 benchmark；
3. 建立 model card、evidence manifest 和版本化协议；
4. 最后再讨论在线策略服务、实验平台、权限和审计。
