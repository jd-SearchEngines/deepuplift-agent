# Lightweight Conversation Notes

## Agreed framing

DeepUplift 不只是模型集合，也不应被包装成只服务某一个行业的工具。它的通用定位是增长、营销、广告、CRM、补贴、推荐和预算分配场景的 causal decision / uplift modeling workbench。

核心表达：

> 传统 CVR 预测回答“谁会转化”，uplift / CATE 估计回答“谁会因为干预而改变”。平台进一步把这个估计转成可审查的人群、策略、预算和证据。

## Decisions to preserve

- 先做 treatment/outcome/feature 诊断，再选模型；
- 不用单个 QINI 或 AUUC 直接决定上线；
- 同时关注 Top-K uplift、policy value、ROI/iROAS、calibration、bootstrap、overlap 和 sensitivity；
- 使用 model registry、model card、preset、run history 和 evidence bundle 管理模型差异；
- 真实生产结论必须来自随机化实验、holdout、灰度和回滚监控；
- 保留真实失败和依赖缺失状态，不用合成结果冒充线上证据。

## Future GPT handoff

后续讨论开源平台时，优先围绕三层接口、数据授权、可复现 benchmark、模型插件机制、决策策略协议和线上实验边界展开。不要重新把项目收缩成简历演示或某一个职位的定制项目。
