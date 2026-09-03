# Contributing to DeepUplift

欢迎提交 issue、文档、数据适配器、模型 adapter 和决策评估改进。

提交前请：

1. 不提交 `.env`、密钥、个人路径、私有数据、模型权重或生成的 `runs/`、`reports/`；
2. 说明数据假设、treatment/outcome 定义和适用 task；
3. 运行 `python3 -m compileall -q app.py deepuplift deepuplift_multiple_treatment scripts`；
4. 如果改动指标或策略逻辑，补充离线 fixture、随机种子和限制说明；
5. 把“框架可运行”“离线证据”“线上业务结论”分开描述。

较大的模型、数据集和后端集成请先开 issue 讨论接口和许可证边界。
