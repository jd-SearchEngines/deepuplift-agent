# DeepUplift

Open-source Uplift & Causal Decision Framework

`Data → Models → Decision`

From heterogeneous treatment effect estimation to deployable business policies.

## DeepUplift Agent

面向增长、营销与资源分配的因果决策工作台（Causal Decision Workbench）。

DeepUplift 的核心问题不是“谁本来就会转化”，而是“谁会因为一次干预而产生增量行为”。它把 uplift / CATE 建模连接到数据诊断、模型比较、人群筛选、策略评估和可复现证据。

> 当前项目是 research / offline workbench。仓库中的离线指标、合成数据和公开样例不能证明线上 lift、因果识别或生产安全性。

## 三层平台边界

```text
数据层   -> schema、预处理、数据集清单、缺失/平衡/overlap/泄漏诊断
模型层   -> 模型注册、统一 adapter、神经 uplift、meta-learner、外部 CATE 后端
决策层   -> QINI/AUUC、Top-K、校准、policy value、ROI/iROAS、OPE、readiness、证据
```

稳定的分层入口位于 `deepuplift.data`、`deepuplift.models` 和 `deepuplift.decision`；旧的 `deepuplift.core` 保留为应用服务和向后兼容层。详细边界见 [`docs/architecture/THREE_LAYER_PLATFORM.md`](docs/architecture/THREE_LAYER_PLATFORM.md)。

## 快速开始

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools
python -m pip install -e ".[dev]"
python examples/coupon_allocation/run.py --rows 10000 --budget 50000
```

这个命令会从合成发券数据跑完：`CausalDataset → diagnostics → S/T/X/DR benchmark → EffectPrediction → coupon policy → experiment plan`。生成物写入 `runs/`，不会进入 Git。

要启动 Streamlit 工作台，额外安装完整应用依赖：

```bash
python -m pip install -r requirements.txt
python scripts/prepare_sample_datasets.py
streamlit run app.py
```

打开 `http://localhost:8501`。无 UI 的旧版轻量检查：

```bash
PYTHON_BIN=python3 scripts/smoke_test_agent.sh
```

最小 Python API：

```python
from deepuplift.application import run_uplift_pipeline

result = run_uplift_pipeline(
    data,
    feature_cols=["recency_days", "orders_30d", "segment"],
    treatment_col="coupon_received",
    outcome_col="conversion",
    id_column="user_id",
    treatment_cost=10,
    outcome_value=35,
    budget=50_000,
)
print(result.policy.summary)
```

如果只想检查语法和 shell 脚本：

```bash
python3 -m compileall -q app.py deepuplift deepuplift_multiple_treatment scripts
bash -n scripts/*.sh
```

可选后端和 Python 3.11 的 CausalML 环境见 [`requirements-optional.txt`](requirements-optional.txt) 和 [`requirements-causalml-py311.txt`](requirements-causalml-py311.txt)。

## v0.3 Trusted Causal Core

DeepUplift supports randomized and observational data, binary treatment (stable reference path), and multi-treatment/continuous treatment as experimental boundaries. The model catalog now distinguishes `STABLE`, `OPTIONAL`, `EXPERIMENTAL`, `LEGACY`, and `INTERFACE_ONLY`; only runnable models are presented as runnable.

| Family | Models | Maturity |
|---|---|---|
| Meta learners | S/T/X/DR, R, IPW | STABLE |
| Tree-backed meta | S/T/X/DR + RandomForest | STABLE |
| External forests/trees | EconML CausalForestDML, CausalML UpliftTree/UpliftRandomForest | OPTIONAL (real adapters; dependency-gated) |
| Multi / continuous | MultiTreatmentOutcome, DoseResponseGBM | EXPERIMENTAL |
| Deep models | TarNet, CFRNet, DragonNet, CEVAE, GANITE and others | EXPERIMENTAL / research compatibility |

For observational binary data, `run_uplift_pipeline()` automatically estimates and
records a cross-fitted nuisance contract. You can override the defaults explicitly:

```python
from deepuplift.application import run_uplift_pipeline

result = run_uplift_pipeline(
    data, feature_cols=["x1", "x2"], treatment_col="treatment",
    outcome_col="outcome", assignment_type="observational",
    model_names=["DR-Learner", "R-Learner"],
    nuisance_config={"estimator": "logistic", "cross_fit": True,
                     "n_splits": 5, "weighting": "overlap",
                     "trim_threshold": 0.05},
)
```

The result records OOF fold IDs, propensity distribution, overlap, clipping/trimming, ESS, balance before/after weighting, and provenance. Observational diagnostics cannot prove absence of hidden confounding. Benchmark policy value is offline evidence only.

Public loaders do not download or redistribute raw data. `synthetic_ground_truth()` is deterministic and carries true unit effects; Hillstrom, Criteo, and retail loaders require a user-supplied upstream file and retain source/license metadata.

## Release Status

`0.4.0a1` Alpha preparation: binary randomized and observational reference paths are stable in core CI; multi-treatment is experimental/reference, continuous treatment is experimental/offline-only, and external causal forests/uplift trees have real optional adapters gated by their dependencies. Public-data loaders are available, but a public-data validation claim is made only after a real local benchmark run.

## 输入数据

框架一级支持三类 treatment：`binary`、`multi_discrete`、`continuous`。当前 reference pipeline 完整跑通 binary；multi-treatment 和 continuous treatment 已冻结 contract 与扩展边界。

最小二元 treatment 数据需要包含：

- 一个 treatment 列（两种取值）；
- 一个 outcome 列（二元分类或连续结果）；
- 一组 treatment 发生前的特征列。

不要把 post-treatment 特征、用户隐私数据或未经授权的第三方数据提交到仓库。数据集登记和授权边界见 [`deepuplift/core/dataset_registry.py`](deepuplift/core/dataset_registry.py)。

## 代码结构

```text
app.py                         Streamlit 交互入口
deepuplift/data/               数据层公开入口
deepuplift/models/             模型层实现与模型目录
deepuplift/decision/           决策层公开入口
deepuplift/core/               训练、适配、诊断、评估、策略与证据服务
deepuplift_multiple_treatment/ 多 treatment 模型实现
scripts/                       数据准备、烟测、回归和证据生成脚本
docs/architecture/             平台分层设计
docs/recovery/                 项目恢复与上下文说明
```

## 证据与发布边界

每次训练运行可以生成配置、指标、预测、readiness、promotion 和 evidence manifest。`runs/`、`reports/`、本地数据、模型权重和截图均属于可再生或环境相关内容，默认不会进入 Git。

模型比较应同时查看 ranking、ground-truth（仅 synthetic/semi-synthetic）、calibration 和 business policy value；单个离线指标胜出不等于可以上线。线上实验、随机化设计、灰度、回滚和隐私审查需要由使用方单独完成。benchmark evidence bundle 写入用户指定的 `runs/<run_id>/`，不提交 Git。

详细协议：[`docs/data/OBSERVATIONAL_PIPELINE.md`](docs/data/OBSERVATIONAL_PIPELINE.md)、[`docs/data/PROPENSITY_AND_OVERLAP.md`](docs/data/PROPENSITY_AND_OVERLAP.md)、[`docs/models/TREE_MODELS.md`](docs/models/TREE_MODELS.md)、[`docs/benchmarks/BENCHMARK_PROTOCOL.md`](docs/benchmarks/BENCHMARK_PROTOCOL.md)、[`docs/open_source/MODEL_MATURITY.md`](docs/open_source/MODEL_MATURITY.md)。

## 开源路线图

建议按以下顺序演进：

1. 稳定三层 API 和数据/模型/决策 schema；
2. 增加公开数据集适配器、可复现 benchmark 和 CI；
3. 将训练运行、模型卡、指标和证据 manifest 统一为版本化协议；
4. 再接入在线实验、策略服务和权限/审计，不把离线结果直接当成生产结论。

## License

MIT，见 [`LICENSE`](LICENSE)。第三方模型、论文、数据集和可选后端仍受各自许可证与使用条款约束。
