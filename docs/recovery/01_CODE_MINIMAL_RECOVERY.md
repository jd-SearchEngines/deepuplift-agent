# Code Recovery

这是 DeepUplift 的最小恢复入口。当前仓库已经包含源码，不需要提交或恢复本地压缩包。

## Source of truth

```text
app.py
requirements*.txt
deepuplift/
deepuplift_multiple_treatment/
scripts/
docs/
```

## Restore

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/prepare_sample_datasets.py
streamlit run app.py
```

## Checks

```bash
python3 -m compileall -q app.py deepuplift deepuplift_multiple_treatment scripts
bash -n scripts/*.sh
PYTHON_BIN=python scripts/smoke_test_agent.sh
```

`torch`、`streamlit`、可选 CATE 后端未安装时，语法检查仍可运行；完整训练和 UI 检查需要先安装依赖。

## Deliberately excluded

`.env`、密钥、个人路径、私有数据、数据集 payload、模型权重、`runs/`、`reports/`、截图和本地对话日志不属于开源恢复集。生成物应在本地重新生成，并在报告中记录环境和数据指纹。
