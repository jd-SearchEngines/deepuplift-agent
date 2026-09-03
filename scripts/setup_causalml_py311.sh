#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-deepuplift-causalml-py311}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

conda create -y -n "${ENV_NAME}" python=3.11
conda run -n "${ENV_NAME}" python -m pip install --upgrade pip
conda run -n "${ENV_NAME}" python -m pip install -r "${ROOT_DIR}/requirements.txt"
conda run -n "${ENV_NAME}" python -m pip install -r "${ROOT_DIR}/requirements-causalml-py311.txt"

cat <<EOF
CausalML environment is ready.

Start the full workbench with:
  conda run -n ${ENV_NAME} python -m streamlit run ${ROOT_DIR}/app.py --server.port 8502

Run a CausalML smoke test with:
  conda run -n ${ENV_NAME} python ${ROOT_DIR}/scripts/run_agent_regression.py --case synthetic_retail_uplift_5k:CausalMLUpliftRandomForest:800
EOF
