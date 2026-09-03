#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
ROWS="${ROWS:-400}"
SENSITIVITY_SAMPLES="${SENSITIVITY_SAMPLES:-1}"

echo "[1/4] Compile app and core modules"
"${PYTHON_BIN}" -m py_compile \
  app.py \
  deepuplift/core/trainer.py \
  deepuplift/core/readiness.py \
  deepuplift/core/evaluator.py \
  deepuplift/core/predictor.py \
  deepuplift/core/registry.py

echo "[2/4] Validate causal knowledge JSON"
"${PYTHON_BIN}" -m json.tool deepuplift/knowledge/causal_agent_kb.json >/dev/null
"${PYTHON_BIN}" scripts/audit_model_catalog.py --strict --min-registered 90 --min-ready 35 >/dev/null

echo "[3/4] Run lightweight uplift regression"
"${PYTHON_BIN}" scripts/run_agent_regression.py \
  --sensitivity-samples "${SENSITIVITY_SAMPLES}" \
  --case "synthetic_retail_uplift_5k:TLearnerGBM:${ROWS}" \
  --case "synthetic_retail_uplift_5k:DRLearnerGBM:${ROWS}"

echo "[4/4] Verify latest regression report"
LATEST_REPORT="$(ls -t reports/agent_regression_*.json | head -1)"
"${PYTHON_BIN}" -m json.tool "${LATEST_REPORT}" >/dev/null
echo "Smoke test passed: ${LATEST_REPORT}"
