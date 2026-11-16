# Hermes Batch 0 测试结果

日期：2026-11-16

## 环境

- Hermes：0.23.1 stable
- Official Codex Python SDK：`openai-codex`，Python 3.11 isolated runtime
- Codex CLI authentication：existing ChatGPT login reused by the SDK
- Evidence class：synthetic engineering fixture only

## Commands and results

```text
pytest -q tests/test_hermes_codex_sdk_adapter.py
2 passed in 0.04s

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONPYCACHEPREFIX=<temporary-cache> PYTHONPATH=. pytest -q
111 passed, 2 skipped, 432 warnings in 35.86s
```

The skipped tests are existing optional-backend skips. The warning count is retained as evidence and was not suppressed to manufacture a pass.

GitHub PR #8 Python 3.10/3.11/3.12 syntax checks and both continuous-frontier checks: PASS.

## Adapter repair evidence

The first attempt successfully created Hermes task A but failed while parsing the pretty-printed `hermes kanban create --json` response because the adapter read only the final line. The parser was changed to parse the complete JSON document. Re-running with the same idempotency keys reused the existing task and did not create a duplicate DAG.

## Safety

The disposable marker fixture, local ledger and SDK runtime were kept outside the Git repository. The committed evidence contains task/thread/turn identifiers and redacted status only; it contains no credentials, participant rows, model weights or personal absolute paths.
