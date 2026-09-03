from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepuplift.core.deep_uplift_designs import (
    deep_design_counts,
    deep_interview_drill_rows,
    deep_loss_catalog,
    deep_source_rows,
    local_causal_report_rows,
    loss_metric_mapping_rows,
    network_architecture_catalog,
)


def table(rows: list[dict[str, str]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(col, "")).replace("\n", " ") for col in columns) + " |")
    return "\n".join([header, sep, *body])


def write(path: Path, content: str) -> None:
    path.parent.mkdir(exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    generated_at = time.strftime("%Y-%m-%d %H:%M:%S %z")
    losses = deep_loss_catalog()
    networks = network_architecture_catalog()
    counts = deep_design_counts()
    drill_rows = deep_interview_drill_rows()

    write(
        Path("docs/UPLIFT_DEEP_LOSS_AND_ARCHITECTURE.md"),
        f"""# Deep Uplift Loss And Architecture Catalog

Generated at: `{generated_at}`

## Why This Exists

Deep uplift modeling is not just `BCE` on conversion. A credible platform needs to separate factual outcome fit, causal robustness, ranking/decision objectives, delayed/full-funnel labels, contrastive representation learning and LLM routing cost-quality objectives.

This project now has two layers of implementation:

- Model-integrated losses: `BaseModel.py`, `CFRNet.py`, `DragonNet.py`, `EFIN.py`, `DESCN.py`.
- Reusable differentiable loss library: `deepuplift/models/deep_losses.py`, validated by `scripts/smoke_deep_loss_functions.py`.

The CFRNet balance term is also covered by both a pure-gradient smoke (`scripts/smoke_cfrnet_differentiable_balance.py`) and a tiny end-to-end training smoke (`scripts/smoke_cfrnet_training_balance.py`). The broader deep-model evidence suite (`scripts/smoke_deep_model_training_evidence.py`) trains `CFRNet`, `DragonNet`, `EFIN` and `DESCN` under the same synthetic interaction-uplift design and writes `reports/deep_model_training_evidence_latest.json`.

## Counts

```json
{json.dumps(counts, ensure_ascii=False, indent=2)}
```

## Loss Catalog

{table(losses, ["loss_id", "family", "status", "objective", "formula", "business_use", "metric_link", "code_path", "source_url", "risk", "interview_line"])}

## Network Architecture Catalog

{table(networks, ["architecture", "status", "core_blocks", "loss_stack", "best_for", "code_path", "source_url", "interview_line"])}

## Deep Algorithm Interview Drill

{table(drill_rows, ["question", "core_answer", "formula_or_objective", "project_evidence", "follow_up"])}

## Loss To Metric Map

{table(loss_metric_mapping_rows(), ["training_signal", "loss_family", "nearest_offline_metric", "business_decision", "code_path"])}

## Source Gate

{table(deep_source_rows(), ["topic", "source", "status", "code_path", "source_url"])}
""",
    )

    write(
        Path("docs/UPLIFT_DEEP_MODEL_INTERVIEW_DEEP_DIVE.md"),
        f"""# Deep Uplift Model Interview Deep Dive

Generated at: `{generated_at}`

## 1. Why BCE/MSE Is Not Enough

Factual BCE/MSE only trains the observed arm. Uplift needs a ranking over treatment effect, but the counterfactual arm is missing. This is why the platform exposes DR/R pseudo-outcomes, representation balance, propensity heads, calibration, bootstrap CI and policy value instead of treating a conversion model as an uplift model.

**Interview phrasing:** "A conversion model predicts who will convert. An uplift model predicts whose conversion probability changes because of treatment. The loss and evaluation stack must reflect that missing-counterfactual problem."

## 2. Representation Balance

CFRNet-style balance uses an IPM penalty between treated and control representations. I fixed the local implementation so this penalty stays in torch and remains differentiable: `deepuplift/models/BaseModel.py:representation_balance_loss`. It is verified by `scripts/smoke_cfrnet_differentiable_balance.py` and a tiny training run in `scripts/smoke_cfrnet_training_balance.py`.

**Tradeoff:** Balance improves robustness when treatment/control covariates shift, but over-balancing can erase true effect modifiers. I tune `alpha` and compare against DR/R LightGBM baselines.

## 3. Targeted Regularization

DragonNet adds a propensity head and targeted regularization. This links treatment assignment and outcome learning in the same network, but it still needs overlap diagnostics because a good propensity head does not create support where no support exists.

## 3.5 Runnable Deep Model Evidence

`scripts/smoke_deep_model_training_evidence.py` connects source-level explanation to executable proof. It trains `CFRNet`, `DragonNet`, `EFIN` and `DESCN`, then stores loss curves, QINI/AUUC, oracle Top-K recall, policy fields and evidence manifests in `reports/deep_model_training_evidence_latest.json` and `reports/deep_model_training_leaderboard_latest.csv`.

**Interview phrasing:** "I do not just list neural uplift architectures. I can open the source, explain the loss, show a smoke training curve, and trace the run to an evidence manifest."

## 4. Full-Funnel And Delayed Feedback

ECUP-style full-funnel uplift and delayed feedback objectives change the data schema: impression/click/conversion stages, D1/D7/D14/D30 labels, censoring and label maturity. The platform already has evaluator/UI/evidence support before claiming a neural plugin is ready.

## 5. Contrastive Uplift

Contrastive learning is useful only if positives and negatives are defined causally. I would use randomized experiments or DR/R pseudo-effects to create effect-sign buckets, then pull together similar persuadables and push apart sure things, lost causes and sleeping dogs. The reusable loss is implemented as `deepuplift/models/deep_losses.py:contrastive_uplift_representation_loss` and wired into `deepuplift/models/ContrastiveUpliftNet.py` as a trainable neural baseline.

## 6. LLM Routing Uplift

LLM routing is treatment-effect estimation: treatment is strong-model/RAG/tool escalation, control is cheap model, outcome is quality or task success. The policy objective is `quality_uplift * value - extra_model_cost - latency_penalty`, implemented as `deepuplift/models/deep_losses.py:llm_routing_cost_quality_loss` and mirrored in the Policy ROI simulator.

The newer multi-action extension treats `strong_model`, `rag`, `tool` and `human_review` as treatment arms with separate cost, latency, hallucination, evidence-failure and budget-risk penalties. It is backed by `synthetic_llm_multi_action_routing_9k`, `llm_multi_action_routing_loss`, Policy multi-action examples, scenario compare smoke and the `llm_multi_action_routing` evaluator contract.

## Interview Drill Table

{table(drill_rows, ["question", "core_answer", "formula_or_objective", "project_evidence", "follow_up"])}
""",
    )

    write(
        Path("docs/UPLIFT_CONTRASTIVE_AND_LLM_UPLIFT.md"),
        f"""# Contrastive And LLM Uplift Playbook

Generated at: `{generated_at}`

## Contrastive Uplift Design

1. Start with randomized or overlap-supported data.
2. Train a DR/R learner or use experimental true uplift to get pseudo-effect buckets.
3. Define positives as users with similar context and same pseudo-effect sign.
4. Define hard negatives across persuadable / sure thing / lost cause / sleeping dog groups.
5. Train a representation encoder with factual loss plus supervised contrastive loss.
6. Validate with Top-K uplift, calibration, segment stability and policy value.

## LLM Routing As Uplift

| Element | Uplift Interpretation |
| --- | --- |
| Control | cheap/small model |
| Treatment | strong model, RAG, tool call or human review |
| Outcome | quality, resolution, conversion or task success |
| Cost | extra model cost, tool cost, latency, human review |
| Policy value | quality uplift * value - incremental cost |

## Multi-Action LLM Routing Extension

`Synthetic LLM Multi-Action Routing 9k` expands the binary strong-vs-cheap framing into action selection:

- action arms: `strong_model`, `rag`, `tool`, `human_review`, with cheap model as fallback control.
- objective: `gain_a = (q_a - q_cheap) * user_value - incremental_cost_a - latency_penalty_a - risk_penalty_a`.
- risk fields: hallucination risk, evidence failure risk, judge noise risk and budget pressure.
- evaluator contract: `llm_multi_action_routing` keeps Top-K routing value, ROI proxy and oracle best-action capture in evidence.
- interview point: production routing needs randomized exploration and judge calibration, otherwise the model only imitates the old router.

## UI Evidence

- `Models -> Deep Uplift Loss Catalog`
- `Models -> Network Architecture Catalog`
- `Models -> 深度算法面试追问 Drill`
- `Evaluation -> Loss-to-Metric Map`
- `Policy -> LLM Routing ROI Simulator`
- `Policy -> 多动作路由策略样例`
- `Evidence -> Deep Loss / Network Source Gate`
- `Story -> Deep Uplift Architecture Interview Card`

## Code Evidence

- `deepuplift/models/deep_losses.py`
- `deepuplift/models/ContrastiveUpliftNet.py`
- `examples/datasets/synthetic_llm_multi_action_routing_9k.csv`
- `scripts/smoke_deep_loss_functions.py`
- `scripts/smoke_cfrnet_differentiable_balance.py`
- `scripts/smoke_cfrnet_training_balance.py`
- `scripts/smoke_contrastive_uplift_training.py`
- `scripts/smoke_deep_model_training_evidence.py`

## Local Causal Report Signals

{table(local_causal_report_rows(), ["title", "source", "trust_level", "url"])}
""",
    )

    print(
        json.dumps(
            {
                "status": "ok",
                "docs": [
                    "docs/UPLIFT_DEEP_LOSS_AND_ARCHITECTURE.md",
                    "docs/UPLIFT_DEEP_MODEL_INTERVIEW_DEEP_DIVE.md",
                    "docs/UPLIFT_CONTRASTIVE_AND_LLM_UPLIFT.md",
                ],
                "counts": counts,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
