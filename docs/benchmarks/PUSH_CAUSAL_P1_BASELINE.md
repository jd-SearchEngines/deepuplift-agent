# Push MRT P1 randomized baseline

**Phase:** P1 — real randomized Push baseline
**Evidence class:** `REAL_RANDOMIZED_MRT`
**Status:** `PASS` for the audited upstream snapshot
**Run:** seed `42`, participant cluster bootstrap `200` repetitions, participant-held-out split `235/59`

The run used the P0 loader and the declared MRT assignment probabilities (`none=0.75`, `activity=0.125`, `diet=0.125`). It did not estimate or replace the randomized assignment mechanism with a propensity model. Rows without explicit assignment remain excluded by the P0 fail-closed rule.

## Outcome-specific results

| Outcome | Complete rows | Arm means: none / activity / diet | ITT contrast | Cluster bootstrap 95% CI |
|---|---:|---:|---:|---:|
| Steps after activity decision, 60 min | 182,539 | 453.970 / 460.784 / 457.219 | activity − none = **+6.287** | [-3.402, 17.119] |
| Low-sodium choices after dietary decision, 24 h | 189,241 | 1.020 / 1.034 / 1.045 | diet − none = **+0.0113** | [-0.0307, 0.0466] |

All three assignment arms are retained in each outcome table. The additional contrasts were:

- Activity outcome: `diet − none = -2.669` (95% CI `[-15.381, 9.805]`); `diet − activity = -8.957` (95% CI `[-24.857, 9.087]`).
- Dietary outcome: `activity − none = +0.0123` (95% CI `[-0.0175, 0.0407]`); `diet − activity = -0.0011` (95% CI `[-0.0395, 0.0374]`).

The estimator is a Horvitz–Thompson assignment ITT on outcome-complete rows, using the known arm probabilities. The uncertainty is a participant-level cluster bootstrap, not a row-level iid bootstrap.

## Context-conditional analysis

For each outcome, arm-specific Ridge outcome models were fit on the participant-held-out training participants using only the P0 pre-treatment feature set. The test split has zero participant overlap with training. The resulting conditional contrasts are descriptive model-based heterogeneity surfaces; they are not individual counterfactual truth:

| Outcome | Activity − none mean predicted effect | Diet − none mean predicted effect |
|---|---:|---:|
| Activity steps, 60 min | +4.609 | +4.426 |
| Dietary low-sodium choices, 24 h | +0.0183 | +0.0307 |

No post-treatment feature was accepted by the adapter. The real MRT has no observed individual counterfactual, so PEHE is `NOT_AVAILABLE` and is not computed.

## Assignment versus delivery

Delivery is retained as a separate descriptive field. Among 47,067 assigned notification rows, 45,966 had an observed notification delivery (`97.6608%`). This run does not estimate a causal delivery/per-protocol effect, because doing so would require an explicit compliance estimand and additional identification assumptions.

## Gate decision and limitations

`P1 PASS`. Both proximal outcomes were analyzed separately; known randomization probabilities were used; participant-clustered uncertainty and participant-held-out context splits were run; all three assignment arms were retained; assignment and delivery were not conflated; and no real-data PEHE or online business-lift claim was made.

The MRT randomizes notification assignment, not continuous notification frequency. It remains unsuitable as a randomized continuous-dose benchmark; that work belongs to P3 semi-synthetic data. The upstream license remains `LICENSE_UNVERIFIED` as recorded in [`PUSH_CAUSAL_P0_AUDIT.md`](PUSH_CAUSAL_P0_AUDIT.md).
