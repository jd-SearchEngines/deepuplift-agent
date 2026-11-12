# Push-derived P2 bias and identification benchmark

**Phase:** P2 — controlled bias and identification
**Evidence class:** `SEMI_SYNTHETIC_KNOWN_TRUTH`
**Status:** `PASS` for the reproducible 5,000-row run
**Run:** seed `42` plus scenario offsets; runtime 5.13 s on the local Python environment

The generator samples pre-treatment context rows from the audited myBPmyLife MRT. It creates `Y(0)`, `Y(1)`, assignment, and selection before producing the observed outcome and selected sample. The true arrays and DGP parameters are held in an evaluation object; they are not columns in the model-facing `CausalDataset`.

## Scenario results

The table reports true ATE and ATE bias. `obs-X` methods receive only the propensity mechanism that is expressible from observed covariates. The MNAR oracle is a sensitivity upper bound and is not an estimable production method.

| Scenario | Observed rows | Selection rate | True ATE | Unadjusted bias | Outcome regression bias | IPW bias | Overlap bias | DR bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Measured confounding | 5,000 | 1.0000 | 0.11979 | +0.14853 | -0.00092 | -0.01352 | -0.00790 | -0.00173 |
| MAR sample selection | 2,642 | 0.5284 | 0.11968 | +0.16375 | +0.02445 | +0.03519 | +0.02805 | +0.02371 |
| Overlap failure | 5,000 | 1.0000 | 0.11990 | +0.28460 | +0.00700 | -0.03741 | -0.01670 | -0.00671 |
| Hidden confounding | 5,000 | 1.0000 | 0.11981 | +0.44249 | +0.34395 | +0.32147 | +0.32920 | +0.34239 |
| MNAR selection sensitivity | 2,633 | 0.5266 | 0.12015 | +0.36088 | +0.30461 | +0.27441 | +0.27908 | +0.30448 |

The MAR scenario additionally ran explicit selection-weighted IPW (`+0.02183` bias) and selection-weighted DR (`+0.00006` bias). This is an identification demonstration under the known MAR selection mechanism, not evidence that arbitrary selection bias is solved.

The overlap-failure scenario had 61.94% of true and observed-X propensities outside `[0.05, 0.95]`. The run records treated/control ESS, weight maxima and variance, and pre/post maximum numeric SMD for every method. Trimming was not silently applied; a future comparison must report the changed target estimand explicitly.

## Hidden-confounding and MNAR interpretation

The hidden-confounding DGP adds an unobserved `U` to both outcome and assignment. All observed-X methods remain materially biased. This is expected and is a limitation result, not a model failure that can be repaired by adding a deeper network.

The MNAR run adds `U` to selection. The report includes `oracle_selection_weighted_dr` with access to true assignment and selection probabilities only as a sensitivity upper bound. It must not be presented as an available estimator for real Push data.

## Gate decision

`P2 PASS`. All four required stress classes are present (measured confounding, MAR SSB, overlap failure, and hidden confounding), with an additional MNAR sensitivity scenario. Potential outcomes are generated before assignment and selection; truth is kept out of model inputs; IPW, stabilized IPW, overlap weighting, outcome regression, DR, and explicit MAR selection weighting are compared; and ESS, weight tails, balance, ATE bias, PEHE on semi-synthetic rows, and policy regret are recorded.

These are known-truth semi-synthetic results derived from MRT context. They are not real Push causal effects, business lift, or evidence that hidden confounding is identified.
