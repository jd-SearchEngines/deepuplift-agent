# Experiment A — Dataset Audit

Audit date: 2026-11-13. Raw data is local-only and is not committed.

| Dataset | Source / local status | Rows used | Treatment / outcome | Audit decision |
|---|---|---:|---|---|
| NHEFS | `causallib.datasets.load_nhefs`, package 0.10.0, locally retrievable | 1,566 binary complete-case; 1,566 continuous dose/outcome rows | `qsmk` / `wt82_71`; `smokeintensity` / `wt82_71` | **RUN** observational bias-repair lane |
| myBPmyLife Push MRT | no local raw file found in bounded audit | 0 | participant-level Push assignment/history required | **BLOCKED_ON_DATA** |
| Hillstrom | no local raw file found | 0 | randomized email treatment/outcome required | **BLOCKED_ON_DATA** |
| Criteo Uplift v2.1 | no local raw file found; license/terms require local verification | 0 | randomized treatment/outcome required | **BLOCKED_ON_DATA** |
| X5 / RetailHero | no local raw file found | 0 | observational assignment metadata required | **BLOCKED_ON_DATA** |
| MegaFon / other observational Push-like data | no local raw file found | 0 | assignment and delivery semantics required | **BLOCKED_ON_DATA** |

## NHEFS provenance and target-trial boundaries

The loader is a public upstream dataset interface; it downloads/caches data in
the experiment environment. DeepUplift stores only aggregate diagnostics in
[`bias_diagnostics.json`](bias_diagnostics.json). The binary loader's
`restrict=True` path is complete-case restricted, so the rows excluded by that
upstream restriction are not treated as missing-at-random evidence. For the
continuous lane, missing dose/outcome rows are excluded and baseline
covariates are retained for model-side imputation.

The continuous feature list is baseline-only. Follow-up variables (`wt82`,
`smkintensity82_71`, prices/taxes and event outcomes) and later `qsmk` are
excluded. This is a data-contract choice, not proof that all baseline
confounding is measured.

## Required next data handoff

For real Push/MRT execution, supply a lawfully obtained local path and source
hash for the raw file plus a field map for participant ID, decision time,
candidate eligibility, assignment probability, actual delivery, outcome
window and censoring. The loader must then run the existing data audit before
any model or policy result is reported.

## Audit update: Hillstrom retrieved and executed (2026-11-14)

The previous audit correctly recorded Hillstrom as absent at audit time. A
subsequent bounded retrieval added the public CSV to the local ignored path
`.local/experiment-a/raw/hillstrom.csv`; it is not committed. SHA-256 is
`0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece`.
The real-randomized lane completed conversion, visit and spend outcomes with
50/50 PASS rows per outcome. This changes Hillstrom from historical
`BLOCKED_ON_DATA` to **RUN / REAL_RANDOMIZED_EXTERNAL** for the current local
environment only. It does not change the status of myBPmyLife, Criteo, X5 or
the Push longitudinal lane.

The raw source was obtained from the MineThatData Hillstrom email experiment
CSV and must be re-checked by a user before redistribution; only its hash,
aggregate metrics and reproduction command are documented in Git.
