# Experiment A — Data Audit (2026-11-14)

Repository: `jd-SearchEngines/deepuplift-agent`
Branch: `codex/push-mrt-causal-benchmark`
Execution: local macOS, native Python 3.11, CPU classical models and sequential MPS neural runs.

## Evidence boundary

This report records data that was actually inspected locally and the exact
experiment lane that was allowed for each source. A source being available on
the web is not treated as a local run. Raw data stays outside Git.

| Dataset | Local status | Rows / schema evidence | Allowed use in this phase | Limitation |
|---|---|---:|---|---|
| Criteo Uplift v2.1 | `AVAILABLE_AND_RUN` | 13,979,592; 12 features; binary treatment | Real randomized A1 at 100k/500k/1m, conversion and visit | 3 seeds, no individual counterfactual truth; exposure excluded from features |
| Hillstrom | `AVAILABLE_AND_RUN` | local CSV, randomized email arms | Real randomized pairwise A1 baselines for conversion, visit and spend | no PEHE; not Push/MRT |
| NHEFS | `AVAILABLE_AND_RUN` | 1,566 complete cases | Observational bias-repair and continuous GPS lane | no randomized truth or individual counterfactuals |
| myBPmyLife MRT | `NOT_AVAILABLE` | no raw archive in bounded local workspace | no real Push result claimed | requires lawfully obtained archive, participant/time/delivery/outcome audit |
| MT-LIFT | `ACCESS_BLOCKED` | official repository inspected; Google Drive download returned login HTML | no multi-treatment real-data run | no authenticated local download; synthetic A2 is not MT-LIFT |
| UVET/CUVET | `AUDIT_ONLY` | 50MB HTTP range; 330,458 parsed prefix rows; official train file is ~10.3GB | schema/support audit only | prefix is not an exact uniform/full-data sample; no formal benchmark claim |
| X5/RetailHero | `NOT_AVAILABLE` | no local licensed file | not run | treatment assignment semantics not verified |

## Criteo audit

The locally downloaded file is the corrected Criteo Uplift v2.1 CSV gzip. The
streamed audit found 13,979,592 rows, columns `f0..f11,treatment,conversion,visit,exposure`,
no missing values, treatment counts 11,882,655 treated and 2,096,937 control,
and treated share 0.8500001288. Conversion rate was 0.0029166803, visit rate
0.0469920009 and exposure rate 0.0306312230. The benchmark uses only `f0..f11`;
`exposure` is retained for audit context and is not a pre-treatment feature.

The source file SHA-256 is
`2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc`.
The original gzip is ignored local data and was not committed.

The treatment-sorted file was not sampled by prefix. Each formal size uses an
exact uniform row sample over the known full row count, then the same sampled
rows and stratified split are reused for all learners within a seed.

## UVET / continuous-treatment boundary

The available UVET schema contains `f0..f4`, `policy`, `spend`, `value` and
`is_val`, with five observed policy values 0.8, 0.9, 1.0, 1.1 and 1.2. A
50MB HTTP range was parsed for schema and support checks only; it is not a
representative full-data draw. The existing DeepUplift continuous contract is
therefore evaluated on the known-truth three-shape synthetic lane, not by
silently treating a truncated UVET prefix as a formal CATE benchmark.

## Data audit gate

**PARTIAL.** Criteo and Hillstrom randomized lanes are runnable and measured;
the real Push MRT and MT-LIFT multi-treatment lanes remain blocked by local
data/access. No business uplift or real Push conclusion is made.
