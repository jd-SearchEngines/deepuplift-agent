# Batch 3 Push E2E v2 Report

Run ID: `push_e2e_v2_1794839941`  
Commit: `9e1e8ff36e38af977e4a842be8c9a74bbadb0eac`  
Evidence class: `SEMI_SYNTHETIC_KNOWN_TRUTH`  
Overall status: **PASS**

## Scope and gates

This report is a known-truth semi-synthetic DGP. It is not a claim about real Push lift or a licensed myBPmyLife analysis.

| Gate | Status |
|---|---|
| code_pass | PASS |
| test_pass | PASS |
| experiment_pass | PASS |
| research_gate_pass | PARTIAL |
| real_mrt_lane | BLOCKED_LICENSE_AND_LOCAL_ARCHIVE |
| full_sequential_dr | NOT_AVAILABLE_CONTINUATION_MODEL_MISSING |

## Model rows

| Seed | Model | Status | PEHE | ATE bias | Qini | AUUC | Runtime (s) |
|---:|---|---|---:|---:|---:|---:|---:|
| 42 | S-Learner | PASS | 0.09660814973629864 | -0.07537767440412449 | 1.4329045638652804 | 11.956696577817084 | 0.540 |
| 42 | T-Learner | PASS | 0.10745847333834288 | -0.08444441819327417 | 0.8922445152130396 | 10.575287392209745 | 0.603 |
| 42 | X-Learner | PASS | 0.09955011295762313 | -0.0796123504622589 | 1.095020374583536 | 11.616142753636177 | 0.766 |
| 42 | DR-Learner | PASS | 0.10769490704520368 | -0.07998250370791854 | 0.9328593663501792 | 10.635655132680538 | 0.689 |
| 52 | S-Learner | PASS | 0.10079324371155837 | -0.08573260692573596 | 2.180764471392845 | 14.229240696236182 | 0.531 |
| 52 | T-Learner | PASS | 0.11648097584405 | -0.09566184596172404 | 1.8430051622402268 | 12.986755655825634 | 0.583 |
| 52 | X-Learner | PASS | 0.11007524654381629 | -0.09226058052486033 | 2.2293577384922405 | 14.045024209266227 | 0.753 |
| 52 | DR-Learner | PASS | 0.14315867381210504 | -0.09872699923727532 | 1.8220382416923837 | 12.565690692510653 | 0.688 |
| 62 | S-Learner | PASS | 0.0983936734313837 | -0.08025190122367637 | 1.3217820787447088 | 13.909399021867712 | 0.534 |
| 62 | T-Learner | PASS | 0.09499042083052477 | -0.07482618457941123 | 1.4134166001533086 | 14.457953512455365 | 0.585 |
| 62 | X-Learner | PASS | 0.09198476893375593 | -0.0756062139058341 | 1.2507414421175622 | 14.14423278112582 | 0.755 |
| 62 | DR-Learner | PASS | 0.09784897919981868 | -0.07456133520939698 | 1.2678810986021403 | 14.099829993680029 | 0.689 |
| 72 | S-Learner | PASS | 0.09488367308470258 | -0.07854134695934847 | 1.0208085214680325 | 14.3694441547744 | 0.531 |
| 72 | T-Learner | PASS | 0.1097805224914573 | -0.07634197056829876 | 0.3347547856585167 | 11.444310316048883 | 0.581 |
| 72 | X-Learner | PASS | 0.09586109103712281 | -0.07150508735902567 | 0.5192789066458174 | 12.444545601838776 | 0.758 |
| 72 | DR-Learner | PASS | 0.11787471269491989 | -0.08056102058736661 | 0.09711297021513676 | 11.42727525348223 | 0.726 |
| 82 | S-Learner | PASS | 0.09643301743006541 | -0.08671228384340203 | 1.553010781923068 | 16.33152948870542 | 0.530 |
| 82 | T-Learner | PASS | 0.1030420021536491 | -0.08115984556232722 | 1.5047700123681897 | 15.747876981203994 | 0.583 |
| 82 | X-Learner | PASS | 0.09656607611275969 | -0.08219105988551322 | 1.4389958305523738 | 16.04907259617674 | 0.753 |
| 82 | DR-Learner | PASS | 0.10713502053618751 | -0.07884687671573382 | 1.2902825863573186 | 16.134620455683773 | 0.697 |

## Optimizer and OPE

| Seed | Solver | Feasible | Objective | Stopping reason | Converged | Runtime (s) |
|---:|---|---|---:|---|---|---:|
| 42 | greedy | True | 8.411357806868622 | sorted_net_value | None | 0.00014587500000029507 |
| 42 | heap | True | 8.411357806868622 | heap_exhausted_static_net_order | None | 0.00020058300000025397 |
| 42 | primal_dual | True | 8.411357806868622 | primal_dual_tolerance | None | 0.0006436669999998479 |
| 42 | line_search | True | 8.411357806868622 | line_search_tolerance | None | 0.0012307919999998695 |
| 42 | coordinate | True | 8.411357806868622 | coordinate_tolerance | None | 0.0031582920000001735 |
| 52 | greedy | True | 9.010450987501915 | sorted_net_value | None | 0.00018854200000006927 |
| 52 | heap | True | 9.010450987501915 | heap_exhausted_static_net_order | None | 0.00023350000000021964 |
| 52 | primal_dual | True | 9.010450987501915 | primal_dual_tolerance | None | 0.0007200840000001207 |
| 52 | line_search | True | 9.010450987501915 | line_search_tolerance | None | 0.0013675830000003941 |
| 52 | coordinate | True | 9.010450987501915 | coordinate_tolerance | None | 0.003488874999999503 |
| 62 | greedy | True | 12.971999688011667 | sorted_net_value | None | 0.00021416700000020938 |
| 62 | heap | True | 12.971999688011667 | heap_exhausted_static_net_order | None | 0.0002607089999999701 |
| 62 | primal_dual | True | 12.971999688011667 | primal_dual_tolerance | None | 0.0007872090000002885 |
| 62 | line_search | True | 12.971999688011667 | line_search_tolerance | None | 0.0014882500000004129 |
| 62 | coordinate | True | 12.971999688011667 | coordinate_tolerance | None | 0.0037396669999996135 |
| 72 | greedy | True | 19.632922236479605 | sorted_net_value | None | 0.0002085419999993121 |
| 72 | heap | True | 19.632922236479605 | heap_exhausted_static_net_order | None | 0.0002584169999995112 |
| 72 | primal_dual | True | 19.632922236479605 | primal_dual_tolerance | None | 0.0007787500000002723 |
| 72 | line_search | True | 19.632922236479605 | line_search_tolerance | None | 0.0015219999999995792 |
| 72 | coordinate | True | 19.632922236479605 | coordinate_tolerance | None | 0.003716457999999534 |
| 82 | greedy | True | 22.20420553906106 | sorted_net_value | None | 0.00023975000000042712 |
| 82 | heap | True | 22.20420553906106 | heap_exhausted_static_net_order | None | 0.00027975000000068917 |
| 82 | primal_dual | True | 22.81071139650015 | primal_dual_tolerance | None | 0.0020258749999992887 |
| 82 | line_search | True | 22.81071139650015 | line_search_tolerance | None | 0.002763708999999892 |
| 82 | coordinate | True | 22.81071139650015 | coordinate_tolerance | None | 0.004034624999999181 |

## Policy truth value

| Policy | Selected | Truth value | Oracle regret |
|---|---:|---:|---:|
| greedy | 13 | 0.358832 | 0.102940 |
| heap | 13 | 0.358832 | 0.102940 |
| primal_dual | 13 | 0.358832 | 0.102940 |
| line_search | 13 | 0.358832 | 0.102940 |
| coordinate | 13 | 0.358832 | 0.102940 |
| random_same_count | 13 | 0.353056 | 0.108716 |
| treat_none | 0 | 0.348860 | 0.112912 |
| treat_all_eligible | 269 | 0.448395 | 0.013377 |
| oracle_constrained | 42 | 0.375219 | 0.086553 |
| greedy | 24 | 0.363665 | 0.103068 |
| heap | 24 | 0.363665 | 0.103068 |
| primal_dual | 24 | 0.363665 | 0.103068 |
| line_search | 24 | 0.363665 | 0.103068 |
| coordinate | 24 | 0.363665 | 0.103068 |
| random_same_count | 24 | 0.357861 | 0.108872 |
| treat_none | 0 | 0.347383 | 0.119350 |
| treat_all_eligible | 282 | 0.457298 | 0.009435 |
| oracle_constrained | 54 | 0.379220 | 0.087514 |
| greedy | 29 | 0.369637 | 0.104620 |
| heap | 29 | 0.369637 | 0.104620 |
| primal_dual | 29 | 0.369637 | 0.104620 |
| line_search | 29 | 0.369637 | 0.104620 |
| coordinate | 29 | 0.369637 | 0.104620 |
| random_same_count | 29 | 0.365291 | 0.108966 |
| treat_none | 0 | 0.351226 | 0.123031 |
| treat_all_eligible | 263 | 0.457607 | 0.016650 |
| oracle_constrained | 51 | 0.383015 | 0.091242 |
| greedy | 28 | 0.372393 | 0.107195 |
| heap | 28 | 0.372393 | 0.107195 |
| primal_dual | 28 | 0.372393 | 0.107195 |
| line_search | 28 | 0.372393 | 0.107195 |
| coordinate | 28 | 0.372393 | 0.107195 |
| random_same_count | 28 | 0.367079 | 0.112510 |
| treat_none | 0 | 0.354773 | 0.124815 |
| treat_all_eligible | 266 | 0.465253 | 0.014335 |
| oracle_constrained | 54 | 0.387691 | 0.091897 |
| greedy | 31 | 0.356768 | 0.113146 |
| heap | 31 | 0.356768 | 0.113146 |
| primal_dual | 34 | 0.357673 | 0.112241 |
| line_search | 34 | 0.357673 | 0.112241 |
| coordinate | 34 | 0.357673 | 0.112241 |
| random_same_count | 31 | 0.348599 | 0.121315 |
| treat_none | 0 | 0.336440 | 0.133474 |
| treat_all_eligible | 277 | 0.459598 | 0.010315 |
| oracle_constrained | 57 | 0.372003 | 0.097910 |

## Limitations and next tasks

- This is known-truth semi-synthetic evidence, not real Push business lift.
- The local myBPmyLife archive remains unavailable/unlicensed; no raw data was downloaded.
- The OPE lane is one-step binary IPS/SNIPS; it is not full sequential DR.
- Three-arm comparison reuses the existing DeepUplift synthetic multi-treatment runner and remains separate from the binary candidate optimizer contract.

Next tasks:
- Add a continuation-value nuisance model before claiming full sequential DR.
- Reopen real MRT lane only after archive fingerprint and license are verified.
- Add a multi-arm candidate/OPE adapter using actual logged arm probabilities.
