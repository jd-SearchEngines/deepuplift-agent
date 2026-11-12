# Push-derived P4 monotonicity and calibration benchmark

**Phase:** P4 — monotonicity and calibration
**Evidence class:** `SEMI_SYNTHETIC_KNOWN_TRUTH`
**Status:** `PASS` for the recorded 500-row monotone/fatigue run

The comparison starts from a fitted `DoseResponseGBM` curve and applies four explicit treatments: no correction, post-hoc PAV/isotonic projection, an iterative one-sided monotonicity penalty, and chain-graph Laplacian smoothing. Curves are scored only on the model's observed dose grid intersection.

| Shape | Method | Violation rate | MISE |
|---|---|---:|---:|
| Monotone | none | 0.2696 | 0.01535 |
| Monotone | PAV/isotonic | 0.0000 | 0.01445 |
| Monotone | explicit monotonic penalty | 0.2100 | 0.01445 |
| Monotone | Laplacian smooth | 0.2776 | 0.01490 |
| Fatigue | none | 0.5344 | 0.00876 |
| Fatigue | PAV/isotonic | 0.0000 | 0.01938 |
| Fatigue | explicit monotonic penalty | 0.9984 | 0.01390 |
| Fatigue | Laplacian smooth | 0.8524 | 0.00842 |

The fatigue result is the required rejection case: forcing a monotone curve can materially increase error. Laplacian smoothing changes roughness but is not a monotonicity guarantee; its high fatigue violation rate is expected and is recorded rather than relabeled as a pass for monotonicity.

`P4 PASS` means all requested comparison mechanisms and both domain shapes ran with finite metrics. It does not mean monotonic calibration is universally beneficial.
