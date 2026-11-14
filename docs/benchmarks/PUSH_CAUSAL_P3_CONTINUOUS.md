# Push MRT P3 multi-treatment and continuous-dose benchmark

**Phase:** P3 — multi-treatment and continuous dose
**Evidence classes:** `REAL_RANDOMIZED_MRT` for multi-arm assignment; `SEMI_SYNTHETIC_KNOWN_TRUTH` for continuous frequency
**Status:** `PASS` for the recorded runs

## Multi-treatment Push baseline

The real MRT arms remain `none`, `activity`, and `diet` with declared probabilities `0.75/0.125/0.125`. A per-arm Ridge outcome model was fit on pre-treatment features with a participant-held-out split (`235/59`, overlap `0`). The design-based ITT contrasts reuse the P1 known-probability estimator. The binary propensity contract was not used.

Held-out predicted arm means for the activity-step outcome were `none=451.420`, `activity=456.029`, and `diet=455.846`. The known-probability ITT contrasts were `activity-none=+6.287`, `diet-none=-2.669`, and `diet-activity=-8.957`, matching the P1 report. There is no individual counterfactual truth on real MRT, so these are not PEHE results.

## Continuous frequency-dose benchmark

The real MRT is not a randomized continuous-dose experiment. The suite therefore samples its pre-treatment context distribution and generates three separate known-truth dose-response DGPs: monotone, saturating, and non-monotone fatigue. All models use the existing continuous contract and decision path; scores are evaluated only inside the model's observed dose support.

Run configuration: 400 rows per shape, seed `42` with shape offsets, CPU, 30 epochs for DRNet/VCNet, 20 factual and 10 augmentation epochs for GIKS. Runtime was 6.26 s for the full three-shape suite.

| Shape | Model | Status | MISE |
|---|---|---|---:|
| Monotone | DoseResponseGBM | PASS | 0.01068 |
| Monotone | DRNet | PASS | 0.01365 |
| Monotone | VCNet | PASS | 0.00699 |
| Monotone | GIKS-DRNet | PASS | 0.00982 |
| Monotone | GIKS-VCNet | PASS | 0.00427 |
| Saturating | DoseResponseGBM | PASS | 0.01488 |
| Saturating | DRNet | PASS | 0.01623 |
| Saturating | VCNet | PASS | 0.01145 |
| Saturating | GIKS-DRNet | PASS | 0.02920 |
| Saturating | GIKS-VCNet | PASS | 0.02498 |
| Fatigue | DoseResponseGBM | PASS | 0.02015 |
| Fatigue | DRNet | PASS | 0.05787 |
| Fatigue | VCNet | PASS | 0.04920 |
| Fatigue | GIKS-DRNet | PASS | 0.04984 |
| Fatigue | GIKS-VCNet | PASS | 0.05437 |

For monotone and saturating shapes, the generated economic optimum was outside at least one model's supported grid, so optimal-dose and policy-regret fields remain unavailable under the support guard rather than being extrapolated. Fatigue had supported policy metrics; DoseResponseGBM economic regret was `0.03679`, VCNet `0.01772`, DRNet `0.03667`, GIKS-DRNet `0.10835`, and GIKS-VCNet `0.05610` on this run.

TransTEE remains `OPTIONAL_NOT_VALIDATED` in the model catalog. No VC-Transformer reproduction claim is made. The real MRT is not relabeled as continuous randomized evidence.
