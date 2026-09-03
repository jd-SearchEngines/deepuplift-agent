# Coupon Allocation Scenario

Coupon allocation is the first complete vertical slice because it expresses
the business decision clearly:

```text
incremental conversion probability × order value − coupon cost
```

The synthetic generator creates historical behavior, randomized coupon
assignment, conversion, GMV, margin and heterogeneous treatment effects. The
example does not use private data.

```bash
python examples/coupon_allocation/run.py --rows 10000 --budget 50000
```

The command runs the full library path:

```text
data -> CausalDataset -> diagnostics -> model benchmark
     -> EffectPrediction -> ranking -> value/cost
     -> budget-constrained PolicyResult -> experiment plan
```

The policy is an offline expected-value policy. The generated experiment plan
is ready for experiment design, not a fabricated power calculation or proof of
online lift. Baseline, MDE, traffic, duration and an accountable rollback
owner still need to be supplied by the business team.
