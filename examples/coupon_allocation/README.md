# Coupon Allocation E2E

This example uses synthetic but realistic user behavior data. It is not a
production benchmark and contains no private or third-party dataset payload.

Run from the repository root:

```bash
python examples/coupon_allocation/run.py --rows 10000 --budget 50000
```

The command executes:

```text
data -> CausalDataset -> diagnostics -> S/T/X/DR benchmark
     -> EffectPrediction -> ranking -> coupon economics
     -> budget policy -> PolicyResult -> experiment plan
```

Generated JSON artifacts go to `runs/` by default and are ignored by Git.
