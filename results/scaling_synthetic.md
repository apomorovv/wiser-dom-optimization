# Historical synthetic scaling snapshot (not current evidence)

This file and its companion CSV predate polished greedy, adaptive exact LNS, repeated
generator seeds, and source-state checkpointing. They are retained only for provenance
and must not be cited as current performance. Regenerate them with the revised command
below before any report uses the results.

Generated with:

```bash
python scripts/run_scaling_study.py \
  --sizes 8,20,50 \
  --noise 0,0.02 \
  --classical-max-orders 20 \
  --repetitions 3 \
  --output results/scaling_synthetic.csv \
  --seed 7
```

The historical hybrid used six iterations, at most eight active orders, a 40-variable configured
QUBO cap, at most five retained assignment candidates per order, 32 reads, 100
sweeps, and four exact-recourse candidates per iteration. Actual maximum QUBO size
was 32 variables. Results are a single deterministic
generator seed and are evidence of workflow behavior, not a performance claim.

| Orders | Method | Noise σ | Objective | Runtime (s) | Gap | Improvement vs hybrid incumbent |
|---:|---|---:|---:|---:|---:|---:|
| 8 | default | 0 | -5,253.8 | 0.057 | — | — |
| 8 | greedy | 0 | -3,322.8 | 0.302 | — | — |
| 8 | bounded MILP | 0 | -3,261.8 | 0.028 | 0.0000 | — |
| 8 | hybrid | 0 | -3,312.0 | 5.248 | — | +10.8 |
| 8 | hybrid | 0.02 | -3,322.8 | 5.131 | — | 0.0 |
| 20 | greedy | 0 | -3,160.6 | 1.875 | — | — |
| 20 | bounded MILP | 0 | -2,977.8 | 0.932 | 0.0097 | — |
| 20 | hybrid | 0 | -3,160.6 | 8.044 | — | 0.0 |
| 20 | hybrid | 0.02 | -3,160.6 | 8.473 | — | 0.0 |
| 50 | greedy | 0 | 2,139.2 | 11.451 | — | — |
| 50 | hybrid | 0 | 2,139.2 | 19.139 | — | 0.0 |
| 50 | hybrid | 0.02 | 2,139.2 | 17.410 | — | 0.0 |

All rows passed the independent validator. The zero-noise hybrid found a small
improvement only at eight orders; 2% coefficient perturbation removed it. At 20 and
50 orders the configured neighborhoods did not improve greedy. This negative result
is useful: the classical incumbent is strong, and a sampler should not receive credit
for merely returning it. The acceptance invariant prevented degradation in every
case.

The 20-order MILP stopped within a 1% configured gap rather than proving the exact
optimum. No global MILP was run at 50 orders under this protocol. More seeds,
neighborhood policies, equal-time comparisons, and approved hardware runs are
required before drawing a general conclusion.
