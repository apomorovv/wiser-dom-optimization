# Experiment protocol

The notebook and `src/domopt/experiments.py` implement one fair, aggregate-only
comparison protocol for the real challenge subsets and an independent synthetic
control.

## 1. Immutable comparison definition

An experiment is identified by:

$$
E=(\text{bundle hash},\text{schema},\text{assumptions},\text{candidate set},
\text{objective},\text{method config},\text{seed},\text{commit}).
$$

Changing any element creates a different experiment. Methods compared within one row
group receive identical canonical data, candidates, objective calculation, and
validator.

## 2. Methods

| Method | Role |
|---|---|
| `default` | Deterministic default-DC policy baseline. |
| `greedy` | Fast sequential reassignment baseline using current residual resources. |
| `classical` | Exact or bounded SciPy/HiGHS MILP reference. |
| `hybrid` | Bounded QUBO assignment search plus exact MILP recourse. |

Simulated annealing is quantum-inspired, not quantum hardware. The optional D-Wave
backend is excluded until restricted-data approval is obtained.

## 3. Common metrics

Every result records feasibility, objective, fulfilled value, penalty cost, shipping
cost, case fill, value fill, reassigned orders, runtime, and—when available—MILP
optimality gap. Hybrid rows also record starting objective, improvement, accepted
moves, recourse calls, and maximum local QUBO width.

$$
J=\text{fulfilled value}-\text{thresholded penalty}-\text{shipping cost},
$$

$$
\operatorname{caseFill}=\frac{\sum_{o,s}f_{os}}{\sum_{o,s}Q_{os}}.
$$

Solver-native objective and QUBO energy are never copied into the comparison table.
The independent evaluator recomputes all three objective components.

## 4. Full experiment grid

### Core solver comparison

Run default, greedy, exact MILP, and hybrid on the same deterministic high-shortage
real-data subset. Exact MILP uses a 60-second limit and 1% requested relative gap;
the achieved gap is reported.

### Size scaling

Run greedy and hybrid at requested subset sizes 8, 20, and 50. Run exact MILP at 8
and 20. Whole assignment groups are expanded, so actual order count can be larger
than the requested seed count. Report runtime, objective, fill, candidate rows, MILP
gap, and local QUBO width.

### Penalty-weight sensitivity

Multiply variable, fixed, per-cut-SKU, minimum, and maximum penalties by 0.5, 1.0,
and 2.0. Compare greedy and hybrid only within the same scale; raw objectives across
different scales do not represent the same economics.

### Candidate-count sensitivity

Retain at most 1, 2, 4, or 6 group options, always preserving the default. Compare
greedy and hybrid to quantify search breadth, runtime, and quality.

### Inventory shocks

Apply deterministic 0%, 10%, and 25% reductions to every protected-ATP value and
compare default, greedy, and hybrid. This is a stress scenario, not a forecast.

### Seed and coefficient-noise sensitivity

Run hybrid at seeds 3, 11, 29, and 47 with relative Gaussian QUBO coefficient noise
0%, 1%, and 5%. Samples are ranked against the original QUBO and pass exact recourse.
This tests local ranking sensitivity; it is not a model of physical QPU noise.

### Pareto-pruning ablation

Run the same base population with and without Pareto pruning. Pruning is useful only
when it reduces candidate count or runtime without reducing validated objective.

### Random-versus-conflict batching

Hold all hybrid settings fixed and compare seeded random whole-group neighborhoods
with shared-resource conflict neighborhoods.

### Additional sampler ablation

Compare random sampling with simulated annealing on the same QUBO workflow. Exact
recourse and acceptance rules remain identical.

### Additional synthetic coordination control

Use an independently generated eight-order seed-2 instance to test whether the hybrid
can revisit a coupled decision that greedy handles myopically. Compare greedy,
zero-gap exact MILP, and hybrid. This control demonstrates algorithmic behavior only;
it cannot establish real-data or quantum advantage.

## 5. Smoke profile

The smoke profile shortens the grid for development:

- sizes 4 and 8;
- penalty scales 0.75 and 1.25;
- candidate limits 1 and 3;
- inventory shocks 0% and 20%;
- seeds 3 and 11;
- coefficient noise 0% and 2%; and
- one hybrid iteration with at most 24 QUBO variables.

The full profile uses four hybrid iterations and at most 40 local QUBO variables.

## 6. Reproducibility and tie-breaking

Default and greedy methods are deterministic. Ties are resolved by incremental
objective, fill increase, lower shipping cost, default before alternate, then
candidate identifier. Every stochastic sampler records its seed. The bundle hash,
configuration, assumption version, and method are included in local run metadata.

Before experiments, the known two-order test must reproduce the exact objective 126:

```bash
pytest -q tests/test_tiny_optimum.py
```

## 7. Execution

```bash
python scripts/run_challenge_study.py \
  --bundle-dir /approved/path/to/challenge-files \
  --profile full \
  --output runs/challenge-study/aggregate_results.csv
```

Or run `notebooks/nestle_challenge_experiments.ipynb` and set `BUNDLE_DIR` in its
configuration cell.

## 8. Interpretation rules

- Never rank an infeasible row as a business solution.
- Compare methods only on the same instance and economic scale.
- A zero MILP gap is a proof; a nonzero gap is not.
- Hybrid improvement is relative to its configured feasible incumbent.
- Zero accepted hybrid moves is a valid negative result.
- Coefficient noise is not physical quantum noise.
- Physical qubits are not interchangeable with logical QUBO variables.
- One stochastic seed is not evidence of robustness.

## 9. Public output contract

The aggregate CSV must not contain order, SKU, DC, candidate, customer, ZIP, or lane
identifiers. `write_experiment_results` and the copilot both reject identifier-like
columns. Row-level assignments and fulfillment stay in ignored local run directories.

