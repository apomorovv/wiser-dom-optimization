# Experiment protocol

This protocol ensures fair and reproducible comparison of baseline, classical, quantum-inspired, and quantum methods.

## 1. Immutable experiment definition

An experiment is defined by

\[
E=(\text{dataset},\text{schema version},\text{assumption version},\text{candidate set},\text{objective convention},\text{method configuration},\text{seed},\text{git commit}).
\]

Changing any component creates a new experiment.

## 2. Common evaluation

Every method receives the same canonical tables and candidate set. Every result is evaluated by:

- `src/domopt/validation.py`;
- `src/domopt/objective.py`;
- `src/domopt/metrics.py`.

Do not copy a method-specific objective directly into a final comparison table.

## 3. Required methods

The minimum comparison contains:

1. `default`: deterministic default-DC baseline;
2. `greedy`: deterministic sequential reassignment baseline;
3. `classical`: exact or bounded classical optimizer;
4. `quantum` or `quantum_inspired`: only after the first three pass validation.

## 4. Required metrics

### Objective

\[
\text{Objective}=\text{FulfilledValue}-\text{PenaltyCost}-\text{ShippingCost}.
\]

Report all components separately.

### Case fill rate

\[
\operatorname{CFR}
=
\frac{\sum_{o,s}F_{os}}{\sum_{o,s}Q_{os}},
\]

where \(F_{os}\) is final fulfilled quantity.

### Value fill rate

\[
\operatorname{VFR}
=
\frac{\sum_{o,s}v_{os}F_{os}}{\sum_{o,s}v_{os}Q_{os}}.
\]

### Reassigned orders

\[
N_{\mathrm{divert}}
=
\sum_o\mathbf 1[d_o^{\mathrm{selected}}\ne d_o^{\mathrm{def}}\land z_o=0].
\]

### Feasibility

A result is feasible only when every validation category has zero violations.

### Classical optimality gap

For maximization with incumbent \(P\) and valid upper bound \(B\ge P\):

\[
\operatorname{gap}
=
\frac{B-P}{\max(1,|P|)}.
\]

Store the solver-native definition too when it differs.

### Quantum sample metrics

Report:

- number of raw samples;
- one-outcome-per-order rate;
- feasible rate before repair;
- feasible rate after repair;
- best feasible objective;
- mean feasible objective;
- standard deviation across seeds;
- repair improvement.

## 5. Determinism and tie-breaking

Default and greedy methods must be deterministic under a fixed order. Unless an experiment explicitly changes it, break ties by:

1. larger incremental objective;
2. larger case-fill increase;
3. lower shipping cost;
4. default DC before alternate DC;
5. lexicographically smaller `candidate_id`.

Record all stochastic seeds. Do not present one stochastic run as representative.

## 6. Tiny-instance gate

Before real-data or quantum experiments:

```bash
pytest -q tests/test_objective.py
pytest -q tests/test_validation.py
pytest -q tests/test_tiny_optimum.py
```

must pass. The tiny exact run must reproduce

\[
O_1\rightarrow D_2,\qquad O_2\rightarrow D_1,
\]

with objective \(126\).

## 7. Scaling protocol

Vary:

- \(|\mathcal O|\): orders;
- \(|\mathcal S|\): SKUs;
- average candidates per order;
- \(|\mathcal D|\): DCs;
- \(|\mathcal T|\): dates;
- scarcity;
- candidate-conflict density.

Report variables, constraints, QUBO variables and couplings, runtime, memory when available, best objective, feasibility, and gap or distance from optimum.

## 8. Sensitivity protocol

At minimum vary:

- penalty multiplier;
- shipping-cost multiplier;
- minimum divert improvement;
- inventory protection;
- candidate-pruning threshold;
- classical time limit;
- QUBO penalty strengths;
- quantum depth, shots, optimizer, and seed.

Use one-factor-at-a-time plots for the primary sensitivity study.

## 9. Run-directory contract

Each run should contain:

```text
config.json
input_fingerprint.json
assignments.csv
fulfillment.csv
metrics.json
validation.json
stdout.log
```

Optional method-specific files include:

```text
solver_model.lp
solver_statistics.json
raw_samples.csv
repaired_samples.csv
qubo.json
```

## 10. Registry

Append one row per method to `experiments/registry.csv` with:

```text
experiment_id
run_id
timestamp_utc
dataset_id
schema_version
assumption_version
method
seed
git_commit
config_path
run_path
feasible
objective_value
case_fill_rate
value_fill_rate
shipping_cost
penalty_cost
reassigned_orders
runtime_seconds
optimality_gap
notes
```

Do not edit reported values by hand. Corrections require a new run.

## 11. Public reporting

Public artifacts must use synthetic or approved anonymized identifiers and aggregate results. Follow `docs/privacy.md`.
