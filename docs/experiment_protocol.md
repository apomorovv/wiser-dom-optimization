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
4. `hybrid`: bounded QUBO candidate generation plus exact classical recourse.

For a quantum claim, run the same hybrid configuration with a remote QPU backend and
at least one tuned local sampler. `simulated_annealing` is quantum-inspired, not a
quantum-hardware result.

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

### Hybrid sample metrics

Report:

- number of raw samples;
- one-outcome-per-order rate;
- sampler calls and actual QPU calls;
- raw one-hot rate;
- unique repaired assignments;
- exact-recourse attempts and successful solves;
- maximum logical QUBO variables and pair couplings;
- accepted improving moves; and
- initial, final, and improved objective.

QUBO feasibility is not global DOM feasibility. Report full feasibility only after
exact recourse and independent validation.

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

Report variables, constraints, QUBO variables and couplings, runtime, memory when
available, best objective, feasibility, and gap or distance from optimum. Keep the
QUBO cap fixed in at least one experiment to demonstrate bounded hardware demand as
the global order count grows.

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

For the implemented annealing path, replace circuit-only settings with reads,
sweeps/anneal schedule, chain strength and embedding statistics when applicable.
Run coefficient-noise levels over multiple seeds; coefficient perturbation is a
sensitivity proxy, not a complete hardware-noise model.

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
solver_metadata.json
```

Optional method-specific files include:

```text
solver_model.lp
solver_statistics.json
raw_samples.csv
repaired_samples.csv
qubo.json
planner_recommendations.csv
planner_view.md
stdout.log
```

The pipeline guarantees the seven core files. A shell or job scheduler captures
`stdout.log`. The hybrid command also writes the planner artifacts.

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

## 11. Fair sampler comparison

Hold constant:

- canonical dataset and candidate columns;
- active neighborhood and QUBO coefficients;
- incumbent warm start;
- repair and top-K recourse policy;
- independent validation and objective calculation; and
- total wall-clock accounting boundary.

Report preprocessing, queue, embedding, sample, repair, and recourse time separately
when the backend exposes them. Compare quality at equal time and time at equal
quality. Exact enumeration is a correctness oracle only for very small QUBOs.

## 12. Public reporting

Public artifacts must use synthetic or approved anonymized identifiers and aggregate results. Follow `docs/privacy.md`.
