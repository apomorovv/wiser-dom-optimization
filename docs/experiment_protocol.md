# Experiment protocol

The notebook and `src/domopt/experiments.py` implement one fair, aggregate-only
comparison protocol for the real challenge subsets and an independent synthetic
control.

## 1. Immutable comparison definition

An experiment is identified by:

$$
E=(\text{profile},\text{bundle/problem hash},\text{schema},\text{assumptions},
\text{objective},\text{method config},\text{seed},\text{source state}).
$$

Changing any element creates a different experiment. Methods compared within one row
group receive identical canonical data, candidates, objective calculation, and
validator. Source state contains the Git commit, a dirty-worktree flag, and a content
hash over tracked changes and untracked files; a commit hash alone is insufficient for
an uncommitted run.

## 2. Methods

| Method | Role |
|---|---|
| `default` | Deterministic default-DC policy baseline. |
| `greedy` | Fast sequential whole-load reassignment using current residual resources. |
| `polished_greedy` | Greedy assignment policy followed by exact fixed-assignment quantity and thresholded-penalty recourse; never accepts degradation. |
| `exact_lns` | Adaptive conflict-aware classical large-neighborhood search; each bounded neighborhood jointly optimizes assignments and fulfillment with an exact/bounded MILP. |
| `classical` | Exact or bounded SciPy/HiGHS MILP reference. |
| `hybrid` | Bounded QUBO assignment search plus exact MILP recourse. |

Simulated annealing is quantum-inspired, and local statevector QAOA is a quantum
algorithm simulation rather than quantum hardware. The optional IBM backend
are excluded until restricted-data approval is obtained.

## 3. Common metrics

Every restricted local result records feasibility, raw source-scale objective,
requested and fulfilled value, penalty, shipping, case/value fill, normalized
objective capture, reassigned orders and assignment groups, runtime, and—when
available—MILP optimality gap, bound, variable count, and constraint count. Public
artifacts use approved aggregates, normalized/indexed economics, or synthetic values;
an aggregate total is not automatically safe to publish.

Polished-greedy rows isolate the raw greedy value and exact quantity-polish gain.
Exact-LNS rows isolate initial polish from assignment-search improvement and record
local solves, accepted assignment moves, maximum active groups/orders, local model
size, and phase timing. Hybrid rows likewise separate raw baseline objective, exact
fixed-assignment polish, sampler-only improvement, and total improvement, and record
accepted moves, recourse calls, maximum local QUBO width, raw one-hot rate, and phase
timing.

$$
J=\text{fulfilled value}-\text{thresholded penalty}-\text{shipping cost},
$$

$$
\operatorname{caseFill}=\frac{\sum_{o,s}f_{os}}{\sum_{o,s}Q_{os}}.
$$

$$
\operatorname{objectiveCapture}=\frac{J}{\sum_{o,s}Q_{os}v_{os}}.
$$

Raw objective totals are compared only within an identical instance. Scaling plots
use objective capture because total value necessarily grows with instance size.

Solver-native objective and QUBO energy are never copied into the comparison table.
The independent evaluator recomputes all three objective components.

## 4. Full experiment grid

### Core solver comparison

Run default, greedy, polished greedy, exact LNS, exact MILP, and hybrid on the same
deterministic high-shortage real-data subset. Exact MILP uses a 60-second limit and 1%
requested relative gap; the achieved gap is reported. Polished greedy is the low-risk
production baseline, while exact LNS is the strongest bounded classical search control
for judging whether a sampler contributes beyond classical neighborhoods.

### Size scaling

The true decision unit is an assignment group, not an order row. On the real bundle,
run greedy and polished greedy at 8, 20, 50, 100, 250, and 372 groups; exact LNS
through 372 groups; hybrid through 50 groups; and the global exact MILP at 8 and 20
groups. The full profile runs three repetitions of every enabled method/size pair and
reports median and dispersion rather than selecting the fastest trial.

The real subsets are nested highest-shortage groups, so their business composition
changes with size. Run a separate independently generated synthetic control at 20, 50,
100, 250, and 500 groups, also with three repetitions, to study algorithmic scaling
under an explicit generator. Report actual groups, orders, lines, SKUs, candidate rows,
runtime, objective capture, reassigned groups, MILP gap/model counts, LNS local model
counts, and maximum local QUBO width. Real and synthetic rows remain separate.

### Candidate-DC scope sensitivity

Build the same focus population under both explicit policies:

- `focus_default_dcs`: only DCs that are defaults within the focus population; and
- `network_intersection`: DCs represented in the shipping, inventory, and dock source
  tables, which is the default.

Compare polished greedy and exact LNS using the same validator. This is a business-rule
sensitivity, not evidence that every technically connected DC is operationally allowed.

### Penalty-weight sensitivity

Multiply variable, fixed, per-cut-SKU, minimum, and maximum penalties by 0.25, 0.5,
1.0, 2.0, and 4.0. Use a separate fixed subset of groups whose default fill is below
the penalty-activation threshold and rank them by worst-case penalty exposure. This
prevents a vacuous sweep over zero-penalty orders. Compare greedy and hybrid only
within the same scale; raw objectives across different scales do not represent the
same economics.

### QUBO-penalty calibration

Cross the one-hot multipliers 1.05, 1.25, 1.5, and 2.0 with resource-conflict
multipliers 0, 0.5, 1.0, and 2.0. Measure raw one-hot rate, repair/recourse work,
accepted moves, improvement, feasibility, and runtime. These are algorithmic
penalties and must not be interpreted as business shortage costs.

### Candidate-count sensitivity

Retain at most 1, 2, 4, or 6 group options, always preserving the default. Compare
greedy and hybrid to quantify search breadth, runtime, and quality.

### Inventory shocks

Apply deterministic 0%, 10%, 25%, and 40% reductions to every protected-ATP value.
Report a nominal greedy routing with only quantity recourse separately from policies
that are fully reoptimized after seeing the shock. The former measures fixed-routing
robustness; the latter is wait-and-see recourse. This is a stress scenario, not a
forecast.

### Seed and coefficient-noise sensitivity

Run hybrid at seeds 3, 11, 29, and 47 with relative Gaussian QUBO coefficient noise
0%, 1%, 3%, and 5%. Samples are ranked against the original QUBO and pass exact recourse.
This tests local ranking sensitivity; it is not a model of physical QPU noise.

### Local QAOA readout-noise proxy

Run the Dicke/XY statevector control at the same four seeds with independent symmetric
measurement bit-flip probabilities 0%, 0.5%, 1%, and 2%. Apply flips after the ideal
circuit and before one-hot repair. Report raw one-hot rate, repair, validated
improvement, and runtime. This is a measurement-channel proxy only; it does not model
gate error, decoherence, crosstalk, compilation, or a physical backend.

### Pareto-pruning ablation

Run the same base population with and without heuristic Pareto pruning. Because
different options consume different resource buckets, this is not a lossless
dominance rule and remains disabled by default. It is useful only when it reduces
candidate count/runtime without reducing validated objective on the tested instance.
Repeat both arms across the common stochastic-seed set.

### Random-versus-conflict batching

Hold all hybrid settings fixed and compare seeded random whole-group neighborhoods
with shared-resource conflict neighborhoods across the common four-seed set.

### Additional sampler ablation

Compare feasible-subspace exact enumeration, random sampling, simulated annealing,
and locally simulated QAOA with product Dicke(1)/W initial states and XY ring mixers.
Exact recourse and acceptance rules remain identical.

### Additional synthetic coordination control

Use an independently generated four-order seed-6 instance with a verified 197.2-unit
assignment gap after fixed-assignment polish. Compare greedy, zero-gap exact MILP,
and hybrid. This prevents classical quantity recourse from being misreported as a
sampler gain. The control demonstrates algorithmic behavior only; it cannot establish
real-data or quantum advantage.

## 5. Smoke profile

The smoke profile shortens the grid for development:

- sizes 4 and 8;
- penalty scales 0.75 and 1.25;
- candidate limits 1 and 3;
- inventory shocks 0% and 20%;
- seeds 3 and 11;
- coefficient noise 0% and 2%; and
- readout bit flips 0% and 2%;
- a 2-by-2 QUBO-penalty grid; and
- one repetition at each real and synthetic scaling point; and
- one hybrid iteration with at most 24 QUBO variables.

The full profile uses three scaling repetitions, three hybrid iterations, two exact
recourse candidates per iteration, a greedy incumbent, at most 40 local QUBO variables,
and bounded adaptive exact-LNS neighborhoods.

## 6. Reproducibility and tie-breaking

Default and greedy methods are deterministic. Ties are resolved by incremental
objective, fill increase, lower shipping cost, default before alternate, then
candidate identifier. Every stochastic sampler records its seed. The bundle hash,
problem hash, profile, configuration, schema/assumption/objective versions, method,
seed, commit, dirty flag, and source-state hash are included in aggregate rows and
checkpoint identity.

Before experiments, the known two-order test must reproduce the exact objective 126:

```bash
pytest -q tests/test_tiny_optimum.py
```

## 7. Execution

Normalize browser-added filename suffixes first:

```bash
python scripts/prepare_challenge_bundle.py \
  --source-dir /approved/path/to/downloads \
  --output-dir data/raw/nestle_challenge
```

```bash
python scripts/run_challenge_study.py \
  --bundle-dir data/raw/nestle_challenge \
  --profile full
```

Or run `notebooks/nestle_challenge_experiments.ipynb` with
`NESTLE_BUNDLE_DIR=data/raw/nestle_challenge`. Each study uses a content-addressed run
directory below
`results/challenge-study/notebook/<profile>/<problem-hash>/<run-hash>/`. The CSV is
paired with a manifest containing its exact identity, row count, and ordered columns.
A checkpoint is loaded only when the manifest, identity, required columns, row count,
and table schema all match; otherwise the study is rerun rather than silently mixing
profiles or code states. Plots live in the same run-scoped directory.

The command-line suite writes separately below
`results/challenge-study/cli/<profile>/`. The optional GPU cell benchmarks synthetic
batched QUBO scoring only. Optional IBM hardware validation sends generated synthetic
circuits and records backend, queue, transpilation, mitigation, raw feasibility,
exact-reference quality, and usage metadata. Neither optional test is part of the
default profile.

## 8. Interpretation rules

- Never rank an infeasible row as a business solution.
- Compare methods only on the same instance and economic scale.
- Summarize repeated scaling with median and dispersion; retain failures rather than
  dropping them from the denominator.
- A zero MILP gap is a proof; a nonzero gap is not.
- Sampler-only hybrid improvement is relative to the exact fixed-assignment-polished
  incumbent; total improvement is relative to the raw configured baseline.
- Zero accepted hybrid moves is a valid negative result.
- Coefficient noise is not physical quantum noise.
- Physical qubits are not interchangeable with logical QUBO variables.
- One stochastic seed is not evidence of robustness.

## 9. Public output contract

The aggregate CSV must not contain order, SKU, DC, candidate, load, material, plant,
customer, ZIP, address, delivery, or lane identifiers. `write_experiment_results`
rejects identifier-like columns, but this is defense in depth rather than publication
approval: values, dates, configuration strings, and small-cell aggregates still need
human review. Row-level assignments, fulfillment, and planner tables stay in ignored
local run directories. Exact source-scale economic totals remain restricted unless the
challenge owner explicitly approves them; public comparisons use normalized/indexed or
synthetic economics.
