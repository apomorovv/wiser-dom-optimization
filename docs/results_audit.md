# Results and implementation audit

Audit updated: 2026-08-07

## Status

The supplied command-line aggregate contains 247 feasible rows across 14 experiment
families and zero independently reported validation violations. It is useful evidence,
but every row reports `git_dirty=True`; numerical conclusions are therefore provisional
until the corrected full profile is rerun from a clean commit. The separately supplied
notebook aggregate has an older 62-column schema, only 86 rows, and two infeasible exact
rows. It is stale and must not be combined with the 105-column command-line aggregate.

No exact source-scale objective, revenue, penalty, shipping, inventory, capacity, or
row-level identifier is published here. Reviewed public results should use normalized
objective capture, baseline-indexed changes, rates, runtimes, model sizes, or
independently generated synthetic values unless the challenge owner approves another
aggregate explicitly.

## Supplied-results evaluation

### Solver decision

The common 20-assignment-group comparison supports **polished greedy as the operational
default**. It matched the zero-gap exact MILP's normalized objective capture (64.9002%)
in 6.63 seconds, versus 6.67 seconds for the full MILP, 16.05 seconds for exact LNS,
and 28.43 seconds for sampler-assisted LNS. Raw greedy was the speed option at 1.73
seconds and reached 64.8961% capture. The hybrid run accepted no sampler move, so it
does not establish sampler value on this real subset.

| Method | Objective capture | Runtime (s) | Evidence-based role |
|---|---:|---:|---|
| Greedy | 64.8961% | 1.73 | Time-critical fallback |
| Polished greedy | 64.9002% | 6.63 | Production default |
| Exact MILP | 64.9002% | 6.67 | Small-instance certificate |
| Exact LNS | 64.9002% | 16.05 | Coordinated-assignment escalation |
| Hybrid sampler LNS | 64.9002% | 28.43 | Research comparator |

This is not a universal solver theorem. The independent synthetic coordination control
contains a deliberately coupled greedy trap: exact LNS and exact MILP improve its
objective by 197.2, while polished greedy cannot. Exact LNS reaches that result in
0.57 seconds versus 8.32 seconds for the sampler-assisted path. This justifies retaining
exact LNS as an escalation, but not making the sampler the production solver.

### Scaling and design choices

- At 372 real assignment groups, median greedy and polished-greedy runtimes were 37.55
  and 133.35 seconds. Polishing improved objective capture by about 3.37 basis points.
- The supplied scaling configuration started exact LNS from raw greedy while the
  comparison showed a separately polished baseline. That was an unfair attribution
  mismatch and explains why exact LNS appeared below polished greedy at larger sizes.
  The corrected experiment always starts exact LNS from the polished incumbent, so its
  accepted-result invariant prevents this artifact.
- Raising the candidate cap from one to two increased capture from 61.5664% to
  64.9002%; caps of four and six added candidates and runtime without improving quality
  on the tested subset. A cap of two is the measured default for this instance, not a
  globally proven optimum.
- Heuristic Pareto pruning reduced the tested hybrid candidate set from 90 to 70 while
  preserving observed quality and modestly reducing runtime. It remains an ablation,
  not a proof of globally safe dominance.
- Penalty scaling from 0.25× through 4× did not change the selected routing on the
  tested subset. That is evidence of local routing stability, not evidence that the
  business penalty scale is correct.
- All exact-feasible, random, simulated-annealing, and local QAOA samplers found the
  synthetic coupled move after repair and exact recourse. With no observed quality
  separation and a faster random control, the supplied simulation does not show a
  quantum-algorithm advantage.

### IBM evidence boundary

The supplied IBM screenshot reports one QPU call and only 19% raw one-hot samples, but
does not identify the backend and reports zero QPU access time. The former hardware
adapter read a provider-specific timing field that IBM does not populate, so zero was a
measurement bug rather than evidence of zero device usage. The corrected study uses a
coupled synthetic instance with an exact reference, discovers and records the current
least-busy eligible IBM backend, compares baseline/DD/DD-plus-measurement-twirling and
$p=1$/$p=2$ variants, and records Runtime timestamps, quantum seconds, raw feasibility,
optimal-hit rate, assignment gain, and transpiled two-qubit cost. No new IBM hardware
result is claimed until that opt-in study is run from an authenticated machine.

## Implemented validity controls

| Control | Current implementation | Evidence to collect in the rerun |
|---|---|---|
| Strict input contract | Five runtime tables are checked for required columns, identifiers, dates, numeric/domain validity, finiteness, and the documented inventory identity before transformation | Successful bundle audit plus any rejected-input diagnostics retained locally |
| Common objective | Default, greedy, polished greedy, exact LNS, exact MILP, and hybrid are recomputed by the same independent evaluator | Component reconciliation and feasibility status for every method row |
| Independent feasibility | Assignment, demand, group cohesion, candidate eligibility, inventory, capacity, diversion, date, and objective checks are solver-independent | Zero violations for every reportable recommendation |
| Thresholded penalties | Exact model, evaluator, baselines/recourse, and planner use the same fixed, variable, per-cut-SKU, floor, and cap semantics | Component tests and aggregate reconciliation |
| Load-atomic decisions | Whole assignment groups share one DC/PGI option; shipping and incremental dock use are charged to a deterministic group leader | Group-level reassignment counts and planner group totals |
| Candidate breadth | The default `network_intersection` policy is compared with the narrower `focus_default_dcs` policy | Candidate breadth, feasibility, objective capture, and runtime by scope |
| Provenance | Rows contain bundle/problem hashes, schema and assumption versions, objective version, configuration, seed, commit, dirty state, and source-state hash | No missing provenance fields in final aggregate tables |
| Checkpoint integrity | CSVs are paired with identity manifests and rejected on profile, problem, configuration, code-state, row-count, or schema mismatch | Only manifest-matched checkpoints combined into final figures |
| Planner economics | The planner uses canonical thresholded penalties, assignment-group totals, expected arrival, and on-time status | Reviewed data-specific one-page planner artifact retained in the approved environment |

## Solver roles to compare

| Method | Valid role in the study | Claim boundary |
|---|---|---|
| Default | Feasible current-policy baseline with shared-resource accounting | Baseline, not an unconstrained per-order preview |
| Greedy | Fast deterministic sequential whole-load reassignment | Heuristic; no optimality claim |
| Polished greedy | Greedy routing followed by exact fixed-assignment quantity recourse | Strong low-risk production baseline; routing remains greedy |
| Exact LNS | Adaptive conflict-aware bounded MILP neighborhoods with global validation | Classical matheuristic; local gaps do not prove a global optimum |
| Exact MILP | Exact or bounded reference on tractable subsets | A proof only when the reported global gap is zero |
| Hybrid | Bounded assignment QUBO proposals followed by exact recourse and validation | Quantum-inspired/classically assisted unless an approved physical backend is actually used |
| Local QAOA simulation | Constraint-preserving Dicke/W initial states and XY mixers on small neighborhoods | Quantum-algorithm simulation, not hardware or quantum advantage |

The solver comparison must show raw greedy, exact quantity-polish gain, exact-LNS
assignment-search gain, and sampler-only hybrid gain separately. This prevents classical
recourse from being attributed to a QUBO sampler.

The optional same-instance exact-cover-versus-deterministic-LP comparison is not
implemented. The local assignment QUBO is a bounded search neighborhood and must not be
reported as that optional formulation comparison.

## Evidence plan

### Common solver comparison

Use one frozen real-data subset and candidate set for all six methods. Report feasibility,
normalized objective capture, case/value fill, penalty and shipping as approved
normalized components, reassigned assignment groups, runtime, model size, bound/gap,
and phase attribution. Rank only feasible rows from this common instance.

### Repeated real scaling

Run the configured nested real assignment-group sizes with three repetitions in the
full profile. Summarize median and dispersion for every enabled method. Report actual
orders, groups, lines, SKUs, candidate rows, local/global model counts, bounded QUBO
width, failures, and achieved gap. Because the subsets are shortage-ranked and their
business composition changes with size, quality levels are descriptive rather than a
pure size effect.

### Repeated synthetic scaling

Run the independent generator at the configured sizes and three seeds/repetitions.
Keep these rows separate from real evidence. This control provides a cleaner view of
algorithmic growth and may be published because it contains no source records or
commercial coefficients.

### Sensitivities and ablations

- compare `network_intersection` and `focus_default_dcs` candidate scopes;
- vary candidate count while always retaining the default option;
- apply penalty scales only to a fixed penalty-active subset;
- distinguish fixed-routing inventory stress from post-shock reoptimization;
- test heuristic Pareto pruning off/on without calling it globally safe;
- compare conflict and seeded-random neighborhoods under the same budgets; and
- run sampler, one-hot/conflict-penalty, seed, and coefficient-perturbation controls on
  an independently generated coupled instance where an assignment move exists.
- run the separate QAOA readout-bit-flip proxy and report raw one-hot feasibility before
  repair without presenting it as a gate or hardware-noise model.

Coefficient perturbation is an algorithmic sensitivity. The readout-bit-flip control
is only a measurement-channel proxy; neither is a complete physical device model.
Uniform inventory reductions are stress scenarios, not calibrated forecasts or a
robust/CVaR recommendation.

## Scalability interpretation rules

- Polished greedy is the measured production default. Exact LNS is the coordinated-
  assignment escalation, global exact MILP supplies tractable certificates, and hybrid
  remains an experimental control.
- Exact LNS bounds each neighborhood by groups, orders, fulfillment variables, local
  time, and requested MILP gap; independent global validation guards every accepted
  move.
- Hybrid bounds local QUBO width and recourse, but preprocessing, neighborhood
  construction, validation, and MILP recourse remain classical end-to-end costs.
- GPU throughput for synthetic dense QUBO scoring is not application speedup unless
  transfer and all surrounding workflow time are included.
- Logical QUBO variables are not physical qubits. IBM hardware claims additionally
  require transpilation depth/two-qubit statistics, queue and execution timing, repeated
  trials, and a matched tuned classical control.
- A zero exact-MILP gap proves optimality only for that exact frozen instance. A local
  neighborhood gap or a feasible repaired sample does not.

## Remaining owner decisions

The implementation exposes rather than silently resolves the following production
questions:

- authoritative enterprise working-day and holiday calendar;
- operationally allowed alternative-DC policy or allowlist;
- physical dock consumption per load versus per order;
- documented throughput maximum or remaining-capacity equation;
- default-versus-alternate protected-ATP horizon;
- customer priority and any all-or-nothing service rules;
- source currency/scale and the public-aggregation approval boundary; and
- provider, region, payload, retention, and timing approval for any remote QPU run.

Until those decisions are signed off, scope and scenario labels must accompany every
affected result.

## Submission artifacts still deferred

The following are required but depend on reviewed full-profile outputs and therefore
remain intentionally unfinished:

1. the separate two-page business/technical summary;
2. the six-to-ten-page final report;
3. the five-to-seven-slide presentation; and
4. the reviewed data-specific one-page planner view.

The runnable repository, notebook workflow, report/planner templates, and aggregate
evidence schema are preparation artifacts; they are not substitutes for those final
deliverables.

## Pre-publication gate

Before promoting any local table or figure:

1. verify its checkpoint manifest and complete provenance identity;
2. retain failed/infeasible rows in the audit denominator and exclude them from ranking;
3. confirm all compared rows share the same problem and economic scale;
4. replace exact source-scale economics with approved normalized/indexed measures;
5. inspect columns, values, configuration strings, labels, logs, paths, dates, and small
   cells for re-identification risk;
6. obtain a second privacy review; and
7. state limitations without claiming quantum or production advantage beyond the
   evidence.
