# Results and implementation audit

Audit updated: 2026-08-07

## Status

The supplied archive contains a clean command-line full study with 247 rows across 14
experiment families. All 18 CSV-manifest hashes pass; every row reports
`feasible=True`, `validation_violation_count=0`, `git_dirty=False`, and the same source
commit. Objective/component, case/value balance, business-cost, and rate identities
reconcile to at most `7.5e-9` absolute error. The separately supplied notebook aggregate
has an older schema and two infeasible exact rows; it is stale and must not be combined
with the clean command-line aggregate.

The clean archive records validity as a Boolean, category/count, and messages rather
than numeric residuals. The updated implementation now emits validator tolerance and
maximum demand-balance, integrality, inventory-excess, and capacity-excess residuals;
those fields require a fresh run and are not retroactively claimed for the archive.

No exact source-scale objective, revenue, penalty, shipping, inventory, capacity, or
row-level identifier is published here. Reviewed public results should use normalized
objective capture, baseline-indexed changes, rates, runtimes, model sizes, or
independently generated synthetic values unless the challenge owner approves another
aggregate explicitly.

## Supplied-results evaluation

### Solver decision

The common 20-assignment-group comparison supports **polished greedy as the operational
default**. It ties exact MILP, exact LNS, and sampler-assisted LNS at 64.9002% normalized
objective capture. The exact MILP is fastest among those tied methods
at 2.518 seconds; polished greedy takes 2.715 seconds, exact LNS 6.565 seconds, and
hybrid 12.149 seconds. Raw greedy takes 0.755 seconds and reaches 64.8961% capture.
The hybrid accepts no sampler move, so all of its improvement over raw greedy comes
from the common classical quantity polish.

| Method | Objective capture | Runtime (s) | Evidence-based role |
|---|---:|---:|---|
| Greedy | 64.8961% | 0.755 | Time-critical fallback |
| Polished greedy | 64.9002% | 2.715 | Production default |
| Exact MILP | 64.9002% | 2.518 | Small-instance certificate |
| Exact LNS | 64.9002% | 6.565 | Coordinated-assignment escalation |
| Hybrid sampler LNS | 64.9002% | 12.149 | Research comparator |

This is not a universal solver theorem. The independent synthetic coordination control
contains a deliberately coupled greedy trap: exact LNS and exact MILP improve its
objective by 197.2, while polished greedy cannot. Exact LNS reaches that result in
0.57 seconds versus 8.32 seconds for the sampler-assisted path. This justifies retaining
exact LNS as an escalation, but not making the sampler the production solver.

### Scaling and design choices

- At 372 real assignment groups (750 orders), median greedy, polished-greedy, and
  exact-LNS runtimes are 15.945, 51.779, and 71.208 seconds. Exact LNS improves the
  polished objective by only `0.000027%` at that point.
- Across the real scaling grid, 98.87% of exact LNS's aggregate improvement over raw
  greedy comes from the initial polish; neighborhood assignment search contributes
  1.13%. The corresponding initial-polish share is 98.44% in synthetic scaling.
- Raising the candidate cap from one to two increased capture from 61.5664% to
  64.9002%; caps of four and six added candidates and runtime without improving quality
  on the tested subset. A cap of two is the measured default for this instance, not a
  globally proven optimum.
- Heuristic Pareto pruning reduced the tested hybrid candidate set from 90 to 70 while
  preserving observed quality and reducing mean runtime by 2.2%. It remains an
  ablation, not a proof of globally safe dominance, and is disabled in the rigorous
  default path.
- Penalty scaling from 0.25× through 4× did not change the selected routing on the
  tested subset. That is evidence of local routing stability, not evidence that the
  business penalty scale is correct.
- All exact-feasible, random, simulated-annealing, and local QAOA samplers found the
  synthetic coupled move after repair and exact recourse. With no observed quality
  separation and a faster random control, the supplied simulation does not show a
  quantum-algorithm advantage.

### IBM evidence boundary

The supplied IBM evidence is a dirty, single-seed, 512-shot study on an independently
generated four-order, 16-logical-qubit control. Baseline p=1 and p=1 with DD plus
measurement twirling each produce one exact feasible-QUBO hit; all variants reach the
same final objective only after exact recourse. The p=1 end-to-end runtimes of 81–83
seconds are not explained by 4.8–5.7 seconds of reported QPU turnaround, while the
deeper p=2 example completes in 15.8 seconds. It is therefore diagnostic only.

The corrected study derives circuit width from the actual QUBO, runs the complete
$p=1/p=2$ by baseline/DD/DD-plus-measurement-twirling matrix, fixes angle and transpiler
seeds across hardware repetitions, records job IDs/timestamps and phase timings, keeps
missing provider timing as missing rather than zero, records package versions, physical
qubit mapping, and available calibration timestamp, resumes successful variants, and
retains/retries failures. No corrected hardware result is claimed until the opt-in
notebook cell is run from this clean branch.

## Implemented validity controls

| Control | Current implementation | Evidence to collect in the rerun |
|---|---|---|
| Strict input contract | Five runtime tables are checked for required columns, identifiers, dates, numeric/domain validity, finiteness, and the documented inventory identity before transformation | Successful bundle audit plus any rejected-input diagnostics retained locally |
| Common objective | Default, greedy, polished greedy, exact LNS, exact MILP, and hybrid are recomputed by the same independent evaluator | Component reconciliation and feasibility status for every method row |
| Independent feasibility | Assignment, demand, group cohesion, candidate eligibility, inventory, capacity, diversion, and date checks are solver-independent; numeric tolerance/residual diagnostics are exported | Zero violations and within-tolerance residuals for every reportable recommendation |
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

1. verify its checkpoint manifest, content hash, and complete provenance identity;
2. retain failed/infeasible rows in the audit denominator and exclude them from ranking;
3. confirm all compared rows share the same problem and economic scale;
4. replace exact source-scale economics with approved normalized/indexed measures;
5. inspect columns, values, configuration strings, labels, logs, paths, dates, and small
   cells for re-identification risk;
6. obtain a second privacy review; and
7. state limitations without claiming quantum or production advantage beyond the
   evidence.
