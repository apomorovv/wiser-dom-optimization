# Modeling assumptions

Assumption version `v2` records every choice that changes feasibility, objective, or
comparability. A changed rule requires a new version, tests, and rerun.

## Active assumptions

### A1. Focus orders and load expansion

The POC focus flag is `IsInvAvail = N`. When any order on a load is a focus order,
the complete load is included so routing cohesion remains enforceable.

### A2. One group option, one DC/date per order

Each order selects one eligible candidate or is unassigned. Lines may be partially
filled at the selected DC but cannot split across DCs. Orders sharing
`assignment_group` select the same `group_option_id`.

### A3. Integer case conversion

Demand and fulfillment are integer cases. Source planning units are divided by
positive planning-units-per-case; pallet planning-unit rows are multiplied by
cases-per-pallet. Any non-integral result outside tolerance stops the adapter.

### A4. Shipping and dock are load-level fixed terms

Source `Shipping_Cost` is total option cost. Shipping and incremental dock use are
charged once to a deterministic group leader. Default loads consume zero incremental
dock because they are already planned; alternate loads consume one unit.

### A5. Thresholded cut penalty

An order penalty is zero when integer fill reaches
$\lceil\theta_oQ_o\rceil$. Below that threshold, penalty equals variable unmet
cost plus fixed cost plus per-cut-SKU cost, then applies an optional minimum and
maximum. The linear unmet model remains available for synthetic data.

### A6. Five-day protected projected ATP

For a candidate PGI, usable inventory is the minimum nonnegative
`Available_inventory` over PGI through PGI plus five calendar days. It may decrease
across checkpoints. Earlier fulfillment consumes every later protected checkpoint.

### A7. Alternative forecast eligibility is hard

Every SKU in an assignment group must be present in an alternate DC's
inventory/forecast table. Absence at the default DC means zero default fill, not a
missing default candidate.

### A8. Lead time and working days

Lead time is `ceil(distance / 500)` calendar days. Alternative PGI is requested
delivery date minus lead time, moved backward over weekends and configured holidays.

### A9. Diversion must improve by both thresholds

A non-default option must improve default protected-ATP fill by at least five
percentage points of total demand and at least 100 cases, capped at demand:

$$
\min\{Q_o,F_o^{\mathrm{def}}+\max(\lceil0.05Q_o\rceil,100)\}.
$$

### A10. Documented dock limit is hard; throughput is scenario-only

`Dock_Remaining` is enforced. The supplied throughput table reports observed
utilization rather than a maximum, so it is not treated as a real constraint.
Analyses may create explicitly labeled case/pallet headroom scenarios.

### A11. Pareto pruning is an opt-in heuristic

The pruning ablation removes an alternate when another group option has no worse
isolated estimated fill/value, no higher shipping cost/lead time, and is strictly
better in at least one dimension. The default is retained. This is not a globally
lossless dominance proof because options can consume different inventory and capacity
buckets, so common-objective experiments leave it disabled by default.

### A12. Independent validation is authoritative

Solver-native objective and QUBO energy are diagnostics. A recommendation is
reportable only after independent demand, assignment, group, eligibility, inventory,
capacity, diversion, and objective recomputation.

### A13. Bounded local search cannot degrade the incumbent

For both exact LNS and sampler-assisted hybrid search, residualized local optimization
and independent global validation precede acceptance. Only a strict feasible
improvement replaces the incumbent. Polished greedy likewise falls back to its feasible
raw greedy solution if fixed-assignment recourse fails or degrades the objective.

### A14. Remote quantum execution is opt-in

Local exact, random, and simulated-annealing samplers are the default. Remote QPU or
managed-hybrid execution requires explicit approval and `allow_remote=true`.

### A15. Required source fields fail closed

The adapter validates the five runtime tables before transformation. Missing columns,
blank required identifiers, invalid dates, malformed or nonfinite required numerics,
invalid domains, and a failed inventory reconciliation stop the load with
`PocDataError`. Required economics or resources are not silently replaced with zero.
Only explicitly documented optional nulls may use a fallback.

### A16. Candidate-DC scope is explicit

The default technical candidate universe is `network_intersection`: DCs represented in
the shipping, inventory, and dock source tables. `focus_default_dcs` is retained as a
narrow comparison policy. Both scopes still require a lane, group-compatible option,
SKU presence, an open PGI date, on-time arrival, and the diversion-improvement rule.
This technical intersection does not prove that every connected DC is operationally
authorized; the scope sensitivity experiment quantifies the decision pending owner
sign-off.

## Remaining decisions for production deployment

### U1. Complete enterprise working-day calendar

Only weekends and explicitly supplied holidays are known. Production must provide
the authoritative plant calendar and exception policy.

### U2. Throughput maximum or remaining-capacity equation

Observed utilization is readable, but no source maximum is documented. Nestlé must
confirm whether another denominator, shift plan, or remaining-capacity table exists.

### U3. Customer-specific all-or-nothing service

The challenge permits partial fulfillment. If particular customer classes require
all-or-nothing service, the source must identify them and the MILP must add the
corresponding equality constraints.

### U4. Commercial scale and approval boundary

The source currency/scale, cost components, and acceptable public aggregates must be
confirmed before a business deployment or external QPU experiment.

### U5. Physical QPU benchmark

The archived IBM study used a generated synthetic QUBO; restricted challenge data were
not sent to hardware. The final path-mixer and linear-W circuit changes require a matched
rerun with backend calibration, timing, transpilation statistics, repetitions, and
uncertainty intervals before hardware claims are updated.

### U6. Operational candidate-DC authorization

Nestlé must confirm whether all DCs in the shipping/inventory/dock intersection are
allowed alternatives for each customer/load, or whether an explicit allowlist,
region/service rule, or the narrower focus-default set is required. Until then,
`candidate_dc_scope` is reported with every scope-sensitive result and neither policy
is presented as a production fact.

## Change checklist

When resolving an assumption, update:

1. this document and `ASSUMPTION_VERSION`;
2. the POC mapping and canonical dictionary;
3. loader, baseline, exact MILP, hybrid preview/recourse, and validator;
4. at least one positive and one negative automated test; and
5. every affected experiment and report.
