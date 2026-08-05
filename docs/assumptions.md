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

### A11. Pareto pruning preserves default options

An alternate option is removed only when another group option has no worse estimated
fill and fulfilled value, no higher shipping cost and lead time, and is strictly
better in at least one dimension. The default option is always retained.

### A12. Independent validation is authoritative

Solver-native objective and QUBO energy are diagnostics. A recommendation is
reportable only after independent demand, assignment, group, eligibility, inventory,
capacity, diversion, and objective recomputation.

### A13. Hybrid search cannot degrade the incumbent

Exact local recourse and global validation precede acceptance. Only a strict feasible
improvement replaces the incumbent.

### A14. Remote quantum execution is opt-in

Local exact, random, and simulated-annealing samplers are the default. Remote QPU or
managed-hybrid execution requires explicit approval and `allow_remote=true`.

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

The repository provides adapters but has not sent the restricted QUBO to hardware.
Any claim needs organizer approval, matched timing, embedding statistics, repeated
trials, and uncertainty intervals.

## Change checklist

When resolving an assumption, update:

1. this document and `ASSUMPTION_VERSION`;
2. the POC mapping and canonical dictionary;
3. loader, baseline, exact MILP, hybrid preview/recourse, and validator;
4. at least one positive and one negative automated test; and
5. every affected experiment and report.

