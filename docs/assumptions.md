# Modeling assumptions

Assumption version `v1` records every choice that changes feasibility, objective,
or comparability. A changed rule requires a new version, tests, and rerun.

## Active assumptions

### A1. Focus orders are preselected

The optimization population is an input. `focus_orders.py` may identify it, but the
MILP does not decide whether an order belongs to the population.

### A2. One DC/date or unassigned

An order selects exactly one eligible DC/PGI candidate or an explicit unassigned
outcome. Lines cannot split across DCs. Partial fulfillment at the selected DC is
allowed.

### A3. Cases are integer

Demand, fulfillment, unfulfilled demand, inventory, and loose-case picks are
nonnegative integer cases. Monetary terms use one declared currency or synthetic
scale.

### A4. Shipping cost is total

`shipping_cost` is the full candidate cost charged once when selected. It is not an
incremental difference from the default lane. Source data must be converted before
loading if it uses another convention.

### A5. Unmet penalty is linear

Penalty cost is unfulfilled cases multiplied by price per case and penalty rate. The
canonical table stores the resulting nonnegative `penalty_per_unfilled_case`.

### A6. Inventory is protected projected ATP

Projected available-to-promise may increase or decrease. A fulfillment at an early
PGI date consumes every later checkpoint, protecting the lowest future amount. The
alternative `cumulative_receipts` metadata policy requires nondecreasing inventory.

### A7. Eligibility is hard preprocessing

Closed dates, late arrivals, forecast restrictions, prohibited sources, and other
confirmed routing rules remove candidates. The validator rejects a selected row that
is not in the eligible candidate table.

### A8. Unmet demand is allowed and penalized

Every requested case is either fulfilled or unfulfilled. An unassigned order has
zero fulfillment and incurs its full unmet-demand penalty.

### A9. Minimum divert improvement is five percentage points

If enabled, a non-default assignment must fill at least

\[
\min\{Q_o,F_o^{\mathrm{def}}+\lceil0.05Q_o\rceil\}
\]

cases. The five percent is based on total ordered cases, not default fill. The
reference `default_fillable_cases` is required when enforcement is enabled.

### A10. Operational capacity has verified units

Enabled resource names are `dock`, `throughput_cases`, `case_pick`, `pallet_pick`,
`weight`, and `volume`. Dock use is fixed per candidate. Throughput is fulfilled
cases. Weight and volume use per-case coefficients. With `pallet_case` pick mode,
full pallets consume pallet-pick capacity and only the remainder consumes case-pick
capacity.

### A11. Independent validation is authoritative

A recommendation is reportable only after independent feasibility and objective
recomputation. QUBO energy and solver-native objective values are diagnostics.

### A12. Hybrid search cannot degrade the incumbent

The local MILP enforces exact residual resources and the global validator checks the
merged result. A move is accepted only for a strict, feasible objective improvement.

### A13. Remote quantum execution is opt-in

Local samplers are the default. Sending QUBO coefficients to a remote service
requires explicit approval and `allow_remote=true`. Remote sampling does not bypass
classical repair, recourse, or validation.

## Unresolved or dataset-dependent rules

### U1. Load grouping

Orders sharing a load may require joint routing, mutual exclusion, or only reporting.
No such relationship is imposed until the business rule is confirmed. Candidate
`dock_units` can represent a verified load-level consumption without guessing.

### U2. Customer-specific partial fulfillment

The current model permits partial fulfillment for every order. If some customer or
SKU classes require all-or-nothing service, preprocessing must add a documented
class and the MILP must add the corresponding equality constraints.

### U3. Protection-horizon construction

The solver enforces supplied projected ATP exactly, but upstream logic that combines
receipts, outgoing plans, reservations, and the five-day protection horizon belongs
to source-data preparation and must be confirmed for the original input pack.

### U4. Commercial scale and rounding

Currency, exchange-rate date, price basis, tax/fuel components, and penalty-rate
rounding must be declared in `metadata.json` for a real run.

### U5. Service-date policy

The candidate table may include PGI, arrival, and requested delivery dates. Exact
working-day calendars and exception policy must be supplied by the data owner.

## Change checklist

When resolving an item, update:

1. this document and `ASSUMPTION_VERSION`;
2. the canonical data dictionary;
3. loader and validator logic;
4. exact MILP and both baselines;
5. local preview/recourse behavior; and
6. at least one positive and one negative test.
