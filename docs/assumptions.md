# Modeling assumptions

This document records assumptions used by the current implementation. Changing an assumption changes feasibility, objective values, or comparability and therefore requires a new assumption version.

## Status labels

- **Active**: implemented and used in reported experiments.
- **Planned**: accepted design, not yet implemented.
- **Unresolved**: requires confirmation from the challenge data or organizers.
- **Excluded from V0**: intentionally postponed.

## V0 active assumptions

### A1. Focus orders are the optimization population

The optimizer receives a set \(\mathcal O\) of focus orders. Focus-order identification is handled by `src/domopt/focus_orders.py`, not by an optimization variable.

### A2. One DC and one PGI date per assigned order

An assigned order may use at most one candidate pair \((d,t)\). SKU lines cannot be split across different DCs. Partial fulfillment from the selected DC is allowed.

A no-assignment variable keeps every order feasible:

\[
\sum_{(d,t)\in\mathcal C_o}x_{odt}+z_o=1.
\]

### A3. Quantity unit is cases

Demand, fulfillment, unmet demand, and inventory are nonnegative integer cases.

### A4. Monetary terms use one scale

Unit value, unmet-demand penalty, and shipping cost must use one declared currency or synthetic monetary scale before optimization.

### A5. Shipping cost is fixed per selected candidate

For candidate \((o,d,t)\), \(c_{odt}\) is charged once when \(x_{odt}=1\). A dataset must use exactly one convention:

- total shipping cost; or
- incremental shipping cost relative to the default assignment.

The conventions cannot be mixed.

### A6. Inventory is cumulative available-to-promise

\(I_{dst}\) is cumulative cases of SKU \(s\) available at DC \(d\) through date \(t\), after reservations outside the modeled focus-order set.

\[
\sum_{\tau\le t}\sum_o f_{osd\tau}\le I_{dst}.
\]

Inventory consumed at an earlier PGI date therefore remains consumed in later cumulative constraints.

### A7. Hard eligibility is preprocessing

Calendar closure, lead-time feasibility, prohibited sources, SKU forecast eligibility, and other hard routing rules are applied by `src/domopt/candidates.py`. Ineligible candidates are absent from \(\mathcal C\).

The validator still checks that every selected candidate belongs to the approved candidate table.

### A8. Unmet demand is permitted and penalized

\[
\sum_{(d,t)\in\mathcal C_o}f_{osdt}+u_{os}=Q_{os}.
\]

If \(z_o=1\), linking constraints force all \(f_{osdt}=0\), so \(u_{os}=Q_{os}\).

### A9. V0 penalties are linear

\[
\max\left[
\sum v_{os}f_{osdt}
-
\sum \pi_{os}u_{os}
-
\sum c_{odt}x_{odt}
\right].
\]

Threshold-triggered, piecewise, or customer-specific nonlinear penalties are excluded from V0.

### A10. The independent validator is authoritative

A result is reportable only after `src/domopt/validation.py` verifies it and `src/domopt/objective.py` independently recomputes every objective component. Solver-reported objective values are diagnostic only.

## Planned assumptions

### A11. Minimum divert improvement

A non-default candidate may be required to improve case fill relative to a documented default reference quantity \(F_o^{\mathrm{def}}\):

\[
\sum_s f_{osdt}
\ge
\left(F_o^{\mathrm{def}}+\left\lceil\delta_oQ_o\right\rceil\right)x_{odt},
\qquad d\ne d_o^{\mathrm{def}}.
\]

This is activated only after confirming whether “5% improvement” means percentage points, 5% of total demand, or a relative percentage increase.

### A12. Protected alternate-DC inventory

Alternate-DC inventory may need to remain sufficient for default demand over a future protection horizon. This should be reflected in processed available-to-promise inventory, not by an informal penalty.

### A13. Dock, throughput, case-pick, and pallet-pick capacities

Optional resources use the generic capacity formulation in `docs/mathematical_formulation.md`. They are enabled only after units and time buckets are verified.

### A14. Load grouping

Orders sharing a load number may require joint assignment or mutual exclusion. The exact rule is unresolved and must not be guessed.

## Unresolved questions

1. Is shipping cost total or incremental relative to the default lane?
2. Is order value measured per case, weight, volume, or another commercial unit?
3. Is penalty linear per unmet case, percentage-based, or threshold-triggered?
4. Is historical COF improvement based on case fill, revenue fill, or another metric?
5. How are in-transit receipts and incoming load plans incorporated into available inventory?
6. Is the inventory-protection horizon measured in calendar or working days?
7. Does one assigned order always consume one dock appointment?
8. What physical units define case-pick and pallet-pick capacity?
9. Must equal load numbers stay together, or are duplicate load numbers forbidden?
10. Is partial fulfillment operationally acceptable for every customer class?

A resolved answer requires updates to this document, the data dictionary, tests, and the assumption version.
