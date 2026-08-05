# Mathematical formulation

## 1. Scope and notation

The deterministic model assigns each focus order to at most one distribution center
(DC) and planned-goods-issue (PGI) date. It may partially fulfill the selected order,
subject to shared inventory and operational capacity.

| Symbol | Meaning |
|---|---|
| \(\mathcal O\) | focus orders, index \(o\) |
| \(\mathcal S\) | stock-keeping units (SKUs), index \(s\) |
| \(\mathcal D\) | distribution centers, index \(d\) |
| \(\mathcal T\) | ordered inventory/capacity dates, index \(t\) |
| \(\mathcal R\) | enabled capacity resources, index \(r\) |
| \(\mathcal S_o\) | SKUs requested by order \(o\) |
| \(\mathcal C_o\) | eligible DC/date candidates for order \(o\) |

Candidate preprocessing creates

\[
\mathcal C\subseteq\mathcal O\times\mathcal D\times\mathcal T.
\]

Closed dates, prohibited sources, impossible lead times, and other hard eligibility
failures are absent from \(\mathcal C\).

## 2. Parameters

### Demand and economics

\[
Q_{os}\in\mathbb Z_{\ge0}
\]

is requested cases, \(v_{os}\ge0\) is value per fulfilled case, and
\(\pi_{os}\ge0\) is penalty per unfulfilled case. The challenge clarification is
implemented as

\[
\pi_{os}=\text{price per case}_{os}\times\text{penalty rate}_{os}
\]

when those two raw fields are available. Canonical data stores the computed
`penalty_per_unfilled_case`.

\[
c_{odt}\ge0
\]

is the **total** shipping cost charged once if candidate \((o,d,t)\) is selected.
It is not treated as an incremental lane-cost difference.

The total cases in an order are

\[
Q_o=\sum_{s\in\mathcal S_o}Q_{os}.
\]

### Inventory

\[
I_{dst}\in\mathbb Z_{\ge0}
\]

is protected projected available-to-promise (ATP) for DC \(d\), SKU \(s\), and
checkpoint \(t\), after demand outside the focus-order population. Projected ATP may
decrease over the horizon. A focus-order fulfillment at \(\tau\) consumes all later
checkpoints \(t\ge\tau\); therefore the smallest future ATP protects later demand.

The optional `cumulative_receipts` policy instead requires a nondecreasing series.
The policy is declared once in metadata and is never inferred silently.

### Capacity

\[
R^r_{dt}\ge0
\]

is the available amount of resource \(r\) at DC \(d\), date \(t\). Supported
resources are `dock`, `throughput_cases`, `case_pick`, `pallet_pick`, `weight`, and
`volume`. Variable use per fulfilled case is \(\alpha^r_{osdt}\), and fixed use when
a candidate is activated is \(\beta^r_{odt}\). A dock candidate normally has
\(\beta^{\mathrm{dock}}_{odt}=1\).

### Minimum alternate fill

Let \(d_o^{\mathrm{def}}\) be the default DC and
\(F_o^{\mathrm{def}}\) the cases fillable under the documented default reference.
The clarified five-percentage-point improvement is

\[
L_o^{\mathrm{div}}
=\min\left\{
Q_o,
F_o^{\mathrm{def}}+\left\lceil0.05Q_o\right\rceil
\right\}.
\]

An order-specific fraction \(\delta_o\) may replace 0.05, but the fraction always
applies to total ordered cases, not to default fill.

## 3. Decision variables

For eligible candidate \((o,d,t)\):

\[
x_{odt}\in\{0,1\}
\]

equals one when the candidate is selected. For every order:

\[
z_o\in\{0,1\}
\]

equals one when no DC is assigned.

For every order line and eligible candidate:

\[
f_{osdt}\in\mathbb Z_{\ge0}
\]

is fulfilled cases, while

\[
u_{os}\in\mathbb Z_{\ge0}
\]

is unfulfilled cases.

If pallet/case-pick resources are enabled, \(p_{osdt}\) and \(k_{osdt}\) are full
pallets and loose cases.

## 4. Objective

The common business objective is

\[
\boxed{
\max J=
\sum_{o,s,(d,t)\in\mathcal C_o}v_{os}f_{osdt}
-\sum_{o,s}\pi_{os}u_{os}
-\sum_{o,d,t}c_{odt}x_{odt}
}.
\]

Every method is compared using this independently recomputed objective. Solver or
QUBO energies are diagnostics, not substitute business metrics.

## 5. Core constraints

### One outcome per order

\[
\boxed{
\sum_{(d,t)\in\mathcal C_o}x_{odt}+z_o=1
\qquad\forall o.
}
\]

This enforces at most one DC/date while preserving feasibility through the explicit
unassigned outcome.

### Demand balance

\[
\boxed{
\sum_{(d,t)\in\mathcal C_o}f_{osdt}+u_{os}=Q_{os}
\qquad\forall o,s\in\mathcal S_o.
}
\]

### Assignment linking

\[
\boxed{
0\le f_{osdt}\le Q_{os}x_{odt}
\qquad\forall o,s,d,t.
}
\]

Partial fulfillment is allowed, but SKU lines cannot split across DCs because only
one candidate is selected for the order.

### Projected-ATP protection

For each inventory checkpoint:

\[
\boxed{
\sum_{\substack{o,\tau\le t:\\(o,d,\tau)\in\mathcal C_o}}
f_{osd\tau}
\le I_{dst}
\qquad\forall d,s,t.
}
\]

No monotonicity of \(I_{dst}\) is required under the `projected_atp` policy.

### Operational capacity

\[
\boxed{
\sum_{o:(o,d,t)\in\mathcal C}
\beta^r_{odt}x_{odt}
+
\sum_{o,s:(o,d,t)\in\mathcal C}
\alpha^r_{osdt}f_{osdt}
\le R^r_{dt}
\qquad\forall d,t,r.
}
\]

Dock use is candidate-fixed. Throughput, weight, and volume are linear in fulfilled
cases. Pick resources use the exact decomposition below.

### Minimum divert improvement

For each candidate whose DC differs from the order's default DC:

\[
\boxed{
\sum_{s\in\mathcal S_o}f_{osdt}
\ge L_o^{\mathrm{div}}x_{odt}.
}
\]

This rule is enabled only when `default_fillable_cases` is supplied and metadata
sets `enforce_min_divert_improvement` to true. Missing reference data causes a load
error instead of silently disabling an asserted rule.

### Pallet and loose-case picks

For cases per pallet \(P_s\in\mathbb Z_{>0}\):

\[
\boxed{f_{osdt}=P_sp_{osdt}+k_{osdt}},
\]

\[
\boxed{0\le k_{osdt}\le(P_s-1)x_{odt}}.
\]

Pallet-pick capacity consumes \(p\); case-pick capacity consumes \(k\). This exact
linear representation avoids applying floor or modulo after optimization.

## 6. Classical methods

The default baseline restricts assignment to the default DC and allocates shared
resources in deterministic order. The greedy baseline evaluates candidate previews
against current residual resources and commits the best incremental feasible choice.

The exact model is solved by SciPy's `milp` interface to HiGHS. It reports incumbent,
dual bound, native optimality gap, variable count, constraint count, and runtime.
When assignment variables are fixed, the same model is an exact fulfillment-recourse
problem for the hybrid solver.

## 7. Relationship to the local QUBO

The QUBO variable \(y_{ok}\) selects one previewed assignment plan \(k\) for active
order \(o\). One-hot and pairwise resource-contention penalties rank combinations.
It does **not** replace the inventory or capacity equations above. After sampling:

1. one-hot structure is repaired deterministically;
2. selected assignments are fixed in the local MILP;
3. all fulfillment quantities are reoptimized;
4. the result is merged with frozen orders; and
5. the complete solution is independently validated.

See [hybrid algorithm](hybrid_algorithm.md) for the QUBO equations and acceptance
invariant.

## 8. Feasibility and reporting

`validation.py` is independent of solver constraints and recomputes assignment,
demand, eligibility, inventory, capacity, and alternate-fill rules from output
tables. `objective.py` independently recomputes fulfilled value, penalty cost, and
shipping cost. A solution is reportable only when validation returns no violations.
