# Mathematical formulation

## 1. Scope

We model a deterministic Distributed Order Management problem for a set of focus orders. Each order may be assigned to at most one feasible distribution center (DC) and planned-goods-issue (PGI) date. The selected DC may partially fulfill the order. Orders share inventory and may also share dock, throughput, case-pick, pallet-pick, weight, or volume capacity.

The primary formulation is a mixed-integer linear program (MILP). A reduced candidate-column model is derived for QUBO and quantum experiments.

## 2. Index sets

\[
\mathcal O
\]

set of focus orders, indexed by \(o\).

\[
\mathcal S
\]

set of SKUs, indexed by \(s\).

\[
\mathcal D
\]

set of DCs, indexed by \(d\).

\[
\mathcal T
\]

ordered set of planning dates, indexed by \(t\).

\[
\mathcal R
\]

set of optional capacity-resource types, indexed by \(r\).

For order \(o\), define

\[
\mathcal S_o=\{s\in\mathcal S:Q_{os}>0\},
\]

the SKUs requested by order \(o\).

Let

\[
\mathcal C\subseteq\mathcal O\times\mathcal D\times\mathcal T
\]

be the set of feasible order–DC–PGI candidates after preprocessing.

For one order,

\[
\mathcal C_o=\{(d,t)\in\mathcal D\times\mathcal T:(o,d,t)\in\mathcal C\}.
\]

The order-specific default DC is

\[
d_o^{\mathrm{def}}\in\mathcal D.
\]

## 3. Parameters

### 3.1 Demand and economics

\[
Q_{os}\in\mathbb Z_{\ge0}
\]

cases of SKU \(s\) requested by order \(o\). Set \(Q_{os}=0\) for \(s\notin\mathcal S_o\).

\[
v_{os}\in\mathbb R_{\ge0}
\]

business value per fulfilled case.

\[
\pi_{os}\in\mathbb R_{\ge0}
\]

penalty per unfulfilled case.

\[
c_{odt}\in\mathbb R_{\ge0}
\]

fixed shipping cost of assigning order \(o\) to DC \(d\) at PGI date \(t\).

A dataset must use exactly one convention:

- **total-cost convention**: \(c_{odt}\) is the full shipping cost; or
- **incremental-cost convention**: \(c_{odt}\) is cost relative to the default assignment.

Define total requested cases:

\[
Q_o=\sum_{s\in\mathcal S_o}Q_{os}.
\]

### 3.2 Inventory

\[
I_{dst}\in\mathbb Z_{\ge0}
\]

cumulative available-to-promise cases of SKU \(s\) at DC \(d\) through date \(t\), after reservations outside the focus-order decision set.

For fixed \((d,s)\), \(I_{dst}\) is normally nondecreasing in \(t\).

### 3.3 Optional operational capacities

\[
R^r_{dt}\in\mathbb R_{\ge0}
\]

available amount of resource \(r\) at DC \(d\) on date \(t\).

\[
\alpha^r_{osdt}\in\mathbb R_{\ge0}
\]

variable consumption of resource \(r\) per fulfilled case of SKU \(s\) when order \(o\) uses candidate \((d,t)\).

\[
\beta^r_{odt}\in\mathbb R_{\ge0}
\]

fixed consumption of resource \(r\) if candidate \((o,d,t)\) is activated.

Examples:

- dock appointment: \(\beta^{\mathrm{dock}}_{odt}=1\);
- throughput in cases: \(\alpha^{\mathrm{throughput}}_{osdt}=1\);
- weight: \(\alpha^{\mathrm{weight}}_{osdt}=w_s\);
- volume: \(\alpha^{\mathrm{volume}}_{osdt}=\nu_s\).

### 3.4 Optional minimum divert threshold

\[
F_o^{\mathrm{def}}\in\mathbb Z_{\ge0}
\]

reference cases fillable under the documented default policy.

\[
\delta_o\in[0,1]
\]

required minimum fill improvement as a fraction of total order demand.

Define

\[
L_o^{\mathrm{div}}
=
\min\left\{
Q_o,
F_o^{\mathrm{def}}+\left\lceil\delta_oQ_o\right\rceil
\right\}.
\]

This threshold is inactive in V0 until the interpretation of the historical improvement rule is confirmed.

## 4. Decision variables

### 4.1 Assignment

For each \((o,d,t)\in\mathcal C\):

\[
x_{odt}\in\{0,1\},
\]

where \(x_{odt}=1\) means that order \(o\) is assigned to DC \(d\) with PGI date \(t\).

For each \(o\in\mathcal O\):

\[
z_o\in\{0,1\},
\]

where \(z_o=1\) means that order \(o\) is not assigned to any DC.

### 4.2 Fulfillment

For each \(o\in\mathcal O\), \(s\in\mathcal S_o\), and \((d,t)\in\mathcal C_o\):

\[
f_{osdt}\in\mathbb Z_{\ge0},
\]

cases of SKU \(s\) fulfilled for order \(o\) from DC \(d\) on date \(t\).

For each \(o\in\mathcal O\), \(s\in\mathcal S_o\):

\[
u_{os}\in\mathbb Z_{\ge0},
\]

unfulfilled cases of SKU \(s\).

## 5. Objective function

Fulfilled value:

\[
V^{\mathrm{fill}}
=
\sum_{o\in\mathcal O}
\sum_{s\in\mathcal S_o}
\sum_{(d,t)\in\mathcal C_o}
v_{os}f_{osdt}.
\]

Unmet-demand penalty:

\[
C^{\mathrm{pen}}
=
\sum_{o\in\mathcal O}
\sum_{s\in\mathcal S_o}
\pi_{os}u_{os}.
\]

Shipping cost:

\[
C^{\mathrm{ship}}
=
\sum_{(o,d,t)\in\mathcal C}
c_{odt}x_{odt}.
\]

The V0 objective is

\[
\boxed{
\max\quad
V^{\mathrm{fill}}-C^{\mathrm{pen}}-C^{\mathrm{ship}}
}
\]

or, expanded,

\[
\boxed{
\max
\left[
\sum_o\sum_{s\in\mathcal S_o}\sum_{(d,t)\in\mathcal C_o}v_{os}f_{osdt}
-
\sum_o\sum_{s\in\mathcal S_o}\pi_{os}u_{os}
-
\sum_{(o,d,t)\in\mathcal C}c_{odt}x_{odt}
\right].
}
\]

## 6. Core constraints

### 6.1 Exactly one modeled outcome per order

\[
\boxed{
\sum_{(d,t)\in\mathcal C_o}x_{odt}+z_o=1
\qquad\forall o\in\mathcal O.
}
\]

This enforces at most one DC/date assignment while keeping the model feasible through \(z_o\).

### 6.2 Demand balance

\[
\boxed{
\sum_{(d,t)\in\mathcal C_o}f_{osdt}+u_{os}=Q_{os}
\qquad\forall o\in\mathcal O,\ s\in\mathcal S_o.
}
\]

Every requested case is either fulfilled or unfulfilled.

### 6.3 Assignment–fulfillment linking

\[
\boxed{
0\le f_{osdt}\le Q_{os}x_{odt}
}
\]

for every valid \((o,s,d,t)\). If a candidate is not selected, it cannot fulfill a line.

### 6.4 Cumulative inventory

For every DC \(d\), SKU \(s\), and checkpoint \(t\):

\[
\boxed{
\sum_{\tau\in\mathcal T:\tau\le t}
\sum_{\substack{o\in\mathcal O:\(o,d,\tau)\in\mathcal C}}
f_{osd\tau}
\le
I_{dst}.
}
\]

Only terms with \(s\in\mathcal S_o\) exist. This prevents multiple orders from consuming the same stock.

## 7. Optional constraints

### 7.1 Generic DC/date capacity

For each enabled resource \(r\), DC \(d\), and date \(t\):

\[
\boxed{
\sum_{o:(o,d,t)\in\mathcal C}\beta^r_{odt}x_{odt}
+
\sum_{o:(o,d,t)\in\mathcal C}\sum_{s\in\mathcal S_o}\alpha^r_{osdt}f_{osdt}
\le
R^r_{dt}.
}
\]

### 7.2 Minimum fulfillment for a divert

For every non-default candidate \((o,d,t)\in\mathcal C\) with \(d\ne d_o^{\mathrm{def}}\):

\[
\boxed{
\sum_{s\in\mathcal S_o}f_{osdt}
\ge
L_o^{\mathrm{div}}x_{odt}.
}
\]

### 7.3 Complete-order requirement

If assignment requires complete fulfillment, replace partial linking by

\[
\boxed{
f_{osdt}=Q_{os}x_{odt}}
\]

for every valid index. This is not active in V0.

### 7.4 Pallet and case-pick decomposition

Let

\[
P_s\in\mathbb Z_{>0}
\]

be cases per pallet. Introduce full pallets \(p_{osdt}\in\mathbb Z_{\ge0}\) and loose cases \(k_{osdt}\in\mathbb Z_{\ge0}\):

\[
\boxed{f_{osdt}=P_sp_{osdt}+k_{osdt}},
\]

\[
\boxed{0\le k_{osdt}\le(P_s-1)x_{odt}}.
\]

This avoids nonlinear floor and modulo operators.

### 7.5 Candidate eligibility

Hard feasibility is enforced by constructing \(\mathcal C\). If an audit indicator \(e_{odt}\in\{0,1\}\) is retained:

\[
x_{odt}\le e_{odt}.
\]

## 8. Baseline definitions

### 8.1 Default baseline

Fix all non-default assignments to zero:

\[
x_{odt}=0
\qquad\text{for }d\ne d_o^{\mathrm{def}}.
\]

Then allocate shared inventory using a documented deterministic order sequence. The baseline must satisfy the same demand and inventory rules.

### 8.2 Sequential greedy baseline

At each iteration:

1. compute feasible incremental choices using current residual resources;
2. select one candidate by the documented score and tie-break rule;
3. reduce residual inventory and capacity;
4. continue until every order has a final outcome.

The greedy method must not score all orders independently and then accept mutually conflicting choices.

## 9. Candidate-column formulation

For QUBO construction, precompute a finite set \(\mathcal K_o\) of fulfillment plans for each order, including a no-assignment plan.

Plan \(k\in\mathcal K_o\) contains a DC \(d_{ok}\), PGI date \(t_{ok}\), fixed line quantities \(q_{osk}\), resource consumption, and shipping cost \(c_{ok}\).

Its value is

\[
w_{ok}
=
\sum_{s\in\mathcal S_o}v_{os}q_{osk}
-
\sum_{s\in\mathcal S_o}\pi_{os}(Q_{os}-q_{osk})
-
c_{ok}.
\]

Define

\[
y_{ok}\in\{0,1\}.
\]

The restricted column model is

\[
\boxed{
\max\sum_o\sum_{k\in\mathcal K_o}w_{ok}y_{ok}
}
\]

subject to

\[
\boxed{
\sum_{k\in\mathcal K_o}y_{ok}=1
\qquad\forall o,
}
\]

and cumulative inventory

\[
\boxed{
\sum_o\sum_{k\in\mathcal K_o}a_{okdst}y_{ok}
\le I_{dst}
\qquad\forall d,s,t,
}
\]

where \(a_{okdst}\) is cumulative inventory consumed by plan \(k\) through \(t\).

Generic capacity constraints are

\[
\boxed{
\sum_o\sum_{k\in\mathcal K_o}b^r_{okdt}y_{ok}
\le R^r_{dt}.
}
\]

The column model is exact only if \(\mathcal K_o\) contains all plans needed to represent an optimal MILP solution. Otherwise it is a restricted master problem.

## 10. QUBO representation

For minimization, begin with

\[
E(y)
=
-\sum_{o,k}w_{ok}y_{ok}
+
\lambda_{\mathrm{one}}
\sum_o\left(1-\sum_{k\in\mathcal K_o}y_{ok}\right)^2
+
E_{\mathrm{resource}}(y).
\]

Resource inequalities require one of:

1. binary slack-variable encoding;
2. conflict penalties when infeasibility is exactly pairwise;
3. a constrained quadratic model;
4. classical feasibility repair with explicit pre- and post-repair reporting.

Pairwise conflicts are not generally sufficient for arbitrary multi-order inventory constraints.

## 11. Model size

The detailed MILP has

\[
|\mathcal C|
\]

binary assignment variables,

\[
|\mathcal O|
\]

binary no-assignment variables,

\[
\sum_o|\mathcal S_o||\mathcal C_o|
\]

integer fulfillment variables, and

\[
\sum_o|\mathcal S_o|
\]

integer unmet-demand variables.

The candidate-column model has

\[
N_y=\sum_o|\mathcal K_o|
\]

logical binary variables before slack variables.

## 12. Tiny-instance optimum

For the standard synthetic instance,

\[
Q_{O1,A}=4,\quad Q_{O1,B}=2,
\]

\[
Q_{O2,A}=3,\quad Q_{O2,B}=4.
\]

Inventory is

\[
I_{D1,A,t_1}=3,\quad I_{D1,B,t_1}=4,
\]

\[
I_{D2,A,t_1}=4,\quad I_{D2,B,t_1}=2.
\]

All case values are \(10\), all unmet-case penalties are \(20\), and assigning \(O_1\) to \(D_2\) costs \(4\).

The unique full-fulfillment assignment is

\[
x_{O1,D2,t_1}=1,
\qquad
x_{O2,D1,t_1}=1.
\]

It produces

\[
V^{\mathrm{fill}}=130,
\qquad
C^{\mathrm{pen}}=0,
\qquad
C^{\mathrm{ship}}=4,
\]

so

\[
\boxed{\text{Objective}=126.}
\]
