# Mathematical formulation

## 1. Scope

The deterministic model assigns each focus order to one eligible distribution-center
(DC) and planned-goods-issue (PGI) option, or explicitly leaves it unassigned. An
assigned order may be partially fulfilled, but its SKU lines cannot split across DCs.
Orders belonging to the same source load choose one common DC/PGI option.

| Symbol | Meaning |
|---|---|
| $\mathcal O$ | focus orders, index $o$ |
| $\mathcal S_o$ | SKUs in order $o$, index $s$ |
| $\mathcal C_o$ | eligible assignment columns for order $o$, index $c$ |
| $d(c),t(c)$ | DC and PGI date of candidate $c$ |
| $\mathcal G$ | assignment groups or loads, index $g$ |
| $\mathcal K_g$ | common DC/PGI options exposed to every member of group $g$ |

Candidate preprocessing removes invalid lanes, missing alternative forecasts, closed
dates, late options, and exhausted alternate dock capacity. The no-assignment outcome
is created by the model and is not a fake candidate row.

## 2. Parameters

### Demand and economics

- $Q_{os}$: requested integer cases;
- $v_{os}$: value per fulfilled case;
- $\pi_{os}$: variable penalty coefficient per unfulfilled case;
- $c_c$: total shipping cost charged when candidate $c$ is selected;
- $\theta_o$: order fill threshold;
- $F_o$: fixed active penalty;
- $K_o$: active penalty per SKU with any cut;
- $m_o,M_o$: optional positive minimum and maximum active penalty.

The fill needed to avoid the order-level penalty is

$$
H_o=\min\left(Q_o,\left\lceil\theta_oQ_o\right\rceil\right),
\qquad Q_o=\sum_{s\in\mathcal S_o}Q_{os}.
$$

### Protected inventory

$I_{dst}$ is protected projected available-to-promise (ATP) for DC $d$, SKU
$s$, and checkpoint $t$. Under the POC adapter it is the minimum nonnegative
source availability over the candidate PGI and the following five calendar days.
Projected ATP may decline across dates.

### Resources

$R^r_{dt}$ is an available amount of resource $r$ at a DC/date. Fixed
candidate use is $\beta^r_c$, and case-dependent use is
$\alpha^r_{osc}$. Supported resources are dock, throughput cases, full-pallet
picks, loose-case picks, weight, and volume. In the supplied data only documented
remaining dock capacity is a source-fact hard limit; pick/throughput limits are
optional labeled scenarios.

### Minimum diversion improvement

Let $F_o^{\mathrm{def}}$ be the default candidate's protected-ATP fill preview.
The minimum non-default fill is

$$
L_o^{\mathrm{div}}=
\min\Bigl\{Q_o,
F_o^{\mathrm{def}}+
\max\left(\bigl\lceil\delta_o \cdot Q_o\bigr\rceil,B_o\right)
\Bigr\},
$$

where the POC uses $\delta_o=0.05$ and $B_o=100$ cases. Thus diversion must
improve by both five percentage points and 100 cases, unless demand itself caps the
requirement.

## 3. Decision variables

For candidate $c\in\mathcal C_o$:

$$
x_c\in\{0,1\}
$$

selects that order-DC-PGI assignment. For every order:

$$
z_o\in\{0,1\}
$$

selects the unassigned outcome. Fulfillment and unmet quantities are

$$
f_{osc}\in\mathbb Z_{\ge0},\qquad
u_{os}\in\mathbb Z_{\ge0}.
$$

When exact pick accounting is active, $p_{osc}$ and $k_{osc}$ are full pallets
and loose cases. Thresholded penalty linearization adds:

- $a_o\in\{0,1\}$: penalty-active indicator;
- $q_{os}$: unmet cases counted while active;
- $h_{os}\in\{0,1\}$: SKU-cut indicator while active; and
- continuous/binary auxiliaries for optional floor and cap.

## 4. Objective

The independently recomputed business objective is

$$
\boxed{
\max J=
\sum_{o,s,c\in\mathcal C_o}v_{os}f_{osc}
-\sum_o P_o
-\sum_c c_cx_c
}.
$$

For the POC thresholded penalty:

$$
a_o=\mathbf 1\left[
\sum_{s,c}f_{osc}<H_o
\right],
$$

$$
R_o=a_oF_o+
\sum_s\left(\pi_{os}q_{os}+K_oh_{os}\right),
$$

$$
P_o=
\begin{cases}
0,&a_o=0,\\
\min\left(\max(R_o,m_o),M_o\right),&a_o=1\text{ and }M_o>0,\\
\max(R_o,m_o),&a_o=1\text{ and }M_o=0.
\end{cases}
$$

The MILP uses standard big-M product, maximum, and minimum linearizations. Bounds
come from each order's total demand and penalty parameters, avoiding an arbitrary
global constant. Synthetic instances may instead use the simpler linear-unmet mode
$P_o=\sum_s\pi_{os}u_{os}$.

## 5. Core constraints

### One outcome per order

$$
\boxed{
\sum_{c\in\mathcal C_o}x_c+z_o=1
\qquad\forall o.
}
$$

### Demand balance

$$
\boxed{
\sum_{c\in\mathcal C_o}f_{osc}+u_{os}=Q_{os}
\qquad\forall o,s\in\mathcal S_o.
}
$$

### Assignment linking

$$
\boxed{
0\le f_{osc}\le Q_{os}x_c
\qquad\forall o,s,c\in\mathcal C_o.
}
$$

### Assignment-group cohesion

For group $g$, leader $\ell(g)$, member $o$, and shared option
$k\in\mathcal K_g$:

$$
\boxed{x_{ok}=x_{\ell(g)k},\qquad z_o=z_{\ell(g)}.}
$$

Candidate generation first guarantees that every member exposes the same option set.
Shipping and dock coefficients are nonzero only on the deterministic leader.

### Projected-ATP protection

For every DC-SKU checkpoint:

$$
\boxed{
\sum_{\substack{o,c:\ d(c)=d,\ t(c)\le t}}
f_{osc}
\le I_{dst}
\qquad\forall d,s,t.
}
$$

An earlier shipment consumes all later checkpoints. No monotonicity assumption is
placed on projected ATP.

### Operational capacity

$$
\boxed{
\sum_{c:\ d(c)=d,t(c)=t}\beta^r_cx_c
+\sum_{o,s,c:\ d(c)=d,t(c)=t}\alpha^r_{osc}f_{osc}
\le R^r_{dt}
\qquad\forall d,t,r.
}
$$

### Diversion threshold

For a non-default candidate $c\in\mathcal C_o$:

$$
\boxed{
\sum_{s\in\mathcal S_o}f_{osc}
\ge L_o^{\mathrm{div}}x_c.
}
$$

### Exact pallet and loose-case accounting

For $P_s$ cases per pallet:

$$
\boxed{f_{osc}=P_sp_{osc}+k_{osc}},
\qquad
\boxed{0\le k_{osc}\le(P_s-1)x_c}.
$$

Full-pallet capacity consumes $p$; loose-case capacity consumes $k$.

## 6. Classical methods

The **default baseline** restricts each decision unit to its default option and
allocates shared resources deterministically. The **greedy baseline** previews
eligible options against current residual resources, commits the best incremental
objective, and updates resources before the next unit. The **exact MILP** is submitted
through SciPy's `milp` interface to HiGHS and returns an incumbent, upper bound,
optimality gap, model size, node count, and runtime.

A zero gap proves optimality. A time-limited solve remains useful when it returns a
feasible incumbent and valid bound.

## 7. Hybrid relationship

The hybrid QUBO selects previewed assignment plans for a bounded active neighborhood.
Its quadratic resource terms are a ranking surrogate, not the final feasibility
model. Every repaired sample is fixed in this exact MILP, which reoptimizes quantities
against residual global resources. The merged solution is accepted only after the
independent validator and objective evaluator pass.

## 8. Independent validation

`validation.py` does not trust solver constraints. It recomputes:

- one outcome per order;
- demand identity and integer/nonnegative quantities;
- assignment-group cohesion;
- candidate eligibility and selected date/DC consistency;
- every protected-ATP checkpoint;
- every enabled capacity;
- minimum diversion fill; and
- the complete business objective.

Only validator-passed solutions enter comparisons, reports, or planner artifacts.
