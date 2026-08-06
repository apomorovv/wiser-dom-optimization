# Hybrid algorithm

## 1. Design objective

The hybrid solver searches coordinated order reassignments without putting the
entire Distributed Order Management (DOM) model on a quantum device. DOM decides
which distribution center (DC) and planned-goods-issue (PGI) date should serve each
order, and how many cases of each stock-keeping unit (SKU) can be fulfilled.

The implementation separates two jobs:

1. a bounded QUBO proposes discrete assignment combinations; and
2. an exact MILP determines fulfillment quantities and enforces hard constraints.

QUBO means *quadratic unconstrained binary optimization*: minimize

$$
E(y)=c+y^\mathsf{T}Qy,\qquad y\in\{0,1\}^n.
$$

MILP means *mixed-integer linear program*: a model with linear constraints and a
mixture of integer and continuous variables. Here all case and assignment variables
are integer.

## 2. Acceptance invariant

Let $S^k$ be the feasible incumbent after iteration $k$, and let $J(S)$ be
the independently recomputed business objective. The solver accepts a proposal
$\widehat S$ only when

$$
\widehat S\text{ is feasible}
\quad\land\quad
J(\widehat S)>J(S^k).
$$

Otherwise $S^{k+1}=S^k$. Therefore

$$
J(S^{k+1})\ge J(S^k)
$$

at every iteration. Sampling quality affects search efficiency, not returned-solution
correctness or monotonicity.

## 3. End-to-end workflow

### Step 1: canonical preprocessing

The loader parses identifiers as strings, dates as timestamps, and cases as
nonnegative integers. Candidate generation removes closed, ineligible, and late
DC/date options before optimization. Every method receives the same canonical
orders, order lines, inventory, candidates, capacities, calendar, and metadata.

### Step 2: feasible classical incumbent

The default or sequential greedy method constructs a globally feasible starting
solution. The greedy method updates residual inventory and capacity after each
decision, so it does not combine independently attractive but mutually infeasible
choices.

### Step 3: conflict-aware neighborhood

Large-neighborhood search (LNS) changes a subset of orders while freezing the rest.
Assignment groups become neighbors when their candidate plans can consume a common
inventory bucket or DC/date capacity. Every neighborhood contains whole groups, so a
source load is never cut in half. The conflict strategy prioritizes:

- current unfulfilled cases;
- number of resource-conflicting neighbors;
- number of eligible assignment choices; and
- a deterministic identifier tie-break.

The batching ablation replaces this rule with seeded random group selection while
holding all other settings constant.

The neighborhood stops at both `neighborhood_orders` and `max_qubo_variables`.
Candidate-column reduction retains the incumbent option and the best isolated
alternatives up to `max_candidates_per_order`, always with an unassigned plan. A
separate Pareto pass removes an alternate only when another group option has no worse
estimated fill/value/cost/lead time and is strictly better in at least one dimension;
the default is always retained. Thus both unusually wide groups and the full order
population remain bounded.

### Step 4: exact residualization

Consumption from frozen orders is subtracted from every inventory checkpoint and
capacity bucket. If $A$ is the active order set and $\bar A$ the frozen set,
the local inventory limit is

$$
\widetilde I_{dst}
=I_{dst}
-\sum_{\substack{o\in\bar A,\,\tau\le t}}
f_{osd\tau}^{\mathrm{inc}}.
$$

The same subtraction is applied to exact-date dock, throughput, pick, weight, and
volume resources. This prevents the local solver from reusing stock or capacity
already committed outside the neighborhood.

### Step 5: assignment-plan QUBO

For active assignment group $g$, the QUBO contains one binary variable $y_{gk}$ for
each common group option $k$, plus an explicit unassigned plan. A group option maps
every member order to the corresponding candidate at the same DC/date. It contains
the aggregate preview fulfillment, isolated business value $w_{gk}$, and resource
usage of all members. Group cohesion is therefore structural in the QUBO instead of
being imposed by majority-vote repair after sampling.

Exactly one outcome per order is encouraged by

$$
P_{\mathrm{one}}
\left(1-\sum_{k\in\mathcal K_g}y_{gk}\right)^2.
$$

The plan value is encoded as $-w_{gk}y_{gk}$, because the QUBO is minimized.
The penalty $P_{\mathrm{one}}$ is calibrated above the largest plan benefit plus
incident resource penalties.

For resource bucket $r$, let $a_{gkr}$ be plan usage and $R_r$ its residual
limit. A quadratic surrogate adds a loss-weighted penalty when two plans have a
direct overload or contribute to higher-order contention. Higher-order pressure is

$$
\rho_r=
\max\left(0,
\frac{\sum_g\max_{k\in\mathcal K_g}a_{gkr}-R_r}
{\sum_g\max_{k\in\mathcal K_g}a_{gkr}}
\right).
$$

For plans $(g,k)$ and $(g',k')$, $g\ne g'$, the surrogate resource amount is

$$
h_{gk,g'k',r}
=\max\left\{
0,
a_{gkr}+a_{g'k'r}-R_r,
\rho_r\min(a_{gkr},a_{g'k'r})
\right\}.
$$

It is multiplied by a conservative marginal business-loss estimate and added as a
quadratic coefficient. The complete local energy is

$$
E(y)=
-\sum_{g,k}\widetilde w_{gk}y_{gk}
+P_{\mathrm{one}}\sum_g
\left(1-\sum_k y_{gk}\right)^2
+\sum_{(gk,g'k')}P_{gk,g'k'}y_{gk}y_{g'k'}.
$$

This is a ranking surrogate, not a proof of capacity feasibility. Aggregate resource
constraints can require higher-order terms or slack encodings that would enlarge the
QUBO. Exact recourse deliberately handles that detail.

### Step 6: warm-started sampling

The active part of the incumbent supplies a valid one-hot initial bitstring. Available
backends are:

| Backend | Purpose | External execution |
|---|---|---:|
| `exact` | Ground-truth QUBO check for at most 24 variables by default | no |
| `random` | Weak reproducible control | no |
| `simulated_annealing` | Scalable local quantum-inspired benchmark | no |
| `dwave-qpu` | Direct quantum annealing experiment | yes, approval required |
| `dwave-hybrid` | D-Wave managed hybrid experiment | yes, approval required |

The QPU adapter uses integer labels, but coefficients may still reveal commercial or
constraint information. Remote calls therefore require `allow_remote=True`.

### Step 7: repair and exact recourse

Raw samples may violate one-hot structure. Deterministic repair keeps the selected
highest-value group plan or, if none is selected, chooses the highest-value plan.
Because every variable already represents a whole group, repair cannot split a load.
The unperturbed QUBO ranks unique repaired assignments.

For each of the best `top_k_recourse` assignments, the local MILP fixes assignment
variables and reoptimizes all fulfillment quantities under residual inventory and
capacity. The local solution is merged with frozen orders, then the global validator
checks:

- exactly one assignment or unassigned outcome;
- line-level demand balance;
- candidate eligibility and date consistency;
- every projected-ATP inventory checkpoint;
- enabled dock, throughput, pick, weight, and volume limits; and
- the five-percentage-point and 100-case alternate-fill improvement; and
- assignment-group cohesion.

Only an improving feasible result replaces the incumbent.

## 4. Why annealing is the primary quantum option

The bounded local problem is naturally binary and quadratic after plan generation.
Quantum annealing accepts a binary quadratic model directly, while gate-model QAOA
requires circuit depth, parameter optimization, constraint-preserving mixers or
penalties, and hardware-specific compilation. The repository keeps the local QUBO
backend-neutral, but the implemented remote adapter targets D-Wave because that is
the shortest path from the mathematical object to available hardware.

This is an engineering choice, not evidence that annealing outperforms QAOA or
classical search. A fair study must compare samplers on identical QUBOs and include
all preprocessing, queue, sampling, repair, and recourse time.

## 5. Scaling

Let $B$ be active assignment groups and $K_g$ common plans for group $g$. The local
logical-variable count is

$$
n_{\mathrm{QUBO}}=\sum_{g\in B}(|K_g|+1),
$$

where one is the unassigned plan and $|K_o|$ is capped before QUBO construction.
Dense pair construction is
$O(n_{\mathrm{QUBO}}^2)$, although zero couplings are omitted by remote adapters.
The outer conflict graph can be built once and updated incrementally in a production
implementation. `max_qubo_variables` makes hardware demand independent of the total
number of orders.

The detailed MILP remains the computational safety net. Global exact MILP size grows
approximately with candidate assignments plus order-line-candidate fulfillment
variables. Local recourse fixes assignments and is much smaller. Production scaling
options are:

1. cache resource signatures and candidate previews;
2. use dual prices from a relaxed MILP to select neighborhoods and price resources;
3. generate only promising assignment columns;
4. run independent neighborhoods in parallel against versioned incumbents;
5. adapt neighborhood size to sampler and recourse time; and
6. retain a classical-only fallback for every remote call.

Physical qubit count is not the same as logical QUBO capacity. Minor embedding,
connectivity, coefficient range, chain strength, and analog noise determine the
usable problem size on a QPU.

## 6. Noise study

`qubo_noise_relative_sigma` applies reproducible symmetric Gaussian perturbations to
QUBO coefficients. This is a coefficient-sensitivity test, not a complete physical
noise model. Samples are still ranked against the original QUBO and evaluated by
exact business recourse. Report performance over multiple seeds and noise levels.
The full profile uses four seeds and four relative-noise levels; do not infer
hardware robustness from a local coefficient perturbation.

## 7. Limitations

- Preview plans use isolated residual resources; the QUBO represents coupling only
  approximately.
- Linear penalty calibration may need rescaling for a particular hardware range.
- The current LNS selector is deterministic; adaptive or dual-informed selection may
  find better neighborhoods.
- Direct D-Wave paths require optional dependencies, credentials, and data approval.
- No QPU benchmark has been executed in this branch, so no quantum advantage or
  business improvement is attributed to quantum hardware.

These limitations are observable in run metadata rather than hidden behind a single
headline objective.
