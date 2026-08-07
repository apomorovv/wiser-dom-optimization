# Exact and sampler-assisted large-neighborhood algorithms

## 1. Design objective

The solvers search coordinated order reassignments without putting the entire
Distributed Order Management (DOM) model on a quantum device. DOM decides which
distribution center (DC) and planned-goods-issue (PGI) date should serve each order,
and how many cases of each stock-keeping unit (SKU) can be fulfilled.

There are two deliberately separate local-search paths:

1. `exact_lns`, the classical assignment-search escalation, solves assignment and
   fulfillment decisions jointly in one bounded local MILP; and
2. `hybrid`, the experimental comparison path, uses a bounded QUBO sampler to propose
   assignments and an exact fixed-assignment MILP to evaluate their fulfillment.

The distinction matters. The QUBO is an intentionally approximate assignment-ranking
model. Exact LNS does not require a sampler, repair, or top-$k$ recourse loop; it
optimizes the detailed local model directly.

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

### Step 2: feasible classical incumbent and attribution baseline

The default or sequential greedy method constructs a globally feasible starting
solution. The greedy method updates residual inventory and capacity after each
decision, so it does not combine independently attractive but mutually infeasible
choices.

`polished_greedy` then fixes that complete routing policy and lets the detailed MILP
reoptimize fulfillment quantities and thresholded penalties. This is an independently
reported classical baseline, not a search move. If the polish fails, is infeasible, or
degrades the independently evaluated objective, the feasible raw greedy incumbent is
retained.

Both local-search paths can start from the same polished incumbent. Their metadata
keeps attribution explicit:

- `raw_initial_objective` is the objective before fixed-assignment quantity polish;
- `initial_polish_improvement` is due only to that classical quantity reoptimization;
- `search_improvement` for exact LNS, or `improvement` for sampler-assisted LNS, begins
  at the polished objective; and
- `total_improvement` includes polish and later search.

Consequently, a sampler is credited only for improvement after the fixed-assignment
polish. A quantity change found by that polish is never described as a quantum or
sampler-generated gain. In exported metrics, `lns_improvement` isolates the exact-LNS
search, `hybrid_improvement` isolates sampler-assisted search, and
`total_search_improvement`/`total_hybrid_improvement` include the classical polish.

### Step 3: precomputed conflict graph and bounded neighborhoods

Large-neighborhood search (LNS) changes a subset of orders while freezing the rest.
Assignment groups become neighbors when any eligible candidate can consume a common
inventory bucket or exact DC/date capacity resource. The solver builds this static
group-level graph once per search run and reuses it across iterations. It also caches
group membership, eligible-choice counts, and an estimate of local fulfillment-variable
count. Every neighborhood contains whole groups, so a source load is never cut in half.

Conflict-focused selection prioritizes:

- approximate recoverable business loss from unfulfilled cases;
- number of resource-conflicting neighbors;
- assignment-group size; and
- a deterministic identifier tie-break.

The exact-LNS selector expands from a high-priority seed through this graph. At the
configured diversification interval it instead uses a seeded random group order. The
target group count grows after a sufficiently fast solve and shrinks after a timeout,
large residual gap, or other expensive solve. Configured limits on groups, member
orders, and estimated local fulfillment variables keep the detailed local MILP bounded.
Because an assignment group is indivisible, the selector keeps at least one complete
group even if that group alone is wider than an order or variable estimate.

The sampler batching ablation similarly replaces conflict selection with seeded random
group selection while holding all other settings constant.

The sampler-assisted neighborhood additionally stops at `neighborhood_orders` and
`max_qubo_variables`. Candidate-column reduction retains the incumbent option and the
best isolated alternatives up to `max_candidates_per_order`, always with an unassigned
plan. An optional heuristic Pareto pass removes an alternate when another group option
has no worse isolated estimated fill/value/cost/lead time and is strictly better in at
least one dimension. Because alternatives can consume different resource buckets, this
pass is not globally lossless and is disabled by default. Candidate caps and bounded
neighborhoods—not pruning—provide the hard QUBO-width guarantee for the comparator.

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

### Step 5: production joint local solve

For `exact_lns`, the residualized neighborhood is passed to `solve_classical` without
fixed assignments. The local MILP therefore chooses, in one solve:

- a candidate DC/date option or unassigned outcome for every active whole group;
- case-level fulfillment and unmet quantity;
- pallet/case decomposition and enabled threshold decisions; and
- all inventory and capacity allocations within the residual limits.

The incumbent's active slice is evaluated against the same residualized problem. A
non-improving local result is rejected immediately. A locally improving result is
merged with the frozen incumbent, then the independent full-problem evaluator and
validator recompute the global objective and every hard constraint. Only a strictly
improving feasible merged solution is accepted. A solver status, local bound, QUBO
energy, or apparent local improvement alone can never replace that acceptance test.

Each local solve has an explicit time limit and relative-gap target. The search records
local size, runtime, gap, objective delta, assignment change, strategy, and the next
adaptive target size in its iteration history. This makes bounded runtime and any
unproven local optimum visible rather than presenting LNS as a global exact solve.

### Step 6: experimental assignment-plan QUBO

For the sampler-assisted comparator, active assignment group $g$ has one QUBO binary
variable $y_{gk}$ for
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

### Step 7: sampler execution and initialization

The active part of the incumbent supplies a valid one-hot bitstring to backends that
accept a classical initial sample, including simulated annealing. The current
`qaoa_statevector` and `ibm-qpu` implementations do **not** warm-start QAOA from that
incumbent and do not transfer previously optimized parameters. They initialize every
group in the uniform feasible weight-one Dicke/W state, use ring-XY mixing, and optimize
their variational angles from configured restarts. Incumbent-amplitude or
parameter-transfer warm starts are future controlled experiments, not current solver
features.

Available backends are:

| Backend | Purpose | External execution |
|---|---|---:|
| `exact` | Ground-truth QUBO check for at most 24 variables by default | no |
| `exact_feasible` | Exact product-of-groups check without invalid bitstrings | no |
| `random` | Weak reproducible control | no |
| `simulated_annealing` | Scalable local quantum-inspired benchmark | no |
| `qaoa_statevector` | Gate-model simulation with product W/Dicke(1) states and XY mixers | no |
| `ibm-qpu` | The same constraint-preserving QAOA circuit on IBM hardware | yes, approval required |

The QPU adapter uses integer labels, but coefficients may still reveal commercial or
constraint information. Remote calls therefore require `allow_remote=True`.

### Step 8: repair and exact recourse for the sampler comparator

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

Only an improving feasible result replaces the incumbent. This top-$k$ fixed-assignment
recourse loop belongs only to `hybrid`; `exact_lns` uses the single joint free-assignment
solve described in Step 5.

## 4. Quantum execution choices

The experimental bounded assignment problem is naturally binary and quadratic after
plan generation. Gate-model QAOA needs parameter optimization and compilation, but
the assignment groups expose useful structure: one
weight-one Dicke (W) state and one XY ring mixer per group keep every ideal state
one-hot. The local statevector implementation tests this quantum component without a
subscription; the optional IBM adapter executes that same circuit on hardware. It does
not replace the classical production path.

This is an engineering choice, not evidence that QAOA outperforms classical search. A
fair study must compare samplers on identical QUBOs and include all preprocessing,
queue, sampling, repair, and recourse time.

### IBM stress matrix and selection rule

The opt-in study deliberately uses a four-group coupled synthetic control whose
polished-greedy assignment is known to be suboptimal. Circuit width is derived from the
actual retained QUBO instead of hardcoded. This avoids paying
for hardware runs on a random instance where no assignment improvement is possible,
and prevents source coefficients or identifiers from leaving the approved environment.

Before submission, authenticated discovery records every accessible operational IBM
backend wide enough for that QUBO and calls IBM Runtime's `least_busy` selector. A named
backend remains available for a controlled rerun, but the queue snapshot distinguishes
the least-busy recommendation from the device actually selected for the study. The
presentation profile holds device, 512-shot budget, angle seed, and transpiler seed
fixed while repeating hardware execution three times across all six variants:

| Layers | Runtime options | Purpose |
|---:|---|---|
| $p=1$ | baseline | Unmitigated hardware reference |
| $p=1$ | dynamical decoupling | Isolate idle-error suppression |
| $p=1$ | dynamical decoupling plus measurement twirling | Test the combined readout strategy |
| $p=2$ | baseline | Deeper unmitigated reference |
| $p=2$ | dynamical decoupling | Isolate depth-dependent idle-error suppression |
| $p=2$ | dynamical decoupling plus measurement twirling | Test extra ansatz depth against added hardware error |

Each variant uses eight seeded transpiler trials; the circuit with the fewest two-qubit
gates, then depth and size, is submitted. The evidence table records backend queue and
job timestamps, package versions, logical and mapped physical qubits, available
calibration timestamp, transpiled depth and two-qubit cost, raw one-hot rate, exact
feasible-QUBO raw hit rate, best feasible normalized gap, Runtime queue/execution/
quantum time, compilation and decode phases, repaired assignment gain, recourse time,
and total solver time. The best-observed successful variant is ranked by median exact
feasible-QUBO hit rate, raw feasibility, and validated gain, with quantum usage and
end-to-end runtime used only as tie-breakers. Failures remain in the denominator and
are resumable. No mitigation strategy is assumed to win in advance.

## 5. Scaling

Exact LNS bounds each solve by a target number of whole assignment groups, a maximum
number of member orders, an estimated maximum number of fulfillment variables, a time
limit, and a relative MIP-gap target. The resource-conflict graph and group-size
estimates are built once per run. Adaptation changes the target group count inside the
configured minimum and maximum. It does not add another group when that addition would
exceed the order or fulfillment-variable limits; one indivisible group remains the
minimum search unit. Total problem size can therefore grow without forcing a
monolithic solve, though the number of neighborhoods needed for a given solution
quality can also grow.

For the sampler comparator, let $B$ be active assignment groups and $K_g$ common plans
for group $g$. Its local logical-variable count is

$$
n_{\mathrm{QUBO}}=\sum_{g\in B}(|K_g|+1),
$$

where one is the unassigned plan and $|K_g|$ is capped before QUBO construction.
Dense pair construction is
$O(n_{\mathrm{QUBO}}^2)$, although zero couplings are omitted by remote adapters.
`max_qubo_variables` makes hardware demand independent of the total number of orders.

Global exact MILP size grows approximately with candidate assignments plus
order-line-candidate fulfillment variables. Exact LNS limits that model to an active
neighborhood and leaves its assignments free; sampler recourse uses smaller
fixed-assignment local models but may solve several of them per neighborhood. Future
scaling experiments include:

1. use dual prices from a relaxed MILP to select neighborhoods and price resources;
2. compress repeated per-order assignment-group variables into group-option variables;
3. sparsify projected-ATP checkpoints or use chained cumulative-consumption rows;
4. run independent neighborhoods in parallel against versioned incumbents; and
5. benchmark integer-scaled CP-SAT and current direct HiGHS interfaces against the
   same validated instances.

Physical qubit count is not the same as logical QUBO capacity. Circuit connectivity,
routing overhead, two-qubit gate count/depth, calibration drift, and noise determine
the usable problem size on a gate-model QPU.

## 6. Noise study

`qubo_noise_relative_sigma` applies reproducible symmetric Gaussian perturbations to
QUBO coefficients. This is a coefficient-sensitivity test, not a complete physical
noise model. Samples are still ranked against the original QUBO and evaluated by
exact business recourse. Report performance over multiple seeds and noise levels.
The full profile uses four seeds and four relative-noise levels; do not infer
hardware robustness from a local coefficient perturbation.

The separate `qaoa_readout_bitflip_probability` control applies independent symmetric
bit flips after ideal statevector sampling and before repair. It measures how a simple
measurement channel changes raw one-hot feasibility and the validated result. It is
still not a full gate/decoherence/hardware noise model.

## 7. Limitations

- Exact LNS is bounded and does not prove global optimality; improvements requiring
  simultaneous changes outside every selected neighborhood can be missed.
- The conflict/loss priority and grow/shrink policy are heuristics. Seeded random
  diversification broadens coverage but does not guarantee exploration of every move.
- A local MILP may stop at its time or gap limit. Its candidate still has to pass the
  independent global objective and feasibility checks.
- Preview plans use isolated residual resources; the QUBO represents coupling only
  approximately.
- Linear penalty calibration may need rescaling for a particular hardware range.
- Direct IBM execution requires optional dependencies, credentials, and data approval.
- No QPU benchmark has been executed in this branch, so no quantum advantage or
  business improvement is attributed to quantum hardware.

These limitations are observable in run metadata rather than hidden behind a single
headline objective.
