# A feasibility-preserving quantum–classical optimizer for Distributed Order Management

## Executive summary

Distributed Order Management (DOM) decides which distribution center (DC) and ship
date should serve each customer order. When the default DC cannot fully serve an
order, diversion may improve fulfillment and avoid penalties, but it can increase
shipping cost or consume inventory and operational headroom needed elsewhere. These
interactions make DOM a coupled combinatorial optimization problem rather than a set
of independent order rankings.

This submission implements the full workflow on the readable Nestlé proof-of-concept
(POC) data. A strict gate opens all ten supplied artifacts and stops on the first
unreadable or structurally invalid file. The source adapter derives cases, order
economics, thresholded penalties, protected inventory, shipping candidates, load
groups, and dock resources. The supplied outputs are used only for reconciliation,
never as optimization labels. Restricted rows, identifiers, DC details, and exact
commercial totals are excluded from the repository.

The common deterministic model is a mixed-integer linear program (MILP). It selects
one eligible DC/planned-goods-issue (PGI) option per order—or an explicit unassigned
outcome—and chooses integer SKU quantities. It enforces demand balance, one-DC
fulfillment, assignment-group cohesion, projected available-to-promise (ATP)
inventory, dock and optional scenario resources, exact pallet/loose-case accounting,
and the POC diversion-uplift rule. Its objective is fulfilled value minus thresholded
unmet penalty minus total shipping cost.

Two transparent baselines provide fair controls. The default baseline keeps loads at
their original option. The sequential greedy baseline previews alternatives against
current residual resources, commits the best incremental choice, and updates the
remaining state. SciPy/HiGHS solves the exact model on tractable subsets and reports a
mathematical bound and optimality gap.

The hybrid method is a feasibility-preserving large-neighborhood search (LNS). It
freezes most orders, selects a bounded set of resource-conflicting loads, and builds a
one-hot quadratic unconstrained binary optimization (QUBO) model over assignment
plans. Exact enumeration, random sampling, simulated annealing, or an approved D-Wave
backend can propose combinations. Deterministic repair restores one common option per
load. Exact local MILP recourse rebuilds SKU quantities using residual resources, and
an independent global validator accepts only a strict feasible improvement. The QUBO
is therefore a proposal mechanism, not the feasibility authority.

The notebook implements the required comparison and all proposed experiments: size
scaling, penalty sensitivity, candidate sensitivity, inventory shocks, seed and local
coefficient-noise sensitivity, Pareto-pruning ablation, and random-versus-conflict
batching. It adds random-versus-annealing sampling and a synthetic coordination
control. The executed smoke profile produced 37 aggregate runs, all feasible, with no
hybrid regression from its configured incumbent. On the smallest real subset the
exact MILP proved a zero gap, and greedy and hybrid matched it. On a larger subset the
safe smoke hybrid improved its default start but did not reach greedy/exact. This is
honest evidence for architectural safety, not quantum advantage. No QPU run or
quantum-advantage claim is made.

## 1. Business problem and decision value

An order normally ships from its default DC. If the default cannot fully serve its
SKUs, a planner may move the load to an alternate DC. A better fill can preserve
sales and customer service, but the decision has network consequences:

- an alternate lane may cost more;
- stock used now may be protected for future demand;
- an alternate shipment may require scarce dock headroom;
- a source load may contain multiple orders that must stay together;
- a partial fill can trigger an order-level shortage penalty; and
- a candidate that looks good alone may conflict with other recommended candidates.

If each of $O$ decision units exposes $K$ options, the assignment space is
approximately $K^O$. Each assignment then induces integer fulfillment quantities
for its SKU lines. Inventory couples orders sharing a DC and SKU, while dock or pick
resources couple otherwise unrelated SKUs. Load cohesion further links orders that
would be independent in a simple assignment problem.

The planner needs more than a selected DC. A usable recommendation must explain its
fill change, penalty avoided, shipping change, service date, binding resource, and
net objective effect. It must also preserve a safe fallback and state any assumption
that still needs operational confirmation.

## 2. Data understanding and strict ingestion

### 2.1 Bundle inventory

All supplied files are readable. The challenge bundle contains:

| Source | Size | Use |
|---|---:|---|
| Order-SKU input | 25,193 rows | Demand, default assignment, dates, values, penalty parameters, units, load, priority |
| Capacity-planning input | 377,504 rows | Daily DC-SKU inventory and forecast availability |
| Shipping input | 12,922 rows | Plant-to-destination lane cost and distance |
| Dock input | 480 rows | Date-specific remaining dock availability |
| Throughput input | 530 rows | Observed case/pallet/order utilization |
| Equations DOCX | 110 equation objects | Source rule definitions |
| Example workbook | 561 rows | Worked POC calculation |
| Order output | 1,109 rows | Aggregate and order-level reconciliation only |
| SKU output | 25,193 rows | Line-level reconciliation only |
| Challenge PDF | 5 pages | Tasks, criteria, privacy, and submission contract |

The format gate parses CSVs, opens the workbook, validates the Word OOXML package,
and extracts every PDF page. It raises before model construction if any artifact
fails. This implements the explicit requirement to stop rather than guess from an
unreadable source.

### 2.2 Quantity conversion

The order input is one row per order-SKU in source planning units. Integer case demand
is

$$
Q_{os}=\operatorname{round}(U_{os}/C_{os})
$$

when positive planning units per case $C_{os}$ exist. For pallet planning-unit
rows it is

$$
Q_{os}=\operatorname{round}(U_{os}P_s),
$$

where $P_s$ is cases per pallet. The adapter rejects a result that is not integral
within tolerance. This rule exactly reconciles to the supplied output case demand.
Unit value, weight, and volume are source totals divided by cases.

### 2.3 Thresholded penalty

The equations and output show an order-level penalty, not a penalty charged on every
unmet case regardless of fill. Let $\theta_o$ be fill threshold. Penalty is inactive
once integer filled cases reach $\lceil\theta_oQ_o\rceil$. Below the threshold:

$$
R_o=F_o+
\sum_s\pi_{os}u_{os}+
K_o\sum_s\mathbf 1[u_{os}>0].
$$

The active amount is raised to an optional minimum and clipped to an optional positive
maximum. $\pi_{os}$ is case value times the source shortage factor. The evaluator
reproduces `PenaltyIfNotDiverted` with maximum absolute error below
$4.5\times10^{-7}$, providing a strong equation-level audit.

### 2.4 Focus population and load cohesion

Rows with unavailable default inventory identify focus orders. If one order on a
source load is a focus order, the entire load enters the population so the solver can
preserve routing cohesion. Missing load identifiers become one-order singleton
groups. The source contains 146 multi-order loads, and the supplied recommendation
never splits one load across DC/date. With the default adapter settings, focus
expansion produces 750 orders, 20,869 order-SKU lines, and 372 assignment groups.

### 2.5 Protected ATP

The inventory table satisfies

$$
\text{Available inventory}=\text{Opening stock}-\text{Reserved quantity}.
$$

Negative values are clipped to zero. For candidate PGI $t$, the solver uses the
minimum availability over $t$ through $t+5$ calendar days. This protects already
committed future consumption. An alternative DC must contain every SKU in the load's
inventory/forecast table. Absence at the default DC means zero preview fill, not a
missing default option.

### 2.6 Lanes, dates, and resources

Shipping lead is `ceil(distance / 500)` calendar days. Alternative PGI is requested
delivery date minus lead time, moved backward over weekends and configured holidays.
Lane cost is treated as total option cost. Shipping and dock use are charged once via
a deterministic group leader.

`Dock_Remaining` is clipped at zero and is the only real operational capacity limit
whose remaining amount is explicit. Default loads consume zero incremental dock;
alternate loads consume one. The throughput file reports observed utilization but no
documented maximum. The adapter therefore does not silently convert it into a hard
constraint. A user may request an explicitly labeled headroom scenario.

### 2.7 Candidates and pruning

Before pruning, the default focus universe exposes 2,182 order candidate rows and
47,075 protected-ATP rows. An option is retained only when every member of a load can
use it. Pareto pruning removes an alternate only if another option has no worse
estimated fill and fulfilled value, no greater shipping cost and lead time, and is
strictly better in at least one dimension. Default options are never removed. The
result has 1,307 candidate rows.

## 3. Deterministic mathematical model

Let $x_c$ select candidate $c$, $z_o$ represent no assignment,
$f_{osc}$ be fulfilled cases, and $u_{os}$ be unmet cases. The objective is

$$
\max J=
\sum_{o,s,c}v_{os}f_{osc}
-\sum_oP_o
-\sum_cc_cx_c.
$$

Each order has one modeled outcome:

$$
\sum_{c\in\mathcal C_o}x_c+z_o=1.
$$

Demand and assignment linking are

$$
\sum_{c\in\mathcal C_o}f_{osc}+u_{os}=Q_{os},
\qquad
0\le f_{osc}\le Q_{os}x_c.
$$

For assignment group $g$, every member selects the same shared option as its
leader, or all are unassigned. This is encoded by equality constraints on corresponding
candidate binaries and unassigned binaries.

Protected ATP at every checkpoint is

$$
\sum_{o,c:d(c)=d,t(c)\le t}f_{osc}\le I_{dst}.
$$

For resource $r$, fixed candidate use $\beta^r_c$, variable use
$\alpha^r_{osc}$, and limit $R^r_{dt}$:

$$
\sum_{c:d(c)=d,t(c)=t}\beta^r_cx_c+
\sum_{o,s,c:d(c)=d,t(c)=t}\alpha^r_{osc}f_{osc}
\le R^r_{dt}.
$$

When pallet/pick scenarios are enabled, exact case decomposition is

$$
f_{osc}=P_sp_{osc}+k_{osc},
\qquad 0\le k_{osc}\le(P_s-1)x_c.
$$

A non-default candidate must fill at least

$$
L_o^{\mathrm{div}}=
\min\{Q_o,F_o^{\mathrm{def}}+
\max(\lceil0.05Q_o\rceil,100)\}.
$$

Threshold activation, unmet-active products, cut-SKU indicators, and floor/cap
selection use bounded big-M linearizations. Bounds are order-specific. The exact
implementation and independent objective evaluator agree within numerical tolerance.

SciPy submits this MILP to HiGHS. A zero optimality gap proves the returned solution
is best for that instance. A time-limited solve may return a strong feasible incumbent
and a bound without proof.

## 4. Classical baselines

### 4.1 Default assignment

The default baseline limits each load to its original option. It allocates quantities
against shared protected ATP and capacity in deterministic order and applies the same
penalty equation and validator as every other method. This answers the operational
question: what is achievable without diversion under the modeled resources?

### 4.2 Sequential greedy reassignment

The greedy baseline previews each currently eligible group option, scores its
incremental business objective, commits the best feasible outcome, and immediately
updates residual resources. Ties prefer larger objective, larger fill, lower shipping
cost, default source, then deterministic candidate key.

Greedy is explainable and fast. Its limitation is path dependence: after it commits a
load, it cannot revisit that decision when another load later reveals a better joint
combination. This makes it an important control for large-neighborhood search.

### 4.3 Exact MILP reference

The MILP is the quality reference on tractable subsets and the fulfillment recourse
engine inside the hybrid. Its recorded diagnostics include variable and constraint
counts, node count, incumbent, upper bound, optimality gap, and wall time. Exact and
heuristic rows are evaluated by the same external code.

## 5. Hybrid quantum-classical design

### 5.1 Why not a monolithic QUBO

A direct QUBO would need binary encodings for SKU quantities, inventory and capacity
slack, thresholded penalty activation, floor/cap logic, and load cohesion. It would
have far more logical variables than the assignment problem, broad coefficient
ranges, many penalty-calibration risks, and embedding overhead on sparse hardware.
Current hardware is better suited to a bounded binary selection problem than the full
operational model.

### 5.2 Bounded neighborhood

The hybrid starts from a feasible default or greedy incumbent. It builds a conflict
view from shared DC-SKU inventory and DC-date resources. The conflict strategy selects
complete groups with unmet demand, high conflict degree, and useful alternatives.
The random-batch ablation selects complete groups by seed. Both active orders and
logical variables are capped.

Frozen-order consumption is subtracted from every inventory checkpoint and capacity
bucket. This residualization prevents local recourse from reusing resources already
committed outside the neighborhood.

### 5.3 Assignment-plan QUBO

For active order $o$ and preview plan $k$, binary $y_{ok}$ selects the plan.
The QUBO minimizes

$$
E(y)=
-\sum_{o,k}\widetilde w_{ok}y_{ok}
+P_{\mathrm{one}}\sum_o\left(1-\sum_ky_{ok}\right)^2
+\sum_{(ok,o'k')}P_{ok,o'k'}y_{ok}y_{o'k'}.
$$

The first term rewards isolated business value, the second encourages one outcome per
order, and the third penalizes candidate pairs contributing to shared-resource
contention. A higher-order pressure estimate adds signal when several individually
compatible plans overload a resource collectively. This remains a surrogate; exact
recourse resolves all real constraints.

The incumbent supplies a one-hot warm start. Available local samplers are exact
enumeration for tiny QUBOs, seeded random sampling, and simulated annealing. Optional
D-Wave QPU and managed-hybrid adapters require credentials and explicit data approval.

### 5.4 Repair, recourse, and safety

Raw samples can violate one-hot structure. Repair selects one plan per order and
projects all members of a load onto one common option. The best unique repaired
assignments are fixed in the exact local MILP. The MILP reoptimizes SKU quantities
under residual resources; infeasible proposals are discarded.

The local solution is merged with frozen orders. An independent validator then checks
assignment uniqueness, load cohesion, demand identity, candidate eligibility, date/DC
consistency, protected ATP, enabled capacities, and diversion uplift. The objective is
recomputed. A proposal replaces incumbent $S^k$ only if

$$
\widehat S\text{ is feasible}\quad\land\quad J(\widehat S)>J(S^k).
$$

Therefore $J(S^{k+1})\ge J(S^k)$. A bad sampler can waste a call, but it cannot
degrade the returned recommendation.

## 6. Evaluation design

Every experiment is defined by bundle hash, schema and assumption versions, candidate
set, common objective, method configuration, seed, and commit. Whole-load expansion
means an 8-order request can contain more than eight actual orders; both counts are
reported.

The full profile contains:

1. a default/greedy/exact/hybrid solver comparison;
2. real-data size scaling at requested sizes 8, 20, and 50;
3. penalty scales 0.5, 1.0, and 2.0;
4. candidate limits 1, 2, 4, and 6;
5. inventory reductions 0%, 10%, and 25%;
6. seeds 3, 11, 29, and 47 with QUBO coefficient noise 0%, 1%, and 5%;
7. Pareto pruning off/on;
8. random/conflict batch selection;
9. random/simulated-annealing sampler comparison; and
10. an eight-order synthetic coordination control.

The smoke profile reduces levels, sampler reads, sweeps, and hybrid iterations for
development. All persisted rows contain aggregate metrics only.

Noise perturbation is reproducible symmetric Gaussian variation of local QUBO
coefficients. Samples are ranked on the original model and passed through exact
recourse. This tests coefficient sensitivity; it does not model analog control error,
embedding chains, readout, thermal effects, or gate noise.

## 7. Results and interpretation

### 7.1 Supplied-output audit

The reference output contains 1,109 orders, 25,193 order-SKU rows, 614 named loads,
631 assignment groups after singleton handling, and three diversions. Selected fill
is 2,413,937 of 2,554,440 cases, or 94.4997%. Order/SKU coverage is complete and the
source penalty equation reconciles. These facts validate transformation logic; they
do not establish that the historical recommendation is optimal.

### 7.2 Executed smoke profile

The smoke notebook produced 37 aggregate experiment rows. Every row passed independent
validation and no hybrid row returned an objective below its starting default
incumbent.

For the common four-decision-unit real subset, normalize the default objective to
100. Greedy, exact MILP, and hybrid each reached 136.5. The exact MILP gap was zero.
The hybrid used a maximum of nine logical QUBO variables and accepted one move. This
shows that the workflow can recover and certify the best subset solution, but it does
not show a sampler advantage because greedy also reached it.

For the requested eight-order scaling point, whole-load expansion produced nine
orders in eight groups. Greedy and zero-gap exact MILP agreed. The one-iteration smoke
hybrid improved its default incumbent but remained below them. The result is valuable
because it demonstrates the safety invariant under an insufficient search budget
rather than hiding a negative outcome.

With Pareto pruning, the base subset candidate rows fell from 12 to 6 while the
validated objective remained unchanged. Random and conflict batches tied on the tiny
base, as did random and simulated-annealing sampling; that instance is too small to
differentiate them. The four seed/noise smoke rows all remained feasible, while one
2% perturbation produced a smaller improvement, confirming ranking sensitivity.

In the independent synthetic coordination control, greedy scored −3895.8 synthetic
units, exact MILP proved −3623.4, and hybrid reached −3872.8. Hybrid improved greedy
by 23 units but remained 249.4 below exact. This supports the value of revisiting
coupled assignments while making clear that the current configuration is not
dominant.

### 7.3 Known-optimum unit test

The two-order synthetic test has proven objective 126, with `O1` assigned to `D2`
and `O2` assigned to `D1`. Exact MILP and exact-QUBO hybrid reproduce it from a
default objective of −50. The example is synthetic and serves as a correctness gate.

## 8. Scaling, robustness, and quantum scope

For active batch $B$ with candidate plans $\mathcal K_o$, logical QUBO width is

$$
n_{\mathrm{QUBO}}=\sum_{o\in B}(|\mathcal K_o|+1).
$$

Pair construction is worst-case $O(n_{\mathrm{QUBO}}^2)$. The configured variable
cap bounds the sampler problem even when global order volume grows. Outer candidate
generation, conflict selection, and repeated recourse still grow and are the practical
classical costs.

Global MILP size grows with candidate assignments plus order-line-candidate quantity
arcs; thresholded penalties and exact pick accounting add auxiliaries. Global exact
optimization is therefore a subset reference or time-limited method. Local recourse
fixes assignment binaries and is materially smaller.

Production improvements include cached inventory profiles and plan signatures,
Pareto and candidate-column reduction, dual-price or reduced-cost neighborhood
selection, parallel neighborhoods against versioned incumbents, adaptive batch size,
and a classical-only fallback for every remote call.

Physical qubit count is not logical QUBO capacity. Minor embedding, chain length,
connectivity, coefficient range, anneal schedule, queue time, and post-processing
affect usable scale. A fair QPU study must compare identical QUBOs, warm starts,
repair, recourse, and validation under both equal time and equal quality. It must
report preprocessing, embedding, queue, QPU access, sampling, repair, and recourse
time separately across multiple seeds. The present work does not make that claim.

## 9. Planner copilot and operational use

The Streamlit planner copilot is justified because the evidence spans methods, sizes,
economics, scarcity, sampler settings, and ablations. It presents aggregate charts and
answers bounded questions such as “Did Pareto pruning help?” and “Is this quantum
advantage?” It calls no external language model and rejects columns named like order,
SKU, DC, customer, candidate, or ZIP identifiers.

The copilot is not a routing agent. A private planner export must still show each
load's default and recommended option, fill change, penalty avoided, shipping change,
net change, service date, and binding reason. A planner verifies extraction time,
holiday calendar, load rule, dock semantics, commercial scale, and validator status.
Manual overrides are recorded with their missing business rule.

## 10. Limitations and next steps

The complete holiday calendar is not supplied. Throughput rows are observed
utilization, not a documented maximum. Customer-specific all-or-nothing service is
not modeled because the challenge allows partial fulfillment and no source class
identifies exceptions. Commercial scale and an external QPU data boundary require
organizer confirmation.

The QUBO uses previewed plan value and quadratic contention; exact aggregate
feasibility remains in recourse. Candidate preview may not capture every joint
effect, and coefficient scaling may need hardware tuning. The current conflict
selector does not use MILP dual prices. The full-profile notebook is implemented but
should be rerun in the approved environment for final timing and uncertainty plots.

Recommended next steps are:

1. confirm the plant working-day calendar and throughput-limit equation;
2. run the full profile and retain aggregate evidence with commit and bundle hash;
3. increase hybrid iterations or use greedy initialization on larger subsets;
4. add dual-informed batch selection and adaptive recourse budgets;
5. have planners review private row-level explanations; and
6. run an approved QPU comparison only after tuned local controls are fixed.

## 11. Conclusion

The submission turns the readable POC into a reproducible and independently
validated DOM optimization workflow. Exact MILP retains responsibility for quantities
and hard operational constraints. The QUBO is confined to a bounded, coordinated
assignment search where quantum sampling could be compared fairly. This partition is
scalable, privacy-aware, and robust to weak samples.

The current evidence supports the architecture and its safety invariant, not a claim
that quantum hardware outperforms classical optimization. That distinction is a
strength: planners keep a certified classical reference and fallback, while the
project exposes a controlled place for future hardware experiments.

## References

1. Tomesh et al., “Quantum Local Search with the Quantum Alternating Operator
   Ansatz,” *Quantum* 6, 781 (2022), https://doi.org/10.22331/q-2022-08-22-781.
2. Egger, Mareček, and Woerner, “Warm-starting quantum optimization,” *Quantum* 5,
   479 (2021), https://doi.org/10.22331/q-2021-06-17-479.
3. Yarkoni et al., “Quantum Annealing for Industry Applications: Introduction and
   Review,” https://arxiv.org/abs/2112.07491.
4. D-Wave, “Hybrid Computing,”
   https://docs.dwavequantum.com/en/latest/concepts/hybrid.html.
5. WISER Global Quantum+AI Program, “WISER <> Nestlé Quantum Optimization for
   Distributed Order Management,” challenge brief (2026).

