# WISER Quantum Challenge Submission Report

## Scalable, safeguarded optimization for distributed order management

**Author:** Andrei Pomorov
**Date:** August 2026

## Portal summary

This submission delivers a runnable distributed order management solver that assigns
whole order groups to eligible distribution-center/date options and optimizes integer
fulfillment under cumulative inventory, dock capacity, diversion, and service-penalty
rules. The production hierarchy combines load-atomic greedy construction, exact
fixed-policy recourse, and adaptive exact large-neighborhood search (LNS). A bounded
QUBO/QAOA path is included as a proposal mechanism behind deterministic repair, exact
recourse, strict improvement, and independent validation.

The final evidence contains 516 aggregate experiment rows across 14 families. All 513
returned plans pass validation with zero recorded demand, integrality, inventory, and
capacity residuals; three frozen-routing controls are correctly proven infeasible under
60-70% inventory reductions. On the common 100-group subset, greedy improves objective
capture by 1.86 percentage points and case fill by 1.62 points over default in 0.58
seconds. All scalable methods reach 372 real groups; greedy and exact LNS reach 100,000
generated orders, while the hybrid reaches 20,000 with a 32-variable local QUBO. An
18-job, 8,192-shot IBM study shows that shallow QAOA retains substantially stronger raw
feasible-subspace quality than the deeper circuit. The recommendation is to deploy the
classical hierarchy in shadow mode and retain QAOA as a controlled, measurable local
proposal path.

## 1. Business and technical framing

The planner must decide where and when an order should ship, then determine how many
cases of each SKU can be fulfilled. A useful recommendation must balance service,
unmet-demand penalties, and shipping cost while respecting rules that make naive
order-by-order assignment unsafe:

- orders on one source load move together;
- alternatives must be eligible for every SKU and date;
- cumulative shipment through each checkpoint cannot exceed protected ATP;
- documented dock and enabled operational capacities cannot be exceeded;
- diversion must improve the protected default preview by the required threshold; and
- all fulfilled and unmet case quantities remain nonnegative integers.

The optimized objective is fulfilled merchandise value minus thresholded unmet-demand
penalty minus shipping cost. To keep the submission privacy-safe, real evidence is
reported as **objective capture** - objective divided by total requested merchandise
value - and **case fill**, fulfilled cases divided by requested cases. Raw source values,
commercial totals, and identifiers are excluded.

The decision is not “classical or quantum.” It is how to build a planner-grade system
that is useful at every stage of the evidence. The implementation therefore keeps a
trusted feasible incumbent and treats every more expensive method as an escalation.

## 2. Data understanding and preparation

The adapter consumes five canonical runtime files. It resolves the documented challenge
columns, normalizes dates and identifiers, validates numeric domains, and stops before
model construction if a required file is absent, empty, or unreadable.

| Runtime input | Optimization role | Key checks |
|---|---|---|
| Order and line demand | Requested cases, values, penalties, loads, defaults | Positive demand, complete load/order keys, valid thresholds |
| Candidate shipping alternatives | Eligible DC/date choices, lead times, shipping | Lane/date validity, SKU coverage, diversion and calendar filters |
| Projected ATP | Time-phased DC/SKU availability | Complete keys, numeric quantities, protected cumulative checkpoints |
| Throughput observations | Optional operational-rate scenarios | Explicit enablement; no invented hard limit from an observation |
| Remaining dock capacity | Source-fact fixed capacity | Valid DC/date joins and nonnegative remaining capacity |

Candidate generation removes invalid lanes, late or closed dates, missing DC/SKU
coverage, and exhausted documented capacity before optimization. Orders sharing a load
are collapsed into assignment groups with a common option set. The model also adds an
explicit unassigned outcome; it does not create a fake candidate row.

The full supplied scope contains 750 modeled orders, 372 assignment groups, 20,869 order
lines, and 2,182 unpruned candidate rows. Heuristic Pareto pruning can reduce the
candidate universe, but remains opt-in because isolated dominance is not a global proof
under shared resources. Candidate-count sensitivity shows that the important quality
transition is from one to two retained candidates; larger caps through six do not
change the common-subset frontier.

## 3. Baselines and evaluation contract

The challenge asks for default and greedy baselines before advanced optimization. Both
are implemented as whole-group policies and measured under the same objective and
validator as every other method.

| Method | Mechanism | Purpose |
|---|---|---|
| Default routing | Retain the planned route and allocate feasible quantities | Business baseline |
| Greedy | Rank eligible group options, commit the best feasible increment, update resources | Fast algorithmic baseline |
| Polished greedy | Fix greedy routes and solve exact quantity/penalty recourse | Routine planning candidate |
| Exact LNS | Reopen bounded, conflict-linked groups in a local MILP | Quality escalation |
| Full MILP | Open all selected decisions and report incumbent, bound, and gap | Small-scope comparator |
| Hybrid sampler | Sample a bounded assignment QUBO; repair and solve exact recourse | Research path |

Every returned plan is independently checked for one outcome per order, group cohesion,
demand identity, integrality, candidate eligibility, cumulative ATP, enabled capacity,
diversion improvement, output schema, and objective recomputation. A solver status is
never accepted as sufficient evidence. The final grid has 513 returned plans and all
513 pass; maximum recorded numeric residuals are zero. The three non-plan rows are not
software failures: the exact recourse model proves the frozen nominal routing infeasible
after inventory is reduced by 60%, 65%, and 70%.

## 4. Mathematical formulation

For each eligible candidate $c\in\mathcal C_o$, binary $x_c$ selects its DC/date. Binary
$z_o$ selects the unassigned outcome. Integer $f_{osc}$ and $u_{os}$ represent fulfilled
and unmet cases for order $o$, SKU $s$. The objective is

$$
\max J = \sum_{o,s,c}v_{os}f_{osc}-\sum_oP_o-\sum_cC_cx_c.
$$

The core constraints are

$$
\sum_{c\in\mathcal C_o}x_c+z_o=1,
\qquad
\sum_{c\in\mathcal C_o}f_{osc}+u_{os}=Q_{os},
\qquad
0\le f_{osc}\le Q_{os}x_c.
$$

Group-linking equalities make all members of a load choose the same outcome. Protected
time-phased inventory is cumulative:

$$
\sum_o\sum_{c:d(c)=d,\tau(c)\le t}f_{osc}\le A_{dst}
\qquad\forall d,s,t.
$$

Capacity rows combine fixed candidate use and case-dependent use. A non-default
candidate must meet the documented service improvement over the protected default
preview. Thresholded penalties, floors, caps, pallet picks, and loose-case picks are
linearized exactly when enabled. The full derivation and assumptions are in
`docs/mathematical_formulation.md` and `docs/assumptions.md`.

## 5. Scalable implementation

### 5.1 Classical hierarchy

Greedy construction compiles immutable line and candidate records once, caches group
options, and updates compact residual inventory/capacity state as each group is
committed. Fixed-policy recourse then prunes every unused candidate column and solves
only the quantity and penalty problem for the selected policy.

Exact LNS avoids a dense all-pairs conflict graph. A sparse inverted index maps each
resource to groups that use it. Neighborhoods are discovered lazily, frozen incumbent
consumption is residualized, and the active assignment and quantity variables are
solved jointly. Limits on groups, orders, fulfillment variables, time, and gap make the
search predictable. The best valid incumbent is retained after timeout or failure.

### 5.2 Bounded QUBO and QAOA

The hybrid changes only the local assignment proposal. For retained choices $m_g$ in
each active group,

$$
n_{\mathrm{QUBO}}=\sum_gm_g\le n_{\max}.
$$

The QUBO includes plan value, one-hot penalties, and pairwise conflict surrogates.
Samples are evaluated before recourse, one-hot violations are repaired, duplicate
assignments are removed, and the best candidates receive exact fixed-assignment
recourse. Only a strict, validator-passed improvement may replace the incumbent.

Gate-model QAOA starts each group in a weight-one Dicke/W state and uses a connected XY
path mixer with $m_g-1$ logical edges. Angles are optimized in the exact feasible
subspace and reused across matched repetitions. Exact enumeration and uniform-feasible
sampling define transparent controls.

QAOA is the optimization proposal method. IBM’s Qiskit `backend` term names the
processor or execution target; it is not a solver. The IBM processor receives only an
independently generated 16-variable control.

## 6. Algorithm and result comparison

### 6.1 Common 100-group subset

| Method | Objective capture | Case fill | Reassigned | Runtime |
|---|---:|---:|---:|---:|
| Default | 74.958% | 78.288% | 0 | 0.244 s |
| Greedy | 76.817% | 79.907% | 13 | 0.583 s |
| Polished greedy | 76.826% | 79.907% | 13 | 2.050 s |
| Exact LNS | 76.826% | 79.907% | 13 | 13.225 s |
| Full MILP, time-limited | 76.800% | 79.908% | 13 | 1.676 s |
| Hybrid simulated annealing | 76.826% | 79.907% | 13 | 55.742 s |

The two required baselines also report the objective's cost components in normalized,
privacy-safe form:

| Baseline | Penalty / requested value | Shipping / requested value |
|---|---:|---:|
| Default | 0.4817% | 0.6754% |
| Greedy | 0.4232% | 0.7060% |

The executable notebook records the corresponding local totals; they are intentionally
excluded from the public branch because the challenge privacy rule prohibits commercial
cost disclosure.

Greedy provides most of the nominal gain: +1.859 percentage points of objective
capture and +1.619 points of case fill over default. Exact policy recourse adds 0.0085
capture points. Exact LNS makes only a negligible further move, and hybrid accepts no
post-polish move on this nominal subset. The full MILP ends with a 0.0545% relative gap
and lower incumbent, so it is correctly labeled a time-limited comparator.

![Common solver comparison](../results/final/figures/submission_solver_summary.png)

### 6.2 Coordinated improvement control

The generated control creates a greedy trap that requires a coordinated assignment
change. Greedy and polished greedy reach 21.49% fill. Exact LNS, full MILP, simulated
annealing, local QAOA, and hardware-derived QAOA proposals reach 30.58% fill after
recourse, a 197.2 synthetic objective-unit gain. Full MILP takes 0.019 seconds, exact
LNS 0.129 seconds, and the simulated-annealing hybrid 27.873 seconds. The result proves
that the decomposition can express coordinated moves and supplies a shared control for
proposal-quality experiments.

![Coordinated control](../results/final/figures/submission_coordination_summary.png)

## 7. Scaling, robustness, noise, and hardware

### 7.1 Real and generated scaling

All scalable methods reach the complete 372-group, 750-order real scope. Median runtime
over three repetitions is 1.305 seconds for greedy, 4.816 for polished greedy, 26.143
for exact LNS, and 101.169 for hybrid simulated annealing. Exact LNS retains the highest
full-scope capture, 82.122%, versus 82.115% for greedy.

![Repeated real scaling](../results/final/figures/submission_scaling_summary.png)

Generated scaling reaches 100,000 orders for greedy (39.163-second median) and exact
LNS (276.707 seconds). Hybrid scaling reaches 20,000 orders in 155.935 seconds while
the local QUBO stays at 32 variables. The operational network can therefore grow
without forcing the quantum-facing representation to grow with it.

### 7.2 Inventory robustness

Frozen routing remains feasible through a 55% inventory reduction and is then proven
infeasible. Adaptive methods remain feasible through 70%. At 70%, objective capture is
72.162% for greedy, 72.204% for hybrid, and 72.741% for exact LNS; corresponding case
fill is 76.560%, 76.560%, and 76.992%. This is the clearest business-facing advantage of
adaptive coordination: it keeps producing valid routes after quantity-only recourse on
the old policy breaks down.

![Inventory robustness](../results/final/figures/submission_robustness_summary.png)

### 7.3 Noise and sampler controls

Exact feasible sampling, simulated annealing, and local constraint-preserving QAOA are
one-hot by construction on the control; a random binary sampler is one-hot for only
0.352% of shots. Simulated annealing hits the exact QUBO optimum on 48.44% of shots and
local QAOA on 1.21%, against a 0.3906% uniform-feasible rate. All recover the same final
move after recourse, demonstrating why raw sampling and downstream plan quality must be
reported separately.

QUBO coefficient noise degrades raw structure after roughly 10% relative perturbation;
at 20% the median one-hot rate is about 29.4%. An independent readout-bit-flip proxy
reduces one-hot feasibility to about 19.7% at a 10% flip probability. Every safeguarded
row still recovers the control improvement. These are sensitivity studies, not a full
physical QPU noise model.

### 7.4 IBM hardware and runtime correction

The final hardware matrix contains 18 successful jobs at 8,192 shots each. Median
$p=1$ raw one-hot feasibility is 65.34%, exact-optimum hit rate is 0.6836%, circuit
depth is 43, and two-qubit-gate count is 40. At $p=2$, these become 10.36%, 0.0610%,
448, and 373. The shallow exact-hit median exceeds the 0.3906% uniform-feasible null;
the deeper circuit does not.

![Hardware depth comparison](../results/final/figures/submission_ibm_depth_summary.png)

All QPU-derived final plans recover the synthetic coordinated gain after exact recourse
and validation. Hardware job latency is queue-sensitive: typical medians are about
42-54 seconds, one baseline job waits roughly 11,926 seconds, and another mitigation
job reaches about 349 seconds. The corrected end-to-end chart uses a log axis, median
labels, and individual-job dots so these outliers remain visible without flattening the
other bars.

## 8. Recommendation, limitations, and submission map

### 8.1 Deployment recommendation

Use **fast** mode for routine planning, with polished greedy as the normal certified
output. Trigger **quality** mode when scarcity, shared resources, high penalty weights,
or planner exceptions justify exact LNS latency. Reserve full MILP for bounded
comparisons and certificates. Keep **hybrid** mode behind an R&D feature gate with the
same exact recourse and validator.

Before production, run a shadow-mode pilot over rolling historical windows. Confirm
authorized alternative DCs, working calendars, throughput limits, customer rules, and
approval thresholds. Measure plan validity, runtime, planner acceptance, and realized
service against default routing.

### 8.2 Limitations

- The real study uses one supplied planning snapshot; business impact needs rolling and
  prospective validation.
- The 100-group full-MILP comparator retains a nonzero gap.
- Hardware evidence uses a 16-variable generated control rather than real operations.
- Exact recourse can equalize final outcomes even when proposal quality differs.
- Queue time is external operational latency and can dominate end-to-end hardware time.

### 8.3 Completed deliverables

| Challenge deliverable | Submission artifact |
|---|---|
| Two-page business/technical summary | `reports/business_technical_summary.pdf` |
| Six-to-ten-page technical report | `reports/challenge_submission_report.pdf` |
| Runnable repository and notebook | Root `README.md`; `notebooks/nestle_challenge_experiments.ipynb` |
| Five-to-seven-slide presentation | `reports/final_presentation.pptx` and `.pdf` |
| One-page planner view | `reports/planner_view.pdf` |
| Mathematical formulation | `docs/mathematical_formulation.md` |
| Data and assumptions | `docs/data_guide.md`; `docs/assumptions.md` |
| Scaling, noise, qubit growth, limitations | This report, notebook, and screened figures |
| Full research treatment | `reports/final_report.pdf` |

The repository also includes unit, integration, notebook, UI, privacy, provenance, and
validator regression tests. Raw restricted data and row-level evidence remain local;
the branch publishes the implementation and screened aggregate artifacts required to
review the approach.
