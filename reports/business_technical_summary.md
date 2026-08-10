# WISER DOM Scalable Solver

**Author:** Andrei Pomorov
**Date:** August 2026

### Executive decision

Adopt the validated classical hierarchy: use **polished greedy** for routine planning and
trigger **exact large-neighborhood search (LNS)** when inventory, capacity, or penalty
conflicts justify additional runtime. Keep the bounded sampler-assisted path as a
controlled research option behind exact recourse and independent validation.

On the common 100-assignment-group subset, greedy improves objective capture from
74.96% to 76.82% and case fill from 78.29% to 79.91% in 0.58 seconds. Exact quantity
polish reaches 76.83% capture in 2.05 seconds. Exact LNS retains the strongest nominal
frontier in 13.22 seconds. A time-limited full MILP returns a lower incumbent with a
nonzero gap, demonstrating the value of preserving the hierarchy’s strong incumbent.

### Business problem and measures

Distributed order management selects a distribution center (DC) and ship date while
orders compete for shared time-phased inventory and capacity. Orders sharing a load
must move together; an alternate DC must improve service; and no recommendation may
exceed cumulative available-to-promise (ATP) inventory or documented capacity.

The optimized objective is fulfilled merchandise value minus unmet-demand penalty minus
shipping cost. **Objective capture** normalizes that value by total requested
merchandise value, avoiding disclosure of commercial totals. **Case fill** is fulfilled
cases divided by requested cases. **Reassignment** means moving an order away from its
default DC.

### Solution hierarchy

| Layer | What it does | Recommended use |
|---|---|---|
| Default | Keeps planned routing and allocates feasible quantities | Business baseline |
| Greedy | Chooses whole-load alternatives while updating resources | Fast incumbent |
| Polished greedy | Fixes routes and exactly re-optimizes quantities and penalties | Routine certified plan |
| Exact LNS | Reopens bounded groups of interacting decisions | Quality escalation |
| Full MILP | Solves all selected decisions and reports incumbent, bound, and gap | Small-scope comparison |
| Hybrid sampler | Proposes bounded local assignments; exact models retain control | Research |

The hybrid QUBO contains only local assignment choices. Exact enumeration, random
sampling, simulated annealing, local QAOA, or hardware-executed QAOA can propose a
bitstring. The workflow repairs one-hot violations, exactly re-optimizes fulfillment,
validates the complete plan, and rejects invalid or non-improving results. QAOA is the
proposal method; an IBM `backend` is a processor/execution target, not a solver.

### Audited evidence

The final study contains **516 aggregate rows across 14 experiment families**. Of these,
513 return plans and all 513 pass the independent validator with zero recorded demand,
integrality, inventory, and capacity residuals. Three frozen-routing controls are
correctly proven infeasible at 60%, 65%, and 70% inventory reductions.

| Common 100-group method | Objective capture | Case fill | Runtime |
|---|---:|---:|---:|
| Default routing | 74.96% | 78.29% | 0.24 s |
| Greedy | 76.82% | 79.91% | 0.58 s |
| Polished greedy | 76.83% | 79.91% | 2.05 s |
| Exact LNS | 76.83% | 79.91% | 13.22 s |
| Full MILP, time-limited | 76.80% | 79.91% | 1.68 s |
| Hybrid simulated annealing | 76.83% | 79.91% | 55.74 s |

All scalable methods reach 372 real assignment groups and 750 orders. Median full-scope
runtime is 1.31 seconds for greedy, 4.82 for polished greedy, 26.14 for exact LNS, and
101.17 for hybrid simulated annealing. On generated instances, greedy and exact LNS
reach 100,000 orders in 39.16 and 276.71 seconds; hybrid reaches 20,000 orders while the
local QUBO remains 32 variables.

The inventory frontier shows why adaptive routing matters. Frozen nominal routing is
feasible through a 55% inventory reduction and then fails. Adaptive methods remain
feasible at 70%; exact LNS achieves 72.74% objective capture and 76.99% case fill,
compared with 72.16% and 76.56% for greedy.

### Quantum and hardware result

A generated coordination control requires a joint move that greedy misses. Exact LNS,
full MILP, simulated annealing, local QAOA, and hardware-derived QAOA proposals all
recover the 197.2 synthetic-unit improvement after recourse, increasing fill from
21.49% to 30.58%.

The final IBM study contains 18 jobs at 8,192 shots each on a 16-variable control. At
$p=1$, median raw one-hot feasibility is 65.34%, exact-optimum hit rate is 0.6836%,
transpiled depth is 43, and the circuit uses 40 two-qubit gates. At $p=2$, these become
10.36%, 0.0610%, depth 448, and 373 gates. Shallow QAOA therefore retains the stronger
signal. All final plans remain protected by exact recourse and validation. End-to-end
runtime is reported on a log scale with individual-job dots because one long queue wait
would otherwise hide the typical 42-54 second medians.

### Deployment and next decisions

The default stack is license-free and runs on Windows, macOS, and Linux with Python
3.10-3.12. SciPy/HiGHS is the portable exact engine; native HiGHS, SCIP, Gurobi, GPU
scoring, and IBM Runtime are opt-in.

Before a pilot, confirm authorized alternative DCs, working calendars, throughput
limits, customer rules, and approval thresholds. Run polished greedy over rolling
historical windows; trigger exact LNS for material conflicts; review validator
diagnostics and planner acceptance; and compare realized service with default routing.
Restrict hardware reruns to approved generated or bounded neighborhoods with exact and
uniform-feasible controls.

The submission delivers a usable scalable solver now and a disciplined test bed for
future proposal methods without making operational quality depend on a license, GPU, or
quantum account.
