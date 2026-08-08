# WISER DOM Hybrid Solver

**Author:** Andrei Pomorov

**Date:** August 2026

### Executive decision

Adopt **polished greedy** as the portable planning default and use **exact
large-neighborhood search (LNS)** when a planning window contains valuable inventory or
dock conflicts. Keep the sampler-assisted hybrid, including quantum hardware, as a
controlled research option behind exact recourse and independent validation.

On the common 20-assignment-group subset, greedy increased objective capture from
61.39% to 64.90% and case fill from 64.21% to 68.50%. Exact polishing retained that
solution in 2.86 seconds. Exact LNS, full exact mixed-integer linear programming (MILP),
and the simulated-annealing hybrid reached the same displayed quality. The full exact
solver certified this small subset with zero optimality gap.

### Business problem

Distributed order management (DOM) selects a distribution center (DC) and planned ship
date while orders compete for shared inventory and capacity. Orders sharing a load must
move together, a non-default DC must improve service, and no recommendation may exceed
cumulative available-to-promise (ATP) inventory or enabled capacity.

The optimized business objective is:

> fulfilled merchandise value − unmet-demand penalty − shipping cost.

**Objective capture** divides that result by total requested merchandise value, making
comparisons interpretable without disclosing commercial totals. **Case fill** is
fulfilled cases divided by requested cases. **Reassignment** means moving an order away
from its default DC.

### Solution hierarchy

| Layer | What it does | Recommended use |
|---|---|---|
| Default baseline | Keeps planned routing and allocates feasible quantities | Business comparison |
| Greedy | Chooses whole-load alternatives while updating shared resources | Fast initial plan |
| Polished greedy | Fixes routing and exactly re-optimizes quantities and penalties | Routine production default |
| Exact LNS | Re-optimizes bounded groups of interacting decisions | Quality escalation |
| Full exact MILP | Solves all selected decisions and reports a bound | Small certificates and benchmarking |
| Hybrid sampler | Proposes coordinated local assignments; exact models retain control | Research only |

The hybrid's quantum unconstrained binary optimization (QUBO) covers only a bounded
local neighborhood. QAOA, simulated annealing, random sampling, or exact enumeration can
propose binary choices. The system repairs one-hot violations, exactly re-optimizes
fulfillment, validates the full plan, and rejects every invalid or non-improving result.

### Evidence

The reviewed study contains 247 aggregate rows across 14 experiment families. All 247
passed the independent validator. Maximum numeric residuals for demand balance,
integrality, cumulative inventory, and enabled capacities were zero.

| Common-subset method | Objective capture | Case fill | Runtime |
|---|---:|---:|---:|
| Default routing | 61.39% | 64.21% | 0.30 s |
| Greedy | 64.90% | 68.50% | 0.76 s |
| Polished greedy | 64.90% | 68.50% | 2.86 s |
| Exact LNS | 64.90% | 68.50% | 7.00 s |
| Full exact MILP | 64.90% | 68.50% | 2.76 s |
| Hybrid simulated annealing | 64.90% | 68.50% | 12.88 s |

Scaling reached 372 assignment groups and 750 orders. Median runtime there was 16.74
seconds for greedy, 54.63 for polished greedy, and 74.03 for exact LNS. LNS produced
small additional gains at several larger scales, supporting time-budgeted escalation.

A synthetic greedy trap was improved by 197.2 units by exact LNS, exact MILP, simulated
annealing, and local QAOA, increasing case fill from 21.49% to 30.58%. This validates
coordinated neighborhoods, not quantum advantage; exact classical methods were faster.

Under a 40% aggregate inventory shock, re-optimized routing achieved 69.66% case fill
versus 67.21% with the original routing fixed. This is the clearest operational
robustness result: the solver can respond to changed availability instead of only
reallocating quantities along stale routes.

### Quantum assessment

The IBM archive contains 18 QPU jobs on a generated 16-qubit control. All QPU-derived
plans were feasible after exact recourse, but native one-hot feasibility fell from
approximately 51%–59% at $p=1$ to 12%–15% at $p=2$ as median two-qubit gates rose from
122 to 420. The QPU did not beat exact MILP in quality or time.

The final code strengthens the next experiment with a lower-edge path mixer, linear
W-state preparation, reusable optimized angles, confidence intervals, near-optimal
rates, and uniform-feasible controls. Those circuit changes require a matched hardware
rerun; they are not retroactively credited to the archived IBM result.

### Deployment and next decisions

The default stack is license-free and runs on Windows, macOS, and Linux with Python
3.10–3.12. SciPy/HiGHS is the default exact backend. Gurobi is a worthwhile optional
comparison because it may improve solve progress on larger MILPs, but it is neither a
dependency nor the default; the notebook skips it cleanly without a package or license.

Before a pilot, owners should confirm authorized alternative DCs, the working-day
calendar, throughput limits, customer rules, and approval thresholds. Run polished
greedy on rolling historical windows; trigger LNS only for material conflicts; review
validator diagnostics and planner acceptance; and compare realized service with
default routing. Restrict quantum reruns to approved synthetic or bounded neighborhoods
with exact and uniform-feasible controls.

The result is a usable classical solver and a safe quantum test bed whose operational
quality does not depend on a license, GPU, or quantum account.
