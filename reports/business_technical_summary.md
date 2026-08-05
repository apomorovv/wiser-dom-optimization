# Business and technical summary

## The decision

Distributed Order Management (DOM) decides how to fulfill customer orders across a
network of distribution centers (DCs). An order normally has a default DC. If that
DC lacks inventory, throughput, labor, dock, or pick capacity, a planner may divert
the complete order to an eligible alternative DC. The choice changes customer fill,
unmet-demand penalty, shipping cost, delivery timing, and the inventory left for
other orders.

This is a combinatorial optimization problem because orders compete for shared
resources. A candidate that looks best for one order in isolation can consume the
last stock or dock slot needed by several more valuable orders. With \(O\) orders
and roughly \(K\) outcomes per order, even the assignment-only search space is
approximately \(K^O\). SKU-level partial quantities and time-indexed resources add
further integer decisions.

The business objective used consistently in this work is:

\[
\text{fulfilled value}-\text{unmet-demand penalty}-\text{total shipping cost}.
\]

Each order selects at most one eligible DC/date; an explicit unassigned outcome keeps
the model feasible. Fulfilled plus unfulfilled cases equal demand. Shared projected
available-to-promise inventory is protected at every future checkpoint. Optional
dock, throughput, pallet-pick, case-pick, weight, and volume limits are enforced.
For a diversion, fill must exceed documented default fill by at least five percentage
points of total ordered cases. Partial fulfillment remains allowed at the one
selected DC.

## Why a hybrid method

A classical mixed-integer linear program (MILP) represents these hard rules exactly
and supplies an optimality bound, but a large operational instance can become slow
as candidate and resource coupling grows. A greedy heuristic is fast and transparent,
but it can commit scarce resources too early. A single global quantum model would
require binary encodings for assignments, quantities, and constraint slack, producing
more logical variables, denser couplings, and delicate penalty scales than current
hardware can reliably support.

The selected architecture uses each tool for the job it handles best:

1. canonical preprocessing creates only eligible candidates;
2. default and sequential greedy methods provide transparent feasible baselines;
3. a conflict-aware large-neighborhood search selects a bounded set of orders;
4. a warm-started QUBO proposes coordinated assignments in that neighborhood;
5. exact MILP recourse rebuilds fulfillment under residual resources; and
6. independent validation accepts only a strict feasible improvement.

QUBO is quadratic unconstrained binary optimization. It is suitable for simulated
annealing and optional quantum annealing. The local model contains one variable per
candidate plan plus an unassigned plan, a one-outcome-per-order penalty, and
loss-weighted resource-contention terms. Its logical size is capped, so total order
volume may grow without requiring a proportionally larger quantum subproblem.

This design has a safety property: a noisy or weak sampler may fail to find an
improvement, but it cannot make the returned solution worse. The incumbent remains
feasible, exact recourse checks sampled assignments, and the validator rejects any
violation. Remote quantum execution is also privacy-gated; local simulated annealing
is the default.

## Evaluation

Every method receives the same canonical tables and candidate set. A separate
objective evaluator reports objective value, fulfilled value, penalty cost, shipping
cost, case fill, value fill, reassigned orders, runtime, and—when available—MILP
optimality gap. A separate validator checks assignment, demand, eligibility,
inventory, capacity, and alternate-fill rules. Infeasible samples are never ranked as
business solutions.

The exact two-order test has a known optimum of 126 synthetic units:
`O1` diverts to `D2`, while `O2` remains at `D1`. The classical MILP and bounded
hybrid method reproduce that optimum, and the hybrid improves its default incumbent
by 176 units. Automated tests cover baselines, the exact optimum, candidate filtering,
projected ATP, the five-percent rule, pick decomposition, QUBO sampling and privacy,
higher-order contention, planner explanation, and independent validation.

The provided recommendation outputs were also audited without exposing identifiers.
They contain 1,109 orders and 25,193 order-SKU rows; the two levels reconcile with no
quantity mismatch. The incumbent selected three diversions and fulfilled 2,413,937
of 2,554,440 requested cases, a 94.4997% case fill rate. Commercial metrics were not
emitted.

The remaining uploaded input names are AppleDouble metadata sidecars rather than the
underlying CSV, workbook, or Word payloads. They do not contain the alternative-DC
inventory, capacity, eligibility, and cost universe needed to reoptimize. Therefore
no real-data improvement is claimed from this upload. The repository supplies a
strict canonical loader and processing contract; the original payloads must be
re-exported before the final private comparison.

## Trade-offs and recommendation

The exact MILP is the quality reference on tractable subsets. Greedy is the rapid
fallback. The hybrid is most useful when the global MILP is time-limited but small,
high-conflict reassignment neighborhoods can still be solved repeatedly. Simulated
annealing establishes a strong local control; exact QUBO enumeration verifies very
small neighborhoods; a D-Wave run is optional after data approval.

Current quantum hardware introduces embedding, coefficient-range, noise, queue, and
latency costs. Physical qubits are not equivalent to dense logical variables. A fair
claim therefore requires multiple seeds, tuned classical competitors, equal QUBOs,
end-to-end time, exact post-processing, and results that scale beyond enumeration.
This implementation makes that comparison reproducible but deliberately makes no
quantum-advantage claim.

For deployment, use the hybrid result only after its validator passes and a planner
reviews the generated explanation of fill uplift, penalty avoided, shipping change,
net objective change, and delivery date. Keep the default incumbent available as a
safe rollback.
