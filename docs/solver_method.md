# Solver method

## Decision hierarchy

The repository exposes three validated modes through `solve_dom`:

| Mode | Method | Intended use |
|---|---|---|
| `fast` | Greedy assignment plus exact fixed-assignment fulfillment recourse | Routine planning |
| `quality` | Adaptive exact large-neighborhood search | Coupled or difficult planning windows |
| `hybrid` | Sampler-assisted neighborhood search plus exact recourse | Research and controlled hardware studies |

Full exact MILP is also available for small-instance certificates and backend
comparisons. SciPy/HiGHS is the default; Gurobi is optional and must be selected
explicitly.

## Fast mode

The greedy stage considers whole assignment groups, ranks eligible choices
deterministically, and commits a choice only when inventory and fixed resource checks
remain feasible. The exact recourse stage then fixes those routing choices and solves
all fulfillment quantities and thresholded penalties exactly. If recourse fails, the
already feasible greedy incumbent is retained.

## Exact large-neighborhood search

Large-neighborhood search (LNS) repeatedly selects interacting assignment groups,
residualizes the rest of the network, and solves a bounded MILP over the active
neighborhood. Neighborhoods begin near resource conflicts and adapt within limits for
order count, fulfillment variables, time, and gap. A candidate move is accepted only
when the merged global solution is valid and improves the common objective.

## Hybrid large-neighborhood search

The hybrid method uses the same incumbent and conflict neighborhoods, but replaces the
local joint-assignment MILP with a proposal QUBO. The sampler may be exact enumeration,
random sampling, simulated annealing, local QAOA statevector simulation, or an IBM QPU.

The proposal stage is deliberately separated from feasibility:

1. Build one binary choice variable for each retained local plan.
2. Add one-hot penalties and pairwise resource-conflict terms.
3. Sample assignment bitstrings.
4. Measure native one-hot feasibility and energy quality.
5. Repair one-hot violations deterministically.
6. Keep unique low-energy assignments.
7. Fix each retained assignment and solve exact fulfillment recourse.
8. Merge with the global incumbent and run the independent validator.
9. Accept only a strictly improving valid solution.

Therefore, final feasibility is a property of repair, exact recourse, and validation; it
is not evidence that a sampler natively respected the constraints.

## Constraint-preserving QAOA

QAOA alternates a problem-cost unitary with a mixer unitary. The assignment encoding
has one-hot groups, so the ideal gate-model implementation starts each group in a
weight-one Dicke state, also called a W state. The final implementation prepares that
state with a linear controlled-rotation construction instead of a generic exponentially
described state-preparation gate.

The XY mixer exchanges the one excitation among choices and preserves group Hamming
weight. A connected path is the default topology:

$$
E_{\mathrm{path}} = \{(0,1),(1,2),\ldots,(m-2,m-1)\}.
$$

It uses `m - 1` logical mixer edges for a group of size `m`, compared with `m` for the
earlier ring when `m > 2`. The path remains connected, so every one-hot choice is
reachable, while reducing logical two-qubit interactions.

Angles are optimized with the matching exact feasible-subspace simulator. One parameter
vector per QUBO and depth is cached and reused across hardware mitigation variants and
repetitions. This prevents repeated local classical optimization from being mistaken
for QPU execution time.

## Quantum evidence metrics

Every one-hot QUBO small enough for exact feasible enumeration is evaluated with:

- raw one-hot feasibility rate;
- exact optimum hit rate and its 95% Wilson confidence interval;
- hit rate conditional on feasibility;
- rate within 1% of the feasible energy span;
- best and mean feasible normalized energy gaps;
- uniform-feasible optimum, near-optimum, and mean-gap null values.

These metrics distinguish native constraint preservation, energy quality, and
downstream recourse. Rare exact-hit counts are not used alone to rank a strategy.

## Safeguards

- IBM access requires `allow_remote=True`.
- The hardware study uses an independently generated synthetic control.
- Multiple transpiler seeds are tried and the lowest two-qubit-gate circuit is used.
- Mitigation options are explicit and include baseline, dynamical decoupling, and
  dynamical decoupling plus measurement twirling.
- Every solution passes a separate validator with categorical violations and numeric
  residual diagnostics.
- The best known feasible incumbent is always retained.

## Complexity and scalability

The main scalability challenge comes from the number of possible routing decisions, not
from the fulfillment calculation alone. If an assignment group has several eligible
DC/PGI alternatives, the solver must choose one of them while simultaneously respecting
shared inventory, dock, throughput, and load-cohesion constraints. As more orders,
SKUs, dates, and alternatives are introduced, these decisions become increasingly
coupled.

### Full MILP

The full deterministic MILP contains binary assignment variables for eligible
order/candidate combinations and fulfillment variables for order-line/candidate
combinations. Its size therefore grows approximately with

$$
N_x \sim \sum_o |\mathcal C_o|,
\qquad
N_f \sim \sum_o |\mathcal S_o|\,|\mathcal C_o|,
$$

in addition to penalty, pallet, capacity, and other auxiliary variables.

The number of variables grows roughly linearly with the number of retained candidate
columns, but the difficulty of the integer search can grow much faster because the
solver must consider combinations of assignments that compete for the same resources.
For this reason, the implementation does **not** rely on solving one unrestricted MILP
over the entire network in production. Full exact MILP is retained primarily for small
instances, benchmarking, and optimality certificates. 

Candidate preprocessing is the first scaling layer. Invalid lanes, unavailable
DC/SKU combinations, closed dates, late options, forecast-ineligible alternatives, and
other impossible candidates are removed before optimization. This reduces both the
number of binary decisions and the number of associated fulfillment variables.

### Scalable classical hierarchy

The implementation uses a solver hierarchy instead of applying the most expensive
method to every problem.

The `fast` mode first constructs a feasible assignment greedily and then fixes those
routing decisions while an exact recourse MILP optimizes fulfillment quantities and
penalties. This avoids the full combinatorial assignment search while still solving the
continuous/integer fulfillment problem exactly for the chosen routing.

The `quality` mode then applies **adaptive exact large-neighborhood search (LNS)**.
Rather than reopening every assignment decision, LNS identifies groups that interact
through inventory or capacity conflicts, freezes the remainder of the network, and
solves an exact MILP only over that bounded neighborhood. The inactive part of the
network is residualized, so its already committed resource usage is accounted for
without being re-optimized. Neighborhood size is explicitly limited by order count,
fulfillment-variable count, runtime, and MIP-gap controls. 

This makes the method an **anytime hierarchy**:

$$
\text{fast feasible solution}
\;\rightarrow\;
\text{exact local improvement}
\;\rightarrow\;
\text{full certificate when tractable}.
$$

The best valid incumbent is never discarded simply because a more expensive search
fails or times out.

### Local QUBO scaling

The hybrid method applies the same localization principle to the quantum component.
The QUBO does not encode the full DOM problem. It contains only the retained assignment
plans of a bounded conflict neighborhood. Continuous fulfillment quantities, cumulative
inventory, and the complete business constraints remain outside the QUBO and are
handled by exact recourse and independent validation.

Suppose neighborhood group \(g\) retains \(m_g\) assignment choices. The number of QUBO
binary variables, and therefore logical qubits in the direct gate-model encoding, is

$$
n=\sum_{g=1}^{G_{\mathrm{local}}}m_g.
$$

An unconstrained \(n\)-qubit statevector has dimension

$$
2^n.
$$

However, the assignment problem requires exactly one choice from each group. The number
of one-hot feasible assignments is therefore only

$$
N_{\mathrm{feasible}}=\prod_{g=1}^{G_{\mathrm{local}}}m_g.
$$

For example, five local groups with three alternatives each require 15 binary variables.
The unrestricted binary space contains

$$
2^{15}=32{,}768
$$

states, whereas the one-hot feasible space contains only

$$
3^5=243
$$

valid assignments.

This reduction is substantial, but it does **not** eliminate exponential scaling:
\(\prod_g m_g\) still grows exponentially with the number of active groups. The exact
feasible-subspace simulator therefore refuses to enumerate neighborhoods whose state
count exceeds the configured `max_feasible_states`. Larger neighborhoods must instead
be reduced, sampled, or handled by the classical LNS path.

### Quantum circuit scaling controls

The gate-model implementation includes additional measures intended to keep the local
quantum problem small and physically meaningful.

Each assignment group starts in a weight-one Dicke (W) state and uses an XY mixer that
moves the single excitation between candidate choices. The final implementation uses a
connected path mixer,

$$
E_g=\{(0,1),(1,2),\ldots,(m_g-2,m_g-1)\},
$$

which requires $(m_g-1)$ logical mixer edges for a group of \(m_g\) choices rather than
the \(m_g\) edges of the earlier ring construction. The path remains connected, so all
one-hot choices remain reachable while reducing two-qubit circuit burden. QAOA
parameters are also optimized once for a QUBO/depth pair and reused across matched
hardware runs rather than repeatedly paying the classical tuning cost. 

The practical quantum scaling controls are therefore:

- candidate pruning before optimization;
- a cap on retained plans per assignment group;
- conflict-based neighborhood selection;
- a cap on the number of active groups and QUBO variables;
- `max_feasible_states` for exact feasible-subspace simulation;
- shallow QAOA depth;
- the lower-edge-count path mixer;
- exact classical recourse after sampling; and
- automatic retention of the best classical incumbent.

Quantum hardware is therefore used as a **bounded local proposal engine**, not as a
representation of the complete distribution network.

### Observed scaling

The implementation was tested on progressively larger real-data subsets containing
from 8 to 372 assignment groups and from 9 to 750 orders. Median greedy runtime
increased from 0.21 s to 16.74 s, polished greedy from 0.52 s to 54.63 s, and exact
LNS from 1.72 s to 74.03 s. Full exact MILP was intentionally restricted to at most
20 groups, while the experimental hybrid search was limited to at most 50 groups.


Importantly, LNS continued to find improvements at larger sizes: relative to the
polished incumbent, it improved the objective at 100, 250, and 372 assignment groups.
This supports the intended design: inexpensive construction handles the full planning
problem, while exact search effort is concentrated only where interactions make it
worthwhile. 

The resulting architecture does not remove the underlying combinatorial complexity of
DOM, nor does it claim that the quantum subproblem scales polynomially. Instead, it
controls complexity through **preprocessing, decomposition, bounded neighborhoods,
candidate reduction, exact recourse, and staged escalation**. This is the key
scalability feature of the implementation: the size of the operational network can grow
without requiring the quantum processor—or even the exact global MILP—to grow at the
same rate.
