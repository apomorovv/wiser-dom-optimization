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

## Complexity

The full assignment MILP grows with eligible candidates and order-line candidate pairs.
The local QUBO grows only with retained assignment plans in one neighborhood, but an
unstructured binary representation still has a `2^n` Hilbert space for `n` qubits.
One-hot feasible-subspace simulation has dimension equal to the product of choices per
group. This remains exponential and is intentionally bounded by
`max_feasible_states`. Candidate caps, conflict neighborhoods, exact LNS, and hardware
depth limits are the practical scaling controls.
