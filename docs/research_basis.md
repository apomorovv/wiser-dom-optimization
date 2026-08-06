# Research basis and solver selection

## Decision

Use classical feasibility and exact recourse around a bounded, warm-started local
QUBO. Treat quantum sampling as a candidate generator, and compare it with exact,
random, and simulated-annealing samplers on the same subproblem.

| Approach | Fit for DOM | Main strength | Main limitation | Role here |
|---|---|---|---|---|
| Global MILP | high | Exact constraints, bounds, mature solvers | Can slow as integer coupling grows | Reference and recourse |
| Greedy | medium | Fast, transparent, feasible | Myopic | Incumbent baseline |
| Monolithic QUBO | low | Uniform binary representation | Quantity/slack expansion and penalty tuning | Rejected |
| QAOA | experimental | Gate-model variational search | Depth, parameter training, constraint handling | Future adapter |
| Quantum annealing | experimental | Direct binary quadratic input | Embedding, coefficient range, noise | Optional local sampler |
| Quantum local search | high as research design | Fixed hardware-sized neighborhoods | No automatic advantage | Chosen architecture |

## Primary sources

### Quantum local search

Tomesh et al., *Quantum Local Search with the Quantum Alternating Operator Ansatz*,
Quantum 6, 781 (2022), demonstrates a hybrid pattern in which a classical outer loop
uses a quantum optimizer for bounded neighborhoods of a larger constrained problem.
That pattern motivates the fixed QUBO budget and incumbent-based outer loop used
here. [Paper](https://doi.org/10.22331/q-2022-08-22-781) and
[preprint](https://arxiv.org/abs/2107.04109).

### Warm starts

Egger, Mareček, and Woerner, *Warm-starting quantum optimization*, Quantum 5, 479
(2021), initializes variational quantum optimization from a classical relaxation and
shows why useful classical information should not be discarded. This repository
uses the feasible incumbent as the common warm start for local samplers. It does not
claim the paper's QAOA guarantees for the annealing adapter.
[Paper](https://doi.org/10.22331/q-2021-06-17-479).

### Industrial quantum annealing evidence

Yarkoni et al., *Quantum Annealing for Industry Applications: Introduction and
Review* (2022), surveys formulation, embedding, hybridization, and benchmarking
issues across industrial use cases. Its cautious assessment supports reporting
logical size, embedding overhead, total workflow time, and classical comparators
instead of equating physical qubits with useful problem scale.
[Preprint](https://arxiv.org/abs/2112.07491).

### Logistics-specific hybrid workflow

The 2026 shipment-selection study by Lopez-Ruiz et al., *Hybrid Quantum-Classical
Optimization Workflows for the Shipment Selection Problem*, uses quantum-generated
assignment candidates with classical refinement and warm starts. It is close in
structure to DOM and supports the candidate-generation/recourse split. Its reported
gains are scenario-specific and do not establish general quantum advantage.
[Preprint](https://arxiv.org/abs/2604.11758).

### Platform behavior

D-Wave documents direct QPU sampling for binary quadratic models and larger managed
hybrid workflows, including classical decomposition around QPU-sized subproblems.
This motivates separate `dwave-qpu` and `dwave-hybrid` adapters and explicit remote
call accounting. [D-Wave hybrid concepts](https://docs.dwavequantum.com/en/latest/concepts/hybrid.html)
and [hybrid reference](https://docs.dwavequantum.com/en/latest/ocean/api_ref_hybrid/reference.html).

IBM's official QAOA tutorial explains the variational gate-model workflow retained
as a future comparison path. [IBM QAOA tutorial](https://quantum.cloud.ibm.com/docs/en/tutorials/quantum-approximate-optimization-algorithm).

### GPU optimization

NVIDIA documents cuOpt as a GPU-accelerated optimization library with a beta MIP
solver. Its current emphasis is finding high-quality feasible solutions quickly;
proving optimality remains under active development. That makes cuOpt a useful
future large-instance incumbent generator, but not yet a drop-in replacement for
the HiGHS reference when a bound or proof matters.
[NVIDIA cuOpt MIP documentation](https://docs.nvidia.com/cuopt/user-guide/latest/cuopt-python/mip/index.html).

The present local QUBOs are deliberately small, so transferring them to a GPU is
unlikely to dominate end-to-end time. The notebook therefore measures a synthetic
CPU/GPU batch-scoring crossover and reports solver stage timing before recommending
any backend change. Candidate preprocessing, group-level greedy construction, and
exact recourse should be optimized first.

### Risk objectives

Inventory protection and calibrated shortage penalties already represent known
business exposure. Adding an arbitrary risk coefficient would double-count that
exposure and make the objective harder to defend. The implemented inventory-shock
frontier is the appropriate first robustness test. A scenario-based expected-value,
worst-case, or CVaR objective becomes justified only when forecast scenarios,
probabilities, and business risk tolerance are available and held out for
validation.

## Claims policy

A result may be described as *quantum-assisted* only when a QPU or quantum service
actually supplied samples. Local simulated annealing is labeled
*quantum-inspired*. Neither label implies advantage. A defensible advantage claim
would require:

1. identical decision model and preprocessing;
2. strong tuned classical baselines;
3. multiple seeds and statistically reported variation;
4. end-to-end wall time, including remote latency and repair;
5. equal feasibility and objective evaluation;
6. scaling beyond instances where enumeration is possible; and
7. a clearly defined quality, time, or cost advantage.

This branch provides the instrumentation for that experiment but does not fabricate
the result.
