# Research basis and solver selection

## Decision

Use adaptive exact-MILP large-neighborhood search (LNS) as the production solver,
starting from a separately reported polished-greedy incumbent. Each bounded
neighborhood leaves both assignment and fulfillment decisions free, so the detailed
business objective and hard constraints are optimized together rather than through a
QUBO ranking surrogate.

Retain the bounded QUBO path as an experimental comparator. Quantum sampling proposes
assignment candidates there and is compared with exact, random, and
simulated-annealing samplers on the same frozen subproblem. It is not the production
optimization path and no improvement produced by classical quantity polishing is
attributed to a sampler.

| Approach | Fit for DOM | Main strength | Main limitation | Role here |
|---|---|---|---|---|
| Global MILP | high | Exact constraints, bounds, mature solvers | Can slow as integer coupling grows | Reference and small-instance solver |
| Polished greedy | high | Fast feasible routing plus exact fixed-assignment quantities | Cannot revise routing | Production incumbent and attribution baseline |
| Adaptive exact-MILP LNS | highest | Joint assignment/quantity optimization with exact local constraints | Bounded neighborhoods can miss global moves | Production choice |
| Sampler-assisted local QUBO | experimental | Backend-neutral, hardware-bounded assignment comparison | Approximate shared-resource model and repeated recourse | Controlled research comparator |
| Monolithic QUBO | low | Uniform binary representation | Quantity/slack expansion and penalty tuning | Rejected |
| Dicke(1)/XY QAOA | experimental | One-hot feasibility is structural | State preparation, depth, parameter training | Local simulator and optional IBM adapter |
| PCE / PB-PCE / QRAO | early research | Qubit compression | Decoding and hard-constraint feasibility | Research backlog |
| Quantum annealing | experimental | Direct binary quadratic input | Embedding, coefficient range, noise | Optional D-Wave adapter |

## Why exact-MILP LNS is the production choice

The important coupling is not merely one-hot assignment. Shared projected ATP,
date/resource capacity, partial fulfillment, pallet/case logic, and thresholded
penalties determine the value of an assignment jointly. A pairwise local QUBO can rank
promising assignments, but it cannot represent all of that recourse without additional
slack variables, higher-order reductions, and penalty calibration. Evaluating several
QUBO samples therefore requires several fixed-assignment MILP solves.

`solve_exact_lns` removes that indirection. It builds the assignment-group resource
conflict graph once, selects whole-group neighborhoods, residualizes frozen global
consumption, and calls the detailed MILP with local assignments and quantities both
free. Neighborhood size is controlled by group count, order count, estimated
fulfillment variables, time, and MIP gap. Its target grows after fast solves, shrinks after
expensive solves, and periodically uses seeded random diversification. A candidate is
accepted only after independent full-problem validation and a strict objective
improvement.

This is a matheuristic rather than an optimality claim. The bounds make runtime
controllable, while the exact local model and validator preserve solution validity.

## Classical alternatives evaluated

- **Local Branching and RINS.** Local Branching adds a linear Hamming-neighborhood cut
  around an incumbent; RINS fixes variables on which an incumbent and LP solution
  agree. Both establish exact sub-MIP search as a mature way to improve feasible MIP
  incumbents. The implemented conflict neighborhoods use the same principle while
  respecting whole assignment groups. [Local Branching](https://doi.org/10.1007/s10107-003-0395-5)
  and [RINS](https://doi.org/10.1007/s10107-004-0518-7).
- **Adaptive LNS.** Hendel's MIP LNS framework dynamically selects among neighborhood
  mechanisms and adapts the fixing rate. It supports the implemented grow/shrink and
  diversification policy, without implying that one policy is universally best.
  [Primary paper](https://optimization-online.org/2018/12/6992/).
- **CP-SAT.** It is a credible integer-model benchmark after every monetary and rate
  coefficient is scaled exactly; CP-SAT requires integer constraints. It is not an
  automatic replacement for the existing MILP and should be selected by reproducible
  end-to-end benchmarks. [Official CP-SAT guide](https://developers.google.com/optimization/cp/cp_solver).
- **Direct HiGHS.** A direct `highspy` benchmark can expose solver controls and newer
  MIP work before they reach the SciPy wrapper. HiGHS 1.15 introduced its prototype
  multithreaded MIP solver, so parallel performance must be measured rather than
  assumed. [Official 1.15 release](https://github.com/ERGO-Code/HiGHS/releases/tag/v1.15.0).
- **Benders decomposition.** Ordinary Benders is attractive when fixed assignments
  leave a continuous LP recourse problem. The current detailed model retains discrete
  case and threshold decisions, so a valid decomposition would require logic-based
  Benders or a changed relaxation. [Logic-based Benders](https://doi.org/10.1007/s10107-003-0375-9).
- **Column generation and Lagrangian relaxation.** Candidate assignment plans are
  already enumerated, so column generation does not address the present bottleneck.
  LP dual or Lagrangian prices remain useful neighborhood-priority signals, but they do
  not replace exact feasibility checks.
- **LP rounding.** Published DOM approximation guarantees rely on structural
  assumptions such as stock being large relative to one order and a bounded line count.
  The relaxation can inform priorities, but strict challenge feasibility still
  requires repair and validation. [Microsoft DOM theory paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2019/11/Theory__DOM.pdf).
- **GPU MIP.** NVIDIA documents cuOpt MIP as beta and emphasizes quickly finding good
  feasible solutions while optimality proof remains under development. It is a future
  incumbent-generator benchmark, not the validity reference.
  [Official FAQ](https://docs.nvidia.com/cuopt/user-guide/latest/faq.html).

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
uses the feasible incumbent to warm-start simulated annealing and the classical outer
loop. The first QAOA implementation uses a uniform feasible state; transferring the
incumbent into feasible amplitudes or parameters is a measured next experiment, not an
assumed benefit.
[Paper](https://doi.org/10.22331/q-2021-06-17-479).

### One-hot constraints, XY mixers, and Dicke states

Every local assignment group must select exactly one plan. Its uniform feasible quantum
state is therefore the weight-one Dicke state $|D^1_K\rangle$, also called a W state, over
the group's $K$ plan qubits. XY mixers preserve Hamming weight, so a product of these
states explores assignments without one-hot penalties or repair in the ideal circuit.
This matches the solver structure more directly than unconstrained X-mixer QAOA.

The implemented `qaoa_statevector` backend simulates that product state and ring-XY
mixer in the reduced categorical basis. The state dimension is
$\prod_g |\mathcal K_g|$, not $2^{\sum_g|\mathcal K_g|}$, which makes it a useful small
control but still exponential in neighborhood groups. The `ibm-qpu` adapter compiles the
corresponding circuit only after explicit remote-data approval.

- Wang et al. analyze one-hot XY mixers and feasible-subspace circuit layouts.
  [Primary paper](https://doi.org/10.1103/PhysRevA.101.012320)
- Bucher et al. (2025) simulate XY enforcement of one-hot constraints in a broader
  multi-constraint QAOA architecture and report better quality and simulated
  time-to-solution than their penalty-QUBO baselines.
  [Preprint](https://arxiv.org/abs/2506.03115)
- Bärtschi and Eidenbenz give short-depth circuits for arbitrary Dicke states.
  [Preprint](https://arxiv.org/abs/2207.09998)
- Kordonowy and Leipold (2026) connect XY topology to trainability and demonstrate
  restricted-ansatz pretraining before restoring RZZ interactions; their results also
  warn that added RZZ expressivity can make the dynamical Lie algebra exponential. This
  is a future ablation, not a description of the current QAOA initialization.
  [Article](https://www.nature.com/articles/s41534-026-01192-4)

### Alternative gate-model encodings

- **Direct one-hot versus binary.** Binary encoding uses fewer qubits, but its cost and
  constraint-preserving mixers generally compile to higher-locality terms. With only a
  few choices per assignment group, the implemented one-hot representation trades one
  or two extra qubits for a two-local XY mixer and direct one-hot samples.
  [Alternating-operator construction](https://arxiv.org/abs/1709.03489).
- **Domain wall.** It uses $K-1$ rather than $K$ qubits for a $K$-valued variable and
  retains two-body structure, but imposes an arbitrary adjacency on otherwise unordered
  DC/date plans. Reported benefits depend strongly on problem structure, so it is an
  encoding ablation rather than the default.
  [Primary paper](https://doi.org/10.1088/2058-9565/ab33c2).
- **Grover mixers.** They can preserve an arbitrary feasible set when its uniform
  superposition is efficiently preparable, but move complexity into state preparation,
  reflection, and controlled operations. Product one-hot feasibility already has a
  shallower two-local XY construction here.
  [Primary paper](https://arxiv.org/abs/2006.00354).
- **Warm-start and multi-angle variants.** The current backend uses uniform Dicke/W
  initialization and shared variational angles, not a warm-started QAOA. Restricted
  pretraining and grouped or multi-angle parameters are defensible future experiments,
  but must receive the same objective-evaluation or shot budget as standard QAOA.
  [Multi-angle QAOA](https://doi.org/10.1038/s41598-022-10555-8).

### Pauli Correlation Encoding (PCE)

PCE encodes binary variables in signs of multi-body Pauli expectation values. The
original construction encodes a polynomial number of classical variables through
Pauli correlations and reports its principal optimization evidence on MaxCut. That
advantage targets a different bottleneck from the current solver: real local QUBOs are
deliberately capped at 40 logical variables, while detailed recourse and preprocessing,
not logical qubit count alone, determine end-to-end runtime.
[Original PCE paper](https://doi.org/10.1038/s41467-024-55346-z).

Standard PCE is also a poor immediate fit for hard assignment constraints because its
decoded variables are signs of continuous expectation values rather than measured
one-hot bitstrings. Padín-Martínez et al. (2026) report that standard PCE does not reliably
enforce a constrained MinCut budget and that tuning does not transfer well. Their
Progressive-Binarization PCE reaches 88–100% feasibility up to 300 variables with
9-qubit noiseless simulations, but requires roughly 10–20 continuation stages. These are
promising research results, not evidence on DOM or noisy hardware.
[PB-PCE preprint](https://arxiv.org/abs/2602.17479)

A 2026 analysis additionally constructs efficiently classically simulable PCE
instances and notes that learning a correlation sign with margin $\gamma$ can require
$O(\gamma^{-2})$ measurements. A PCE experiment must therefore include the best
dequantized PCE baseline and decoding/measurement cost.
[Primary preprint](https://arxiv.org/abs/2607.20409).

QRAO similarly compresses relaxed binary variables into quantum random access codes,
but its rounding guarantees concern objectives such as MaxCut and do not preserve
DOM's one-hot, inventory, capacity, or integer-recourse feasibility. It is not a
production encoding for these bounded local problems.
[QRAO analysis](https://arxiv.org/abs/2302.09481).

Recommendation: keep direct one-hot Dicke/XY QAOA as the near-term gate-model baseline.
Add PB-PCE only as a separate synthetic adapter once local QUBO width, rather than
recourse or preprocessing, is empirically the limiting resource. It must use the same
decoded-assignment repair, exact recourse, validator, and end-to-end timing protocol.

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

IBM's official QAOA material provides the gate-model execution path used by the optional
adapter. IBM's Open Plan currently offers limited free QPU time, so D-Wave is not the only
hardware route. [IBM QAOA tutorial](https://quantum.cloud.ibm.com/docs/en/tutorials/quantum-approximate-optimization-algorithm)
and [IBM plan overview](https://quantum.cloud.ibm.com/docs/en/guides/plans-overview).

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
actually supplied samples. Local simulated annealing is *quantum-inspired*; local
statevector QAOA is a *quantum algorithm simulation*, not a hardware run. None of these
labels implies advantage. A defensible advantage claim
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
