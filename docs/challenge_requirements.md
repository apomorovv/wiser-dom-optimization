# Challenge requirement matrix

This matrix maps every task in the five-page challenge brief to an implemented and
reviewable artifact. ``Implemented`` means the code path and automated checks exist;
it does not imply that the final full-profile evidence or submission artifact has
already been generated and reviewed.

| Brief requirement | Implementation | Evidence |
|---|---|---|
| Explain DOM and combinatorial trade-offs | Business framing, exact-versus-heuristic and hardware discussion | `README.md`, `docs/hybrid_algorithm.md`, `docs/research_basis.md` |
| Separate two-page business/technical summary | Required deliverable explicitly tracked; intentionally deferred until reviewed full-profile evidence exists | `reports/README.md` |
| Document every POC table/file | Strict file inventory, required-column/type/domain audit, inventory-identity check, and field-level transformations | `docs/challenge_data.md`, `docs/poc_data_mapping.md`, `docs/data_dictionary.md`, `src/domopt/poc.py` |
| Default-assignment baseline | Deterministic shared-resource default allocator | `src/domopt/baselines.py`, notebook solver comparison |
| Greedy/sequential baseline | Residual-resource sequential reassignment | `src/domopt/baselines.py`, notebook solver comparison |
| Strong scalable classical controls | Greedy plus exact fixed-assignment quantity polish and adaptive conflict-aware exact-MILP LNS | `src/domopt/baselines.py`, `src/domopt/hybrid.py`, solver and scaling comparisons |
| Report objective, fill, diverts, penalty, shipping | Common independent metrics for every method | `src/domopt/metrics.py`, `src/domopt/objective.py`, experiment CSV |
| Deterministic binary or column-selection model | Exact MILP with assignment columns and SKU quantities | `src/domopt/classical.py`, `docs/mathematical_formulation.md` |
| Define variables, objective, and constraints | Complete thresholded-penalty and load-cohesion formulation | `docs/mathematical_formulation.md` |
| Implement on tractable real-data subset | Deterministic assignment-group subsets with whole-load expansion | `src/domopt/poc.py`, `src/domopt/experiments.py`, notebook |
| Explain candidate generation and validation | Strict source adapter, explicit network-versus-default-DC scope, optional Pareto pruning, and independent validator | `docs/poc_data_mapping.md`, `src/domopt/poc.py`, `src/domopt/validation.py` |
| Compare best candidate with identical checks | Default, greedy, polished greedy, exact LNS, exact MILP, and hybrid use one objective and validator | notebook `solver_comparison` experiment |
| Estimate variable/qubit growth | Bounded local QUBO width plus local/global MILP variable and constraint counts | `docs/hybrid_algorithm.md`, size-scaling experiment |
| Analyze runtime, complexity, and robustness | Repeated real and synthetic scaling, candidate/scope, noise, and ablation experiments | notebook and `src/domopt/experiments.py` |
| Propose scalability improvements | Atomic greedy, exact quantity polish, adaptive exact LNS, cached conflict indexes, candidate limits, conflict batches, and bounded local recourse; heuristic pruning is optional | implementation and `docs/hybrid_algorithm.md` |
| 6–10 page report | Intentionally deferred until reviewed full-profile results exist | `reports/README.md` |
| Runnable notebook/repository | Installation/execution instructions and content-addressed checkpoints that reject changed profile, problem, configuration, source state, or table schema | `README.md`, `notebooks/nestle_challenge_experiments.ipynb`, `src/domopt/checkpoints.py` |
| 5–7 slide presentation | Intentionally deferred until reviewed full-profile results exist | `reports/README.md` |
| One-page planner view | Template retained; generator now uses the canonical thresholded penalty, load-group totals, and delivery-date status; reviewed data-specific artifact remains deferred | `reports/planner_view_template.md`, `src/domopt/planner.py` |
| Compare exact cover and deterministic model (optional) | **Not implemented.** The local assignment QUBO is a search neighborhood, not a same-instance exact-cover-versus-deterministic-LP study | Optional future experiment |
| Repair/post-processing (optional) | One-hot repair plus exact MILP recourse | `src/domopt/hybrid.py` |
| Explore batching (optional) | Conflict-versus-random batch ablation | notebook `batch_strategy_ablation` |
| Add uncertainty (optional) | Inventory-shock and coefficient-noise scenarios | notebook experiments |
| Planner dashboard (optional) | Aggregate-only Streamlit copilot | `apps/planner_copilot.py` |
| GPU acceleration study (optional) | Synthetic CPU/GPU QUBO-scoring crossover and capability audit | notebook and `src/domopt/hardware.py` |
| Remote QPU test (optional) | Opt-in IBM hardware study using generated synthetic coefficients only, least-busy discovery, repeated mitigation variants, and exact references | notebook, `scripts/run_ibm_hardware_study.py`, and `src/domopt/experiments.py` |
| Reproducible evidence | Aggregate rows record problem/bundle hashes, schema and assumption versions, objective version, commit, dirty state, source-state hash, configuration, and seed | `src/domopt/experiments.py`, `src/domopt/pipeline.py`, `src/domopt/checkpoints.py` |
| Privacy-safe public package | No raw challenge tables or identifiers committed; public result writers reject identifier-like columns | `.gitignore`, `docs/privacy.md`, aggregate-output guards |

## Implemented experiment grid

| Experiment | Full-profile levels | Question answered |
|---|---|---|
| Repeated real size scaling | 8, 20, 50, 100, 250, and 372 assignment groups; three repetitions per method/size | How do runtime, model width, dispersion, and solution quality grow on nested real subsets? |
| Repeated synthetic scaling | 20, 50, 100, 250, and 500 independently generated groups; three repetitions | Which runtime/quality trend remains when size is varied under a controlled generator? |
| Candidate-DC scope | focus-order default DCs versus shipping/inventory/dock network intersection | What breadth and value are lost by restricting alternatives to focus-order default DCs? |
| Penalty sensitivity | 0.25×, 0.5×, 1×, 2×, and 4× on penalty-active groups | Are routing choices stable under economic weighting? |
| QUBO-penalty sensitivity | 4 one-hot multipliers × 4 conflict multipliers | Are constraint surrogates calibrated rather than guessed? |
| Candidate sensitivity | 1, 2, 4, 6 options per decision unit | Is extra search breadth worth its cost? |
| Inventory shocks | 0%, 10%, 25%, and 40% | How gracefully does the solver respond to shortfalls? |
| Seed/coefficient noise | 4 seeds × 4 coefficient-noise levels | Is the local sampler ranking robust? |
| QAOA readout proxy | 4 seeds × 4 measurement bit-flip levels | How does a simple readout channel change raw feasibility and validated quality? |
| Pareto ablation | heuristic pruning off/on across 4 seeds | Does optional column reduction improve observed width/runtime without loss on the tested instance? |
| Batch ablation | random/conflict across 4 seeds | Does resource-aware composition improve neighborhood search? |
| Sampler control | feasible exact/random/simulated annealing/Dicke-XY QAOA simulation | Which proposal generators find the coupled move, and which preserve one-hot feasibility? |
| Synthetic coordination control | greedy/polished greedy/exact LNS/exact/hybrid | Can each architecture revisit a deliberately coupled greedy trap without misattributing exact quantity recourse to sampling? |

Coefficient perturbation and the local readout channel are not full physical QPU noise. A hardware claim requires an
approved QPU run with matched QUBOs, end-to-end timing, transpilation statistics,
multiple trials, and uncertainty bounds.

The real size subsets are nested and deliberately shortage-focused, so their quality
levels also reflect changing business composition. The independently generated
synthetic scaling control separates that composition caveat from algorithmic growth.
No result table in this repository is final evidence until its checkpoint manifest and
provenance fields match the requested run and the full-profile output has been reviewed.
