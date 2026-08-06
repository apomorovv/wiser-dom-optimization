# Challenge requirement matrix

This matrix maps every task in the five-page challenge brief to an implemented and
reviewable artifact.

| Brief requirement | Implementation | Evidence |
|---|---|---|
| Explain DOM and combinatorial trade-offs | Business framing, exact-versus-heuristic and hardware discussion | `README.md`, `docs/hybrid_algorithm.md`, `docs/research_basis.md` |
| Document every POC table/file | Strict file inventory and field-level transformations | `docs/challenge_data.md`, `docs/poc_data_mapping.md`, `docs/data_dictionary.md` |
| Default-assignment baseline | Deterministic shared-resource default allocator | `src/domopt/baselines.py`, notebook solver comparison |
| Greedy/sequential baseline | Residual-resource sequential reassignment | `src/domopt/baselines.py`, notebook solver comparison |
| Report objective, fill, diverts, penalty, shipping | Common independent metrics for every method | `src/domopt/metrics.py`, `src/domopt/objective.py`, experiment CSV |
| Deterministic binary or column-selection model | Exact MILP with assignment columns and SKU quantities | `src/domopt/classical.py`, `docs/mathematical_formulation.md` |
| Define variables, objective, and constraints | Complete thresholded-penalty and load-cohesion formulation | `docs/mathematical_formulation.md` |
| Implement on tractable real-data subset | Deterministic assignment-group subsets with whole-load expansion | `src/domopt/poc.py`, `src/domopt/experiments.py`, notebook |
| Explain candidate generation and validation | Source adapter, Pareto pruning, independent validator | `docs/poc_data_mapping.md`, `src/domopt/validation.py` |
| Compare best candidate with identical checks | Default, greedy, exact MILP, and hybrid use one objective and validator | notebook `solver_comparison` experiment |
| Estimate variable/qubit growth | Bounded local QUBO width and MILP scaling discussion | `docs/hybrid_algorithm.md`, size-scaling plots |
| Analyze runtime, complexity, and robustness | Size, candidate, noise, and ablation experiments | notebook and `src/domopt/experiments.py` |
| Propose scalability improvements | Atomic greedy, copy-on-write inventory, Pareto pruning, candidate limits, conflict batches, local recourse | implementation and `docs/hybrid_algorithm.md` |
| 6–10 page report | Intentionally deferred until reviewed full-profile results exist | `reports/README.md` |
| Runnable notebook/repository | Installation and execution instructions | `README.md`, `notebooks/nestle_challenge_experiments.ipynb` |
| 5–7 slide presentation | Intentionally deferred until reviewed full-profile results exist | `reports/README.md` |
| One-page planner view | Template retained; data-specific artifact deferred | `reports/planner_view_template.md` |
| Compare exact cover and deterministic model (optional) | Local assignment QUBO/column selection compared with exact MILP | hybrid workflow and sampler ablation |
| Repair/post-processing (optional) | One-hot repair plus exact MILP recourse | `src/domopt/hybrid.py` |
| Explore batching (optional) | Conflict-versus-random batch ablation | notebook `batch_strategy_ablation` |
| Add uncertainty (optional) | Inventory-shock and coefficient-noise scenarios | notebook experiments |
| Planner dashboard (optional) | Aggregate-only Streamlit copilot | `apps/planner_copilot.py` |
| GPU acceleration study (optional) | Synthetic CPU/GPU QUBO-scoring crossover and capability audit | notebook and `src/domopt/hardware.py` |
| Remote QPU test (optional) | Opt-in D-Wave adapter using generated synthetic coefficients only | notebook and `src/domopt/experiments.py` |
| Privacy-safe public package | No raw challenge tables or identifiers committed | `.gitignore`, `docs/privacy.md`, aggregate-output guards |

## Implemented experiment grid

| Experiment | Full-profile levels | Question answered |
|---|---|---|
| Size scaling | 8, 20, 50, 100, 250, and 372 assignment groups | How do runtime, model width, and solution quality grow? |
| Penalty sensitivity | 0.25×, 0.5×, 1×, 2×, and 4× on penalty-active groups | Are routing choices stable under economic weighting? |
| QUBO-penalty sensitivity | 4 one-hot multipliers × 4 conflict multipliers | Are constraint surrogates calibrated rather than guessed? |
| Candidate sensitivity | 1, 2, 4, 6 options per decision unit | Is extra search breadth worth its cost? |
| Inventory shocks | 0%, 10%, 25%, and 40% | How gracefully does the solver respond to shortfalls? |
| Seed/noise | 4 seeds × 4 coefficient-noise levels | Is the local sampler ranking robust? |
| Pareto ablation | pruning off/on | Does safe column reduction improve width/runtime without loss? |
| Batch ablation | random/conflict | Does resource-aware composition improve neighborhood search? |
| Sampler control | random/simulated annealing | Does the annealing sampler add value beyond weak random search? |
| Synthetic coordination control | greedy/exact/hybrid | Can the architecture revisit a deliberately coupled greedy trap? |

Coefficient perturbation is not physical QPU noise. A hardware claim requires an
approved QPU run with matched QUBOs, end-to-end timing, embedding statistics,
multiple trials, and uncertainty bounds.
