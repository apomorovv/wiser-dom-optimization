# Challenge requirement matrix

This matrix maps every task in the five-page challenge brief to an implemented and
reviewable artifact.

| Brief requirement | Implementation | Evidence |
|---|---|---|
| Explain DOM and combinatorial trade-offs | Business framing, exact-versus-heuristic and hardware discussion | `reports/business_technical_summary.md`, `reports/business_technical_summary.pdf` |
| Document every POC table/file | Strict file inventory and field-level transformations | `docs/challenge_data.md`, `docs/poc_data_mapping.md`, `docs/data_dictionary.md` |
| Default-assignment baseline | Deterministic shared-resource default allocator | `src/domopt/baselines.py`, notebook solver comparison |
| Greedy/sequential baseline | Residual-resource sequential reassignment | `src/domopt/baselines.py`, notebook solver comparison |
| Report objective, fill, diverts, penalty, shipping | Common independent metrics for every method | `src/domopt/metrics.py`, `src/domopt/objective.py`, experiment CSV |
| Deterministic binary or column-selection model | Exact MILP with assignment columns and SKU quantities | `src/domopt/classical.py`, `docs/mathematical_formulation.md` |
| Define variables, objective, and constraints | Complete thresholded-penalty and load-cohesion formulation | `docs/mathematical_formulation.md` |
| Implement on tractable real-data subset | Deterministic shortage-based 8/20/50-order subsets with whole-load expansion | `src/domopt/poc.py`, `src/domopt/experiments.py`, notebook |
| Explain candidate generation and validation | Source adapter, Pareto pruning, independent validator | `docs/poc_data_mapping.md`, `src/domopt/validation.py` |
| Compare best candidate with identical checks | Default, greedy, exact MILP, and hybrid use one objective and validator | notebook `solver_comparison` experiment |
| Estimate variable/qubit growth | Bounded local QUBO width and MILP scaling discussion | `docs/hybrid_algorithm.md`, technical report |
| Analyze runtime, complexity, and robustness | Size, candidate, noise, and ablation experiments | notebook and `src/domopt/experiments.py` |
| Propose scalability improvements | Pareto pruning, candidate limits, conflict batches, local recourse | implementation and report |
| 6–10 page report | Submission-ready technical report | `reports/technical_report.md`, `reports/technical_report.pdf` |
| Runnable notebook/repository | Installation and execution instructions | `README.md`, `notebooks/nestle_challenge_experiments.ipynb` |
| 5–7 slide presentation | Seven-slide reviewed deck | `reports/wiser_dom_submission_deck.pptx` |
| One-page planner view | Interpretation, evidence, sign-off, and limits | `reports/planner_view.md`, `reports/planner_view.pdf` |
| Compare exact cover and deterministic model (optional) | Local assignment QUBO/column selection compared with exact MILP | hybrid workflow and sampler ablation |
| Repair/post-processing (optional) | One-hot repair plus exact MILP recourse | `src/domopt/hybrid.py` |
| Explore batching (optional) | Conflict-versus-random batch ablation | notebook `batch_strategy_ablation` |
| Add uncertainty (optional) | Inventory-shock and coefficient-noise scenarios | notebook experiments |
| Planner dashboard (optional) | Aggregate-only Streamlit copilot | `apps/planner_copilot.py` |
| Privacy-safe public package | No raw challenge tables or identifiers committed | `.gitignore`, `docs/privacy.md`, aggregate-output guards |

## Implemented experiment grid

| Experiment | Full-profile levels | Question answered |
|---|---|---|
| Size scaling | 8, 20, 50 requested orders | How do runtime, model width, and solution quality grow? |
| Penalty sensitivity | 0.5×, 1×, 2× | Are routing choices stable under economic weighting? |
| Candidate sensitivity | 1, 2, 4, 6 options per decision unit | Is extra search breadth worth its cost? |
| Inventory shocks | 0%, 10%, 25% | How gracefully does the solver respond to shortfalls? |
| Seed/noise | 4 seeds × 3 coefficient-noise levels | Is the local sampler ranking robust? |
| Pareto ablation | pruning off/on | Does safe column reduction improve width/runtime without loss? |
| Batch ablation | random/conflict | Does resource-aware composition improve neighborhood search? |
| Sampler control | random/simulated annealing | Does the annealing sampler add value beyond weak random search? |
| Synthetic coordination control | greedy/exact/hybrid | Can the architecture revisit a deliberately coupled greedy trap? |

Coefficient perturbation is not physical QPU noise. A hardware claim requires an
approved QPU run with matched QUBOs, end-to-end timing, embedding statistics,
multiple trials, and uncertainty bounds.

