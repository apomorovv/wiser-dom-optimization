# WISER–Nestlé Distributed Order Management optimizer

This repository is a complete, privacy-safe submission for the 2026 WISER–Nestlé
Distributed Order Management (DOM) challenge. It reads the supplied proof-of-concept
data, constructs a validated optimization instance, compares transparent classical
baselines with an exact solver and a hybrid quantum-classical search, and produces a
notebook, planner copilot, report, planner view, and seven-slide presentation.

The central safety rule is simple: a QUBO sample is only a proposal. Exact
mixed-integer recourse rebuilds SKU quantities, an independent validator checks all
hard constraints, and a move is accepted only when it is feasible and improves the
current solution. A weak or noisy sampler can consume runtime, but it cannot worsen
the returned incumbent.

## What is implemented

- strict readability checks for all ten supplied challenge files;
- a real-data adapter for orders, inventory, lanes, dock availability, throughput
  observations, the equations document, workbook, and two reference outputs;
- source-accurate case conversion and thresholded order-penalty logic;
- load-cohesive candidate generation, five-day protected ATP, shipping lead times,
  working-day PGI adjustment, forecast eligibility, and dock usage;
- deterministic default and sequential-greedy baselines;
- an exact SciPy/HiGHS mixed-integer linear program (MILP);
- bounded large-neighborhood search with QUBO assignment proposals and exact MILP
  fulfillment recourse;
- exact enumeration, random sampling, simulated annealing, and optional D-Wave
  backends behind an explicit privacy gate;
- an independent objective evaluator and feasibility validator;
- all requested experiments plus two additional solver controls;
- a runnable Jupyter notebook and aggregate-only Streamlit planner copilot; and
- submission-ready Markdown, PDF, and PowerPoint artifacts.

No QPU experiment has been run and no quantum advantage is claimed.

## Verified supplied-data audit

The repository intentionally publishes only non-identifying aggregates.

| Measure | Verified value |
|---|---:|
| Supplied artifacts opened | 10 of 10 |
| Order-level output rows | 1,109 |
| Order-SKU output rows | 25,193 |
| Named loads | 614 |
| Assignment groups after singleton handling | 631 |
| Diverted orders in supplied output | 3 |
| Requested cases | 2,554,440 |
| Selected fulfilled cases | 2,413,937 |
| Selected case fill | 94.4997% |

The input order table contains 25,193 order-SKU rows. The capacity-planning table
contains 377,504 DC-SKU-date rows, shipping contains 12,922 lanes, dock contains 480
rows, and throughput contains 530 observed-utilization rows. The workbook is a
561-row worked example and the Word document contains the source business equations.
The output files are used only for reconciliation; they are never treated as labels
for optimization.

## Solver architecture

```mermaid
flowchart TD
    A["Readable challenge bundle"] --> B["Canonical orders, lines, candidates, resources"]
    B --> C["Feasible default or greedy incumbent"]
    C --> D["Conflict-aware bounded neighborhood"]
    D --> E["QUBO assignment sampler"]
    E --> F["One-hot repair and exact MILP recourse"]
    F --> G{"Validated strict improvement?"}
    G -->|Yes| C
    G -->|No| D
```

The exact model assigns each order to one eligible DC/PGI option or leaves it
unassigned. Partial fulfillment is allowed at the selected DC, but an order cannot
split across DCs. Orders sharing a source load move together. The objective is:

$$
\text{fulfilled value}-\text{thresholded unmet penalty}-\text{shipping cost}.
$$

Hard constraints cover demand balance, candidate eligibility, projected inventory,
dock capacity, optional scenario capacity, the diversion-uplift rule, pallet/case
accounting, and assignment-group cohesion. See the
[mathematical formulation](docs/mathematical_formulation.md) and
[source mapping](docs/poc_data_mapping.md).

## Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,notebook,app]"
```

Install `.[qpu]` only inside an approved environment configured for D-Wave access.
The default workflow is entirely local.

## Run the challenge study

Keep the supplied raw files outside Git. Point the command to a directory containing
the exact filenames listed in [challenge data](docs/challenge_data.md).

```bash
python scripts/run_challenge_study.py \
  --bundle-dir /approved/path/to/challenge-files \
  --profile full \
  --output runs/challenge-study/aggregate_results.csv
```

The command first opens every required file and stops on the first missing,
unreadable, empty, or structurally invalid artifact. `--profile smoke` runs a short
development grid; `--profile full` runs the submission grid.

## Run the notebook

```bash
jupyter lab notebooks/nestle_challenge_experiments.ipynb
```

Set `BUNDLE_DIR` in the first configuration cell. The notebook performs:

1. strict bundle and output reconciliation audits;
2. default, greedy, exact MILP, and hybrid comparison;
3. size scaling;
4. penalty-weight sensitivity;
5. candidate-count sensitivity;
6. inventory shocks;
7. sampler seed and local coefficient-noise sensitivity;
8. Pareto-pruning ablation;
9. random-versus-conflict batching ablation;
10. random-versus-simulated-annealing sampler ablation; and
11. a synthetic coordination control with a known exact reference.

All persisted experiment rows are aggregate-only.

## Run the planner copilot

The copilot is useful because the experiment matrix is multi-dimensional and planners
need explanations, not raw solver logs. It is deliberately rules-based: no external
language model receives the data, and identifier-bearing uploads are rejected.

```bash
python -m streamlit run apps/planner_copilot.py
```

It explains solver comparisons, scaling, sensitivities, ablations, and the limits of
the evidence. See [app instructions](apps/README.md).

## Reproduce the known-optimum test

```bash
python scripts/make_tiny_instance.py
python scripts/run_experiment.py \
  --data-dir data/synthetic/tiny \
  --methods default greedy classical hybrid \
  --hybrid-config configs/hybrid_tiny.yaml \
  --output-dir runs/tiny-comparison \
  --experiment-id tiny-comparison
pytest
```

The tiny instance has a proven optimum of 126 synthetic units. The exact MILP and
hybrid exact-QUBO configuration both reproduce it.

## Repository map

| Path | Purpose |
|---|---|
| `src/domopt/poc.py` | strict real-data audit and source-to-canonical adapter |
| `src/domopt/classical.py` | exact MILP and fixed-assignment recourse |
| `src/domopt/hybrid.py` | conflict batching, QUBO search, repair, and acceptance |
| `src/domopt/validation.py` | independent feasibility authority |
| `src/domopt/experiments.py` | complete experiment matrix |
| `notebooks/` | runnable challenge analysis |
| `apps/` | aggregate-only planner copilot |
| `docs/` | data, assumptions, formulation, algorithm, and requirement mapping |
| `reports/` | submission sources and rendered deliverables |
| `tests/` | unit, model, sampler, audit, privacy, and Markdown checks |

## Submission artifacts

- [challenge requirement matrix](docs/challenge_requirements.md)
- [two-page business and technical summary](reports/business_technical_summary.md)
- [6–10 page technical report](reports/technical_report.md)
- [one-page planner view](reports/planner_view.md)
- [presentation source](reports/presentation.md)
- `reports/wiser_dom_submission_deck.pptx`

Rendered PDFs are generated from the Markdown sources and kept next to them.

## Privacy and interpretation

Raw operational rows, identifiers, exact DC-level resources, and commercial totals
are excluded from Git. QUBO variable labels are integers for optional remote runs,
but coefficients can still encode sensitive economics; remote execution therefore
requires explicit approval. See [privacy policy](docs/privacy.md).

The supplied throughput file reports observed utilization rather than a documented
maximum. It is not enforced as a real hard limit. Experiments may add explicit
scenario headroom, and those rows are labeled as scenarios. Likewise, local QUBO
coefficient perturbation is a robustness test, not physical QPU noise.

