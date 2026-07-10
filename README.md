# WISER Distributed Order Management Optimization

Hybrid of classical and quantum-inspired optimization methods for the
WISER 2026 Distributed Order Management challenge.

## Problem

Assign focus orders to eligible distribution centers while balancing:

- fulfillment value,
- unfulfilled-demand penalties,
- shipping cost,
- inventory,
- throughput,
- dock capacity,
- case-pick capacity,
- pallet-pick capacity,
- PGI and delivery-date feasibility.

## Repository layout

- `src/domopt/`: reusable implementation
- `notebooks/`: exploratory and presentation notebooks
- `configs/`: standard run configurations
- `scripts/`: command-line entry points
- `tests/`: correctness and feasibility tests
- `experiments/`: tracked experiment definitions
- `runs/`: generated run artifacts; not committed
- `results/`: sanitized tables and figures
- `reports/`: final report, presentation, and planner view

## Installation

```bash
conda env create -f environment.yml
conda activate wiser-dom
python -m pip install -e ".[classical,quantum,notebook,dev]"
