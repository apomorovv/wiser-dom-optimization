# Remote jobs

This directory documents remote-server and scheduler commands. Do not store data, credentials, private keys, or logs containing restricted records here.

## Direct execution

From the repository root:

```bash
conda activate wiser-dom
python -m pip install -e ".[dev]"
pytest -q
```

Run the tiny comparison:

```bash
python scripts/make_tiny_instance.py \
  --output-dir data/synthetic/tiny

python scripts/run_experiment.py \
  --data-dir data/synthetic/tiny \
  --methods default greedy classical hybrid \
  --hybrid-config configs/hybrid_tiny.yaml \
  --output-dir runs/tiny/all_methods \
  --experiment-id tiny-v1 \
  --seed 7
```

## Persistent terminal

```bash
tmux new -s wiser-dom
```

Detach with `Ctrl-b d` and reconnect with:

```bash
tmux attach -t wiser-dom
```

## Logging

```bash
mkdir -p runs/example
python scripts/run_experiment.py \
  --data-dir data/synthetic/tiny \
  --methods default greedy classical hybrid \
  --hybrid-config configs/hybrid_tiny.yaml \
  --output-dir runs/example \
  --experiment-id tiny-v1 \
  --seed 7 \
  > runs/example/stdout.log 2>&1
```

## Compute selection

Classical baselines and MILP solving generally do not require a GPU. Reserve GPU resources for quantum simulation only when the chosen backend benefits from them.

Record hostname, CPU/core count, GPU model when used, Python/package versions, and wall-clock runtime in run metadata.

## Scheduler template

```bash
#!/usr/bin/env bash
set -euo pipefail

cd /path/to/repository
source /path/to/conda.sh
conda activate wiser-dom

python -m pip install -e ".[dev]"
pytest -q tests/test_tiny_optimum.py

python scripts/run_experiment.py \
  --data-dir data/synthetic/tiny \
  --methods default greedy classical hybrid \
  --hybrid-config configs/hybrid_tiny.yaml \
  --output-dir runs/tiny/all_methods \
  --experiment-id tiny-v1 \
  --seed 7
```

Do not submit real-data jobs to an unapproved external platform.
