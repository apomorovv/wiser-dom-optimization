# Reproducibility and privacy protocol

## Execution profiles

`smoke` is a quick installation and pipeline check. `full` is the reviewed submission
grid with repeated real and synthetic scaling, sensitivity studies, robustness tests,
sampler ablations, and the coordination control. IBM hardware has separate `quick` and
`presentation` profiles because QPU access is optional and remote.

The reviewed `full` profile uses a 100-order common comparison, reaches all 372 real
assignment groups, scales generated greedy and exact-LNS workloads to 100,000 orders,
and scales the hybrid workflow to 20,000 generated orders. The final `presentation`
hardware profile contains 18 jobs at 8,192 shots each. The notebook records those final
settings while leaving remote submission disabled by default.

## Fair comparison rules

All compared solvers receive the same normalized problem, candidate universe, business
objective, time limit where applicable, seed policy, metrics, and independent validator.
Polished greedy and hybrid search separate assignment changes from quantity recourse.
Exact MILP backend comparison compiles one matrix and changes only the backend adapter.

Results are interpreted in this order:

1. validation and residuals;
2. normalized objective and operational metrics;
3. runtime and model size;
4. optimality or search evidence;
5. native sampler quality;
6. limitations and statistical uncertainty.

## Checkpoints and manifests

Each experiment checkpoint has a CSV and JSON manifest containing:

- experiment and checkpoint schema versions;
- profile and complete configuration;
- problem and bundle fingerprints;
- package and dependency versions;
- Git commit;
- computation-relevant source hash;
- relevant-source and full-worktree dirty flags;
- CSV row count, columns, and SHA-256 hash.

Notebook output and execution-count changes do not affect the computation source hash.
Code-cell changes do. This avoids source drift during a Run All operation while keeping
general worktree dirtiness visible.

## Randomness

Synthetic generators, sampler reads, QAOA parameter starts, and transpiler trials use
recorded seeds. Hardware repetitions hold angle and transpiler seeds fixed so device
variation is not confounded with parameter or compilation variation. Stochastic results
are summarized across repetitions; exact or deterministic controls remain explicit.

## Validation contract

No result enters an aggregate comparison unless the independent validator confirms:

- one modeled assignment outcome per order;
- group-cohesive routing;
- nonnegative integral quantities;
- exact demand balance;
- eligible DC and date choices;
- cumulative inventory feasibility;
- capacity feasibility;
- diversion-improvement compliance;
- consistent output schemas.

Numeric residual maxima are recorded even when categorical checks pass.

## Privacy contract

Real source files, row-level outputs, evidence CSV/JSON files, manifests, IBM job
tables, checkpoints, identifiers, source-currency commercial totals, and charts of
absolute commercial totals are local-only. Committed reports and figures contain only
reviewed aggregate, normalized, or synthetic evidence. Synthetic controls are clearly
labeled and never represented as business impact.

Remote IBM execution is disabled by default. The provided hardware study constructs an
independent synthetic instance and sends only its compiled circuit after explicit
approval. A real-data QUBO must not be sent without separate data-owner authorization.
In Qiskit, an IBM `backend` names the quantum processor/execution target; QAOA is the
optimization proposal method.

## Reproduction commands

```bash
python -m pip install -e ".[full]"
python -m pytest
python scripts/run_challenge_study.py \
  --bundle-dir data/raw/nestle_challenge \
  --profile smoke
```

After the smoke profile succeeds, replace `smoke` with `full`. The notebook is an
equivalent execution surface and adds the optional Gurobi comparison and hardware cells.
Publication figures are regenerated from audited local table directories with
`scripts/create_submission_figures.py`; the source tables remain outside the branch.
