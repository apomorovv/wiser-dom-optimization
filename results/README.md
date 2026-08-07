# Results

This directory is reserved for aggregate tables, figures, and planner-facing summaries.
Raw per-order solver artifacts still belong in ignored `runs/` directories.
Historical pre-LNS snapshots have been removed; only newly generated, manifest-verified
evidence should be considered for publication.

## Suggested structure

```text
results/
  challenge-study/
    cli/
      full/
        aggregate_results.csv
        aggregate_results.manifest.json
        figures/
    notebook/
      <profile>/
        aggregate_results.csv
        aggregate_results.manifest.json
        tables/
        figures/
      ibm-quick/
        tables/
        figures/
  planner/
    synthetic_planner_example.md
```

`challenge-study/` is ignored by Git until a human selects publication-safe aggregates.
Separating `cli` from `notebook` prevents either execution surface from loading or
overwriting the other's checkpoints.

## Required provenance

Every final result must identify:

- experiment ID;
- dataset ID;
- git commit;
- assumption version;
- method configuration;
- source-state hash and dirty/clean provenance;
- runtime-environment versions; and
- stable profile directory.

## Required comparison columns

```text
method
feasible
objective_value
fulfilled_value
penalty_cost
shipping_cost
case_fill_rate
value_fill_rate
reassigned_orders
runtime_seconds
optimality_gap
```

Do not rank infeasible solutions by objective value as though they were valid.

## Planner output

A planner-facing recommendation should state:

- order;
- default DC;
- recommended DC;
- default and recommended fill;
- incremental fulfilled cases or value;
- penalty avoided;
- shipping-cost change;
- net objective change;
- delivery-date status;
- main binding constraint or reason;
- next-best feasible option when available.

Use synthetic or approved anonymized examples only.

`scripts/run_scaling_study.py` can create a repeated synthetic scaling table in a chosen
output directory. Do not mix those objectives with real challenge evidence.
