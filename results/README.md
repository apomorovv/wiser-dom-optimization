# Results

This directory is reserved for reviewed aggregate tables, figures, and planner-facing
summaries selected from validated runs. Raw per-run artifacts belong in `runs/`.
The existing `scaling_synthetic.*` pair is a historical development snapshot from the
pre-LNS experiment schema; it is not current challenge evidence and must be regenerated
before inclusion in a report.

## Suggested structure

```text
results/
  tables/
    baseline_comparison.csv
    scaling_synthetic.csv
    sensitivity_summary.csv
  figures/
    objective_by_method.png
    runtime_scaling.png
    feasible_sample_rate.png
  planner/
    synthetic_planner_example.md
```

## Required provenance

Every final result must identify:

- experiment ID;
- dataset ID;
- git commit;
- assumption version;
- method configuration;
- source run directories.

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

`scripts/run_scaling_study.py` creates a repeated `scaling_synthetic.csv` from
independently generated data. Do not mix those objectives with real challenge evidence.

