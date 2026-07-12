# Results

This directory contains final aggregate tables, figures, and planner-facing summaries selected from validated runs. Raw per-run artifacts belong in `runs/`.

## Suggested structure

```text
results/
  tables/
    baseline_comparison.csv
    scaling_summary.csv
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

