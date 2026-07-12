# Experiment registry

`experiments/registry.csv` is the append-only index of reproducible method runs.

## One row per method

An integrated comparison using default, greedy, and classical methods creates three registry rows with the same `experiment_id` and different `method` values.

## Required columns

```text
experiment_id
run_id
timestamp_utc
dataset_id
schema_version
assumption_version
method
seed
git_commit
config_path
run_path
feasible
objective_value
case_fill_rate
value_fill_rate
shipping_cost
penalty_cost
reassigned_orders
runtime_seconds
optimality_gap
notes
```

## Example

```text
tiny-v0,tiny-v0-default-7,2026-07-12T22:00:00Z,tiny,0.1.0,v0,default,7,<sha>,<config>,runs/tiny/baselines/default,true,<value>,<rate>,<rate>,<cost>,<cost>,<count>,<time>,,deterministic default baseline
tiny-v0,tiny-v0-greedy-7,2026-07-12T22:00:01Z,tiny,0.1.0,v0,greedy,7,<sha>,<config>,runs/tiny/baselines/greedy,true,126,1.0,1.0,4,0,1,<time>,,sequential greedy
tiny-v0,tiny-v0-classical-7,2026-07-12T22:00:02Z,tiny,0.1.0,v0,classical,7,<sha>,<config>,runs/tiny/classical,true,126,1.0,1.0,4,0,1,<time>,0.0,proven optimum
```

The default-baseline numerical values depend on the documented deterministic allocation sequence and must be produced by code rather than copied from this README.

## Rules

- Do not overwrite old rows.
- Do not hand-correct objective values.
- Compared methods must share one dataset fingerprint.
- Record infeasible runs instead of silently discarding them.
- Store detailed logs in `runs/`, not in the registry.
- Use aggregate or synthetic information only.

