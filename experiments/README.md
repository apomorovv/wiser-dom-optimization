# Experiment registry

`experiments/registry.csv` is the append-only index of reproducible method runs.

## One row per method

An integrated comparison using default, greedy, classical, and hybrid methods creates
four registry rows with the same `experiment_id` and different `method` values.

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
tiny-v1,tiny-v1-default-7,2026-08-05T22:00:00Z,tiny,0.2.0,v1,default,7,<sha>,<config>,runs/tiny/default,true,-50,<rate>,<rate>,0,<cost>,0,<time>,,deterministic default baseline
tiny-v1,tiny-v1-greedy-7,2026-08-05T22:00:01Z,tiny,0.2.0,v1,greedy,7,<sha>,<config>,runs/tiny/greedy,true,126,1.0,1.0,4,0,1,<time>,,sequential greedy
tiny-v1,tiny-v1-classical-7,2026-08-05T22:00:02Z,tiny,0.2.0,v1,classical,7,<sha>,<config>,runs/tiny/classical,true,126,1.0,1.0,4,0,1,<time>,0.0,proven optimum
tiny-v1,tiny-v1-hybrid-7,2026-08-05T22:00:03Z,tiny,0.2.0,v1,hybrid,7,<sha>,<config>,runs/tiny/hybrid,true,126,1.0,1.0,4,0,1,<time>,,bounded exact-QUBO test
```

The default-baseline numerical values depend on the documented deterministic allocation sequence and must be produced by code rather than copied from this README.

## Rules

- Do not overwrite old rows.
- Do not hand-correct objective values.
- Compared methods must share one dataset fingerprint.
- Record infeasible runs instead of silently discarding them.
- Store detailed logs in `runs/`, not in the registry.
- Use aggregate or synthetic information only.

