# Synthetic data

Synthetic instances support unit tests, solver debugging, examples, and public demonstrations. They must be independently generated and must not reproduce operational identifiers or confidential numerical records.

## Generate the standard tiny instance

```bash
python scripts/make_tiny_instance.py \
  --output-dir data/synthetic/tiny
```

## Exact tiny-instance definition

The instance has one PGI date $t_1$, two orders, two SKUs, and two DCs.

### Orders

| `order_id` | `default_dc` | `requested_delivery_date` |
|---|---|---|
| `O1` | `D1` | `2026-07-15` |
| `O2` | `D1` | `2026-07-15` |

### Order lines

All values are synthetic monetary units.

| `order_id` | `sku_id` | `demand_cases` | `unit_value` | `penalty_per_unfilled_case` |
|---|---|---:|---:|---:|
| `O1` | `A` | 4 | 10 | 20 |
| `O1` | `B` | 2 | 10 | 20 |
| `O2` | `A` | 3 | 10 | 20 |
| `O2` | `B` | 4 | 10 | 20 |

### Cumulative inventory at $t_1$

| `dc_id` | `sku_id` | `date` | `cumulative_available_cases` |
|---|---|---|---:|
| `D1` | `A` | `2026-07-14` | 3 |
| `D1` | `B` | `2026-07-14` | 4 |
| `D2` | `A` | `2026-07-14` | 4 |
| `D2` | `B` | `2026-07-14` | 2 |

### Feasible assignment candidates

| `candidate_id` | `order_id` | `dc_id` | `pgi_date` | `shipping_cost` | `is_default` |
|---|---|---|---|---:|---|
| `O1_D1_T1` | `O1` | `D1` | `2026-07-14` | 0 | `true` |
| `O1_D2_T1` | `O1` | `D2` | `2026-07-14` | 4 | `false` |
| `O2_D1_T1` | `O2` | `D1` | `2026-07-14` | 0 | `true` |
| `O2_D2_T1` | `O2` | `D2` | `2026-07-14` | 4 | `false` |

Each order also has a no-assignment option with zero shipping cost and zero fulfillment.

## Exact optimum

The complete demands are

$$
Q_{O1,A}=4,\quad Q_{O1,B}=2,
$$

$$
Q_{O2,A}=3,\quad Q_{O2,B}=4.
$$

DC $D_2$ exactly matches order $O_1$, while DC $D_1$ exactly matches order $O_2$. Therefore

$$
O_1\rightarrow D_2,\qquad O_2\rightarrow D_1
$$

fulfills all $13$ cases. Its objective is

$$
(13)(10)-0-4=126.
$$

Expected summary:

```text
selected assignments:
  O1 -> D2
  O2 -> D1
fulfilled cases: 13
unfulfilled cases: 0
fulfilled value: 130
penalty cost: 0
shipping cost: 4
objective value: 126
feasible: true
```

## Generated files

```text
orders.csv
order_lines.csv
inventory.csv
candidates.csv
capacities.csv
metadata.json
```

`capacities.csv` may be empty but should contain the canonical header.

## Required tests

- `tests/test_objective.py`: verifies objective $126$.
- `tests/test_validation.py`: verifies feasibility.
- `tests/test_tiny_optimum.py`: verifies the split assignment.
- `tests/test_baselines.py`: verifies deterministic baseline behavior.
- `tests/test_candidate_generation.py`: verifies all four assignment candidates.




