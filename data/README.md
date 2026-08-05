# Data directory

This directory separates restricted source data, reproducible transformations, canonical solver-ready data, and public synthetic examples.

## Directory policy

- `data/raw/`: original challenge-approved files. Restricted; do not commit contents.
- `data/interim/`: intermediate joins, filters, diagnostics, and extracts. Normally restricted.
- `data/processed/`: canonical normalized solver tables. Normally restricted.
- `data/synthetic/`: independently generated public-safe instances.

## Processing flow

\[
\text{raw}
\rightarrow
\text{normalized interim tables}
\rightarrow
\text{focus orders}
\rightarrow
\text{feasible candidates}
\rightarrow
\text{processed solver tables}.
\]

Implementation mapping:

| Stage | Module |
|---|---|
| Load and normalize | `src/domopt/data.py` |
| Validate schemas | `src/domopt/schemas.py` |
| Identify focus orders | `src/domopt/focus_orders.py` |
| Generate candidates | `src/domopt/candidates.py` |
| Compute objective | `src/domopt/objective.py` |
| Validate solutions | `src/domopt/validation.py` |

## Canonical tables

The canonical contract is defined in [`../docs/data_dictionary.md`](../docs/data_dictionary.md). Initial datasets should contain:

- `orders.csv`
- `order_lines.csv`
- `inventory.csv`
- `candidates.csv`
- `capacities.csv`
- `calendar.csv` when multiple dates are enabled
- `metadata.json`

All identifiers must be loaded as strings. Never allow order, load, material, or DC identifiers to become floating-point values.

## Integrity requirements

A processed dataset is valid only when:

1. every `order_id` in `order_lines.csv` exists in `orders.csv`;
2. every order has at least one assignment candidate or an explicit no-assignment option;
3. `(order_id, sku_id)` is unique in `order_lines.csv`;
4. `(dc_id, sku_id, date)` is unique in `inventory.csv`;
5. case quantities are nonnegative integers;
6. monetary fields use one declared scale;
7. dates use `YYYY-MM-DD`;
8. every default DC exists in the network;
9. no solver-facing candidate is ineligible;
10. public synthetic data contain no copied operational identifiers or values.

## Metadata

Every generated dataset should include:

```json
{
  "dataset_id": "string",
  "schema_version": "0.2.0",
  "created_utc": "ISO-8601 timestamp",
  "source_fingerprint": "hash or approved source version",
  "generator_commit": "git commit SHA",
  "assumption_version": "v1",
  "currency": "declared currency or synthetic units",
  "quantity_unit": "cases",
  "inventory_policy": "projected_atp",
  "pick_capacity_mode": "auto",
  "enforce_min_divert_improvement": false
}
```

Do not store confidential source paths or source-file contents in public metadata.
