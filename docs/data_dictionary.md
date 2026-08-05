# Canonical data dictionary

This document defines the solver-facing data contract. Source columns may differ, but `src/domopt/data.py` must normalize them to these names before any method runs.

## Conventions

- Identifiers are strings.
- Dates use ISO `YYYY-MM-DD` after parsing.
- Case quantities are nonnegative integers.
- Monetary values are finite and use one dataset-level scale.
- Booleans are normalized to `true`/`false`.
- Primary keys contain no missing values.
- Strings such as `"null"`, `"None"`, and `"nan"` are not valid identifiers.

## 1. `orders.csv`

One row per order.

| Column | Type | Required | Definition |
|---|---|---:|---|
| `order_id` | string | yes | Stable anonymized order or grouping identifier \(o\). |
| `default_dc` | string | yes | Default DC \(d_o^{\mathrm{def}}\). |
| `requested_delivery_date` | date | yes | Requested delivery date. |
| `default_pgi_date` | date | no | Original PGI date. |
| `default_fillable_cases` | integer | conditional | Default-policy fill reference \(F_o^{\mathrm{def}}\); required when minimum-divert enforcement is enabled. |
| `min_divert_improvement_fraction` | float | no | Fraction of total order demand added to default fill; defaults to metadata, normally 0.05. |
| `load_id` | string | no | Anonymized load/group identifier. |
| `priority` | integer/string | no | Documented customer or allocation priority. |
| `is_top_customer` | boolean | no | Priority-customer indicator. |
| `focus_reason` | string | no | Why the order entered the optimization set. |

Primary key:

\[
(\texttt{order\_id}).
\]

## 2. `order_lines.csv`

One row per order–SKU pair.

| Column | Type | Required | Definition |
|---|---|---:|---|
| `order_id` | string | yes | Order \(o\). |
| `sku_id` | string | yes | SKU \(s\). |
| `demand_cases` | integer | yes | Requested cases \(Q_{os}\). |
| `unit_value` | float | yes | Fulfillment value per case \(v_{os}\). |
| `penalty_per_unfilled_case` | float | yes | Penalty per unmet case \(\pi_{os}\). |
| `unit_weight` | float | no | Weight per case. |
| `unit_volume` | float | no | Volume per case. |
| `cases_per_pallet` | integer | no | Cases per full pallet \(P_s\). |
| `forecast_required` | boolean | no | Whether candidate-DC forecast eligibility is required. |

Primary key:

\[
(\texttt{order\_id},\texttt{sku\_id}).
\]

Foreign key:

\[
\texttt{order\_lines.order\_id}\rightarrow\texttt{orders.order\_id}.
\]

## 3. `inventory.csv`

One row per DC–SKU–date protected inventory checkpoint.

| Column | Type | Required | Definition |
|---|---|---:|---|
| `dc_id` | string | yes | DC \(d\). |
| `sku_id` | string | yes | SKU \(s\). |
| `date` | date | yes | Inventory checkpoint \(t\). |
| `cumulative_available_cases` | integer | yes | Protected projected ATP \(I_{dst}\) available to focus orders through this checkpoint. |
| `reserved_default_cases` | integer | no | Cases protected outside the focus-order set. |
| `source_inventory_cases` | integer | no | Pre-reservation inventory for audit. |

Primary key:

\[
(\texttt{dc\_id},\texttt{sku\_id},\texttt{date}).
\]

Required invariant:

\[
I_{dst}\ge0.
\]

Under metadata policy `projected_atp`, values may decrease because later committed
demand can reduce future availability. An earlier focus-order fulfillment consumes
every later checkpoint. Under `cumulative_receipts`, the series must be
nondecreasing.

## 4. `candidates.csv`

One row per feasible order–DC–PGI assignment candidate.

| Column | Type | Required | Definition |
|---|---|---:|---|
| `candidate_id` | string | yes | Unique candidate key. |
| `order_id` | string | yes | Order \(o\). |
| `dc_id` | string | yes | Candidate DC \(d\). |
| `pgi_date` | date | yes | Candidate PGI date \(t\). |
| `shipping_cost` | float | yes | Fixed candidate cost \(c_{odt}\). |
| `is_default` | boolean | yes | Whether this uses the default DC. |
| `lead_time_days` | integer | no | Lane lead time. |
| `arrival_date` | date | no | Calculated arrival date. |
| `eligible` | boolean | yes | Must be true in solver-facing data. |
| `eligibility_reason` | string | no | Audit explanation. |
| `distance` | float | no | Lane distance in declared units. |
| `dock_units` | float | no | Fixed dock consumption when selected; defaults to 1. |

Primary key:

\[
(\texttt{candidate\_id}).
\]

Natural uniqueness:

\[
(\texttt{order\_id},\texttt{dc\_id},\texttt{pgi\_date}).
\]

The no-assignment variable \(z_o\) may be created programmatically.

## 5. `capacities.csv`

One row per DC–date–resource capacity.

| Column | Type | Required | Definition |
|---|---|---:|---|
| `dc_id` | string | yes | DC \(d\). |
| `date` | date | yes | Capacity date \(t\). |
| `resource` | string | yes | One of `dock`, `throughput_cases`, `case_pick`, `pallet_pick`, `weight`, or `volume`. |
| `capacity` | float | yes | Available capacity \(R^r_{dt}\). |
| `unit` | string | yes | Physical unit. |

Primary key:

\[
(\texttt{dc\_id},\texttt{date},\texttt{resource}).
\]

An empty table with a valid header means that no optional capacities are enabled.

## 6. `calendar.csv`

One row per DC–date status.

| Column | Type | Required | Definition |
|---|---|---:|---|
| `dc_id` | string | yes | DC \(d\). |
| `date` | date | yes | Date \(t\). |
| `is_open` | boolean | yes | Whether PGI is allowed. |
| `closure_reason` | string | no | Weekend, holiday, maintenance, etc. |

Primary key:

\[
(\texttt{dc\_id},\texttt{date}).
\]

Closed dates must be removed during candidate generation.

## 7. Solver output: `assignments.csv`

One row per order.

| Column | Type | Definition |
|---|---|---|
| `order_id` | string | Order \(o\). |
| `candidate_id` | string/null | Selected candidate. |
| `selected_dc` | string/null | Selected DC. |
| `selected_pgi_date` | date/null | Selected PGI date. |
| `is_unassigned` | boolean | Value of \(z_o\). |
| `is_divert` | boolean | Selected DC differs from default DC. |
| `method` | string | `default`, `greedy`, `classical`, or `hybrid`. |

## 8. Solver output: `fulfillment.csv`

One row per order–SKU pair.

| Column | Type | Definition |
|---|---|---|
| `order_id` | string | Order \(o\). |
| `sku_id` | string | SKU \(s\). |
| `fulfilled_cases` | integer | \(\sum_{d,t}f_{osdt}\). |
| `unfulfilled_cases` | integer | \(u_{os}\). |
| `selected_dc` | string/null | Fulfillment DC. |
| `selected_pgi_date` | date/null | Fulfillment date. |

Required identity:

\[
\texttt{fulfilled\_cases}+\texttt{unfulfilled\_cases}=\texttt{demand\_cases}.
\]

## 9. Solver output: `metrics.json`

```json
{
  "method": "string",
  "dataset_id": "string",
  "experiment_id": "string",
  "seed": 0,
  "feasible": true,
  "objective_value": 0.0,
  "fulfilled_value": 0.0,
  "penalty_cost": 0.0,
  "shipping_cost": 0.0,
  "case_fill_rate": 0.0,
  "value_fill_rate": 0.0,
  "reassigned_orders": 0,
  "unassigned_orders": 0,
  "runtime_seconds": 0.0,
  "best_bound": null,
  "optimality_gap": null,
  "initial_objective": null,
  "hybrid_improvement": null,
  "sampler_calls": null,
  "qpu_calls": null,
  "maximum_qubo_variables": null,
  "recourse_solves": null,
  "violations": {}
}
```

## 10. `metadata.json`

| Field | Required | Definition |
|---|---:|---|
| `dataset_id` | yes | Non-sensitive dataset/version label. |
| `schema_version` | yes | Canonical contract version, currently `0.2.0`. |
| `assumption_version` | yes | Modeling assumption version, currently `v1`. |
| `currency` | yes | Currency code or `synthetic_units`. |
| `quantity_unit` | yes | Must be `cases` for this implementation. |
| `inventory_policy` | yes | `projected_atp` or `cumulative_receipts`. |
| `pick_capacity_mode` | no | `auto`, `cases`, or `pallet_case`. |
| `enforce_min_divert_improvement` | yes | Whether the alternate-fill threshold is hard. |
| `min_divert_improvement_fraction` | no | Dataset default, normally 0.05. |
| `source_fingerprint` | recommended | Hash or approved source version, never a restricted path. |

For `pallet_case` mode, every order line must contain a positive
`cases_per_pallet`. For real data, metadata should also state the economic scale,
source cutoff, timezone, working-day calendar version, and protection-horizon logic.

## Source-to-canonical examples

| Source concept | Canonical field |
|---|---|
| Sales document / grouping indicator | `order_id` |
| Plant / DC | `dc_id` or `default_dc` |
| Material number | `sku_id` |
| Requested delivery date | `requested_delivery_date` |
| PGI / expected PGI | `default_pgi_date` or `pgi_date` |
| Projected inventory / ATP | processed into protected `cumulative_available_cases` checkpoints |
| Ordered quantity | `demand_cases` |
| SKU price | `unit_value` |
| Shipping cost | `shipping_cost` |
| Default fill quantity | `default_fillable_cases` |
| Price × penalty rate | `penalty_per_unfilled_case` |
| Lead time | `lead_time_days` |
| Load number | `load_id` |

Optimization modules must use canonical names rather than raw source names.
