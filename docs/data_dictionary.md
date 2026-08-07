# Canonical data dictionary

This is the solver-facing contract after source-specific transformation. The POC
adapter builds these objects in memory; restricted row-level tables are not written
to Git.

## Conventions

- Identifiers are strings.
- Dates are normalized timestamps and serialize as `YYYY-MM-DD`.
- Case quantities are nonnegative integers.
- Monetary values are finite and use one dataset-level scale.
- Booleans are normalized to `true` or `false`.
- Primary keys contain no missing or textual-null values.
- The current schema is `0.3.0`; the current assumption version is `v2`.

## Strict source gate

The five runtime CSVs are validated before canonical defaults, clipping, or type
coercion are applied. `src/domopt/poc.py:POC_REQUIRED_COLUMNS` is the executable
required-column contract. The audit rejects missing/empty files, missing columns,
blank required identifiers, invalid dates, malformed or nonfinite required numerics,
negative quantities/costs where prohibited, unsupported `Y`/`N` values, and fill
thresholds outside `[0, 1]`. It also verifies, within numeric tolerance,

$$
\texttt{Available_inventory}
=\texttt{OpeningStock}-\texttt{Total_Reserved_Qty}.
$$

Optional nulls are allowed only for fields whose transformation defines an explicit
fallback. Failure raises `PocDataError`; malformed required business values are not
silently converted to zero. Bundle normalization also refuses unrelated stale files in
the destination directory so an old document or table cannot be mistaken for part of
the current input contract.

## `orders`

One row per order.

| Column | Type | Required | Definition |
|---|---|---:|---|
| `order_id` | string | yes | Stable order identifier. |
| `default_dc` | string | yes | Default distribution center. |
| `requested_delivery_date` | date | yes | Customer requested delivery date. |
| `default_pgi_date` | date | POC | Original planned-goods-issue date. |
| `assignment_group` | string | POC | Orders that must select one common DC/PGI option. |
| `default_fillable_cases` | integer | diversion rule | Protected-ATP preview at the default candidate. |
| `min_divert_improvement_fraction` | float | no | Fractional uplift, normally `0.05`. |
| `priority` | integer/string | no | Documented allocation priority. |
| `is_top_customer` | boolean | no | Priority-customer indicator. |
| `penalty_threshold_fraction` | float | thresholded penalty | Fill threshold required to avoid penalty. |
| `penalty_fixed` | float | thresholded penalty | Fixed penalty when active. |
| `penalty_per_cut_sku` | float | thresholded penalty | Amount per SKU with unmet cases. |
| `penalty_minimum` | float | no | Positive active-penalty floor; zero disables. |
| `penalty_maximum` | float | no | Positive active-penalty cap; zero disables. |
| `destination_zip` | string | candidate generation | Anonymized lane destination. |

Primary key: `order_id`.

## `order_lines`

One row per order-SKU.

| Column | Type | Required | Definition |
|---|---|---:|---|
| `order_id` | string | yes | Parent order. |
| `sku_id` | string | yes | Stock-keeping unit. |
| `demand_cases` | integer | yes | Requested cases $Q_{os}$. |
| `unit_value` | float | yes | Fulfillment value per case $v_{os}$. |
| `penalty_per_unfilled_case` | float | yes | Variable active-penalty coefficient $\pi_{os}$. |
| `unit_weight` | float | no | Weight per case. |
| `unit_volume` | float | no | Volume per case. |
| `cases_per_pallet` | integer | pick mode | Cases per full pallet. |
| `forecast_required` | boolean | no | Whether alternative-DC SKU presence is required. |

Primary key: `(order_id, sku_id)`. `order_id` references `orders`.

## `inventory`

One row per DC-SKU-date protected checkpoint.

| Column | Type | Required | Definition |
|---|---|---:|---|
| `dc_id` | string | yes | Distribution center. |
| `sku_id` | string | yes | Stock-keeping unit. |
| `date` | date | yes | Inventory checkpoint. |
| `cumulative_available_cases` | integer | yes | Protected projected ATP available to focus orders. |
| `reserved_default_cases` | integer | no | Amount protected outside the focus set. |
| `source_inventory_cases` | integer | no | Pre-protection quantity for audit. |

Primary key: `(dc_id, sku_id, date)`. Values are nonnegative. Under
`projected_atp`, they may decrease; an earlier fulfillment consumes every later
checkpoint. Under `cumulative_receipts`, they must be nondecreasing.

## `candidates`

One row per order-DC-PGI option.

| Column | Type | Required | Definition |
|---|---|---:|---|
| `candidate_id` | string | yes | Unique assignment-column key. |
| `order_id` | string | yes | Parent order. |
| `dc_id` | string | yes | Candidate DC. |
| `pgi_date` | date | yes | Candidate PGI date. |
| `shipping_cost` | float | yes | Total fixed option cost; group followers carry zero. |
| `is_default` | boolean | yes | Whether this is the original DC option. |
| `eligible` | boolean | yes | Must be true in solver-facing data. |
| `group_option_id` | string | grouped decisions | Common DC/date option within an assignment group. |
| `lead_time_days` | integer | no | Calendar-day lane lead. |
| `arrival_date` | date | no | PGI plus lead time. |
| `distance` | float | no | Source lane distance. |
| `dock_units` | float | no | Fixed incremental dock use. |
| `estimated_fill_cases` | integer | pruning | Isolated protected-ATP preview. |
| `estimated_fulfilled_value` | float | pruning | Isolated fulfillment-value preview. |

Primary key: `candidate_id`; natural key: `(order_id, dc_id, pgi_date)`. The
unassigned outcome is created by the solver and is not represented as a fake DC.

Candidate DC breadth is explicit configuration, not inferred silently. The default
`network_intersection` scope considers DCs present in shipping, inventory, and dock
sources; `focus_default_dcs` restricts the universe to DCs that are defaults in the
focus population. Lane, forecast/inventory, calendar, group compatibility, delivery,
and diversion checks still filter both scopes. The sensitivity experiment compares the
two policies; operational permission for every connected DC still requires owner
confirmation.

## `capacities`

One row per DC-date-resource limit.

| Column | Type | Required | Definition |
|---|---|---:|---|
| `dc_id` | string | yes | Distribution center. |
| `date` | date | yes | Capacity date. |
| `resource` | string | yes | `dock`, `throughput_cases`, `case_pick`, `pallet_pick`, `weight`, or `volume`. |
| `capacity` | float | yes | Available amount. |
| `unit` | string | yes | Physical unit. |

Primary key: `(dc_id, date, resource)`. The readable POC uses documented
`Dock_Remaining`. Pick values derived from observed throughput are enabled only in an
explicitly labeled scenario.

## `calendar`

One row per DC-date status.

| Column | Type | Required | Definition |
|---|---|---:|---|
| `dc_id` | string | yes | Distribution center. |
| `date` | date | yes | Date. |
| `is_open` | boolean | yes | Whether PGI is allowed. |
| `closure_reason` | string | no | Weekend, holiday, or other closure. |

Primary key: `(dc_id, date)`. Closed candidates are removed before optimization.

## Assignment output

One row per order.

| Column | Type | Definition |
|---|---|---|
| `order_id` | string | Order. |
| `candidate_id` | string/null | Selected candidate. |
| `selected_dc` | string/null | Selected distribution center. |
| `selected_pgi_date` | date/null | Selected PGI date. |
| `is_unassigned` | boolean | Explicit no-assignment outcome. |
| `is_divert` | boolean | Selected DC differs from default. |
| `method` | string | `default`, `greedy`, `polished_greedy`, `exact_lns`, `classical`, or `hybrid`. |

## Fulfillment output

One row per order-SKU.

| Column | Type | Definition |
|---|---|---|
| `order_id` | string | Order. |
| `sku_id` | string | SKU. |
| `fulfilled_cases` | integer | Cases fulfilled at the selected option. |
| `unfulfilled_cases` | integer | Requested cases not fulfilled. |
| `selected_dc` | string/null | Fulfillment DC. |
| `selected_pgi_date` | date/null | Fulfillment date. |

Required identity:

$$
\text{fulfilled cases}+\text{unfulfilled cases}=\text{demand cases}.
$$

## Aggregate metrics

Every method is evaluated independently into the following common fields.

```json
{
  "method": "hybrid",
  "feasible": true,
  "objective_value": 0.0,
  "fulfilled_value": 0.0,
  "penalty_cost": 0.0,
  "shipping_cost": 0.0,
  "requested_value": 0.0,
  "objective_capture_rate": 0.0,
  "objective_per_assignment_group": 0.0,
  "case_fill_rate": 0.0,
  "value_fill_rate": 0.0,
  "reassigned_orders": 0,
  "reassigned_assignment_groups": 0,
  "runtime_seconds": 0.0,
  "optimality_gap": null,
  "raw_initial_objective": null,
  "polished_initial_objective": null,
  "initial_polish_improvement": null,
  "hybrid_improvement": null,
  "search_improvement": null,
  "lns_improvement": null,
  "total_hybrid_improvement": null,
  "maximum_qubo_variables": null,
  "maximum_active_groups": null,
  "maximum_local_variables": null,
  "maximum_local_constraints": null,
  "assignment_moves": null,
  "accepted_moves": null
}
```

Aggregate experiment rows additionally carry `experiment_schema_version`,
`bundle_sha256`, `problem_sha256`, `objective_version`, `git_commit`, `git_dirty`,
`source_state_sha256`, and the serialized method configuration. These fields are part
of reproducibility identity, not business features. Exact source-scale values remain
restricted even when aggregated; public artifacts require approval or
normalized/indexed values.

## Metadata

| Field | Required | Definition |
|---|---:|---|
| `dataset_id` | yes | Non-sensitive dataset/version label. |
| `schema_version` | yes | `0.3.0`. |
| `assumption_version` | yes | `v2`. |
| `currency` | yes | Declared source or synthetic scale. |
| `quantity_unit` | yes | `cases`. |
| `inventory_policy` | yes | `projected_atp` or `cumulative_receipts`. |
| `penalty_mode` | yes | `linear_unmet` or `thresholded_cut`. |
| `enforce_assignment_group` | no | Whether group options must match. |
| `enforce_min_divert_improvement` | yes | Whether alternate uplift is hard. |
| `min_divert_improvement_fraction` | no | Normally `0.05`. |
| `min_divert_improvement_cases` | no | POC value `100`. |
| `pick_capacity_mode` | no | `auto`, `cases`, or `pallet_case`. |
| `throughput_capacity_is_scenario` | no | Distinguishes assumed headroom from source limits. |
| `candidate_dc_scope` | yes for POC | `network_intersection` or `focus_default_dcs`. |
| `candidate_dc_count` | yes for POC | Number of DCs retained after candidate construction, for audit only. |
| `source_fingerprint` or `bundle_sha256` | recommended | Reproducibility hash; never a restricted path. |
| `raw_data_export_permitted` | recommended | Privacy guard. |
