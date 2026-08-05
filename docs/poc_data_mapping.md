# POC source-to-model mapping

This document records how the readable Nestlé challenge fields become solver
parameters. It separates source facts from explicit modeling assumptions.

## Order and line mapping

| Source field | Canonical field | Transformation |
|---|---|---|
| `Group_Flag` | `order_id` | String-preserving normalized identifier. |
| `MaterialNumber` | `sku_id` | String-preserving normalized identifier. |
| `Plant` | `default_dc` | Default distribution center. |
| `transportationplanningdate` | `default_pgi_date` | Parsed timestamp. |
| `RequestedDeliveryDate` | `requested_delivery_date` | Parsed timestamp. |
| `LoadNumber` | `assignment_group` | Missing load becomes `ORDER::<order_id>` singleton. |
| `DeliveryPriority` | `priority` | Integer. |
| `IsTopCust` | `is_top_customer` | `Y` becomes `true`. |
| `ZipCode` | `destination_zip` | String-preserving lane destination. |
| `Order_SKU_Revenue` | `unit_value` | Divide by derived case demand. |
| `Penaltyforpotentialcuts` | `penalty_per_unfilled_case` | Multiply by `unit_value`. |
| `ProductCasesPerPallet` | `cases_per_pallet` | Rounded positive integer. |
| `OrderedWeight` | `unit_weight` | Divide by case demand. |
| `OrderedVolume` | `unit_volume` | Divide by case demand. |

Case demand is derived from `OrderedQty_converted` using
`ProductPlanningUnitsPerCase` when positive; otherwise the quantity is interpreted in
pallet planning units and multiplied by `ProductCasesPerPallet`.

## Penalty mapping

| Source field | Canonical field | Meaning |
|---|---|---|
| `FillRateThreshold` | `penalty_threshold_fraction` | Integer case fill required to avoid the order-level penalty. |
| `FixedPenalty` | `penalty_fixed` | Fixed amount when penalty activates. |
| `FixedPenaltyPerSKU` | `penalty_per_cut_sku` | Added for each SKU with any cut. |
| `MinimumPenalty` | `penalty_minimum` | Optional positive floor. |
| `MaximumPenalty` | `penalty_maximum` | Optional positive cap; zero means no cap. |

For order $o$, penalty activates when integer fulfilled cases are below
$\lceil\theta_oQ_o\rceil$. If active:

$$
\operatorname{rawPenalty}_o=
\sum_s\pi_{os}u_{os}+F_o+K_o\sum_s\mathbf 1[u_{os}>0].
$$

The raw value is raised to the optional minimum and then clipped to the optional
maximum. If the threshold is met, the penalty is zero. This calculation reproduces
the supplied `PenaltyIfNotDiverted` values to floating-point tolerance.

## Inventory mapping

| Source field | Canonical field | Transformation |
|---|---|---|
| `LocationID` | `dc_id` | String-preserving normalized identifier. |
| `MaterialID` | `sku_id` | String-preserving normalized identifier. |
| `DATE` | `date` | Parsed daily checkpoint. |
| `Available_inventory` | `cumulative_available_cases` | Minimum nonnegative value over the configured five-day protection window. |

The source identity `Available_inventory = OpeningStock - Total_Reserved_Qty` is
verified. The canonical name is retained for compatibility with the exact projected
ATP constraint; values may decrease over time.

## Candidate and resource mapping

| Source | Candidate/resource field | Rule |
|---|---|---|
| `Plant`, `TargetZip` | `dc_id`, destination match | Lane must exist. |
| `Distance` | `lead_time_days` | `ceil(distance / 500)`. |
| `Shipping_Cost` | `shipping_cost` | Total load-option cost, charged once via group leader. |
| `Dock_Remaining` | `dock` capacity | Nonnegative incremental alternate-load capacity. |
| Inventory SKU presence | forecast eligibility | Every group SKU must be present at an alternate DC. |
| Requested delivery date | alternative PGI | Subtract lead time and roll backward over closed days. |

The default DC/date is always represented when its shipping lane exists. A default
SKU absent from inventory receives a zero-fill preview rather than causing the
candidate to disappear.

## Assignment groups

Orders sharing `LoadNumber` are one routing decision. They must select the same
`group_option_id`, which is the same DC and PGI date. Every candidate option is kept
only when it exists for every member. Shipping cost and dock use are charged once to
the lexicographically first member; all other members carry zero fixed use.

The input contains 146 multi-order loads. The supplied output never splits one of
these loads across DC/date, supporting this interpretation.

## Diversion rule

For order demand $Q_o$ and default protected-ATP preview
$F_o^{\mathrm{def}}$, a non-default candidate must fulfill at least

$$
L_o^{\mathrm{div}}=
\min\left{Q_o,F_o^{\mathrm{def}}+
\max\left(\lceil0.05Q_o\rceil,100\right)\right\}.
$$

The default reference is computed from the same protected-ATP candidate preview used
by all methods. It is not copied from the supplied recommendation output.

## Explicit scenarios, not source facts

The throughput table is observed utilization, not a stated limit. When
`throughput_headroom_fraction` is set, the adapter creates scenario capacities for
case and pallet picks using the observed value times that fraction. Reports and
experiment configuration label these as scenarios.

Configured holidays beyond weekends are also explicit assumptions; the supplied
bundle does not provide a complete holiday calendar.

