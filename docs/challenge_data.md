# Challenge data status and processing

## What was received

The uploaded pack contains two usable recommendation outputs and the five-page WISER
challenge brief:

- order-level recommendations: 1,109 rows and 64 columns;
- order-SKU recommendations: 25,193 rows and 32 columns; and
- the WISER–Nestlé DOM challenge PDF.

The files named as raw order, shipping-cost, throughput, dock, capacity-planning,
workbook, equations, and DOM data are only 170–326 byte AppleDouble metadata
sidecars. AppleDouble files normally accompany a real file copied from macOS and do
not contain its tabular/document payload. Their small metadata forks cannot be used
to recover the original tables.

## Privacy-preserving incumbent audit

`scripts/audit_challenge_outputs.py` validates key uniqueness, order/SKU coverage,
numeric quantities, and order-to-SKU rollups. It returns aggregates only.

| Audit measure | Result |
|---|---:|
| Unique orders | 1,109 |
| Order-SKU rows | 25,193 |
| Unique loads | 615 |
| DC labels across default/recommended fields | 8 |
| Default orders | 1,106 |
| Diverted orders | 3 |
| Requested cases | 2,554,440 |
| Selected fulfilled cases | 2,413,937 |
| Case fill rate | 94.4996555% |
| Order/SKU quantity mismatches | 0 |

Commercial totals are excluded unless an authorized private analyst explicitly
passes `--include-commercial-metrics`. The command never returns order, load, SKU,
or DC identifiers.

## Why outputs are not enough to reoptimize

An incumbent output can show what was recommended, but not the full opportunity set.
Counterfactual optimization needs, for every focus order and alternative DC/date:

- SKU demand and economic coefficients;
- eligibility and lead-time/calendar status;
- protected inventory at every relevant checkpoint;
- shipping cost;
- dock, throughput, pick, weight, and volume capacity; and
- default fill used by the five-percentage-point rule.

The two outputs do not establish unused alternatives or their residual resources.
Inferring those values would create a plausible-looking but unsupported result.

## Required re-export

Re-upload or export the original payload files, not filenames beginning with `._` and
not Finder metadata. A safe check is that CSVs contain readable header rows, the
workbook opens as an `.xlsx` ZIP container, and the equations document opens as a
Word document. Keep original operational files outside git.

Transform the approved source into this canonical directory:

```text
processed-instance/
  orders.csv
  order_lines.csv
  inventory.csv
  candidates.csv
  capacities.csv
  calendar.csv
  metadata.json
```

The exact field contract is in [data dictionary](data_dictionary.md). Run
`load_problem_data` before any solver; it rejects duplicate keys, textual null IDs,
invalid dates/numbers, missing references, incorrect default flags, unsupported
inventory policy, and incomplete minimum-divert inputs.

## Source-to-canonical mapping checklist

| Business concept | Canonical destination | Required transformation |
|---|---|---|
| Sales document/group | `orders.order_id` | preserve as string; anonymize only in approved copy |
| Material | `order_lines.sku_id` | preserve leading zeros as string |
| Ordered cases | `order_lines.demand_cases` | integer cases |
| Price and penalty rate | `penalty_per_unfilled_case` | multiply on one declared monetary scale |
| Default source | `orders.default_dc` | preserve as string |
| Candidate source/date | `candidates.dc_id`, `pgi_date` | retain only eligible combinations |
| Lane cost | `candidates.shipping_cost` | total cost for selected candidate |
| Projected inventory | `inventory.cumulative_available_cases` | protected ATP by DC/SKU/checkpoint |
| Default fill | `orders.default_fillable_cases` | same policy used for 5% comparison |
| Dock/throughput/picks | `capacities` | normalize resource name, date, unit, limit |

Any source ambiguity must be resolved in metadata or assumptions before a real
result is reported.
