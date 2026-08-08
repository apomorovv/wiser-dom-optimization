# Data guide

This guide describes the five runtime tables expected by the challenge adapter and the
canonical in-memory model created from them. Raw challenge files are restricted and
must remain under an ignored local directory such as `data/raw/nestle_challenge/`.

## Runtime files

| Canonical filename | Source grain | Main use |
|---|---|---|
| `input_order_data.csv` | Order and SKU line | Demand, value, penalties, default DC, dates, priority, weight, volume, grouping |
| `input_capacity_planning.csv` | DC, SKU, date | Cumulative available-to-promise inventory |
| `input_shipping_cost_data.csv` | DC and destination region | Distance, cost, and transit estimate |
| `input_dock_capacity.csv` | DC and date | Remaining dock capacity |
| `input_throughput_capacity.csv` | DC and planning date | Case-pick and pallet-pick utilization |

Two optional files, `output_order_level_data.csv` and
`output_order_sku_level_data.csv`, are treated only as reference outputs for an audit.
They are never used as training labels or solver inputs.

Run the readability and schema gate before solving:

```bash
python scripts/prepare_challenge_bundle.py \
  --source-dir /path/to/downloads \
  --output-dir data/raw/nestle_challenge
```

## Canonical tables

The adapter in `src/domopt/poc.py` converts source columns to six normalized tables.

| Table | Unique key or grain | Important fields |
|---|---|---|
| `orders` | One row per order | `order_id`, `assignment_group`, `default_dc`, requested date, priority |
| `order_lines` | One row per order and SKU | demand cases, unit value, cut penalty, cases per pallet, weight, volume |
| `inventory` | One row per DC, SKU, checkpoint date | cumulative available cases |
| `candidates` | One row per eligible order, DC, and PGI date | shipping cost, default flag, eligibility, group option |
| `capacities` | One row per DC, date, and resource | capacity and unit |
| `calendar` | One row per DC and date | open or closed flag |

An assignment group represents orders that must select a common load-level option.
PGI means planned goods issue, the modeled ship date. SKU means stock-keeping unit.
Cumulative ATP means the total stock available through a checkpoint date after the
configured inventory protection.

## Unit conversion and dates

- Planning units are converted to integer cases using the supplied units-per-case field.
- Requested delivery dates and transportation planning dates are parsed as calendar
  dates, not free-form text.
- Shipping distance is converted to an estimated transit duration with the documented
  miles-per-lead-day assumption.
- A candidate PGI date must be open and early enough to meet the requested delivery date.
- Missing inventory coverage at or after a PGI date means zero modeled availability;
  it is never interpreted as unlimited stock.
- Case-pick and pallet-pick resources use the supplied cases-per-pallet conversion.

## Candidate generation

The default candidate universe is the intersection of DCs present in shipping,
inventory, and dock-capacity data. Each order receives eligible combinations of DC and
open PGI date. Candidate ranking can cap alternatives per assignment group for local
search, but the full real-data comparison uses the unpruned universe. Score-based
Pareto pruning is isolated in its own ablation because shared resources can invalidate
an apparently dominated local option.

## Business rules

- Orders in the same assignment group move together.
- Fulfilled and unfulfilled cases sum exactly to demand for every order-SKU line.
- Cumulative inventory use cannot exceed cumulative ATP at any checkpoint.
- Dock, case-pick, pallet-pick, weight, and volume consumption cannot exceed capacity.
- A diversion must satisfy the configured minimum fill improvement.
- Thresholded penalties can contain activation, fixed, per-cut-SKU, minimum, and maximum
  components. These business penalties are distinct from QUBO constraint penalties.

The exact formulas are in [the mathematical formulation](mathematical_formulation.md),
and explicit modeling assumptions are in [the assumptions file](assumptions.md).

## Privacy

Committed evidence contains aggregate counts, normalized rates, runtimes, residuals,
and synthetic controls only. Do not commit source rows, customer information, order
identifiers, SKU identifiers, DC identifiers, destination details, or commercial totals.
The aggregate writer rejects identifier-like columns, and `.gitignore` excludes raw and
private output directories.
