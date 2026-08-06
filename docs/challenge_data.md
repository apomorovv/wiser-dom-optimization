# Challenge data inventory and status

## Readability gate

The runtime gate parses five required CSV files and raises `PocDataError` immediately
if an input is missing, empty, malformed, or unreadable. Documentation and challenge
briefs are reviewed separately; the optimizer never requires a PDF, DOCX, or XLSX to
run.

## Supplied files

| File | Rows/pages | Role in the workflow |
|---|---:|---|
| `input_order_data.csv` | 25,193 rows, 39 columns | Order-SKU demand, default source/date, customer priority, economics, unit conversion, and penalty parameters. |
| `input_capacity_planning.csv` | 377,504 rows, 23 columns | Daily inventory/forecast by DC and SKU; used to construct protected ATP. |
| `input_shipping_cost_data.csv` | 12,922 rows, 7 columns | Plant-to-destination-ZIP lanes, distance, and total shipping cost. |
| `input_dock_capacity.csv` | 480 rows, 13 columns | Date-specific `Dock_Remaining`; used as incremental alternate-load capacity. |
| `input_throughput_capacity.csv` | 530 rows, 7 columns | Observed case-pick, pallet-pick, and order utilization; not a documented maximum. |

The clean bundle may also contain two optional reconciliation files:

| File | Rows | Role |
|---|---:|---|
| `output_order_level_data.csv` | 1,109 | Supplied order-level recommendation output used only for audit. |
| `output_order_sku_level_data.csv` | 25,193 | Supplied SKU-level recommendation output used only for audit. |

`DOM Equations.docx`, `Example.xlsx`, and the five-page challenge PDF are source
references. They are not copied into the runtime bundle and are not required by the
loader.

An all-null trailing column in the order input is ignored. Identifiers are loaded as
strings so leading zeros and exact joins are preserved.

## Privacy-safe source audit

| Measure | Verified value |
|---|---:|
| Orders in supplied recommendation output | 1,109 |
| Order-SKU rows | 25,193 |
| Named loads | 614 |
| Assignment groups after missing-load singleton handling | 631 |
| Diverted orders | 3 |
| Requested cases | 2,554,440 |
| Selected fulfilled cases | 2,413,937 |
| Selected case fill | 94.4996555% |
| Order/SKU key coverage | complete |
| Maximum default-penalty reproduction error | less than $4.5\times10^{-7}$ |

Raw identifiers, DC details, and commercial totals are not emitted.

## What each source contributes

### Order-SKU data

`input_order_data.csv` is the main fact table. One row is one order-SKU line. The
adapter derives integer case demand as follows:

$$
Q_{os}=
\begin{cases}
\operatorname{round}(\text{OrderedQty converted}/\text{planning units per case}),
&\text{when units per case}>0,\\
\operatorname{round}(\text{OrderedQty converted}\times\text{cases per pallet}),
&\text{otherwise.}
\end{cases}
$$

The second branch is required for the rows represented in pallet planning units. The
calculation reconciles exactly with the supplied output case quantities.

The same table supplies order-level fill thresholds, fixed penalty, penalty per cut
SKU, optional minimum and maximum penalty, top-customer flag, priority, default DC,
default PGI, requested delivery date, load number, destination ZIP, SKU value,
weight, volume, and cases per pallet.

### Inventory and forecast

`Available_inventory` equals `OpeningStock - Total_Reserved_Qty` in the provided
table. Negative values are clipped to zero. For candidate PGI $t$, the adapter
uses the minimum available inventory over $t$ through $t+5$ calendar days:

$$
I_{dst}=\max\left(0,\left\lfloor
\min_{\tau\in[t,t+5]}\text{AvailableInventory}_{ds\tau}
\right\rfloor\right).
$$

This protects already-planned future consumption. An alternative candidate is
eligible only when every SKU in its assignment group exists in the alternative DC's
inventory/forecast table. A missing default SKU is interpreted as zero fill, which
keeps the default comparison available without inventing inventory.

### Shipping and dates

Candidate lane lead time is:

$$
\operatorname{leadDays}=\left\lceil\frac{\operatorname{distance}}{500}\right\rceil.
$$

Alternative PGI is requested delivery date minus lead time, rolled backward over
weekends and any configured holidays. The shipping value is treated as total cost
for the selected load option, not an incremental difference.

### Dock and throughput

`Dock_Remaining` is clipped at zero. A default load consumes zero *incremental* dock
capacity because it is already booked; an alternate load consumes one unit. Cost and
dock use are attached to one deterministic group leader, preventing double counting
across orders on the same load.

The throughput file contains observed `util_case_picks`, `util_pallets`, and
`order_count`. Because no maximum or remaining-capacity equation is documented, the
real adapter does not treat these observations as hard capacity. An analyst may set
`throughput_headroom_fraction` to create an explicitly labeled scenario.

### Supplied outputs

The two output CSVs are audit benchmarks. They establish reconciliation targets for
case quantities and penalties, but they are not training labels and do not constrain
the optimizer to reproduce the supplied recommendation.

## Canonical instance produced

With the default focus/load settings and before Pareto pruning, the readable bundle
produces:

| Canonical object | Aggregate size |
|---|---:|
| Focus orders after whole-load expansion | 750 |
| Focus order-SKU lines | 20,869 |
| Assignment groups | 372 |
| Candidate rows | 2,182 |
| Candidate rows after Pareto pruning | 1,307 |
| Protected-ATP rows | 47,075 |
| Dock-capacity rows | 50 |

The exact numbers are generated by code and may change only when the source bundle,
assumption version, or candidate rules change.

## Run the gate and adapter

First normalize browser-added suffixes such as `(1)` into a clean local directory:

```bash
python scripts/prepare_challenge_bundle.py \
  --source-dir /approved/path/to/downloads \
  --output-dir data/raw/nestle_challenge
```

```python
from domopt.poc import PocConfig, audit_poc_bundle, load_poc_problem

audit = audit_poc_bundle("data/raw/nestle_challenge")
problem = load_poc_problem(
    "data/raw/nestle_challenge",
    config=PocConfig(pareto_prune=False),
)
```

See [POC source mapping](poc_data_mapping.md) for field-level transformations and
[canonical data dictionary](data_dictionary.md) for the solver contract.
