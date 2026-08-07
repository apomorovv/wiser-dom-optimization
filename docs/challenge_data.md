# Challenge data inventory and status

## Readability gate

The runtime gate parses five required CSV files and raises `PocDataError` immediately
if an input is missing, empty, malformed, or unreadable. Documentation and challenge
briefs are reviewed separately; the optimizer never requires a PDF, DOCX, or XLSX to
run.

## Supplied files

| File | Publication-safe status | Role in the workflow |
|---|---|---|
| `input_order_data.csv` | Shape validated at runtime | Order-SKU demand, default source/date, customer priority, economics, unit conversion, and penalty parameters. |
| `input_capacity_planning.csv` | Shape validated at runtime | Daily inventory/forecast by DC and SKU; used to construct protected ATP. |
| `input_shipping_cost_data.csv` | Shape validated at runtime | Plant-to-destination-ZIP lanes, distance, and total shipping cost. |
| `input_dock_capacity.csv` | Shape validated at runtime | Date-specific `Dock_Remaining`; used as incremental alternate-load capacity. |
| `input_throughput_capacity.csv` | Shape validated at runtime | Observed case-pick, pallet-pick, and order utilization; not a documented maximum. |

The clean bundle may also contain two optional reconciliation files:

| File | Status | Role |
|---|---|---|
| `output_order_level_data.csv` | Optional, validated locally | Supplied order-level recommendation output used only for audit. |
| `output_order_sku_level_data.csv` | Optional, validated locally | Supplied SKU-level recommendation output used only for audit. |

`DOM Equations.docx`, `Example.xlsx`, and the five-page challenge PDF are source
references. They are not copied into the runtime bundle and are not required by the
loader.

An all-null trailing column in the order input is ignored. Identifiers are loaded as
strings so leading zeros and exact joins are preserved.

## Restricted source audit

The local audit records source row counts, assignment-group counts, requested and
fulfilled cases, output fill, key coverage, and penalty-reproduction error. Those
exact source-scale values remain in ignored run artifacts pending publication review.
Public evidence should retain only approved rates or indexed values. Raw identifiers,
DC details, and commercial totals are never emitted by the aggregate writer.

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

The adapter reports focus orders after whole-load expansion, order-SKU lines,
assignment groups, candidates before/after optional pruning, protected-ATP rows, and
dock-capacity rows. Exact sizes are generated locally and fingerprinted with the
source bundle, assumption version, and candidate rules; they are withheld here until
publication review.

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
