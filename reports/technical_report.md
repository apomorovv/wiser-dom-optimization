# A feasibility-preserving quantum–classical optimizer for Distributed Order Management

## Executive summary

Distributed Order Management (DOM) assigns customer orders to distribution centers
(DCs) when the default source cannot fully serve demand. The decision must account
for SKU inventory, operational capacity, service dates, shipping cost, fulfilled
value, and the cost of unfulfilled demand. The choices are coupled: using scarce
stock for one order changes which assignments remain useful for every other order.

This work implements a hybrid large-neighborhood search (LNS). A transparent default
or sequential greedy method first produces a feasible incumbent. The algorithm then
selects a bounded neighborhood of resource-coupled orders and builds a warm-started
quadratic unconstrained binary optimization (QUBO) model over assignment plans. A
local classical sampler, exact enumerator, or explicitly approved D-Wave backend may
propose combinations. Deterministic repair restores one outcome per order. Finally,
an exact mixed-integer linear program (MILP) reoptimizes SKU fulfillment using the
resources left by frozen orders. A complete independent validator accepts only a
strict feasible improvement.

This separation preserves operational correctness while leaving a technically sound
place for quantum sampling. It avoids encoding every quantity and slack variable in
a monolithic QUBO, keeps logical quantum size bounded as the global population grows,
and guarantees that poor samples cannot degrade the returned incumbent. The system
makes no quantum-advantage claim; it supplies the controls and instrumentation needed
to test one.

The attached recommendation outputs were audited at aggregate level: 1,109 orders
and 25,193 order-SKU rows reconcile with zero quantity mismatch, and the selected
incumbent fills 94.4997% of 2,554,440 requested cases. However, the files uploaded as
raw inputs are AppleDouble metadata sidecars rather than their underlying contents.
They do not contain the counterfactual inventory, costs, eligibility, or capacity
needed for reoptimization. Real-data improvement results must wait for a correct
re-export. The code is validated on a known-optimum tiny instance and independently
generated scaling instances.

## 1. Business problem

An order normally ships from its default DC. When that DC has insufficient stock or
operational capacity, an alternative DC can improve service. Diversion is not free:
it may increase transport expense, consume inventory protected for another market,
or move work into a constrained dock or picking shift. A useful recommendation must
therefore explain both the customer benefit and the network trade-off.

DOM is combinatorial. If each of \(O\) orders has \(K\) candidate outcomes, the
assignment search space is on the order of \(K^O\). Each assignment also determines
integer SKU quantities. Inventory couples different PGI dates, and throughput, dock,
pick, weight, and volume resources couple otherwise unrelated SKUs. An independent
order-by-order ranking cannot represent those interactions.

The challenge requires two classical baselines, a mathematical model, a quantum or
quantum-inspired comparison, fair metrics, scaling and noise analysis, and business
communication. The implementation treats all methods through a common data,
objective, and validation pipeline.

## 2. Canonical data and preparation

The solver consumes seven versioned objects: orders, order lines, inventory,
candidates, capacities, calendar, and metadata. Identifiers remain strings to retain
leading zeros and avoid accidental numeric coercion. Dates are parsed, case
quantities are nonnegative integers, and monetary values use one declared scale.

Candidate preprocessing is part of the model. A row identifies one order, DC, and
PGI date; it carries total shipping cost, default status, and an eligibility flag.
Closed dates, prohibited sources, late arrivals, forecast restrictions, and other
confirmed hard rules should remove a row before optimization. The no-assignment
outcome is created in code rather than represented as a fake DC.

Inventory is protected projected available-to-promise (ATP) by DC, SKU, and date.
Unlike cumulative receipts, projected ATP may decline because of future demand
outside the focus-order population. If a focus order consumes cases at date \(\tau\),
that use persists at every checkpoint \(t\ge\tau\). The model therefore protects a
low future ATP value instead of assuming inventory must be nondecreasing.

The clarified diversion rule requires an alternate assignment to fill at least the
default-reference quantity plus five percent of total order demand, capped at total
demand. Total shipping cost is charged once for the selected candidate. Unmet
penalty is unfulfilled cases multiplied by price per case and penalty rate. These
clarifications are versioned as assumption `v1`.

The loader rejects duplicate keys, missing references, malformed dates or quantities,
incorrect default flags, closed eligible candidates, unknown inventory policies, and
missing default-fill references when the five-percent rule is enabled. Restricted
source files and generated row-level runs are excluded from git.

## 3. Deterministic optimization model

Let \(x_{odt}\) select a candidate DC/date, \(z_o\) represent no assignment,
\(f_{osdt}\) be fulfilled cases, and \(u_{os}\) be unfulfilled cases. With value
\(v_{os}\), penalty \(\pi_{os}\), and total shipping cost \(c_{odt}\), maximize

\[
J=
\sum_{o,s,d,t}v_{os}f_{osdt}
-\sum_{o,s}\pi_{os}u_{os}
-\sum_{o,d,t}c_{odt}x_{odt}.
\]

Exactly one modeled outcome is required:

\[
\sum_{(d,t)\in\mathcal C_o}x_{odt}+z_o=1.
\]

Demand balance and assignment linking are

\[
\sum_{(d,t)\in\mathcal C_o}f_{osdt}+u_{os}=Q_{os},
\qquad
0\le f_{osdt}\le Q_{os}x_{odt}.
\]

For protected ATP \(I_{dst}\), consumption through each checkpoint is

\[
\sum_{o,\tau\le t}f_{osd\tau}\le I_{dst}.
\]

Operational resource \(r\) with limit \(R^r_{dt}\), fixed use \(\beta^r\), and
case-variable use \(\alpha^r\) satisfies

\[
\sum_o\beta^r_{odt}x_{odt}
+\sum_{o,s}\alpha^r_{osdt}f_{osdt}
\le R^r_{dt}.
\]

For pallet size \(P_s\), full-pallet and loose-case variables give the exact linear
pick identity \(f=P_sp+k\), with \(0\le k\le(P_s-1)x\). The diversion threshold is

\[
\sum_sf_{osdt}\ge
\min\{Q_o,F_o^{\mathrm{def}}+\lceil0.05Q_o\rceil\}x_{odt}
\]

for a non-default candidate.

SciPy submits this MILP to HiGHS. A successful solve records model dimensions,
incumbent, dual bound, native gap, nodes, and runtime. The independent evaluator
recomputes the business objective from output tables, and the validator separately
recomputes every constraint family.

## 4. Baselines

The **default baseline** limits each order to its eligible default-DC candidates and
allocates resources in deterministic order. It represents the operational policy
the optimizer must beat while remaining feasible.

The **sequential greedy baseline** previews every currently eligible candidate,
scores its incremental objective, commits the best choice using documented
tie-breaks, and immediately reduces residual inventory and capacity. It is stronger
than selecting each order independently because every later preview sees earlier
consumption. It remains fast and interpretable, but it cannot undo a locally
attractive early decision.

The **exact MILP** is the reference on tractable instances. A time-limited MILP can
also provide an incumbent and bound. The same formulation becomes the hybrid's local
recourse solver when assignment variables are fixed.

## 5. Hybrid large-neighborhood search

The hybrid starts from the default or greedy solution. It builds a conflict graph:
orders are adjacent when at least one candidate can touch a common DC/SKU inventory
bucket or DC/date capacity. Neighborhood seeds prioritize current unmet cases,
conflict degree, and candidate count. Both active orders and QUBO variables are
capped.

All frozen-order consumption is subtracted from the original limits. This exact
residualization creates a self-contained local problem. The QUBO then contains one
binary \(y_{ok}\) for each previewed candidate plan and an unassigned plan. It
minimizes negative isolated plan value, an exactly-one penalty

\[
P_{\mathrm{one}}(1-\sum_ky_{ok})^2,
\]

and quadratic resource-contention terms. The contention calculation uses both direct
pair overload and an estimate of higher-order pressure, so three plans that overload
a resource collectively can receive a signal even when each pair fits.

The incumbent's local assignments form a one-hot warm start. Exact enumeration is
available for tiny QUBOs, simulated annealing is the scalable local control, and
D-Wave direct or managed-hybrid sampling is optional. A remote call requires an
explicit privacy flag. Remote variables use integer labels, although coefficient
sensitivity still requires data-owner approval.

Repair retains one plan per order. The best distinct repaired combinations are fixed
in the detailed local MILP, which chooses exact fulfillment and may reject a QUBO
preview as infeasible. The merged global result is independently validated. A move
is accepted only if it improves the recomputed objective. Consequently the sequence
of incumbent values is monotone nondecreasing.

The hybrid is not “quantum instead of classical.” It is classical decomposition and
validation with a replaceable assignment-sampling component. This makes the quantum
experiment narrower, more reproducible, and safer.

## 6. Verification and current results

The known two-order synthetic instance has four assignment candidates and constrained
SKU inventory. Its exact optimum is:

\[
O_1\rightarrow D_2,\qquad O_2\rightarrow D_1,
\]

with objective 126 synthetic units. The exact MILP proves a zero native gap. Starting
from the default solution at −50, the hybrid exact-QUBO configuration reaches 126, an
improvement of 176, while using no more than five local QUBO variables in the tested
iteration. The independent validator reports no violation.

The automated suite covers default and greedy feasibility, candidate filtering,
objective arithmetic, the exact optimum, decreasing projected ATP, diversion
threshold rejection, exact pallet/loose-case consumption, one-hot QUBO ground state,
annealing reproducibility, the remote privacy gate, hybrid monotonicity, higher-order
resource pressure, public-safe synthetic generation, aggregate output auditing, and
planner explanations.

For the two readable challenge outputs, aggregate audit results are:

| Metric | Audited incumbent |
|---|---:|
| Orders | 1,109 |
| Order-SKU rows | 25,193 |
| Selected diversions | 3 |
| Requested cases | 2,554,440 |
| Fulfilled cases | 2,413,937 |
| Case fill | 94.4997% |
| Order/SKU reconciliation mismatches | 0 |

This is not a solver comparison. The source alternatives and resources are absent,
so the audit cannot compute an optimized counterfactual, shipping/penalty comparison,
or optimality gap. Reporting an improvement would be unsupported.

## 7. Scaling and noise

For active neighborhood \(B\) with plan set \(\mathcal K_o\), logical QUBO size is

\[
n=\sum_{o\in B}(|\mathcal K_o|+1).
\]

Pair construction is worst-case \(O(n^2)\), while sparse zero couplings are omitted
from remote binary quadratic models. Because `max_qubo_variables` is fixed, hardware
demand does not grow with the global order count. Outer-loop candidate previews and
conflict selection still grow and should be cached or indexed in production.

MILP size grows with assignments plus order-line-candidate fulfillment variables;
exact pallet/case mode adds two variables per fulfillment arc. The global exact model
is therefore best used for small instances, bounds, or a time-limited reference. Local
recourse fixes assignments and is materially smaller.

The synthetic generator varies orders, DCs, SKUs, candidates, demand, value,
penalties, inventory scarcity, dock, and throughput. The scaling script compares
default, greedy, exact MILP below a configured size, and hybrid search. It also adds
reproducible symmetric Gaussian coefficient perturbations to the QUBO. This tests
ranking sensitivity but is not a complete physical model of annealer or gate noise.

A seed-7 study at 8, 20, and 50 orders kept the actual local QUBO at or below 32
variables. Every method was feasible. At eight orders, zero-noise hybrid search
improved greedy by 10.8 objective units but remained 50.2 below the proven MILP
optimum; 2% coefficient noise removed that improvement. At 20 and 50 orders, neither
noise setting improved the greedy incumbent. Hybrid runtime was about 5.1–19.1
seconds across these sizes, versus 0.3–11.5 seconds for greedy. The 20-order MILP
returned a 0.97% gap, and the protocol skipped global MILP at 50 orders. This
single-seed result is deliberately reported as a negative/limited outcome: the
hybrid safety invariant worked, but sampler benefit was not consistent.

For a fair quantum experiment, the neighborhood, QUBO, warm start, repair, recourse,
and validation must remain identical. Report multiple seeds, exact or best-known
distance, end-to-end wall time, raw one-hot rate, logical variables and couplings,
embedding/chain statistics, QPU access time, and total remote latency. Physical qubit
count alone is not a scaling result.

## 8. Research rationale

Quantum local search is supported by Tomesh et al.'s 2022 work on bounded quantum
neighborhoods for larger constrained problems. Egger et al.'s warm-start research
supports retaining useful classical information rather than starting every quantum
search uniformly. Yarkoni et al.'s industrial quantum-annealing review emphasizes
formulation, embedding, benchmarking, and the absence of an automatic advantage.
A 2026 shipment-selection workflow similarly combines quantum assignment candidates
with classical refinement. D-Wave's documented interfaces accept the local binary
quadratic model directly, while QAOA remains a future gate-model comparator.

The resulting solver choice is conservative: direct binary quadratic sampling is a
better immediate fit than a deep constrained gate circuit, but the QUBO is
backend-neutral and classical simulated annealing remains mandatory as a control.

## 9. Limitations and next steps

The local QUBO uses previewed plans and quadratic contention as a surrogate. Exact
aggregate capacity may require higher-order interactions that the recourse MILP—not
the QUBO—resolves. Penalty coefficients may need hardware-range scaling. The current
neighborhood selector is deterministic and does not yet use MILP dual prices. Load
grouping and customer-specific all-or-nothing rules remain unresolved until source
semantics are confirmed.

The highest-priority next step is to obtain the actual raw payloads and map them to
the canonical tables. Then:

1. reproduce the incumbent with the exact objective convention;
2. run default, greedy, bounded MILP, and hybrid methods on identical focus orders;
3. measure gaps on tractable batches and distance to best known on larger batches;
4. sweep neighborhood composition, size, QUBO penalties, noise, reads, and seeds;
5. add dual-informed neighborhood selection and candidate-column reduction;
6. run a privacy-approved QPU comparison only after the local controls are tuned; and
7. issue the generated planner view for business review.

## 10. Conclusion

The implementation turns DOM into a rigorously validated optimization workflow. Its
novelty is not a claim that quantum hardware solves the full logistics model today;
it is a practical partition that reserves bounded, coordinated assignment search for
a quantum-capable backend while exact classical optimization retains quantity,
resource, and safety responsibility. This architecture is scalable, testable, and
honest about current hardware and data limitations.

## References

1. Tomesh et al., “Quantum Local Search with the Quantum Alternating Operator
   Ansatz,” *Quantum* 6, 781 (2022),
   https://doi.org/10.22331/q-2022-08-22-781.
2. Egger, Mareček, and Woerner, “Warm-starting quantum optimization,” *Quantum* 5,
   479 (2021), https://doi.org/10.22331/q-2021-06-17-479.
3. Yarkoni et al., “Quantum Annealing for Industry Applications: Introduction and
   Review,” https://arxiv.org/abs/2112.07491.
4. Lopez-Ruiz et al., “Hybrid Quantum-Classical Optimization Workflows for the
   Shipment Selection Problem,” https://arxiv.org/abs/2604.11758.
5. D-Wave, “Hybrid Computing,”
   https://docs.dwavequantum.com/en/latest/concepts/hybrid.html.
