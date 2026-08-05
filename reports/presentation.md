# Slide 1 — DOM: a shared-resource decision

- Decide one distribution center (DC) and ship date per focus order.
- Balance fulfillment value, unmet penalty, total shipping cost, and service date.
- Orders compete for SKU inventory, dock, throughput, and pick capacity.
- Goal: better feasible recommendations than the default policy—not attractive but
  incompatible order-by-order choices.

---

# Slide 2 — Exact business model

\[
\max\;\text{fulfilled value}-\text{penalty cost}-\text{shipping cost}
\]

- One eligible DC/date or unassigned; partial fill at the selected DC.
- Demand balance for every order-SKU.
- Protected projected ATP at every future checkpoint.
- Dock, throughput, pallet/case picks, weight, and volume.
- Alternate fill ≥ default fill + 5% of total ordered cases.
- Independent validator and objective recomputation.

---

# Slide 3 — Feasibility-preserving hybrid

1. Build a feasible default or greedy incumbent.
2. Select a bounded neighborhood of orders sharing scarce resources.
3. Sample a warm-started assignment QUBO.
4. Repair one-hot outcomes.
5. Run exact local MILP recourse for fulfillment.
6. Accept only a globally feasible strict improvement.

**Invariant:** returned objective never falls below the feasible incumbent.

---

# Slide 4 — Why this quantum/classical split

| Classical layer | Quantum/annealing opportunity |
|---|---|
| Data integrity and candidate eligibility | Coordinated binary assignment proposals |
| Exact quantities and hard resources | Bounded high-conflict neighborhood search |
| Validation and business objective | Alternative samples around a warm start |
| Safe incumbent and fallback | Optional D-Wave QPU/hybrid adapter |

Avoids a monolithic quantity/slack QUBO. Local simulated annealing is the control;
remote execution is privacy-gated. No quantum advantage is assumed.

---

# Slide 5 — Evidence and reproducibility

- Known two-order optimum: 126 synthetic units.
- Exact MILP and hybrid reproduce `O1→D2`, `O2→D1`.
- Hybrid default-incumbent improvement: +176 synthetic units.
- Automated feasibility, rule, sampler, recourse, audit, and planner tests.
- Scaling generator varies orders, DCs, SKUs, candidates, scarcity, and QUBO noise.
- Run artifacts include input hash, configuration, seed, metrics, and violations.

---

# Slide 6 — Provided-data audit and limitation

- Usable outputs: 1,109 orders; 25,193 order-SKU rows; zero reconciliation mismatch.
- Incumbent: 3 diversions; 2,413,937 / 2,554,440 cases; 94.4997% case fill.
- Raw identifiers and commercial totals excluded from reporting.
- Remaining “input” uploads are AppleDouble metadata sidecars, not source payloads.
- Re-export originals before any honest alternative-DC optimization or commercial
  improvement claim.

---

# Slide 7 — Decision and next experiment

- Use exact MILP on tractable subsets as the quality reference.
- Use greedy as the fast operational fallback.
- Use bounded hybrid LNS when global integer search is time-limited.
- Compare exact, simulated annealing, and approved QPU samples on identical QUBOs.
- Report end-to-end runtime, feasible objective, gap/distance to optimum, logical
  variables/couplings, noise sensitivity, and results across seeds.
- Deploy only validator-passed recommendations with the generated planner view.
