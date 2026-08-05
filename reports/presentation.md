# Slide 1 — DOM optimizer

**Feasibility-preserving hybrid search for WISER–Nestlé**

- One DC/date decision per order load.
- Exact resource recourse after every sampled assignment.
- Classical baselines, bounded QUBO experiments, and planner explanations.

---

# Slide 2 — The POC bundle is readable

| Audit | Verified aggregate |
|---|---:|
| Challenge artifacts opened | 10/10 |
| Order-level outputs | 1,109 |
| Order-SKU rows | 25,193 |
| Selected output fill | 94.50% |

- Inputs include orders, 377,504 inventory rows, 12,922 lanes, dock and throughput observations, equations, and workbook.
- Case conversion and default-penalty equations reproduce supplied outputs.
- Raw rows, identifiers, DC details, and commercial totals stay outside Git.

---

# Slide 3 — Exact model owns feasibility

$$
\max\;\text{fulfilled value}-\text{thresholded penalty}-\text{shipping cost}
$$

- One eligible DC/PGI or unassigned; partial fill at one DC.
- Orders on one load select the same option.
- Five-day protected ATP and documented remaining dock capacity.
- Diversion requires +5 percentage points and +100 cases.
- Independent objective and constraint recomputation.

---

# Slide 4 — Hybrid large-neighborhood search

1. Construct a feasible default or greedy incumbent.
2. Select whole loads sharing scarce resources.
3. Sample a capped one-hot assignment QUBO.
4. Repair load options.
5. Reoptimize SKU quantities with exact local MILP recourse.
6. Accept only a globally feasible strict improvement.

**Invariant:** the returned objective never falls below its feasible incumbent.

---

# Slide 5 — Smoke evidence establishes safety

- 37 aggregate experiment rows.
- 100% independently feasible.
- 0 hybrid regressions from the configured incumbent.
- The real four-unit exact MILP reached zero gap.
- Hybrid matched exact and greedy there with a nine-variable local QUBO.
- The larger smoke subset shows a safe but incomplete hybrid search.

No QPU was used and no quantum advantage is claimed.

---

# Slide 6 — Experiment matrix

| Study | Full levels |
|---|---|
| Size | 8 / 20 / 50 |
| Penalty | 0.5× / 1× / 2× |
| Candidates | 1 / 2 / 4 / 6 |
| Inventory shock | 0% / 10% / 25% |
| Seed/noise | 4 seeds × 3 local coefficient levels |
| Ablations | Pareto, batch strategy, sampler |

An additional synthetic control tests a known coordination trap.

---

# Slide 7 — Recommendation

- Use exact MILP as the tractable quality reference.
- Keep greedy as the fast operational fallback.
- Use hybrid search for bounded, high-conflict neighborhoods when global exact search is time-limited.
- Run the full profile in the approved environment and confirm calendar/throughput semantics.
- Consider QPU testing only with matched QUBOs, full timing, embedding metrics, repeated trials, and approval.
- Release only validator-passed recommendations through the planner view.

