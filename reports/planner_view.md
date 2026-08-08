# Planner Decision View

**Author:** Andrei Pomorov

**Planning scope:** Reviewed 20-assignment-group comparison

**Recommended plan:** Polished greedy with exact fulfillment recourse

**Status:** Validated; use exact LNS only when the escalation trigger is met

## Decision summary

| Planning metric | Default routing | Recommended plan | Change |
|---|---:|---:|---:|
| Objective capture | 61.39% | 64.90% | +3.51 pp |
| Case fill rate | 64.21% | 68.50% | +4.29 pp |
| Reassigned orders | 0 | 4 | +4 |
| Runtime | 0.30 s | 2.86 s | +2.56 s |
| Validation violations | 0 | 0 | No change |

**Objective capture** is fulfilled value minus penalties and shipping, divided by total
requested merchandise value. **Case fill** is fulfilled cases divided by requested
cases. **pp** means percentage points.

## Planner actions

1. Review the four reassignments for lane authorization and service rationale.
2. Confirm calendar exceptions and any capacity event after the data snapshot.
3. Release the plan when the candidate DCs are authorized.
4. Trigger exact LNS for material conflicts or a failed approval threshold; re-optimize
   after material inventory changes.

## What the solver guarantees

- exact assignment, load cohesion, case balance, and integer quantities;
- candidate eligibility, protected ATP, capacity, and diversion thresholds; and
- independent objective recomputation plus rejection of invalid or non-improving moves.

## Escalation guide

| Signal | Action |
|---|---|
| Routine window; recommended plan passes review | Release polished-greedy plan |
| Coupled inventory/dock conflict or high-value exception | Run exact LNS with a fixed time budget |
| Small disputed subset needs a certificate | Run full exact MILP and inspect the bound |
| Quantum experiment requested | Use only an approved bounded or synthetic neighborhood; retain exact recourse and validation |

## Known limitations

This is one challenge snapshot. DC authorization, authoritative calendars, throughput
maxima, customer service rules, and commercial approvals require owner confirmation.
Quantum hardware remains experimental and has not shown a speed or quality advantage.

**Recommended decision:** approve the polished-greedy plan subject to planner review,
retain exact LNS as the quality escalation, and treat quantum runs as controlled R&D.
