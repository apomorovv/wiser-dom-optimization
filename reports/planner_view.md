# Planner Decision View

**Author:** Andrei Pomorov
**Planning scope:** Reviewed 100-assignment-group comparison
**Recommended plan:** Polished greedy with exact fulfillment recourse
**Status:** Independently validated; exact LNS is the quality escalation

## Decision summary

| Planning metric | Default routing | Recommended plan | Change |
|---|---:|---:|---:|
| Objective capture | 74.96% | 76.83% | +1.87 pp |
| Case fill | 78.29% | 79.91% | +1.62 pp |
| Reassigned orders | 0 | 13 | +13 |
| Runtime | 0.24 s | 2.05 s | +1.81 s |
| Validation violations | 0 | 0 | No change |

**Objective capture** is fulfilled value minus penalties and shipping, divided by total
requested merchandise value. **Case fill** is fulfilled cases divided by requested
cases. **pp** means percentage points.

## Planner actions

1. Review the 13 reassignments for lane authorization and service rationale.
2. Confirm calendar exceptions and capacity events after the data snapshot.
3. Release only while the certification panel remains green.
4. Re-optimize after a material inventory change; trigger exact LNS when conflicts or
   penalty exposure are material.

## What the solver certifies

- one assignment outcome per order and cohesive routing for each load;
- exact demand balance, nonnegative integral quantities, and eligible DC/date choices;
- protected cumulative ATP, documented capacity, and diversion-improvement compliance;
- independent objective recomputation and rejection of invalid or non-improving moves.

## Escalation guide

| Signal | Action |
|---|---|
| Routine window; plan passes review | Release polished-greedy plan |
| Coupled inventory/dock conflict, severe scarcity, or high penalty exposure | Run exact LNS with a fixed budget |
| Small disputed subset needs a bound | Run full MILP and inspect incumbent plus gap |
| Research proposal requested | Use an approved bounded/generated neighborhood; retain exact recourse and validation |

## Evidence boundary

All 513 returned plans in the final 516-row study pass validation; three frozen-routing
controls are proven infeasible after 55% inventory reduction. At 70% reduction, adaptive
methods remain feasible and exact LNS retains the strongest tested frontier. The study
uses one supplied planning snapshot, so DC authorization, calendars, throughput maxima,
customer rules, and commercial approvals still require owner confirmation.

**Recommended decision:** approve the polished-greedy plan subject to planner review,
retain exact LNS as the quality escalation, and keep sampler-assisted runs behind the
same certification gate.
