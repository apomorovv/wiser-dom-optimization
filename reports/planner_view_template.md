# Planner view — interpretation and sign-off

`scripts/run_hybrid.py` creates the data-specific one-page view and
`planner_recommendations.csv`. Review it inside the approved environment.

## Summary to report

| Measure | Validated result |
|---|---:|
| Orders reviewed | generated |
| Recommended diversions | generated |
| Diverted assignment groups/loads | generated |
| Case-fill change vs default | generated |
| Penalty avoided | detailed CSV |
| Shipping-cost change | detailed CSV |
| Net objective change vs default | generated |

## How to read a recommendation

- **DIVERT**: the alternate DC/date passed all rules and improved the common
  objective after exact shared-resource recourse.
- **KEEP DEFAULT**: no tested feasible alternative produced a strict improvement.
- **UNASSIGNED**: no assignment was selected after feasibility and economic
  evaluation; the order carries unmet-demand penalty.
- **Fill change**: recommended fulfilled cases minus feasible default-policy cases.
- **Penalty avoided**: default unmet penalty minus recommended unmet penalty.
- **Shipping change**: recommended total shipping cost minus default total cost.
- **Net change**: fulfilled-value gain + penalty avoided − shipping-cost increase.
- **Group totals**: sum member-order changes once per atomic assignment group; use the
  leader/member marker to avoid interpreting one load as independent order decisions.
- **Delivery status**: expected arrival is derived from the selected PGI and lane lead
  time and must be on or before the requested delivery date.
- **Constraint note**: reports the tightest modeled ATP or capacity row touched by the
  selected option; a binding row has no remaining modeled slack.
- **Screened alternative**: is the strongest remaining eligible row-level candidate,
  not a feasible fallback. Reoptimize its entire assignment group against shared
  resources before accepting it.

## Planner sign-off

- [ ] Requested delivery and PGI dates remain operationally acceptable.
- [ ] Expected arrival/on-time status was reviewed for every recommended diversion.
- [ ] Load grouping or appointment rules not present in solver data were reviewed.
- [ ] Inventory and capacity extracts use the approved cutoff and protection horizon.
- [ ] Commercial scale and five-percent default-fill reference are correct.
- [ ] Every recommendation has `validation.json: is_feasible = true`.
- [ ] No restricted identifiers or commercial rows will leave the approved workspace.

The optimizer is decision support. A planner should record any manual rejection and
the business rule that caused it so the next assumption version can model that rule.
