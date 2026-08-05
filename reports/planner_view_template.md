# Planner view — interpretation and sign-off

`scripts/run_hybrid.py` creates the data-specific one-page view and
`planner_recommendations.csv`. Review it inside the approved environment.

## Summary to report

| Measure | Validated result |
|---|---:|
| Orders reviewed | generated |
| Recommended diversions | generated |
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

## Planner sign-off

- [ ] Requested delivery and PGI dates remain operationally acceptable.
- [ ] Load grouping or appointment rules not present in solver data were reviewed.
- [ ] Inventory and capacity extracts use the approved cutoff and protection horizon.
- [ ] Commercial scale and five-percent default-fill reference are correct.
- [ ] Every recommendation has `validation.json: is_feasible = true`.
- [ ] No restricted identifiers or commercial rows will leave the approved workspace.

The optimizer is decision support. A planner should record any manual rejection and
the business rule that caused it so the next assumption version can model that rule.
