# Planner view: how to use the DOM recommendation

## Recommendation status

Use only a solution whose independent validation result is **feasible**. The hybrid
never replaces its starting policy unless exact recourse finds a strict improvement,
so the default recommendation remains the safe rollback.

| Planner measure | What to compare |
|---|---|
| Orders and loads reviewed | Complete assignment groups, not isolated order lines |
| Recommended diversions | Alternate DC differs from default and meets the uplift rule |
| Fill change | Recommended fulfilled cases minus protected default preview |
| Penalty avoided | Default thresholded penalty minus recommended penalty |
| Shipping change | Recommended total load cost minus default load cost |
| Net change | Fulfilled-value gain + penalty avoided − shipping increase |
| Service date | Candidate PGI plus lane lead remains compatible with requested delivery |

## Read each outcome

- **DIVERT** — One alternate DC/date is feasible for every order on the load. It
  improves default protected-ATP fill by at least both five percentage points and 100
  cases, respects inventory and dock limits, and improves the common objective after
  exact quantity recourse.
- **KEEP DEFAULT** — No tested eligible alternate produced a strict validated
  improvement under the current data and assumptions.
- **UNASSIGNED** — No candidate is selected; fulfillment is zero and the order incurs
  its thresholded unmet penalty.

The supplied output is an audit benchmark, not an instruction to reproduce three
historical diversions. The optimizer compares all generated eligible options under
one current resource state.

## Evidence available to the planner

The app and run artifacts show aggregate method, feasibility, objective index, case
fill, penalty, shipping, runtime, exact gap where available, candidate count, and
local QUBO width. A private planner export can additionally explain each load's
default and recommended fill, cost changes, binding resource, and next feasible
option. That row-level export remains inside the approved environment.

## Required sign-off

- [ ] The extraction timestamp and five-day protection horizon are current.
- [ ] The plant holiday calendar and requested-delivery exception policy are correct.
- [ ] Orders sharing a source load may move together as modeled.
- [ ] `Dock_Remaining` represents usable alternate-load headroom.
- [ ] Any throughput or pick limit is labeled as an approved scenario, not source fact.
- [ ] Commercial scale, fill threshold, penalty floor/cap, and 100-case uplift are correct.
- [ ] The selected solution has zero validator violations.
- [ ] No restricted row, identifier, DC resource, or commercial detail leaves the approved workspace.

## Human override

This optimizer is decision support. Record a manual rejection with its reason—such as
a customer restriction, appointment rule, plant closure, or service exception—so the
next assumption version can model the rule. Do not silently change solver quantities
or objective components after validation.

