# Branch comparison and final solver decision

Audit date: 2026-08-07. GitHub commit relationships and changed files were compared
across every repository branch. The solver branches form one linear implementation
history rather than several independent alternatives:

`main → feature/dom-model-v0 → hybrid-quantum-classical-dom → challenge-experiments-v2 → adaptive-exact-lns → results-ibm-hardware-study`

| Branch | Material contribution | Final disposition |
|---|---|---|
| `main` | Scaffold; core solver modules are empty. | Not a usable solver base. |
| `Updates_AGupta` | One assumptions Q&A, no code or results. | Do not merge; maintained docs already cover it more precisely. |
| `feature/dom-model-v0` | Canonical data model, objective, baselines, MILP, basic QUBO, validator, tiny optimum. | Retained through descendants. |
| `hybrid-quantum-classical-dom` | QUBO-assisted LNS, repair, exact recourse, strict improvement, planner and experiments; older D-Wave path. | Retain safety architecture; keep D-Wave and premature generated reports removed. |
| `challenge-experiments-v2` | Clean five-table bundle, expanded real/synthetic experiments, privacy controls and visualization. | Retained through descendants. |
| `adaptive-exact-lns` | Polished greedy, conflict-aware exact-MILP LNS, bounded/residualized neighborhoods, global validation and phase attribution. | Retain as the decisive classical improvement. |
| `results-ibm-hardware-study` | Current IBM Runtime adapter, backend discovery, mitigation variants, hardware metadata, progress persistence and plots. | Final base, with the corrections in this branch. |

## Evidence-backed method hierarchy

The supplied clean full-study archive contains 247 feasible rows; all 18 manifest
hashes pass. On the common 20-assignment-group subset, polished greedy, exact LNS,
global exact MILP, and hybrid all reach 64.9002% normalized objective capture. Their
runtimes are 2.715, 6.565, 2.518, and 12.149 seconds respectively. Hybrid accepts no
move; all of its improvement over raw greedy is the shared classical quantity polish.

At 372 groups/750 orders, polished greedy takes 51.779 seconds. Exact LNS takes 71.208
seconds for only `0.000027%` extra objective. Across real scaling, 98.87%
of LNS's aggregate gain over raw greedy is the initial polish. Hybrid is tested only to
50 real groups and adds no post-polish gain in any repeated row.

Therefore the final stack is:

1. `fast`: polished greedy, the production default;
2. `quality`: adaptive exact-MILP LNS, a coordinated-assignment escalation;
3. global exact MILP: tractable certificates and tiny validation cases; and
4. `hybrid`: bounded sampler/IBM proposals followed by exact recourse and independent
   validation, retained as an experimental comparator.

This hierarchy is exposed by `src/domopt/solver.py`. Every mode refuses to return an
invalid incumbent. Nothing in the current evidence supports a quantum-advantage claim.

## Corrections made on the final branch

- All notebook controls are explicit in one configuration cell immediately below the
  automatic dependency bootstrap; no terminal commands or environment flags are
  required, and each experiment has **Purpose** and **Why it matters**.
- Stable profile directories replace nested problem/run-hash directories; complete
  identities and content hashes remain in adjacent manifests.
- Dirty/clean source state remains provenance only and never blocks an experiment.
- The IBM matrix is the full two-depth by three-mitigation factorial, with derived
  circuit width, separate angle/transpiler seeds, phase timings, job provenance,
  resumable successful variants, and retained/retried failures.
- Validator tolerance and numeric residuals are exported with result rows.
- IBM adapter tests run in a credential-free CI job with the IBM dependencies present.
- Historical pre-LNS scaling artifacts were deleted.

No corrected IBM result is committed. The top-cell hardware switch must be enabled on
an approved authenticated environment to generate that remaining evidence.
