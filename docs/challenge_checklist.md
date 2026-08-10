# Challenge completion checklist

**Author:** Andrei Pomorov

| Challenge requirement | Final implementation or artifact |
|---|---|
| Two-page business and technical summary | `reports/business_technical_summary.md` and `.pdf` |
| Five runtime tables documented | `docs/data_guide.md` |
| Default and greedy baselines | `src/domopt/baselines.py` |
| Objective, fill, reassignment, penalty, and shipping metrics | `src/domopt/metrics.py` and the notebook |
| Binary or column optimization model | `docs/mathematical_formulation.md` and `src/domopt/classical.py` |
| Candidate generation and tractability strategy | `src/domopt/candidates.py`, `src/domopt/poc.py`, and `docs/solver_method.md` |
| Independent validation | `src/domopt/validation.py` and validator tests |
| Common solver comparison | Notebook implementation plus screened aggregate figures and final report |
| Real and synthetic scaling | Notebook implementation plus screened aggregate and synthetic figures |
| Qubit growth, noise, and limitations | Final report, QAOA noise studies, and IBM hardware section |
| Exact-cover or exact comparison | Exact LNS controls plus full-MILP incumbent, bound, and gap |
| Repair and feasibility recovery | Deterministic one-hot repair plus exact recourse |
| Batching and decomposition | Conflict-based and random batch ablation |
| Uncertainty or robustness | Inventory shocks, coefficient noise, readout proxy, and repeated runs |
| Runnable notebook and repository | `notebooks/nestle_challenge_experiments.ipynb` and root README |
| Six-to-ten-page challenge report | `reports/challenge_submission_report.md` and `.pdf` |
| Full academic paper | `reports/final_report.md` and `.pdf` |
| Five-to-seven-slide presentation | Seven-slide `reports/final_presentation.pdf` and `.pptx` |
| One-page planner view | `reports/planner_view.md` and `.pdf` |
| Planner-facing interactive copilot | `apps/solver_cockpit.py` plus README interaction guide |
| Optional solver comparison | License-free HiGHS plus opt-in Gurobi notebook experiment |
| Hardware evidence | Synthetic-only IBM QPU study with exact, local, and uniform-feasible controls |

## Final quality gates

- [x] Commercial totals and operational identifiers are excluded from submission prose.
- [x] Raw inputs, identifiers, evidence CSV/JSON files, raw IBM job tables, manifests,
  checkpoints, commercial totals, and commercial-total charts remain excluded.
- [x] Submission reports and figures contain only reviewed aggregate or synthetic evidence.
- [x] SciPy/HiGHS remains the default on every operating system.
- [x] Gurobi and GPU dependencies are optional and isolated.
- [x] Remote QPU execution requires explicit approval.
- [x] The 18-job, 8,192-shot IBM study uses the final Dicke/W-state and XY-path circuit.
- [x] QAOA is named as the proposal method; IBM `backend` is documented as a processor
  or Qiskit execution target, not as a solver.
- [x] The end-to-end hardware runtime figure uses a logarithmic scale, medians, and
  individual-job dots so queue outliers do not compress the comparison.
