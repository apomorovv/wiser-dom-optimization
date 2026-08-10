# Results

`final/figures/` contains the screened figure bundle for submission. Generated reruns
and their evidence tables belong under ignored `results/challenge-study/` or
`results/runs/` paths until they have been audited.

## Bundle contents

| Path | Contents |
|---|---|
| `final/figures/` | Aggregate normalized summaries, scaling, robustness, coordination, and QAOA figures |
| `final/ibm/figures/` | Synthetic IBM hardware and queue-control figures |

The underlying CSV/JSON evidence, checkpoint manifests, IBM job tables, backend
snapshots, strategy rankings, solver logs, and checkpoints remain local-only. The two
charts that expose absolute commercial objective or cost totals are also excluded.
These private files are not needed to run the repository or review the submission
narrative.

The technical paper, summary, planner view, and presentation contain screened aggregate
metrics. Synthetic controls demonstrate the execution and validation workflow and are
not represented as business impact.

## Reproducing or replacing evidence

Run the smoke profile first to create a new local evidence bundle:

```bash
python scripts/run_challenge_study.py \
  --bundle-dir data/raw/nestle_challenge \
  --profile smoke
```

Then run `full`. Review feasibility, residuals, manifests, privacy, source state, and
figure consistency before promoting new privacy-safe figures. The publication command
reads audited local tables but writes figures only:

```bash
python scripts/create_submission_figures.py \
  --study-root results/challenge-study/notebook/full \
  --ibm-root results/challenge-study/notebook/ibm-presentation
```

The final audited grid contains 516 rows: 513 validated plans and three intentionally
infeasible frozen-routing controls at 60%, 65%, and 70% inventory reductions. The IBM
hardware figure summarizes 18 synthetic-only QPU jobs at 8,192 shots each; its runtime
panel uses a log axis and individual job dots to expose queue-time outliers.
