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

The final paper, summary, planner view, and presentation contain screened aggregate
metrics. Synthetic controls demonstrate the execution and validation workflow, not
business impact or quantum advantage.

## Reproducing or replacing evidence

Run the smoke profile first to create a new local evidence bundle:

```bash
python scripts/run_challenge_study.py \
  --bundle-dir data/raw/nestle_challenge \
  --profile smoke
```

Then run `full`. Do not overwrite the published figures automatically. Review
feasibility, residuals, manifests, privacy, source state, and figure consistency before
promoting any new privacy-safe figures.
