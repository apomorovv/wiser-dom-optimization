# Planner Copilot

The Streamlit app explains aggregate challenge experiments without sending data to an
external language model. It rejects direct identifier columns and does not accept raw
order, customer, SKU, DC, ZIP, or lane tables.

Run the experiment suite first:

```bash
python scripts/run_challenge_study.py \
  --bundle-dir /approved/path/to/challenge-files \
  --profile full \
  --output results/challenge-study/cli/full/aggregate_results.csv
```

Then start the app:

```bash
python -m streamlit run apps/planner_copilot.py
```

The app answers bounded questions about solver comparison, scaling, penalties,
candidate counts, inventory shocks, simulator seed/noise robustness, Pareto pruning,
batching, and evidence limitations. It is a decision-support interface, not an
autonomous routing agent.
