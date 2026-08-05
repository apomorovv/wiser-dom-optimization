# Data privacy and publication policy

## Core rule

Use only challenge-approved or independently generated synthetic data. Do not publish or commit raw operational records, customer/order identifiers, confidential commercial costs, or identifiable DC-level details.

## Directory treatment

| Path | Default treatment |
|---|---|
| `data/raw/` | Restricted; never commit contents. |
| `data/interim/` | Restricted; never commit contents. |
| `data/processed/` | Restricted unless approved and fully anonymized. |
| `data/synthetic/` | Commit only independently generated safe data. |
| `runs/` | Local by default; may contain restricted decoded outputs. |
| `results/` | Commit only aggregate or approved anonymized results. |

## Do not publish

- source customer names or addresses;
- real sales-document, load, material, plant, or customer identifiers;
- exact commercially sensitive shipping costs or penalties;
- raw order-level or SKU-level exports;
- unrestricted source-spreadsheet screenshots;
- confidential DC inventory or capacity;
- public cloud links exposing restricted data.

## Normally safe public content

Subject to challenge approval:

- synthetic IDs such as `O1`, `SKU_A`, and `D1`;
- independently generated synthetic costs and demands;
- aggregate fill rates and objective improvements;
- normalized or indexed costs;
- runtime, model size, and solver methodology;
- synthetic planner examples.

## Anonymization

Renaming an identifier is insufficient when a row remains matchable by dates, quantities, costs, and locations. For public artifacts:

1. remove source identifiers;
2. aggregate or regenerate sensitive values;
3. remove precise locations;
4. use synthetic dates when exact dates are unnecessary;
5. inspect plots, notebook output, logs, and exception traces;
6. verify that git history contains no deleted sensitive files.

## External computing

Do not upload restricted data to public notebooks, external quantum platforms, third-party APIs, or unapproved clouds. Use synthetic data, approved aggregates, or an approved restricted environment.

### Quantum-service gate

The QPU adapters are disabled unless `allow_remote=true`. They replace order, SKU,
DC, and candidate labels with local integer indices before transmission. That avoids
literal identifiers in remote variable names, but it does not make the payload
non-sensitive: objective and coupling coefficients can encode shipping economics,
penalties, scarcity, and network structure.

Before enabling a remote backend, record approval for the provider, region, account,
retention policy, coefficient payload, and experiment window. Never include raw
tables, source paths, decoded identifiers, or planner outputs in sampler labels,
metadata, logs, or support requests.

`simulated_annealing`, `exact`, and `random` execute locally and are the safe default
for restricted inputs.

## Git checks

Before every push:

```bash
git status --short
git diff --cached --name-only
git diff --cached | less
```

If restricted data were committed, deleting them in a later commit is insufficient because git history retains them. Stop and rewrite history before pushing further.

## Publication checklist

- [ ] IDs are synthetic or approved.
- [ ] No raw source rows are reproduced.
- [ ] Costs are aggregate, normalized, synthetic, or approved.
- [ ] No confidential DC inventory/capacity appears.
- [ ] Logs do not reveal restricted paths.
- [ ] External links have correct permissions.
- [ ] Remote QUBO execution is disabled or explicitly approved.
- [ ] Remote variable labels contain integers only.
- [ ] Figures cannot be reverse-matched to records.
- [ ] A second team member completed privacy review.

