# SOS1 local-dominance experiment

Copy these files into `local_dom/general_form/`:

- `local_dominance_radius_sos1.py`
- `run_local_dominance_sos1.py`
- `run_sos1_scalability.py`

The baseline SP/EV workflow must already have produced, for each instance:

- `output_data/stochastic_results.json`
- `output_data/expectedvalue_results.json`
- `output_data/cost_gap_diagnostics.json`

## Single instance

From `local_dom/general_form/`:

```bash
python run_local_dominance_sos1.py lands_instance_small_1_1_1 \
    --time-limit 300 \
    --threads 1
```

Explicit scenario:

```bash
python run_local_dominance_sos1.py lands_instance_small_1_1_1 \
    --scenario scenario_3 \
    --time-limit 300 \
    --threads 1
```

Naive instance:

```bash
python run_local_dominance_sos1.py lands_instance_naive \
    --time-limit 300 \
    --threads 1
```

Results are written to:

```text
<instance>/output_data/sos1_exp/
    sos1_result.json
    sos1_result_<scenario>.json
    gurobi_<scenario>.log
```

## All generated instances

```bash
python run_sos1_scalability.py \
    --time-limit 300 \
    --threads 1 \
    --rerun-policy missing
```

Include the naive instance too:

```bash
python run_sos1_scalability.py \
    --include-naive \
    --time-limit 300 \
    --threads 1 \
    --rerun-policy missing
```

Aggregate reports are written to:

```text
general_form/sos1_scalability_summary/
    sos1_scalability_summary.csv
    sos1_scalability_summary.json
    sos1_scalability_summary.md
    sos1_scalability_grouped.csv
    sos1_scalability_grouped.json
```

## Recommended timing protocol

Use 300 seconds per instance for the first full sweep.  With 30 generated
instances this caps the worst-case solver time at about 2.5 hours.  Any instance
that hits the limit still records the incumbent, best bound and MIP gap.

Then rerun only non-optimal cases at 600 seconds:

```bash
python run_sos1_scalability.py \
    --time-limit 600 \
    --threads 1 \
    --rerun-policy nonoptimal
```

## Important SOS1 detail

The code sets Gurobi `PreSOS1BigM=0`.  This is deliberate: it prevents Gurobi
presolve from replacing the SOS1 constraints with a big-M reformulation, so the
experiment is genuinely testing native SOS1 branching rather than an implicit
big-M model.

The star-side KKT system needs both complementarity families:

1. `(row_slack[r], mu[r])` with `mu = -pi_star >= 0`;
2. `(y_star[j], reduced_cost[j])`.

Only implementing the first family (as in the screenshot fragment) does not
fully enforce optimality of the star recourse LP.
