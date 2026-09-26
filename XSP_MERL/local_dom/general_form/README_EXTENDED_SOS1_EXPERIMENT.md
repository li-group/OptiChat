# Extended/symmetric SOS1 local-dominance experiment

This bundle mirrors the existing one-sided SOS1 experiment, but enforces KKT
optimality for **both** recourse LPs.

Copy these files into `local_dom/general_form/` beside
`local_dominance_radius_unbounded.py`:

- `local_dominance_radius_extended_sos1.py`
- `run_local_dominance_extended_sos1.py`
- `run_extended_sos1_scalability.py`

The baseline SP/EV workflow must already have produced, for every instance:

- `output_data/stochastic_results.json`
- `output_data/expectedvalue_results.json`
- `output_data/cost_gap_diagnostics.json`

## 1. Mathematical formulation being tested

For a selected realized scenario `xi_hat` with `Delta(xi_hat) > gap_tol`, the
model solves

```text
min ||xi - xi_hat||_1
```

and makes both recourse solutions optimal at the candidate `xi`.

For each side `k in {EV, star}`:

```text
y_k >= 0
W y_k + row_slack_k = H xi - T x_k
mu_k >= 0                           (mu_k = -pi_k)
reduced_cost_k = d + W^T mu_k >= 0
SOS1(row_slack_k[r], mu_k[r])       for every row r
SOS1(y_k[j], reduced_cost_k[j])     for every recourse variable j
```

The comparison constraint is

```text
c^T x_EV + d^T y_EV + epsilon <= c^T x_star + d^T y_star.
```

- `epsilon = 0` computes the nearest point satisfying `Delta(xi) <= 0`.
- `epsilon > 0` computes the nearest strict reversal with margin
  `Delta(xi) <= -epsilon`; this is useful for auditing the infimum/minimum issue.

With `epsilon = 0`, the model is the native-SOS1 counterpart of the symmetric
primal-dual/strong-duality formulation, but it contains no bilinear terms.

### Model-size comparison

Let `m = dim(xi)`, `ny = dim(y)`, and `r = rows(W)`.

| Formulation | Variables | Linear constraints | SOS1 sets |
|---|---:|---:|---:|
| Existing one-sided SOS1 | `2m + 3ny + 2r` | `2m + 2r + ny + 1` | `r + ny` |
| Extended/symmetric SOS1 | `2m + 4ny + 4r` | `2m + 2r + 2ny + 1` | `2(r + ny)` |

The extended model therefore doubles the complementarity/SOS1 families because
both value functions are represented exactly.

## 2. Single-instance runs

Run from `local_dom/general_form/`.

### Generated LandS instance

```bash
python run_local_dominance_extended_sos1.py lands_instance_small_1_1_1 \
    --time-limit 300 \
    --threads 1
```

### Explicit positive-gap scenario

```bash
python run_local_dominance_extended_sos1.py lands_instance_small_1_1_1 \
    --scenario scenario_3 \
    --time-limit 300 \
    --threads 1
```

### Naive instance

```bash
python run_local_dominance_extended_sos1.py lands_instance_naive \
    --time-limit 300 \
    --threads 1
```

### Optional strict-reversal audit

For example, require `Delta(xi) <= -1e-6`:

```bash
python run_local_dominance_extended_sos1.py lands_instance_small_1_1_1 \
    --violation-margin 1e-6 \
    --time-limit 300 \
    --threads 1
```

Per-instance results are written to

```text
<instance>/output_data/sos1_extended_exp/
    extended_sos1_result.json
    extended_sos1_result_<scenario>.json
    gurobi_<scenario>.log
```

The JSON includes independent LP checks for **both** KKT systems.

## 3. Scalability sweep: small + medium by default

The batch runner is deliberately sequential, matching the existing experiment.
It defaults to all generated **small and medium** instances and skips hard ones.

```bash
python run_extended_sos1_scalability.py \
    --time-limit 300 \
    --threads 1 \
    --rerun-policy missing
```

Include the naive instance as well:

```bash
python run_extended_sos1_scalability.py \
    --include-naive \
    --time-limit 300 \
    --threads 1
```

## 4. How to include the hard instances

Add `--include-hard`:

```bash
python run_extended_sos1_scalability.py \
    --include-hard \
    --time-limit 300 \
    --threads 1 \
    --rerun-policy missing
```

You may also target only one hard instance explicitly, even without the flag:

```bash
python run_extended_sos1_scalability.py \
    --instances lands_instance_hard_1_1_1 \
    --time-limit 300 \
    --threads 1
```

Given the existing one-sided experiment already times out on all five
`med_2_1_*` and all five `med_3_1_*` cases at 300 seconds, a sensible first
extended-formulation sweep is small + medium only.

## 5. Aggregate summary files

The batch runner updates the summary **after every instance**, so an interrupted
experiment retains all completed rows.

```text
general_form/extended_sos1_scalability_summary/
    extended_sos1_scalability_summary.csv
    extended_sos1_scalability_summary.json
    extended_sos1_scalability_summary.md
    extended_sos1_scalability_grouped.csv
    extended_sos1_scalability_grouped.json
```

The tables report instance dimensions, raw model size, SOS1 count, solver status,
runtime, incumbent radius, best bound, MIP gap, node count, true gap at the
returned `xi`, and the independent LP-vs-KKT validation errors on both sides.

## 6. Recommended experiment protocol

First pass:

```bash
python run_extended_sos1_scalability.py \
    --time-limit 300 \
    --threads 1 \
    --rerun-policy missing
```

Then rerun only non-optimal cases at 600 seconds:

```bash
python run_extended_sos1_scalability.py \
    --time-limit 600 \
    --threads 1 \
    --rerun-policy nonoptimal
```

Only after observing the medium behavior should the hard class be enabled.

## 7. Native-SOS1 requirement

The code sets

```text
PreSOS1BigM = 0
```

by default. This is deliberate. It prevents Gurobi presolve from replacing the
SOS1 sets by a hidden big-M reformulation, so the experiment is genuinely a
native-SOS1 scalability test.

Each recourse LP needs **both** complementarity families:

1. `SOS1(row_slack[r], mu[r])`;
2. `SOS1(y[j], reduced_cost[j])`.

The extended formulation has those two families on both the EV and `x_star`
sides, for a total of `2(r + ny)` SOS1 sets.
