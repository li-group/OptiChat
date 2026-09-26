# Hadamard-Cramer Big-M local-dominance experiment

Place these files in `local_dom/general_form/`:

- `local_dominance_radius_hadamard_bigm.py` — core formulation, bound construction, solve, validation, per-instance outputs.
- `run_local_dominance_hadamard.py` — thin single-instance CLI.
- `run_hadamard_scalability.py` — sequential scalability benchmark over all generated LANDS instances.
- `test_hadamard_bigm_bounds.py` — unit tests for the pure bound formulas.

The existing `local_dominance_radius_unbounded.py` must remain beside them because the experiment reuses `LocalDominanceData`.

## Formulation being tested

The model keeps the reduced/asymmetric logic:

- EV side: primal-feasible `y_EV` only.
- `x_star` side: KKT optimality only.
- no EV dual/KKT copy.

The star recourse inequality system is converted to standard form with row slacks:

`Wbar = [W I_r]`, `u = [y_star; row_slack]`, `qbar = [d; 0]`.

The two KKT complementarity families are linearized with Fortuny-Amat binaries:

1. `(row_slack_i, mu_i)` where `mu=-pi_star >= 0`;
2. `(y_star_j, reduced_cost_j)` where `reduced_cost = d + W^T mu >= 0`.

## Hadamard-Cramer bounds

The code first finds an independently verified violating point `xi_bar` and sets

`U = ||xi_bar - xi_hat||_1`.

It adds the safe cutoff `sum(t) <= U` and explicit implied bounds `xi_hat-U <= xi <= xi_hat+U`.

For every `xi` in this L1 ball,

`|b_star_i(xi)| <= |b_star_i(xi_hat)| + U * ||H_i,:||_inf`.

Let `Bmax` be the maximum of these rowwise bounds, `Wmax=max(1,max|W|)`, and `qmax=max|d|`.
For recourse rank `r`, the dense Hadamard coefficient is `r^(r/2)`.

The MILP uses

- `M_primal = r^(r/2) * Bmax * Wmax^(r-1)` for `y_star` and row slacks;
- `M_mu = r^(r/2) * qmax * Wmax^(r-1)` for `mu=-pi_star`;
- `M_rc[j] = |d_j| + ||W[:,j]||_1 * M_mu` for each reduced cost.

The code also reports the corresponding factorial/Cramer reference constants, but does not use them in the solve.

### Why the primal bound is rank-r rather than the fully general optimistic-bilevel `ell` bound

In this local-dominance application, `y_star` enters the upper-level certificate only through its optimal objective value `d^T y_star`. We do not need to preserve a particular nonbasic follower point selected by extra componentwise upper-level coupling constraints. Therefore, at any candidate `xi`, one optimal basic recourse solution is sufficient. This permits the sharper rank-`r` Hadamard-Cramer vertex bound after the safe radius `U` bounds the follower RHS.

The code verifies that `W` is integral before using the determinant-denominator argument. The generated LANDS instances have integer `W` entries.

## Output location

Every single-instance run writes to exactly:

`<instance_name>/output_data/hadamard_exp/`

including:

- `hadamard_bounds.json`
- `hadamard_bounds_scenario_k.json`
- `hadamard_result.json`
- `hadamard_result_scenario_k.json`
- `gurobi_scenario_k.log`

The result includes the Hadamard and factorial reference bounds, LP relaxation objective, model size, runtime, node count, incumbent, best bound, MIP gap, and independent recourse validation.

Aggregate reports from the batch runner are written to:

`general_form/hadamard_scalability_summary/`

with CSV, JSON, and Markdown per-instance and grouped summaries.

## Run one generated instance

From `local_dom/general_form/`:

```bash
python run_local_dominance_hadamard.py \
    lands_instance_small_1_1_1 \
    --time-limit 300 \
    --threads 1
```

Explicit scenario:

```bash
python run_local_dominance_hadamard.py \
    lands_instance_small_1_1_1 \
    --scenario scenario_3 \
    --time-limit 300 \
    --threads 1
```

## Run the naive instance

```bash
python run_local_dominance_hadamard.py \
    lands_instance_naive \
    --time-limit 300 \
    --threads 1
```

## Run all generated LANDS instances

```bash
python run_hadamard_scalability.py \
    --time-limit 300 \
    --threads 1
```

Include the naive instance too:

```bash
python run_hadamard_scalability.py \
    --include-naive \
    --time-limit 300 \
    --threads 1
```

Rerun only non-optimal prior results:

```bash
python run_hadamard_scalability.py \
    --time-limit 600 \
    --threads 1 \
    --rerun-policy nonoptimal
```

## Recommended primary benchmark protocol

Use `300 s`, one Gurobi thread, seed 1, and `NumericFocus=0` for the primary table so it is directly comparable to the earlier native-SOS1 study. If very large Hadamard constants trigger numerical difficulty, run a *separate sensitivity experiment* with e.g. `--numeric-focus 2`; do not mix those settings into the primary scalability table.

The batch report records both MIP runtime and total runtime including the safe-incumbent/bound-preprocessing step.

## Unit tests

```bash
pytest -q test_hadamard_bigm_bounds.py
```

The supplied tests verify the rank-2 equality of factorial and Hadamard factors, strict Hadamard improvement from rank 3 onward, the componentwise reduced-cost refinement, and inclusion of the identity slack block in `Wbar`.
