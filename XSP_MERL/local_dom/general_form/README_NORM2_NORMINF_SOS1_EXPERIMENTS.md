# Local-dominance SOS1 experiments for L2 and L-infinity norms

This bundle extends the existing LandS local-dominance experiments to two additional perturbation metrics:

1. **L2 / Euclidean radius**
2. **L-infinity / uniform componentwise radius**

For each norm there are two exact SOS1 models:

- **extended**: KKT + SOS1 on both the EV and `x_star` recourse LPs;
- **one-sided**: EV is only primal feasible, while `x_star` is KKT-optimal via SOS1.

The one-sided reduction remains exact for both norms because it changes the representation of the violation set, not the distance metric used to reach that set.

---

## 1. Why the new norms make sense

For the same violation set

```text
V = {xi : Delta(xi) <= 0},
```

the local radius is simply the distance from the realized positive-gap scenario `xi_hat` to `V`.

### L2

`L2` asks for the smallest Euclidean displacement from the realized scenario to a tie/reversal. It is a natural aggregate measure when multiple demand components can move simultaneously and Euclidean geometry is meaningful after scaling.

The code minimizes the **squared** Euclidean distance

```text
sum_i (xi_i - xi_hat_i)^2
```

because this has exactly the same minimizers as the Euclidean norm. The reported radius is the square root of the solver objective.

### L-infinity

`L-infinity` is especially interpretable as a robustness tolerance. It asks for the smallest `rho` such that a tie/reversal is possible while every component satisfies

```text
|xi_i - xi_hat_i| <= rho.
```

Equivalently, for every radius strictly below the optimum, the ordering is protected against **all simultaneous componentwise deviations inside the box** `[-rho,+rho]` around the realized scenario.

For LandS, the active uncertain coordinates are demand components in common demand units, so this is a meaningful absolute demand-tolerance interpretation. If future models mix variables with different units or very different scales, use weighted/normalized norms.

A useful cross-norm check is

```text
gamma_inf <= gamma_2 <= gamma_1
```

when all three radii use the same center and violation set.

---

## 2. Exact recourse KKT system

For fixed first-stage `x`, the recourse problem is

```text
min  d^T y
s.t. W y <= H xi - T x
     y >= 0.
```

With `mu = -pi >= 0`, define

```text
row_slack   = H xi - T x - W y >= 0
reduced_cost = d + W^T mu        >= 0.
```

KKT complementarity is

```text
mu[r] * row_slack[r] = 0
 y[j] * reduced_cost[j] = 0.
```

The implementation replaces those products exactly by native Gurobi SOS1 pairs:

```text
SOS1(row_slack[r], mu[r])
SOS1(y[j], reduced_cost[j]).
```

By default, `PreSOS1BigM=0` is imposed so the benchmark measures the native SOS1 formulation rather than an automatic big-M presolve conversion.

---

## 3. Files

### Shared implementation

```text
local_dominance_sos1_norm_common.py
sos1_norm_scalability_common.py
```

### L2 extended

```text
local_dominance_radius_norm2_extended_sos1.py
run_norm2_extended_sos1_scalability.py
```

### L2 one-sided

```text
local_dominance_radius_norm2_one_sided_sos1.py
run_norm2_one_sided_sos1_scalability.py
```

### L-infinity extended

```text
local_dominance_radius_norminf_extended_sos1.py
run_norminf_extended_sos1_scalability.py
```

### L-infinity one-sided

```text
local_dominance_radius_norminf_one_sided_sos1.py
run_norminf_one_sided_sos1_scalability.py
```

Place all files beside the existing `local_dominance_radius_unbounded.py` in `local_dom/general_form/`.

---

## 4. Manual single-instance runs

### L2 extended

```bash
python run_local_dominance_norm2_extended_sos1.py \
    lands_instance_small_1_1_1 \
    --time-limit 300 \
    --threads 1
```

### L2 one-sided

```bash
python run_local_dominance_norm2_one_sided_sos1.py \
    lands_instance_small_1_1_1 \
    --time-limit 300 \
    --threads 1
```

### L-infinity extended

```bash
python run_local_dominance_norminf_extended_sos1.py \
    lands_instance_small_1_1_1 \
    --time-limit 300 \
    --threads 1
```

### L-infinity one-sided

```bash
python run_local_dominance_norminf_one_sided_sos1.py \
    lands_instance_small_1_1_1 \
    --time-limit 300 \
    --threads 1
```

### Naive instance

Any of the four model files accepts

```bash
python run_local_dominance_norminf_one_sided_sos1.py \
    lands_instance_naive \
    --time-limit 300 \
    --threads 1
```

### Explicit positive-gap scenario

```bash
python run_local_dominance_norm2_extended_sos1.py \
    lands_instance_small_1_1_1 \
    --scenario scenario_3 \
    --time-limit 300
```

The script rejects a requested scenario unless its saved initial cost gap satisfies `Delta(xi_hat) > gap_tol`.

### Optional strict-reversal audit

The default computational boundary is `Delta(xi) <= 0`. To instead require a strict negative margin numerically, use

```bash
--violation-margin 1e-6
```

which enforces

```text
Delta(xi) <= -1e-6.
```

---

## 5. Scalability sweeps

The default sweep intentionally runs **small + medium only**. On the supplied benchmark tree this is 30 generated instances.

### L2 extended

```bash
python run_norm2_extended_sos1_scalability.py \
    --time-limit 300 \
    --threads 1 \
    --rerun-policy missing
```

### L2 one-sided

```bash
python run_norm2_one_sided_sos1_scalability.py \
    --time-limit 300 \
    --threads 1 \
    --rerun-policy missing
```

### L-infinity extended

```bash
python run_norminf_extended_sos1_scalability.py \
    --time-limit 300 \
    --threads 1 \
    --rerun-policy missing
```

### L-infinity one-sided

```bash
python run_norminf_one_sided_sos1_scalability.py \
    --time-limit 300 \
    --threads 1 \
    --rerun-policy missing
```

### Add the naive instance

```bash
--include-naive
```

### Include all hard instances

```bash
--include-hard
```

This expands the generated-instance sweep from 30 small+medium cases to all 45 small+medium+hard cases.

### Run one selected hard instance without enabling the whole hard family

```bash
python run_norminf_one_sided_sos1_scalability.py \
    --instances lands_instance_hard_1_1_1 \
    --time-limit 300 \
    --threads 1
```

### Preview the batch without solving

```bash
python run_norm2_extended_sos1_scalability.py --dry-run
```

This is useful for checking instance discovery before starting a long Gurobi experiment.

---

## 6. Summary directories and files

The requested summary roots are used exactly:

```text
Norm_2_extended_sos1/
Norm_2_one_sided_sos1/
Norm_inf_extended_sos1/
Norm_inf_one_sided_sos1/
```

For example, the L2 extended runner creates

```text
Norm_2_extended_sos1/
├── Norm_2_extended_sos1_scalability_summary.md
├── Norm_2_extended_sos1_scalability_summary.csv
├── Norm_2_extended_sos1_scalability_summary.json
├── Norm_2_extended_sos1_scalability_grouped.csv
└── Norm_2_extended_sos1_scalability_grouped.json
```

The other three experiments use the analogous names.

The summary is rewritten after **every processed instance**, so a time-consuming batch can be interrupted without losing earlier results.

---

## 7. Per-instance outputs

Each model has its own per-instance output folder:

```text
output_data/norm2_extended_sos1_exp/
output_data/norm2_one_sided_sos1_exp/
output_data/norminf_extended_sos1_exp/
output_data/norminf_one_sided_sos1_exp/
```

Each result records:

- solver status and runtime;
- returned radius and radius best bound;
- raw solver objective and raw best bound;
- MIP gap and node count;
- model dimensions and SOS1 count;
- selected positive-gap scenario;
- returned `xi`;
- independent EV and `x_star` recourse LP values at the returned `xi`;
- true cost gap from those independent LP solves;
- KKT/SOS1 residual diagnostics.

For L2, the raw solver objective/bound are in **squared-radius units** and the corresponding square-root radius values are stored separately.

---

## 8. Structural sizes

Let

```text
m  = dim(xi)
ny = number of second-stage variables
r  = number of recourse rows.
```

Then the raw formulations have:

| Model | Variables | Linear constraints | SOS1 sets | Quadratic objective |
|---|---:|---:|---:|---:|
| L2 extended | `m + 4 ny + 4 r` | `2 r + 2 ny + 1` | `2(r+ny)` | `m` terms |
| L2 one-sided | `m + 3 ny + 2 r` | `2 r + ny + 1` | `r+ny` | `m` terms |
| L-inf extended | `m + 1 + 4 ny + 4 r` | `2m + 2r + 2ny + 1` | `2(r+ny)` | none |
| L-inf one-sided | `m + 1 + 3 ny + 2 r` | `2m + 2r + ny + 1` | `r+ny` | none |

Thus the one-sided formulation retains the same structural advantage for both new norms: it removes the EV-side KKT system and halves the number of SOS1 complementarity sets relative to the extended formulation.

---

## 9. Recommended experiment order

Given the current L1 SOS1 evidence, use the same controlled protocol:

```text
small -> medium -> hard only if the medium results justify going further.
```

Do not assume in advance that L-infinity will solve faster merely because its distance objective is linear, or that L2 will solve slower because its objective is quadratic. Once SOS1 branching dominates, the relative solve behavior is empirical.

---

## 10. Validation performed on this bundle

The bundle was checked for:

- Python syntax (`py_compile`) for all files;
- LandS instance discovery on the supplied benchmark tree;
- default discovery = 30 small+medium instances;
- `--include-hard` discovery = 45 generated instances;
- `--include-naive` with default size policy = 31 instances;
- successful loading of naive, small, medium, and hard LandS data;
- successful positive-gap scenario selection on those representative cases;
- CLI help and dry-run behavior.

An actual SOS1 optimization solve was not executed in the artifact environment because `gurobipy` is not installed there. Run the optimization in the existing licensed Gurobi environment used for the L1 experiments.
