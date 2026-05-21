# General-form two-stage stochastic programming workflow

This folder implements the deterministic equivalent of

```text
min_x,y_s c^T x + sum_s p_s d^T y_s
s.t.    A x >= b
        x >= 0
        T x + W y_s <= H xi_s, y_s >= 0, for all scenarios s.
```

The current instance folder is `lands_instance/`.

## Main TSSP workflow

Files:

- `general_model.py`: reusable model classes and workflow functions.
- `run_workflow.py`: command-line wrapper for the stochastic/EV/gap workflow.
- `test_general_model.py`: pytest unit tests against `lands_instance/test_data.json`.
- `lands_instance/instance_data.json`: input data.

Run from inside `general_form/`:

```bash
python run_workflow.py lands_instance --solver gurobi --backend pyomo
```

Use `--backend auto` to fall back to SciPy when Pyomo/Gurobi is unavailable.
The workflow writes:

- `lands_instance/stochastic_results.json`
- `lands_instance/expectedvalue_results.json`
- `lands_instance/cost_gap_diagnostics.json`
- `lands_instance/cost_gap_diagnostiocs.json` compatibility alias for the filename typo in the prompt

## Isolated local-dominance-radius workflow

This second-stage workflow assumes the files above already exist. It does not
import or reuse `general_model.py`; it reconstructs the needed matrices and
saved first-stage decisions directly from JSON files.

Files:

- `local_dominance_radius.py`: isolated implementation of the local dominance radius model.
- `run_local_dominance.py`: command-line wrapper.
- `test_local_dominance_radius.py`: pytest tests for scenario selection, dimensions, and a small-instance LP cross-check.

Default Gurobi/Pyomo run:

```bash
python run_local_dominance.py lands_instance --solver gurobi --method pyomo
```

To manually choose a positive-gap scenario:

```bash
python run_local_dominance.py lands_instance --scenario scenario_3 --solver gurobi --method pyomo
```

The Pyomo model contains bilinear strong-duality equations, so the Gurobi option
`NonConvex=2` is set automatically.

The local-dominance workflow writes:

- `lands_instance/cost_gap_dominating_scenarios.json`
- `lands_instance/local_dominance_radius.json` when solved with `--method pyomo`

There is also an optional small-instance validation method:

```bash
python run_local_dominance.py lands_instance --scenario scenario_3 --method enumeration
```

For this dummy instance, the enumeration cross-check writes
`lands_instance/local_dominance_radius_enumeration_check.json` and gives

```text
gamma_l1_radius = 1.7619047619047619
nearest_violation_xi = [0, 0, 0, 0, 5.238095238095238, 3, 2]
```

This optional enumeration is not intended to replace the requested Pyomo/Gurobi
primal-dual model for larger instances.

## Test

```bash
pytest -q
```

The main TSSP tests use `backend="auto"`, so they run with Pyomo+Gurobi when
available and SciPy otherwise. The local-dominance Pyomo dimension test is
skipped automatically when Pyomo is not installed.

## Directional upper bound workflow

The isolated directional workflow computes the coordinate-ray upper bound

\[
\bar\gamma = \min_{i,\sigma\in\{-1,+1\}} \inf\{t\ge 0 : \Delta(\hat\xi + \sigma t e_i) \le 0\}.
\]

It assumes the main stochastic/EV workflow has already produced
`stochastic_results.json`, `expectedvalue_results.json`, and
`cost_gap_diagnostics.json` inside the selected instance folder.

Run:

```bash
python run_directional_upper_bound.py lands_instance
```

Optionally select a specific positive-gap scenario:

```bash
python run_directional_upper_bound.py lands_instance --scenario scenario_3
```

Output is written to:

```text
lands_instance/directional_upper_bound_results.json
```

The implementation is isolated in `directional_upper_bound.py`. It converts the
recourse LP into standard form, enumerates dual-feasible bases of `[W I]`, and
tracks the optimal basis along each coordinate ray. At each breakpoint it uses a
deterministic lexicographic anti-cycling rule: among all dual-feasible,
primal-feasible bases at the breakpoint, it selects the lexicographically first
basis that is feasible for a right-neighborhood of the ray parameter. The output
stores every interval, the basis active on that interval, leaving variables at
breakpoints, and all basis-change events for both `x_EV` and `x_star`.
