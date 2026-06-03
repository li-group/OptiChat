"""Reusable deterministic-equivalent workflow for the LandS TSSP.

The model convention is

    min c^T x + sum_s p_s d^T y_s
    s.t. A x >= b, x >= 0,
         T x + W y_s <= H xi_s, y_s >= 0.

The module provides a strict Pyomo solve path and a scipy.optimize.linprog
fallback path for testing without Pyomo/Gurobi.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


Number = float
JsonDict = Dict[str, Any]


# ---------------------------------------------------------------------------
# Basic data utilities
# ---------------------------------------------------------------------------

def load_instance(path: str | Path) -> JsonDict:
    """Load a LandS JSON instance from disk."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _matrix_from_dict(obj: Mapping[str, Any], name: str) -> np.ndarray:
    """Convert a row-major matrix dictionary to a numpy array."""
    if not isinstance(obj, Mapping):
        raise ValueError(f"Matrix {name!r} must be a dictionary with rows, cols, data.")
    for key in ("rows", "cols", "data"):
        if key not in obj:
            raise ValueError(f"Matrix {name!r} is missing required key {key!r}.")
    rows = int(obj["rows"])
    cols = int(obj["cols"])
    data = list(obj["data"])
    if rows < 0 or cols < 0:
        raise ValueError(f"Matrix {name!r} has invalid shape ({rows}, {cols}).")
    if len(data) != rows * cols:
        raise ValueError(
            f"Matrix {name!r} has {len(data)} entries but shape ({rows}, {cols}) "
            f"requires {rows * cols} entries."
        )
    return np.asarray(data, dtype=float).reshape((rows, cols))


def _as_float_vector(data: Any, name: str) -> np.ndarray:
    arr = np.asarray(data, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"Vector {name!r} must be one-dimensional.")
    return arr


def _scenario_key_sort_value(key: str) -> Tuple[int, str]:
    try:
        return (int(key.split("_", 1)[1]), key)
    except Exception:
        return (10**12, key)


def get_scenario_ids(data: Mapping[str, Any]) -> List[str]:
    """Return scenario identifiers such as ['xi_1', 'xi_2', ...]."""
    ids = [k for k in data if isinstance(k, str) and k.startswith("xi_")]
    ids.sort(key=_scenario_key_sort_value)
    return ids


def _extract_arrays(data: Mapping[str, Any]) -> Dict[str, Any]:
    """Extract matrices/vectors from raw JSON data as numpy arrays."""
    arrays = {
        "A": _matrix_from_dict(data["A"], "A"),
        "T": _matrix_from_dict(data["T"], "T"),
        "W": _matrix_from_dict(data["W"], "W"),
        "H": _matrix_from_dict(data["H"], "H"),
        "b": _as_float_vector(data["b"], "b"),
        "c": _as_float_vector(data["c"], "c"),
        "d": _as_float_vector(data["d"], "d"),
        "p_s": _as_float_vector(data["p_s"], "p_s"),
    }
    scenario_ids = get_scenario_ids(data)
    arrays["scenario_ids"] = scenario_ids
    arrays["xis"] = [_as_float_vector(data[sid], sid) for sid in scenario_ids]
    return arrays


def _to_float_list(values: Sequence[float] | np.ndarray, ndigits: Optional[int] = None) -> List[float]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if ndigits is None:
        return [float(v) for v in arr]
    return [round(float(v), ndigits) for v in arr]


def _json_safe(value: Any) -> Any:
    """Recursively convert numpy scalar/array values to JSON-safe Python values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return str(value)
    return value


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(dict(payload)), f, indent=2, sort_keys=False)
        f.write("\n")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_instance(data: Mapping[str, Any], tolerance: float = 1e-8) -> bool:
    """Validate dimensions and probabilities of a LandS instance.

    Raises
    ------
    ValueError
        If any consistency check fails.
    """
    required = ["A", "b", "c", "d", "p_s", "T", "W", "H"]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"Instance is missing required key(s): {missing}.")

    arrays = _extract_arrays(data)
    A, b, c = arrays["A"], arrays["b"], arrays["c"]
    T, W, H = arrays["T"], arrays["W"], arrays["H"]
    d, p_s = arrays["d"], arrays["p_s"]
    scenario_ids, xis = arrays["scenario_ids"], arrays["xis"]

    if A.shape[0] != b.shape[0]:
        raise ValueError(
            f"A row count ({A.shape[0]}) must equal length of b ({b.shape[0]})."
        )
    if A.shape[1] != c.shape[0]:
        raise ValueError(
            f"A column count ({A.shape[1]}) must equal number of first-stage variables "
            f"len(c) ({c.shape[0]})."
        )
    if T.shape[1] != c.shape[0]:
        raise ValueError(
            f"T column count ({T.shape[1]}) must equal number of first-stage variables "
            f"len(c) ({c.shape[0]})."
        )
    if W.shape[1] != d.shape[0]:
        raise ValueError(
            f"W column count ({W.shape[1]}) must equal number of second-stage variables "
            f"len(d) ({d.shape[0]})."
        )
    if not (T.shape[0] == W.shape[0] == H.shape[0]):
        raise ValueError(
            "T, W, and H must have the same number of rows; got "
            f"T={T.shape[0]}, W={W.shape[0]}, H={H.shape[0]}."
        )
    if H.shape[1] <= 0:
        raise ValueError("H must have at least one column so that H xi_s is defined.")
    if len(scenario_ids) == 0:
        raise ValueError("Instance must contain at least one scenario key xi_1, xi_2, ... .")
    if p_s.shape[0] != len(scenario_ids):
        raise ValueError(
            f"Length of p_s ({p_s.shape[0]}) must equal number of scenarios "
            f"({len(scenario_ids)})."
        )
    for sid, xi in zip(scenario_ids, xis):
        if xi.shape[0] != H.shape[1]:
            raise ValueError(
                f"Scenario vector {sid!r} has length {xi.shape[0]}, but H has "
                f"{H.shape[1]} columns."
            )
    if np.any(p_s < -tolerance):
        bad = [(i + 1, float(p)) for i, p in enumerate(p_s) if p < -tolerance]
        raise ValueError(f"Scenario probabilities must be nonnegative; bad entries: {bad}.")
    if abs(float(np.sum(p_s)) - 1.0) > tolerance:
        raise ValueError(
            f"Scenario probabilities must sum to 1; got {float(np.sum(p_s)):.16g}."
        )
    return True


# ---------------------------------------------------------------------------
# Solver backends
# ---------------------------------------------------------------------------


def _linprog_solve(
    c_obj: np.ndarray,
    A_ub: Optional[np.ndarray],
    b_ub: Optional[np.ndarray],
    bounds: Sequence[Tuple[float, Optional[float]]],
) -> Dict[str, Any]:
    try:
        from scipy.optimize import linprog
    except Exception as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "backend='scipy' requires scipy.optimize.linprog to be installed."
        ) from exc

    res = linprog(
        c=np.asarray(c_obj, dtype=float),
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=bounds,
        method="highs",
    )
    termination = "optimal" if res.success else str(res.message)
    return {
        "success": bool(res.success),
        "status": int(res.status),
        "status_message": str(res.message),
        "termination_condition": termination,
        "objective": None if res.fun is None else float(res.fun),
        "x": None if res.x is None else np.asarray(res.x, dtype=float),
        "raw_result": res,
    }


def _check_pyomo_solver(solver: str):
    try:
        import pyomo.environ as pyo
    except Exception as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "backend='pyomo' was requested, but Pyomo is not installed."
        ) from exc
    opt = pyo.SolverFactory(solver)
    if opt is None or not opt.available(False):
        raise RuntimeError(
            f"backend='pyomo' was requested, but solver {solver!r} is not available."
        )
    return pyo, opt


def _is_optimal_termination(term: str) -> bool:
    term_lower = str(term).lower()
    return "optimal" in term_lower


# ---------------------------------------------------------------------------
# Deterministic equivalent solves
# ---------------------------------------------------------------------------


def _solve_de_scipy(
    data: Mapping[str, Any],
    scenario_ids: Sequence[str],
    xis: Sequence[np.ndarray],
    probabilities: Sequence[float],
) -> Dict[str, Any]:
    arrays = _extract_arrays(data)
    A, b, c = arrays["A"], arrays["b"], arrays["c"]
    T, W, H = arrays["T"], arrays["W"], arrays["H"]
    d = arrays["d"]

    n_x = c.size
    n_y = d.size
    n_s = len(xis)
    n_vars = n_x + n_s * n_y

    obj = np.zeros(n_vars, dtype=float)
    obj[:n_x] = c
    for s_idx, p in enumerate(probabilities):
        lo = n_x + s_idx * n_y
        obj[lo : lo + n_y] = float(p) * d

    rows: List[np.ndarray] = []
    rhs: List[float] = []

    # A x >= b -> -A x <= -b
    for r in range(A.shape[0]):
        row = np.zeros(n_vars, dtype=float)
        row[:n_x] = -A[r, :]
        rows.append(row)
        rhs.append(float(-b[r]))

    # T x + W y_s <= H xi_s
    for s_idx, xi in enumerate(xis):
        hxi = H @ xi
        y_lo = n_x + s_idx * n_y
        for r in range(T.shape[0]):
            row = np.zeros(n_vars, dtype=float)
            row[:n_x] = T[r, :]
            row[y_lo : y_lo + n_y] = W[r, :]
            rows.append(row)
            rhs.append(float(hxi[r]))

    A_ub = np.vstack(rows) if rows else None
    b_ub = np.asarray(rhs, dtype=float) if rhs else None
    solve = _linprog_solve(obj, A_ub, b_ub, bounds=[(0.0, None)] * n_vars)

    if not solve["success"]:
        return {
            "success": False,
            "status": solve["status"],
            "status_message": solve["status_message"],
            "termination_condition": solve["termination_condition"],
            "objective": None,
            "x": None,
            "y_by_scenario": {},
        }

    sol = solve["x"]
    x = sol[:n_x]
    y_by_scenario = {}
    for s_idx, sid in enumerate(scenario_ids):
        y_lo = n_x + s_idx * n_y
        y_by_scenario[sid] = sol[y_lo : y_lo + n_y]

    return {
        "success": True,
        "status": solve["status"],
        "status_message": solve["status_message"],
        "termination_condition": solve["termination_condition"],
        "objective": float(solve["objective"]),
        "x": x,
        "y_by_scenario": y_by_scenario,
    }


def _solve_de_pyomo(
    data: Mapping[str, Any],
    scenario_ids: Sequence[str],
    xis: Sequence[np.ndarray],
    probabilities: Sequence[float],
    solver: str,
) -> Dict[str, Any]:
    pyo, opt = _check_pyomo_solver(solver)
    arrays = _extract_arrays(data)
    A, b, c = arrays["A"], arrays["b"], arrays["c"]
    T, W, H = arrays["T"], arrays["W"], arrays["H"]
    d = arrays["d"]

    n_x = c.size
    n_y = d.size
    n_s = len(xis)
    hxi_by_idx = [H @ xi for xi in xis]

    m = pyo.ConcreteModel()
    m.I = pyo.RangeSet(0, n_x - 1)
    m.J = pyo.RangeSet(0, n_y - 1)
    m.S = pyo.RangeSet(0, n_s - 1)
    m.R1 = pyo.RangeSet(0, A.shape[0] - 1)
    m.R2 = pyo.RangeSet(0, T.shape[0] - 1)

    m.x = pyo.Var(m.I, domain=pyo.NonNegativeReals)
    m.y = pyo.Var(m.S, m.J, domain=pyo.NonNegativeReals)

    def first_stage_rule(mm, r):
        return sum(float(A[r, i]) * mm.x[i] for i in mm.I) >= float(b[r])

    def second_stage_rule(mm, s, r):
        lhs = sum(float(T[r, i]) * mm.x[i] for i in mm.I) + sum(
            float(W[r, j]) * mm.y[s, j] for j in mm.J
        )
        return lhs <= float(hxi_by_idx[s][r])

    def objective_rule(mm):
        return sum(float(c[i]) * mm.x[i] for i in mm.I) + sum(
            float(probabilities[s]) * float(d[j]) * mm.y[s, j]
            for s in mm.S
            for j in mm.J
        )

    m.first_stage = pyo.Constraint(m.R1, rule=first_stage_rule)
    m.second_stage = pyo.Constraint(m.S, m.R2, rule=second_stage_rule)
    m.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    result = opt.solve(m, tee=False)
    status = str(result.solver.status)
    term = str(result.solver.termination_condition)
    if not _is_optimal_termination(term):
        return {
            "success": False,
            "status": status,
            "status_message": status,
            "termination_condition": term,
            "objective": None,
            "x": None,
            "y_by_scenario": {},
        }

    x = np.asarray([pyo.value(m.x[i]) for i in range(n_x)], dtype=float)
    y_by_scenario = {}
    for s_idx, sid in enumerate(scenario_ids):
        y_by_scenario[sid] = np.asarray([pyo.value(m.y[s_idx, j]) for j in range(n_y)], dtype=float)

    return {
        "success": True,
        "status": status,
        "status_message": status,
        "termination_condition": term,
        "objective": float(pyo.value(m.obj)),
        "x": x,
        "y_by_scenario": y_by_scenario,
    }


def _solve_de(
    data: Mapping[str, Any],
    scenario_ids: Sequence[str],
    xis: Sequence[np.ndarray],
    probabilities: Sequence[float],
    backend: str,
    solver: str,
) -> Dict[str, Any]:
    backend = backend.lower()
    if backend == "scipy":
        return _solve_de_scipy(data, scenario_ids, xis, probabilities)
    if backend == "pyomo":
        return _solve_de_pyomo(data, scenario_ids, xis, probabilities, solver=solver)
    raise ValueError("backend must be either 'pyomo' or 'scipy'.")


# ---------------------------------------------------------------------------
# Public recourse and workflow functions
# ---------------------------------------------------------------------------


def evaluate_recourse(
    data: Mapping[str, Any],
    x: Sequence[float],
    xi: Sequence[float],
    backend: str = "pyomo",
    solver: str = "gurobi",
) -> JsonDict:
    """Evaluate Q(x, xi) = min { d^T y : T x + W y <= H xi, y >= 0 }."""
    validate_instance(data)
    arrays = _extract_arrays(data)
    T, W, H, d = arrays["T"], arrays["W"], arrays["H"], arrays["d"]
    x_arr = _as_float_vector(x, "x")
    xi_arr = _as_float_vector(xi, "xi")
    if x_arr.shape[0] != T.shape[1]:
        raise ValueError(f"x has length {x_arr.shape[0]}, but T has {T.shape[1]} columns.")
    if xi_arr.shape[0] != H.shape[1]:
        raise ValueError(f"xi has length {xi_arr.shape[0]}, but H has {H.shape[1]} columns.")

    rhs = H @ xi_arr - T @ x_arr
    backend = backend.lower()

    if backend == "scipy":
        solve = _linprog_solve(
            c_obj=d,
            A_ub=W,
            b_ub=rhs,
            bounds=[(0.0, None)] * d.size,
        )
        infeasible = not solve["success"] and solve["status"] == 2
        return {
            "recourse_cost": solve["objective"],
            "y": None if solve["x"] is None else _to_float_list(solve["x"]),
            "status": solve["status"],
            "status_message": solve["status_message"],
            "termination_condition": solve["termination_condition"],
            "infeasible": bool(infeasible),
            "success": bool(solve["success"]),
        }

    if backend == "pyomo":
        pyo, opt = _check_pyomo_solver(solver)
        n_y = d.size
        m = pyo.ConcreteModel()
        m.J = pyo.RangeSet(0, n_y - 1)
        m.R = pyo.RangeSet(0, W.shape[0] - 1)
        m.y = pyo.Var(m.J, domain=pyo.NonNegativeReals)

        def recourse_rule(mm, r):
            return sum(float(W[r, j]) * mm.y[j] for j in mm.J) <= float(rhs[r])

        def obj_rule(mm):
            return sum(float(d[j]) * mm.y[j] for j in mm.J)

        m.cons = pyo.Constraint(m.R, rule=recourse_rule)
        m.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)
        result = opt.solve(m, tee=False)
        status = str(result.solver.status)
        term = str(result.solver.termination_condition)
        success = _is_optimal_termination(term)
        infeasible = "infeasible" in term.lower()
        return {
            "recourse_cost": None if not success else float(pyo.value(m.obj)),
            "y": None if not success else [float(pyo.value(m.y[j])) for j in range(n_y)],
            "status": status,
            "status_message": status,
            "termination_condition": term,
            "infeasible": bool(infeasible),
            "success": bool(success),
        }

    raise ValueError("backend must be either 'pyomo' or 'scipy'.")


def _scenario_recourse_evaluations(
    data: Mapping[str, Any],
    x: Sequence[float],
    backend: str,
    solver: str,
) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, float], float]:
    arrays = _extract_arrays(data)
    p_s = arrays["p_s"]
    scenario_ids = arrays["scenario_ids"]
    xis = arrays["xis"]
    recourse_evaluations: Dict[str, Any] = {}
    costs: Dict[str, float] = {}
    weighted_costs: Dict[str, float] = {}

    for idx, (sid, xi) in enumerate(zip(scenario_ids, xis)):
        rec = evaluate_recourse(data, x=x, xi=xi, backend=backend, solver=solver)
        rec["scenario_id"] = sid
        rec["probability"] = float(p_s[idx])
        rec["xi"] = _to_float_list(xi)
        recourse_evaluations[sid] = rec
        if not rec["success"] or rec["recourse_cost"] is None:
            raise RuntimeError(
                f"Recourse evaluation failed for scenario {sid}: "
                f"{rec.get('termination_condition')} {rec.get('status_message')}"
            )
        costs[sid] = float(rec["recourse_cost"])
        weighted_costs[sid] = float(p_s[idx]) * costs[sid]

    expected_second_stage = float(sum(weighted_costs.values()))
    return recourse_evaluations, costs, weighted_costs, expected_second_stage


def solve_stochastic_program(
    data: Mapping[str, Any],
    backend: str = "pyomo",
    solver: str = "gurobi",
    output_path: Optional[str | Path] = None,
) -> JsonDict:
    """Solve the full deterministic-equivalent stochastic program."""
    validate_instance(data)
    arrays = _extract_arrays(data)
    scenario_ids = arrays["scenario_ids"]
    xis = arrays["xis"]
    p_s = arrays["p_s"]
    c, d = arrays["c"], arrays["d"]

    solve = _solve_de(data, scenario_ids, xis, p_s, backend=backend, solver=solver)
    if not solve["success"]:
        raise RuntimeError(
            "Stochastic program solve failed: "
            f"{solve.get('termination_condition')} {solve.get('status_message')}"
        )

    x_star = solve["x"]
    y_by_scenario = solve["y_by_scenario"]
    first_stage_cost = float(c @ x_star)
    y_solution = {sid: _to_float_list(y_by_scenario[sid]) for sid in scenario_ids}
    de_second_stage_costs = {sid: float(d @ y_by_scenario[sid]) for sid in scenario_ids}

    rec_evals, q_costs, weighted_q_costs, expected_second_stage = _scenario_recourse_evaluations(
        data, x_star, backend=backend, solver=solver
    )
    total_objective = first_stage_cost + expected_second_stage

    result: JsonDict = {
        "solution_type": "stochastic_program",
        "backend": backend,
        "solver": solver,
        "solver_status": solve["status"],
        "termination_condition": solve["termination_condition"],
        "optimal_first_stage_solution": _to_float_list(x_star),
        "x_star": _to_float_list(x_star),
        "optimal_second_stage_solution_by_scenario": y_solution,
        "y_star_by_scenario": y_solution,
        "total_stochastic_objective": total_objective,
        "z_SP": total_objective,
        "deterministic_equivalent_objective": float(solve["objective"]),
        "first_stage_cost": first_stage_cost,
        "total_expected_second_stage_cost": expected_second_stage,
        "scenario_second_stage_costs": q_costs,
        "probability_weighted_second_stage_costs": weighted_q_costs,
        "deterministic_equivalent_second_stage_costs": de_second_stage_costs,
        "recourse_evaluations": rec_evals,
        "scenario_probabilities": {
            sid: float(p_s[idx]) for idx, sid in enumerate(scenario_ids)
        },
        "scenario_ids": list(scenario_ids),
    }

    if output_path is not None:
        _write_json(output_path, result)
    return result


def _mean_scenario(data: Mapping[str, Any]) -> np.ndarray:
    arrays = _extract_arrays(data)
    p_s = arrays["p_s"]
    xis = arrays["xis"]
    xi_bar = np.zeros_like(xis[0], dtype=float)
    for p, xi in zip(p_s, xis):
        xi_bar += float(p) * xi
    return xi_bar


def compute_expected_value_solution_metrics(
    data: Mapping[str, Any],
    x_ev: Sequence[float],
    backend: str = "pyomo",
    solver: str = "gurobi",
) -> JsonDict:
    """Compute EV/EEV components for a fixed expected-value first-stage decision."""
    validate_instance(data)
    arrays = _extract_arrays(data)
    c = arrays["c"]
    xi_bar = _mean_scenario(data)
    x_ev_arr = _as_float_vector(x_ev, "x_ev")
    if x_ev_arr.shape[0] != c.shape[0]:
        raise ValueError(f"x_ev has length {x_ev_arr.shape[0]}, but len(c) is {c.shape[0]}.")

    first_stage_cost = float(c @ x_ev_arr)
    mean_rec = evaluate_recourse(data, x=x_ev_arr, xi=xi_bar, backend=backend, solver=solver)
    if not mean_rec["success"] or mean_rec["recourse_cost"] is None:
        raise RuntimeError(
            "Mean-scenario recourse evaluation failed: "
            f"{mean_rec.get('termination_condition')} {mean_rec.get('status_message')}"
        )
    rec_evals, q_costs, weighted_q_costs, expected_second_stage = _scenario_recourse_evaluations(
        data, x_ev_arr, backend=backend, solver=solver
    )

    return {
        "x_EV": _to_float_list(x_ev_arr),
        "xi_bar": _to_float_list(xi_bar),
        "first_stage_cost": first_stage_cost,
        "mean_scenario_second_stage_cost": float(mean_rec["recourse_cost"]),
        "mean_scenario_recourse_evaluation": mean_rec,
        "total_expected_second_stage_cost_based_on_x_EV": expected_second_stage,
        "scenario_second_stage_costs": q_costs,
        "probability_weighted_second_stage_costs": weighted_q_costs,
        "recourse_evaluations": rec_evals,
        "EV": first_stage_cost + float(mean_rec["recourse_cost"]),
        "EEV": first_stage_cost + expected_second_stage,
    }


def _read_z_sp_from_sibling(output_path: Optional[str | Path]) -> Optional[float]:
    if output_path is None:
        return None
    path = Path(output_path)
    candidate = path.parent / "stochastic_results.json"
    if not candidate.exists():
        return None
    try:
        data = load_instance(candidate)
        for key in ("z_SP", "total_stochastic_objective"):
            if key in data and data[key] is not None:
                return float(data[key])
    except Exception:
        return None
    return None

def solve_expected_value_problem(
    data: Mapping[str, Any],
    backend: str = "pyomo",
    solver: str = "gurobi",
    output_path: Optional[str | Path] = None,
) -> JsonDict:
    """Solve the expected-value problem and compute EV, EEV, and VSS."""
    validate_instance(data)
    arrays = _extract_arrays(data)
    xi_bar = _mean_scenario(data)
    c = arrays["c"]

    solve = _solve_de(
        data,
        scenario_ids=["mean_scenario"],
        xis=[xi_bar],
        probabilities=[1.0],
        backend=backend,
        solver=solver,
    )
    if not solve["success"]:
        raise RuntimeError(
            "Expected value problem solve failed: "
            f"{solve.get('termination_condition')} {solve.get('status_message')}"
        )

    x_ev = solve["x"]
    y_ev = solve["y_by_scenario"]["mean_scenario"]
    metrics = compute_expected_value_solution_metrics(
        data, x_ev=x_ev, backend=backend, solver=solver
    )

    z_sp = _read_z_sp_from_sibling(output_path)
    if z_sp is None:
        # Best effort for standalone use. In normal workflow, solve_stochastic_program
        # is run first and the sibling stochastic_results.json is read instead.
        sp = solve_stochastic_program(data, backend=backend, solver=solver, output_path=None)
        z_sp = float(sp["z_SP"])

    vss = float(metrics["EEV"] - z_sp)
    result: JsonDict = {
        "solution_type": "expected_value_problem",
        "backend": backend,
        "solver": solver,
        "solver_status": solve["status"],
        "termination_condition": solve["termination_condition"],
        "xi_bar": metrics["xi_bar"],
        "optimal_expected_value_first_stage_solution": _to_float_list(x_ev),
        "x_EV": _to_float_list(x_ev),
        "optimal_second_stage_solution_mean_scenario": _to_float_list(y_ev),
        "y_EV_mean_scenario": _to_float_list(y_ev),
        "expected_value_objective": float(metrics["EV"]),
        "EV": float(metrics["EV"]),
        "expected_result_of_using_expected_value_solution": float(metrics["EEV"]),
        "EEV": float(metrics["EEV"]),
        "stochastic_objective_reference": z_sp,
        "z_SP": z_sp,
        "value_of_stochastic_solution": vss,
        "VSS": vss,
        "first_stage_cost": float(metrics["first_stage_cost"]),
        "mean_scenario_second_stage_cost": float(metrics["mean_scenario_second_stage_cost"]),
        "total_expected_second_stage_cost_based_on_x_EV": float(
            metrics["total_expected_second_stage_cost_based_on_x_EV"]
        ),
        "scenario_second_stage_costs": metrics["scenario_second_stage_costs"],
        "probability_weighted_second_stage_costs": metrics[
            "probability_weighted_second_stage_costs"
        ],
        "mean_scenario_recourse_evaluation": metrics["mean_scenario_recourse_evaluation"],
        "recourse_evaluations": metrics["recourse_evaluations"],
        "scenario_ids": arrays["scenario_ids"],
    }

    if output_path is not None:
        _write_json(output_path, result)
    return result


def write_cost_gap_diagnostics(
    stochastic_results_path: str | Path,
    expectedvalue_results_path: str | Path,
    output_path: str | Path,
    tolerance: float = 1e-8,
) -> JsonDict:
    """Write scenario-wise cost-gap diagnostics from saved JSON results only.

    This function intentionally performs no optimization solves.
    """
    sp = load_instance(stochastic_results_path)
    ev = load_instance(expectedvalue_results_path)

    sp_first = float(sp["first_stage_cost"])
    ev_first = float(ev["first_stage_cost"])
    sp_costs = sp["scenario_second_stage_costs"]
    ev_costs = ev["scenario_second_stage_costs"]
    probabilities = sp.get("scenario_probabilities", {})
    if not probabilities:
        # Fall back to probabilities embedded in recourse evaluations.
        probabilities = {
            sid: rec.get("probability")
            for sid, rec in sp.get("recourse_evaluations", {}).items()
        }
    weighted_sum = 0.0
    scenario_ids = sp.get("scenario_ids") or list(sp_costs.keys())
    nonnegative: List[Dict[str, Any]] = []
    negative: List[Dict[str, Any]] = []

    for sid in scenario_ids:
        if sid not in sp_costs:
            raise ValueError(f"Scenario {sid!r} missing from stochastic results costs.")
        if sid not in ev_costs:
            raise ValueError(f"Scenario {sid!r} missing from expected-value results costs.")
        stochastic_policy_cost = sp_first + float(sp_costs[sid])
        expected_value_policy_cost = ev_first + float(ev_costs[sid])
        gap = expected_value_policy_cost - stochastic_policy_cost
        weighted_gap = float(probabilities[sid]) * gap
        weighted_sum += weighted_gap
        item = {
            "scenario_id": sid,
            "probability": None if probabilities.get(sid) is None else float(probabilities[sid]),
            "cost_gap": float(gap),
            "weighted_cost_gap": float(weighted_gap),
            "stochastic_policy_cost": float(stochastic_policy_cost),
            "expected_value_policy_cost": float(expected_value_policy_cost),
        }
        if gap >= -tolerance:
            nonnegative.append(item)
        else:
            negative.append(item)

    result = {
        "source_files": {
            "stochastic_results_path": str(stochastic_results_path),
            "expectedvalue_results_path": str(expectedvalue_results_path),
        },
        "stochastic_first_stage_cost": sp_first,
        "expected_value_first_stage_cost": ev_first,
        "computed_without_additional_optimization_solves": True,
        "tolerance": float(tolerance),
        "total_number_of_evaluated_scenarios": len(scenario_ids),
        "number_of_scenarios_with_nonnegative_cost_gap": len(nonnegative),
        "number_of_scenarios_with_negative_cost_gap": len(negative),
        "scenarios_with_nonnegative_cost_gap": nonnegative,
        "scenarios_with_negative_cost_gap": negative,
        "weighted_cost_gap_sum": weighted_sum,
    }
    _write_json(output_path, result)
    return result


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------


def _default_output_dir(instance_path: Path) -> Path:
    # Expected instance path: instance/input_data/name.json
    if instance_path.parent.name == "input_data":
        return instance_path.parent.parent / "output_data"
    return instance_path.parent / "output_data"



def resolve_instance_path(instance_arg: str | Path) -> Path:
    path = Path(instance_arg)

    if path.is_file():
        return path

    if path.is_dir():
        candidate = path / "input_data" / f"{path.name}.json"
        if candidate.is_file():
            return candidate

        raise FileNotFoundError(
            f"Could not find instance JSON at expected path: {candidate}"
        )

    raise FileNotFoundError(f"Instance path does not exist: {path}")

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Solve a LandS TSSP instance.")
    parser.add_argument(
        "--instance",
        required=True,
        help=(
            "Path to instance JSON, or path to an instance directory containing "
            "input_data/<instance_name>.json."
        ),
    )
    parser.add_argument("--backend", choices=["pyomo", "scipy"], default="pyomo")
    parser.add_argument("--solver", default="gurobi")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument(
        "--skip_diagnostics",
        action="store_true",
        help="Do not write cost_gap_diagnostics.json.",
    )
    args = parser.parse_args(argv)

    instance_path = resolve_instance_path(args.instance)

    data = load_instance(instance_path)
    out_dir = Path(args.output_dir) if args.output_dir else _default_output_dir(instance_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    sp_path = out_dir / "stochastic_results.json"
    ev_path = out_dir / "expectedvalue_results.json"
    gap_path = out_dir / "cost_gap_diagnostics.json"

    sp = solve_stochastic_program(
        data,
        backend=args.backend,
        solver=args.solver,
        output_path=sp_path,
    )

    ev = solve_expected_value_problem(
        data,
        backend=args.backend,
        solver=args.solver,
        output_path=ev_path,
    )

    summary = {
        "z_SP": sp["z_SP"],
        "EV": ev["EV"],
        "EEV": ev["EEV"],
        "VSS": ev["VSS"],
        "stochastic_first_stage_cost": sp["first_stage_cost"],
        "expected_value_first_stage_cost": ev["first_stage_cost"],
        "first_stage_cost_difference_cTxEV_minus_cTxStar": (
            ev["first_stage_cost"] - sp["first_stage_cost"]
        ),
    }

    if not args.skip_diagnostics:
        diags = write_cost_gap_diagnostics(sp_path, ev_path, gap_path)

        if abs(diags["weighted_cost_gap_sum"] - ev["VSS"]) > 1e-6:
            print(
                "WARNING: Weighted cost gap sum from diagnostics does not match VSS "
                "from expected value results. This may indicate an inconsistency "
                "in the results or a bug in the diagnostics."
            )

        summary["weighted_cost_gap_sum (must be same as VSS)"] = diags["weighted_cost_gap_sum"]

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
