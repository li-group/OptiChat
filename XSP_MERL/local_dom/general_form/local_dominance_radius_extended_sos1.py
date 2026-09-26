"""
Extended/symmetric SOS1 formulation for the local-dominance-radius problem.

Place this file beside ``local_dominance_radius_unbounded.py`` in
``local_dom/general_form/``.

The baseline SP/EV workflow must already have produced, for each instance:

    output_data/stochastic_results.json
    output_data/expectedvalue_results.json
    output_data/cost_gap_diagnostics.json

Mathematical model
------------------
For a selected realized scenario xi_hat with Delta(xi_hat) > gap_tol, solve

    min ||xi - xi_hat||_1

subject to BOTH recourse LPs being optimal at the candidate xi:

    Q(x,xi) = min d^T y
              s.t. W y <= H xi - T x,
                   y >= 0.

For each side k in {EV, star}, use mu^k = -pi^k >= 0 and introduce

    row_slack^k = H xi - T x^k - W y^k >= 0,
    reduced_cost^k = d + W^T mu^k >= 0.

KKT complementarity is enforced with native SOS1 pairs:

    SOS1(row_slack^k[r], mu^k[r])
    SOS1(y^k[j], reduced_cost^k[j]).

The loss-of-strict-dominance constraint is

    c^T x_EV + d^T y_EV + violation_margin
        <= c^T x_star + d^T y_star.

With violation_margin = 0, this is Delta(xi) <= 0.  A positive margin epsilon
instead enforces Delta(xi) <= -epsilon and is useful as a numerical audit for
the infimum-versus-minimum issue in the weak-dominance interpretation.

There are deliberately:
    - no artificial bounds on xi,
    - no artificial bounds on mu / pi,
    - no bilinear strong-duality equations,
    - no big-M complementarity constraints,
    - no explicit binary variables.

By default Gurobi parameter PreSOS1BigM is set to 0 so presolve retains native
SOS1 constraints rather than converting them to a big-M reformulation.

Outputs are written to:
    <instance_dir>/output_data/sos1_extended_exp/
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import math
import re
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception as exc:  # pragma: no cover - depends on local Gurobi install
    gp = None
    GRB = None
    _GUROBI_IMPORT_ERROR = exc
else:
    _GUROBI_IMPORT_ERROR = None

try:  # package import
    from .local_dominance_radius_unbounded import LocalDominanceData
except Exception:  # direct script import from general_form/
    from local_dominance_radius_unbounded import LocalDominanceData


Number = Union[int, float]


class ExtendedSOS1LocalDominanceError(RuntimeError):
    """Raised when the extended SOS1 workflow cannot be completed."""


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.floating, np.integer)):
        return _json_safe(value.item())
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, indent=2, sort_keys=False)
        f.write("\n")


def _scenario_index(scenario: Union[int, str]) -> int:
    if isinstance(scenario, int):
        if scenario <= 0:
            raise ValueError("Scenario indices are 1-based and must be positive.")
        return scenario
    text = str(scenario).strip()
    if text.isdigit():
        return int(text)
    match = re.match(r"^(?:scenario|xi)_(\d+)$", text)
    if match:
        return int(match.group(1))
    raise ValueError(
        f"Cannot parse scenario {scenario!r}. Use 3, '3', 'scenario_3', or 'xi_3'."
    )


def _status_name(status: int) -> str:
    if GRB is None:
        return str(status)
    candidates = [
        "LOADED", "OPTIMAL", "INFEASIBLE", "INF_OR_UNBD", "UNBOUNDED",
        "CUTOFF", "ITERATION_LIMIT", "NODE_LIMIT", "TIME_LIMIT",
        "SOLUTION_LIMIT", "INTERRUPTED", "NUMERIC", "SUBOPTIMAL",
        "INPROGRESS", "USER_OBJ_LIMIT", "WORK_LIMIT", "MEM_LIMIT",
    ]
    for name in candidates:
        if hasattr(GRB, name) and status == getattr(GRB, name):
            return name
    return f"STATUS_{status}"


def _safe_model_attr(model: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(model, name)
    except Exception:
        return default


def _positive_part_max(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.maximum(x, 0.0).max())


def _max_abs(x: np.ndarray) -> float:
    return 0.0 if x.size == 0 else float(np.abs(x).max())


@dataclass(frozen=True)
class ScenarioChoice:
    index_1_based: int
    name: str
    xi_hat: np.ndarray
    cost_gap: float
    probability: float


def choose_positive_gap_scenario(
    data: LocalDominanceData,
    scenario: Optional[Union[int, str]] = None,
    policy: str = "first_positive",
    gap_tol: float = 1e-8,
) -> ScenarioChoice:
    """Choose a realized scenario satisfying Delta(xi_hat) > gap_tol."""
    gap_payload = data.cost_gap_diagnostics.get("scenario_cost_gaps", {})
    if not isinstance(gap_payload, Mapping):
        raise ExtendedSOS1LocalDominanceError(
            "cost_gap_diagnostics.json is missing the 'scenario_cost_gaps' mapping."
        )

    records: List[ScenarioChoice] = []
    for idx in range(1, data.num_scenarios + 1):
        name = f"scenario_{idx}"
        rec = gap_payload.get(name)
        if rec is None:
            continue
        gap = float(rec["cost_gap"])
        if gap > gap_tol:
            records.append(
                ScenarioChoice(
                    index_1_based=idx,
                    name=name,
                    xi_hat=np.asarray(data.xi[idx - 1], dtype=float),
                    cost_gap=gap,
                    probability=float(rec.get("probability", data.probabilities[idx - 1])),
                )
            )

    if scenario is not None:
        idx = _scenario_index(scenario)
        if idx < 1 or idx > data.num_scenarios:
            raise ExtendedSOS1LocalDominanceError(
                f"Scenario {idx} is outside the available range 1..{data.num_scenarios}."
            )
        name = f"scenario_{idx}"
        rec = gap_payload.get(name)
        if rec is None:
            raise ExtendedSOS1LocalDominanceError(f"No cost-gap record was found for {name}.")
        gap = float(rec["cost_gap"])
        if gap <= gap_tol:
            raise ExtendedSOS1LocalDominanceError(
                f"Selected {name} has cost gap {gap:.8g}, not > gap_tol={gap_tol:.3g}."
            )
        return ScenarioChoice(
            index_1_based=idx,
            name=name,
            xi_hat=np.asarray(data.xi[idx - 1], dtype=float),
            cost_gap=gap,
            probability=float(rec.get("probability", data.probabilities[idx - 1])),
        )

    if not records:
        raise ExtendedSOS1LocalDominanceError(
            f"Instance {data.instance_name!r} has no positive-gap scenarios."
        )

    if policy == "first_positive":
        return min(records, key=lambda r: r.index_1_based)
    if policy == "max_gap":
        return max(records, key=lambda r: r.cost_gap)
    if policy == "min_positive":
        return min(records, key=lambda r: r.cost_gap)
    raise ValueError(
        f"Unknown scenario policy {policy!r}. Use first_positive, max_gap, or min_positive."
    )


class ExtendedSOS1LocalDominanceProblem:
    """Symmetric nearest-boundary model with KKT optimality on both recourse LPs."""

    def __init__(
        self,
        instance_name: str,
        *,
        base_dir: Optional[Union[str, Path]] = None,
        scenario: Optional[Union[int, str]] = None,
        scenario_policy: str = "first_positive",
        gap_tol: float = 1e-8,
        violation_margin: float = 0.0,
    ):
        if gp is None or GRB is None:
            raise ExtendedSOS1LocalDominanceError(
                "gurobipy could not be imported. Install/configure Gurobi's Python package. "
                f"Original import error: {_GUROBI_IMPORT_ERROR}"
            )
        if violation_margin < 0:
            raise ValueError("violation_margin must be nonnegative.")

        self.data = LocalDominanceData.load(instance_name, base_dir=base_dir)
        self.choice = choose_positive_gap_scenario(
            self.data,
            scenario=scenario,
            policy=scenario_policy,
            gap_tol=gap_tol,
        )
        self.gap_tol = float(gap_tol)
        self.violation_margin = float(violation_margin)
        self.output_dir = self.data.instance_dir / "output_data" / "sos1_extended_exp"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._Tx_EV = np.asarray(self.data.T @ self.data.x_EV, dtype=float)
        self._Tx_star = np.asarray(self.data.T @ self.data.x_star, dtype=float)

        self._W_row_nz: List[np.ndarray] = [
            np.flatnonzero(np.abs(self.data.W[r, :]) > 0.0)
            for r in range(self.data.num_recourse_constraints)
        ]
        self._W_col_nz: List[np.ndarray] = [
            np.flatnonzero(np.abs(self.data.W[:, j]) > 0.0)
            for j in range(self.data.num_second_stage_vars)
        ]
        self._H_row_nz: List[np.ndarray] = [
            np.flatnonzero(np.abs(self.data.H[r, :]) > 0.0)
            for r in range(self.data.num_recourse_constraints)
        ]

    @property
    def structural_metrics(self) -> Dict[str, Any]:
        d = self.data
        m = d.xi_dim
        ny = d.num_second_stage_vars
        r = d.num_recourse_constraints
        return {
            "num_scenarios": d.num_scenarios,
            "n_x_first_stage": d.num_first_stage_vars,
            "n_y_second_stage": ny,
            "n_recourse_rows": r,
            "xi_dim": m,
            "n_first_stage_rows": int(d.A.shape[0]),
            "nnz_A": int(np.count_nonzero(d.A)),
            "nnz_T": int(np.count_nonzero(d.T)),
            "nnz_W": int(np.count_nonzero(d.W)),
            "nnz_H": int(np.count_nonzero(d.H)),
            # xi,t + [y,mu,row_slack,reduced_cost] for EV and star.
            "theoretical_extended_sos1_model_num_vars": int(2 * m + 4 * ny + 4 * r),
            # 2m abs constraints + two primal-slack systems + two rc systems + comparison.
            "theoretical_extended_sos1_model_num_linear_constraints": int(2 * m + 2 * r + 2 * ny + 1),
            "theoretical_extended_sos1_model_num_sos1": int(2 * (r + ny)),
            "theoretical_explicit_binary_vars": 0,
        }

    def _W_row_expr(self, vars_by_j: Any, r: int) -> Any:
        return gp.quicksum(
            float(self.data.W[r, j]) * vars_by_j[int(j)] for j in self._W_row_nz[r]
        )

    def _W_col_mu_expr(self, mu: Any, j: int) -> Any:
        return gp.quicksum(
            float(self.data.W[r, j]) * mu[int(r)] for r in self._W_col_nz[j]
        )

    def _Hxi_expr(self, xi: Any, r: int) -> Any:
        return gp.quicksum(
            float(self.data.H[r, k]) * xi[int(k)] for k in self._H_row_nz[r]
        )

    def _add_kkt_side(
        self,
        model: Any,
        *,
        xi: Any,
        side: str,
        tx_fixed: np.ndarray,
    ) -> Dict[str, Any]:
        """Add primal feasibility, dual stationarity and both SOS1 families for one LP."""
        d = self.data
        ny = d.num_second_stage_vars
        nr = d.num_recourse_constraints

        y = model.addVars(ny, lb=0.0, vtype=GRB.CONTINUOUS, name=f"y_{side}")
        mu = model.addVars(
            nr, lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"mu_{side}"
        )
        row_slack = model.addVars(
            nr, lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"row_slack_{side}"
        )
        reduced_cost = model.addVars(
            ny, lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"reduced_cost_{side}"
        )

        for r in range(nr):
            model.addConstr(
                self._W_row_expr(y, r) + row_slack[r]
                == self._Hxi_expr(xi, r) - float(tx_fixed[r]),
                name=f"primal_{side}_slack[{r}]",
            )

        # pi = -mu <= 0 and W^T pi <= d  <=>  d + W^T mu >= 0.
        for j in range(ny):
            model.addConstr(
                reduced_cost[j] == float(d.d[j]) + self._W_col_mu_expr(mu, j),
                name=f"reduced_cost_{side}_def[{j}]",
            )

        for r in range(nr):
            model.addSOS(GRB.SOS_TYPE1, [row_slack[r], mu[r]], [1.0, 2.0])
        for j in range(ny):
            model.addSOS(GRB.SOS_TYPE1, [y[j], reduced_cost[j]], [1.0, 2.0])

        return {
            "y": y,
            "mu": mu,
            "row_slack": row_slack,
            "reduced_cost": reduced_cost,
        }

    def build_model(self) -> Tuple[Any, Dict[str, Any]]:
        """Build the all-linear + SOS1 extended/symmetric local-dominance model."""
        d = self.data
        m_dim = d.xi_dim
        ny = d.num_second_stage_vars

        model = gp.Model(f"local_dom_extended_sos1_{d.instance_name}_{self.choice.name}")

        xi = model.addVars(
            m_dim, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="xi"
        )
        t = model.addVars(m_dim, lb=0.0, vtype=GRB.CONTINUOUS, name="t")
        model.setObjective(gp.quicksum(t[k] for k in range(m_dim)), GRB.MINIMIZE)

        for k in range(m_dim):
            hat = float(self.choice.xi_hat[k])
            model.addConstr(xi[k] - hat <= t[k], name=f"abs_upper[{k}]")
            model.addConstr(hat - xi[k] <= t[k], name=f"abs_lower[{k}]")

        ev = self._add_kkt_side(model, xi=xi, side="EV", tx_fixed=self._Tx_EV)
        star = self._add_kkt_side(model, xi=xi, side="star", tx_fixed=self._Tx_star)

        lhs = float(d.first_stage_cost_EV) + gp.quicksum(
            float(d.d[j]) * ev["y"][j] for j in range(ny) if d.d[j] != 0.0
        )
        rhs = float(d.first_stage_cost_star) + gp.quicksum(
            float(d.d[j]) * star["y"][j] for j in range(ny) if d.d[j] != 0.0
        )
        model.addConstr(
            lhs + float(self.violation_margin) <= rhs,
            name="loss_of_dominance",
        )

        model.update()
        handles = {
            "xi": xi,
            "t": t,
            "EV": ev,
            "star": star,
        }
        return model, handles

    def _solve_recourse_lp(self, x_fixed: np.ndarray, xi_value: np.ndarray) -> Dict[str, Any]:
        """Independent LP validation of Q(x_fixed, xi_value)."""
        d = self.data
        ny = d.num_second_stage_vars
        nr = d.num_recourse_constraints
        b = np.asarray(d.H @ xi_value - d.T @ x_fixed, dtype=float)

        lp = gp.Model("recourse_validation")
        lp.Params.OutputFlag = 0
        y = lp.addVars(ny, lb=0.0, vtype=GRB.CONTINUOUS, name="y")
        constrs = []
        for r in range(nr):
            c = lp.addConstr(self._W_row_expr(y, r) <= float(b[r]), name=f"rec[{r}]")
            constrs.append(c)
        lp.setObjective(
            gp.quicksum(float(d.d[j]) * y[j] for j in range(ny) if d.d[j] != 0.0),
            GRB.MINIMIZE,
        )
        lp.optimize()

        out: Dict[str, Any] = {"status": _status_name(int(lp.Status)), "objective": None, "y": None, "pi": None}
        if lp.Status == GRB.OPTIMAL:
            out["objective"] = float(lp.ObjVal)
            out["y"] = [float(y[j].X) for j in range(ny)]
            out["pi"] = [float(c.Pi) for c in constrs]
        lp.dispose()
        return out

    def _side_validation(
        self,
        *,
        side: str,
        xi: np.ndarray,
        x_fixed: np.ndarray,
        y: np.ndarray,
        mu: np.ndarray,
        row_slack: np.ndarray,
        reduced_cost: np.ndarray,
    ) -> Dict[str, Any]:
        d = self.data
        rhs = d.H @ xi - d.T @ x_fixed
        slack_eq_resid = d.W @ y + row_slack - rhs
        reduced_cost_eq_resid = reduced_cost - (d.d + d.W.T @ mu)
        row_pair_min = np.minimum(np.abs(row_slack), np.abs(mu))
        var_pair_min = np.minimum(np.abs(y), np.abs(reduced_cost))
        row_products = np.abs(row_slack * mu)
        var_products = np.abs(y * reduced_cost)

        independent_lp = self._solve_recourse_lp(x_fixed, xi)
        kkt_obj = float(d.d @ y)
        lp_obj = independent_lp.get("objective")

        return {
            "side": side,
            "max_primal_slack_equality_abs_residual": _max_abs(slack_eq_resid),
            "max_reduced_cost_equality_abs_residual": _max_abs(reduced_cost_eq_resid),
            "max_row_sos_pair_min_abs": _max_abs(row_pair_min),
            "max_variable_sos_pair_min_abs": _max_abs(var_pair_min),
            "max_row_complementarity_product_abs": _max_abs(row_products),
            "max_variable_complementarity_product_abs": _max_abs(var_products),
            "min_row_slack": float(row_slack.min()) if row_slack.size else 0.0,
            "min_mu": float(mu.min()) if mu.size else 0.0,
            "min_y": float(y.min()) if y.size else 0.0,
            "min_reduced_cost": float(reduced_cost.min()) if reduced_cost.size else 0.0,
            "independent_recourse_LP": independent_lp,
            "recourse_objective_from_KKT_y": kkt_obj,
            "KKT_minus_independent_LP": None if lp_obj is None else kkt_obj - float(lp_obj),
            "pi_recovered_as_minus_mu": (-mu).tolist(),
        }

    def _validate_incumbent(self, vals: Dict[str, Any]) -> Dict[str, Any]:
        d = self.data
        xi = vals["xi"]
        t = vals["t"]

        ev = vals["EV"]
        star = vals["star"]

        ev_val = self._side_validation(
            side="EV", xi=xi, x_fixed=d.x_EV,
            y=ev["y"], mu=ev["mu"], row_slack=ev["row_slack"], reduced_cost=ev["reduced_cost"],
        )
        star_val = self._side_validation(
            side="star", xi=xi, x_fixed=d.x_star,
            y=star["y"], mu=star["mu"], row_slack=star["row_slack"], reduced_cost=star["reduced_cost"],
        )

        kkt_gap = (
            d.first_stage_cost_EV + float(d.d @ ev["y"])
            - d.first_stage_cost_star - float(d.d @ star["y"])
        )

        ev_lp_obj = ev_val["independent_recourse_LP"].get("objective")
        star_lp_obj = star_val["independent_recourse_LP"].get("objective")
        true_gap = None
        if ev_lp_obj is not None and star_lp_obj is not None:
            true_gap = (
                d.first_stage_cost_EV + float(ev_lp_obj)
                - d.first_stage_cost_star - float(star_lp_obj)
            )

        l1_dist = float(np.abs(xi - self.choice.xi_hat).sum())
        sum_t = float(t.sum())
        margin_resid_kkt = float(kkt_gap + self.violation_margin)
        margin_resid_true = None if true_gap is None else float(true_gap + self.violation_margin)

        return {
            "l1_distance_from_xi_hat": l1_dist,
            "sum_t": sum_t,
            "sum_t_minus_l1_distance": float(sum_t - l1_dist),
            "violation_margin": self.violation_margin,
            "KKT_gap_EV_minus_star": float(kkt_gap),
            "KKT_margin_constraint_residual": margin_resid_kkt,
            "true_gap_from_independent_recourse_LPs": true_gap,
            "true_margin_constraint_residual": margin_resid_true,
            "EV": ev_val,
            "star": star_val,
        }

    def solve(
        self,
        *,
        time_limit: float = 300.0,
        mip_gap: float = 1e-4,
        threads: int = 1,
        seed: int = 1,
        int_feas_tol: Optional[float] = 1e-8,
        feasibility_tol: Optional[float] = 1e-8,
        tee: bool = True,
        keep_native_sos1: bool = True,
        write_model: bool = False,
        extra_gurobi_params: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Solve and persist one extended SOS1 experiment."""
        model, h = self.build_model()

        scenario_tag = self.choice.name
        margin_tag = "" if self.violation_margin == 0.0 else f"_eps_{self.violation_margin:.3g}".replace("+", "")
        log_path = self.output_dir / f"gurobi_{scenario_tag}{margin_tag}.log"
        result_path = self.output_dir / f"extended_sos1_result_{scenario_tag}{margin_tag}.json"
        canonical_path = self.output_dir / "extended_sos1_result.json"

        model.Params.TimeLimit = float(time_limit)
        model.Params.MIPGap = float(mip_gap)
        model.Params.Threads = int(threads)
        model.Params.Seed = int(seed)
        model.Params.LogFile = str(log_path)
        model.Params.OutputFlag = 1
        model.Params.LogToConsole = 1 if tee else 0

        if int_feas_tol is not None:
            model.Params.IntFeasTol = float(int_feas_tol)
        if feasibility_tol is not None:
            model.Params.FeasibilityTol = float(feasibility_tol)
        if keep_native_sos1:
            model.Params.PreSOS1BigM = 0
        if extra_gurobi_params:
            for key, value in extra_gurobi_params.items():
                model.setParam(str(key), value)

        if write_model:
            model.write(str(self.output_dir / f"extended_model_{scenario_tag}{margin_tag}.lp"))

        model.update()
        raw_model_stats = {
            "num_vars": int(_safe_model_attr(model, "NumVars", 0)),
            "num_linear_constraints": int(_safe_model_attr(model, "NumConstrs", 0)),
            "num_sos": int(_safe_model_attr(model, "NumSOS", 0)),
            "num_binary_vars": int(_safe_model_attr(model, "NumBinVars", 0)),
            "num_integer_vars": int(_safe_model_attr(model, "NumIntVars", 0)),
            "num_continuous_vars": int(_safe_model_attr(model, "NumVars", 0))
            - int(_safe_model_attr(model, "NumIntVars", 0)),
            "num_nonzeros": int(_safe_model_attr(model, "NumNZs", 0)),
            "is_mip": bool(_safe_model_attr(model, "IsMIP", True)),
        }

        py_start = time.perf_counter()
        model.optimize()
        python_wall = float(time.perf_counter() - py_start)

        status_code = int(model.Status)
        status = _status_name(status_code)
        sol_count = int(model.SolCount)
        has_incumbent = sol_count > 0

        perf: Dict[str, Any] = {
            "status_code": status_code,
            "status": status,
            "has_incumbent": has_incumbent,
            "solution_count": sol_count,
            "runtime_sec_gurobi": float(_safe_model_attr(model, "Runtime", python_wall)),
            "runtime_sec_python_wall": python_wall,
            "work": _safe_model_attr(model, "Work", None),
            "node_count": float(_safe_model_attr(model, "NodeCount", 0.0)),
            "simplex_iterations": float(_safe_model_attr(model, "IterCount", 0.0)),
            "barrier_iterations": int(_safe_model_attr(model, "BarIterCount", 0)),
            "objective_value": None,
            "best_bound": None,
            "mip_gap": None,
        }
        try:
            perf["best_bound"] = float(model.ObjBound)
        except Exception:
            pass

        solution: Optional[Dict[str, Any]] = None
        validation: Optional[Dict[str, Any]] = None

        if has_incumbent:
            try:
                perf["objective_value"] = float(model.ObjVal)
            except Exception:
                pass
            try:
                perf["mip_gap"] = float(model.MIPGap)
            except Exception:
                pass

            def side_values(side: str) -> Dict[str, np.ndarray]:
                hs = h[side]
                return {
                    "y": np.asarray([hs["y"][j].X for j in range(self.data.num_second_stage_vars)], dtype=float),
                    "mu": np.asarray([hs["mu"][r].X for r in range(self.data.num_recourse_constraints)], dtype=float),
                    "row_slack": np.asarray([hs["row_slack"][r].X for r in range(self.data.num_recourse_constraints)], dtype=float),
                    "reduced_cost": np.asarray([hs["reduced_cost"][j].X for j in range(self.data.num_second_stage_vars)], dtype=float),
                }

            vals: Dict[str, Any] = {
                "xi": np.asarray([h["xi"][k].X for k in range(self.data.xi_dim)], dtype=float),
                "t": np.asarray([h["t"][k].X for k in range(self.data.xi_dim)], dtype=float),
                "EV": side_values("EV"),
                "star": side_values("star"),
            }

            solution = {
                "gamma_l1_radius_incumbent": float(vals["t"].sum()),
                "nearest_boundary_xi": vals["xi"].tolist(),
                "absolute_deviation_t": vals["t"].tolist(),
                "violation_margin": self.violation_margin,
                "EV": {
                    "y": vals["EV"]["y"].tolist(),
                    "mu_equals_minus_pi": vals["EV"]["mu"].tolist(),
                    "pi": (-vals["EV"]["mu"]).tolist(),
                    "row_slack": vals["EV"]["row_slack"].tolist(),
                    "reduced_cost": vals["EV"]["reduced_cost"].tolist(),
                },
                "star": {
                    "y": vals["star"]["y"].tolist(),
                    "mu_equals_minus_pi": vals["star"]["mu"].tolist(),
                    "pi": (-vals["star"]["mu"]).tolist(),
                    "row_slack": vals["star"]["row_slack"].tolist(),
                    "reduced_cost": vals["star"]["reduced_cost"].tolist(),
                },
            }
            validation = self._validate_incumbent(vals)

        result: Dict[str, Any] = {
            "instance_name": self.data.instance_name,
            "method": "extended_symmetric_both_sides_KKT_native_SOS1_no_xi_or_dual_bounds",
            "formulation": {
                "EV_side": "primal y_EV + KKT with mu_EV=-pi_EV, row slacks, reduced costs",
                "star_side": "primal y_star + KKT with mu_star=-pi_star, row slacks, reduced costs",
                "complementarity": "four SOS1 families: row and variable complementarity on both sides",
                "violation_condition": "Delta(xi) <= -violation_margin",
                "violation_margin": self.violation_margin,
                "artificial_xi_bounds": False,
                "artificial_dual_bounds": False,
                "big_M_complementarity": False,
                "native_SOS1_preserved_via_PreSOS1BigM_0": bool(keep_native_sos1),
                "nonconvex_bilinear_terms": False,
            },
            "selected_scenario": {
                "scenario_index_1_based": self.choice.index_1_based,
                "scenario_name": self.choice.name,
                "xi_hat": self.choice.xi_hat.tolist(),
                "initial_cost_gap": self.choice.cost_gap,
                "probability": self.choice.probability,
                "selection_requires_strict_positive_gap": True,
                "gap_tol": self.gap_tol,
            },
            "instance_structure": self.structural_metrics,
            "raw_gurobi_model_structure": raw_model_stats,
            "solver_configuration": {
                "solver": "gurobi",
                "gurobi_version": list(gp.gurobi.version()),
                "time_limit_sec": float(time_limit),
                "target_mip_gap": float(mip_gap),
                "threads": int(threads),
                "seed": int(seed),
                "int_feas_tol": int_feas_tol,
                "feasibility_tol": feasibility_tol,
                "PreSOS1BigM": 0 if keep_native_sos1 else "Gurobi default",
                "extra_gurobi_params": dict(extra_gurobi_params or {}),
            },
            "performance": perf,
            "solution": solution,
            "validation": validation,
            "files": {
                "gurobi_log": str(log_path),
                "scenario_result": str(result_path),
                "canonical_result": str(canonical_path),
            },
        }

        _write_json(result_path, result)
        _write_json(canonical_path, result)
        model.dispose()
        return result


def run_extended_sos1_local_dominance(
    instance_name: str,
    *,
    base_dir: Optional[Union[str, Path]] = None,
    scenario: Optional[Union[int, str]] = None,
    scenario_policy: str = "first_positive",
    gap_tol: float = 1e-8,
    violation_margin: float = 0.0,
    time_limit: float = 300.0,
    mip_gap: float = 1e-4,
    threads: int = 1,
    seed: int = 1,
    int_feas_tol: Optional[float] = 1e-8,
    feasibility_tol: Optional[float] = 1e-8,
    tee: bool = True,
    keep_native_sos1: bool = True,
    write_model: bool = False,
    extra_gurobi_params: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    problem = ExtendedSOS1LocalDominanceProblem(
        instance_name,
        base_dir=base_dir,
        scenario=scenario,
        scenario_policy=scenario_policy,
        gap_tol=gap_tol,
        violation_margin=violation_margin,
    )
    return problem.solve(
        time_limit=time_limit,
        mip_gap=mip_gap,
        threads=threads,
        seed=seed,
        int_feas_tol=int_feas_tol,
        feasibility_tol=feasibility_tol,
        tee=tee,
        keep_native_sos1=keep_native_sos1,
        write_model=write_model,
        extra_gurobi_params=extra_gurobi_params,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Solve the extended/symmetric local-dominance-radius model with native Gurobi SOS1 KKT constraints."
    )
    parser.add_argument("instance_name", help="e.g. lands_instance_small_1_1_1 or lands_instance_naive")
    parser.add_argument("--base-dir", default=None, help="Directory containing instance folders; defaults to this file's directory.")
    parser.add_argument("--scenario", default=None, help="Optional explicit positive-gap scenario: 3, scenario_3, or xi_3.")
    parser.add_argument(
        "--scenario-policy",
        choices=["first_positive", "max_gap", "min_positive"],
        default="first_positive",
        help="Selection rule when --scenario is omitted.",
    )
    parser.add_argument("--gap-tol", type=float, default=1e-8)
    parser.add_argument(
        "--violation-margin",
        type=float,
        default=0.0,
        help="epsilon >= 0 in Delta(xi) <= -epsilon. Use 0 for the closure/tie formulation.",
    )
    parser.add_argument("--time-limit", type=float, default=300.0, help="Seconds. Default: 300 (5 minutes).")
    parser.add_argument("--mip-gap", type=float, default=1e-4)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--int-feas-tol", type=float, default=1e-8)
    parser.add_argument("--feasibility-tol", type=float, default=1e-8)
    parser.add_argument("--quiet", action="store_true", help="Suppress Gurobi console output; log file is still written.")
    parser.add_argument("--write-model", action="store_true", help="Also write the LP/SOS model file for inspection.")
    parser.add_argument(
        "--allow-sos1-reformulation",
        action="store_true",
        help="Do NOT set PreSOS1BigM=0. Leave this flag off for the native-SOS1 experiment.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    result = run_extended_sos1_local_dominance(
        args.instance_name,
        base_dir=args.base_dir,
        scenario=args.scenario,
        scenario_policy=args.scenario_policy,
        gap_tol=args.gap_tol,
        violation_margin=args.violation_margin,
        time_limit=args.time_limit,
        mip_gap=args.mip_gap,
        threads=args.threads,
        seed=args.seed,
        int_feas_tol=args.int_feas_tol,
        feasibility_tol=args.feasibility_tol,
        tee=not args.quiet,
        keep_native_sos1=not args.allow_sos1_reformulation,
        write_model=args.write_model,
    )

    perf = result["performance"]
    print("\n=== Extended SOS1 local-dominance result ===")
    print(f"Instance:      {result['instance_name']}")
    print(f"Scenario:      {result['selected_scenario']['scenario_name']}")
    print(f"Margin eps:    {result['formulation']['violation_margin']}")
    print(f"Status:        {perf['status']}")
    print(f"Runtime (s):   {perf['runtime_sec_gurobi']:.3f}")
    print(f"Incumbent:     {perf['objective_value']}")
    print(f"Best bound:    {perf['best_bound']}")
    print(f"MIP gap:       {perf['mip_gap']}")
    if result["validation"] is not None:
        print(f"True gap @ xi: {result['validation']['true_gap_from_independent_recourse_LPs']}")
    print(f"Saved to:      {result['files']['canonical_result']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
