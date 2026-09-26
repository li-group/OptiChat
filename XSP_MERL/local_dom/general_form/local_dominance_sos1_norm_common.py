"""Common implementation for L2 and L-infinity local-dominance SOS1 models.

This module is intended to be copied into the same directory as
``local_dominance_radius_unbounded.py`` in the existing LandS codebase.

It supports two exact formulations of the nearest-boundary problem

    min ||xi - xi_hat||_p
    s.t. Delta(xi) <= -epsilon,

where p is either 2 or infinity and epsilon >= 0 is an optional strict-reversal
margin.  The realized scenario xi_hat is required to satisfy Delta(xi_hat)>0.

Formulations
------------
``extended``
    KKT optimality is imposed for BOTH the EV and x_star recourse LPs.

``one_sided``
    The EV side is only primal feasible (an upper certificate), while the
    x_star side is KKT optimal.  This is exact because the EV certificate is
    existentially chosen; whenever Delta(xi)<=-epsilon, an optimal EV recourse
    solution can be selected.

For KKT complementarity, native Gurobi SOS1 constraints are used:

    SOS1(row_slack[r], mu[r])
    SOS1(y[j], reduced_cost[j])

with mu=-pi>=0 for the recourse dual multipliers.

Norm handling
-------------
L2:
    Minimize the squared Euclidean distance
        sum_j (xi_j-xi_hat_j)^2.
    This has exactly the same minimizers as the Euclidean norm itself, and the
    reported radius is sqrt(objective).  The quadratic objective is convex.

L-infinity:
    Introduce rho>=0 and impose
        -rho <= xi_j-xi_hat_j <= rho
    for every component j, then minimize rho.

There are deliberately no artificial xi bounds, no artificial dual bounds,
no big-M complementarity constraints, and no bilinear strong-duality terms.
By default ``PreSOS1BigM=0`` is set so Gurobi retains the SOS1 structure.
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


class NormSOS1LocalDominanceError(RuntimeError):
    """Raised when a norm-SOS1 local-dominance experiment cannot be completed."""


@dataclass(frozen=True)
class ExperimentConfig:
    """Immutable description of one of the four requested experiments."""

    norm_kind: str                  # "2" or "inf"
    formulation_kind: str           # "extended" or "one_sided"
    slug: str                       # e.g. "norm2_extended_sos1"
    summary_stem: str               # e.g. "Norm_2_extended_sos1"
    output_subdir: str              # per-instance output directory
    canonical_filename: str         # canonical JSON result
    display_name: str

    def __post_init__(self) -> None:
        if self.norm_kind not in {"2", "inf"}:
            raise ValueError("norm_kind must be '2' or 'inf'.")
        if self.formulation_kind not in {"extended", "one_sided"}:
            raise ValueError("formulation_kind must be 'extended' or 'one_sided'.")


@dataclass(frozen=True)
class ScenarioChoice:
    index_1_based: int
    name: str
    xi_hat: np.ndarray
    cost_gap: float
    probability: float


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


def _radius_from_delta(delta: np.ndarray, norm_kind: str) -> float:
    if norm_kind == "2":
        return float(np.linalg.norm(delta, ord=2))
    if norm_kind == "inf":
        return float(np.linalg.norm(delta, ord=np.inf)) if delta.size else 0.0
    raise ValueError(norm_kind)


def _radius_from_raw_objective(value: Optional[float], norm_kind: str) -> Optional[float]:
    if value is None:
        return None
    value = float(value)
    if norm_kind == "2":
        # The exact model objective is the squared Euclidean radius.
        return math.sqrt(max(value, 0.0))
    return value


def choose_positive_gap_scenario(
    data: LocalDominanceData,
    scenario: Optional[Union[int, str]] = None,
    policy: str = "first_positive",
    gap_tol: float = 1e-8,
) -> ScenarioChoice:
    """Choose a realized scenario with Delta(xi_hat)>gap_tol."""
    gap_payload = data.cost_gap_diagnostics.get("scenario_cost_gaps", {})
    if not isinstance(gap_payload, Mapping):
        raise NormSOS1LocalDominanceError(
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
            raise NormSOS1LocalDominanceError(
                f"Scenario {idx} is outside 1..{data.num_scenarios}."
            )
        name = f"scenario_{idx}"
        rec = gap_payload.get(name)
        if rec is None:
            raise NormSOS1LocalDominanceError(f"No cost-gap record was found for {name}.")
        gap = float(rec["cost_gap"])
        if gap <= gap_tol:
            raise NormSOS1LocalDominanceError(
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
        raise NormSOS1LocalDominanceError(
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


class SOS1NormLocalDominanceProblem:
    """Exact L2/L-infinity nearest-boundary model with native SOS1 KKT."""

    def __init__(
        self,
        config: ExperimentConfig,
        instance_name: str,
        *,
        base_dir: Optional[Union[str, Path]] = None,
        scenario: Optional[Union[int, str]] = None,
        scenario_policy: str = "first_positive",
        gap_tol: float = 1e-8,
        violation_margin: float = 0.0,
    ):
        if gp is None or GRB is None:
            raise NormSOS1LocalDominanceError(
                "gurobipy could not be imported. Install/configure Gurobi's Python package. "
                f"Original import error: {_GUROBI_IMPORT_ERROR}"
            )
        if violation_margin < 0:
            raise ValueError("violation_margin must be nonnegative.")

        self.config = config
        self.data = LocalDominanceData.load(instance_name, base_dir=base_dir)
        self.choice = choose_positive_gap_scenario(
            self.data,
            scenario=scenario,
            policy=scenario_policy,
            gap_tol=gap_tol,
        )
        self.violation_margin = float(violation_margin)
        self.output_dir = self.data.instance_dir / "output_data" / config.output_subdir
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
        nr = d.num_recourse_constraints

        if self.config.formulation_kind == "one_sided":
            recourse_vars = 3 * ny + 2 * nr
            recourse_linear = 2 * nr + ny + 1
            nsos = nr + ny
        else:
            recourse_vars = 4 * ny + 4 * nr
            recourse_linear = 2 * nr + 2 * ny + 1
            nsos = 2 * (nr + ny)

        if self.config.norm_kind == "2":
            norm_vars = 0
            norm_linear = 0
            quadratic_obj_terms = m
        else:
            norm_vars = 1  # rho
            norm_linear = 2 * m
            quadratic_obj_terms = 0

        return {
            "num_scenarios": d.num_scenarios,
            "n_x_first_stage": d.num_first_stage_vars,
            "n_y_second_stage": ny,
            "n_recourse_rows": nr,
            "xi_dim": m,
            "n_first_stage_rows": int(d.A.shape[0]),
            "nnz_A": int(np.count_nonzero(d.A)),
            "nnz_T": int(np.count_nonzero(d.T)),
            "nnz_W": int(np.count_nonzero(d.W)),
            "nnz_H": int(np.count_nonzero(d.H)),
            "theoretical_sos1_model_num_vars": int(m + norm_vars + recourse_vars),
            "theoretical_sos1_model_num_linear_constraints": int(norm_linear + recourse_linear),
            "theoretical_sos1_model_num_sos1": int(nsos),
            "theoretical_quadratic_objective_terms": int(quadratic_obj_terms),
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

    def _add_kkt_side(self, model: Any, xi: Any, x_fixed: np.ndarray, label: str) -> Dict[str, Any]:
        """Add primal feasibility, stationarity and SOS1 complementarity for one recourse LP."""
        d = self.data
        ny = d.num_second_stage_vars
        nr = d.num_recourse_constraints
        Tx = np.asarray(d.T @ x_fixed, dtype=float)

        y = model.addVars(ny, lb=0.0, vtype=GRB.CONTINUOUS, name=f"y_{label}")
        mu = model.addVars(nr, lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"mu_{label}")
        row_slack = model.addVars(
            nr, lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"row_slack_{label}"
        )
        reduced_cost = model.addVars(
            ny, lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"reduced_cost_{label}"
        )

        for r in range(nr):
            model.addConstr(
                self._W_row_expr(y, r) + row_slack[r]
                == self._Hxi_expr(xi, r) - float(Tx[r]),
                name=f"primal_{label}_slack[{r}]",
            )

        for j in range(ny):
            model.addConstr(
                reduced_cost[j] == float(d.d[j]) + self._W_col_mu_expr(mu, j),
                name=f"reduced_cost_{label}_def[{j}]",
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
        d = self.data
        m_dim = d.xi_dim
        ny = d.num_second_stage_vars
        nr = d.num_recourse_constraints

        model = gp.Model(f"{self.config.slug}_{d.instance_name}_{self.choice.name}")
        xi = model.addVars(
            m_dim,
            lb=-GRB.INFINITY,
            ub=GRB.INFINITY,
            vtype=GRB.CONTINUOUS,
            name="xi",
        )

        handles: Dict[str, Any] = {"xi": xi}

        # Norm objective.
        if self.config.norm_kind == "2":
            qobj = gp.QuadExpr()
            for k in range(m_dim):
                hat = float(self.choice.xi_hat[k])
                qobj += xi[k] * xi[k] - 2.0 * hat * xi[k] + hat * hat
            model.setObjective(qobj, GRB.MINIMIZE)
        else:
            rho = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="rho_inf")
            for k in range(m_dim):
                hat = float(self.choice.xi_hat[k])
                model.addConstr(xi[k] - hat <= rho, name=f"inf_upper[{k}]")
                model.addConstr(hat - xi[k] <= rho, name=f"inf_lower[{k}]")
            model.setObjective(rho, GRB.MINIMIZE)
            handles["rho"] = rho

        if self.config.formulation_kind == "extended":
            ev = self._add_kkt_side(model, xi, d.x_EV, "EV")
            star = self._add_kkt_side(model, xi, d.x_star, "star")
            handles["EV"] = ev
            handles["star"] = star

            lhs = float(d.first_stage_cost_EV) + gp.quicksum(
                float(d.d[j]) * ev["y"][j] for j in range(ny) if d.d[j] != 0.0
            )
            rhs = float(d.first_stage_cost_star) + gp.quicksum(
                float(d.d[j]) * star["y"][j] for j in range(ny) if d.d[j] != 0.0
            )
            model.addConstr(
                lhs + self.violation_margin <= rhs,
                name="violation_exact_value_comparison",
            )
        else:
            # EV: primal-feasible upper certificate only.
            y_EV = model.addVars(ny, lb=0.0, vtype=GRB.CONTINUOUS, name="y_EV")
            for r in range(nr):
                model.addConstr(
                    self._W_row_expr(y_EV, r)
                    <= self._Hxi_expr(xi, r) - float(self._Tx_EV[r]),
                    name=f"primal_EV[{r}]",
                )
            star = self._add_kkt_side(model, xi, d.x_star, "star")
            handles["y_EV"] = y_EV
            handles["star"] = star

            lhs = float(d.first_stage_cost_EV) + gp.quicksum(
                float(d.d[j]) * y_EV[j] for j in range(ny) if d.d[j] != 0.0
            )
            rhs = float(d.first_stage_cost_star) + gp.quicksum(
                float(d.d[j]) * star["y"][j] for j in range(ny) if d.d[j] != 0.0
            )
            model.addConstr(
                lhs + self.violation_margin <= rhs,
                name="violation_one_sided_certificate",
            )

        model.update()
        return model, handles

    def _solve_recourse_lp(self, x_fixed: np.ndarray, xi_value: np.ndarray) -> Dict[str, Any]:
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

    def _extract_kkt_values(self, side: Mapping[str, Any]) -> Dict[str, np.ndarray]:
        ny = self.data.num_second_stage_vars
        nr = self.data.num_recourse_constraints
        return {
            "y": np.asarray([side["y"][j].X for j in range(ny)], dtype=float),
            "mu": np.asarray([side["mu"][r].X for r in range(nr)], dtype=float),
            "row_slack": np.asarray([side["row_slack"][r].X for r in range(nr)], dtype=float),
            "reduced_cost": np.asarray([side["reduced_cost"][j].X for j in range(ny)], dtype=float),
        }

    def _validate_kkt_side(
        self,
        label: str,
        x_fixed: np.ndarray,
        xi: np.ndarray,
        vals: Mapping[str, np.ndarray],
        independent_lp: Mapping[str, Any],
    ) -> Dict[str, Any]:
        d = self.data
        b = d.H @ xi - d.T @ x_fixed
        y = vals["y"]
        mu = vals["mu"]
        slack = vals["row_slack"]
        rc = vals["reduced_cost"]

        primal_eq = d.W @ y + slack - b
        rc_eq = rc - (d.d + d.W.T @ mu)
        row_pair = np.minimum(np.abs(slack), np.abs(mu))
        var_pair = np.minimum(np.abs(y), np.abs(rc))
        lp_obj = independent_lp.get("objective")
        kkt_obj = float(d.d @ y)

        return {
            "label": label,
            "max_primal_slack_equality_abs_residual": _max_abs(primal_eq),
            "max_reduced_cost_equality_abs_residual": _max_abs(rc_eq),
            "max_row_sos_pair_min_abs": _max_abs(row_pair),
            "max_variable_sos_pair_min_abs": _max_abs(var_pair),
            "max_row_complementarity_product_abs": _max_abs(slack * mu),
            "max_variable_complementarity_product_abs": _max_abs(y * rc),
            "min_row_slack": float(slack.min()) if slack.size else 0.0,
            "min_mu": float(mu.min()) if mu.size else 0.0,
            "min_y": float(y.min()) if y.size else 0.0,
            "min_reduced_cost": float(rc.min()) if rc.size else 0.0,
            "recourse_objective_from_KKT_y": kkt_obj,
            "KKT_minus_independent_LP": None if lp_obj is None else kkt_obj - float(lp_obj),
            "pi_recovered_as_minus_mu": (-mu).tolist(),
        }

    def _validate_incumbent(self, model: Any, h: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        d = self.data
        xi = np.asarray([h["xi"][k].X for k in range(d.xi_dim)], dtype=float)
        delta = xi - self.choice.xi_hat
        radius = _radius_from_delta(delta, self.config.norm_kind)

        ev_lp = self._solve_recourse_lp(d.x_EV, xi)
        star_lp = self._solve_recourse_lp(d.x_star, xi)
        true_gap = None
        if ev_lp["objective"] is not None and star_lp["objective"] is not None:
            true_gap = (
                d.first_stage_cost_EV + float(ev_lp["objective"])
                - d.first_stage_cost_star - float(star_lp["objective"])
            )

        validation: Dict[str, Any] = {
            "radius_recomputed_from_xi": radius,
            "true_gap_from_independent_recourse_LPs": true_gap,
            "true_margin_constraint_residual": None if true_gap is None else float(true_gap + self.violation_margin),
            "independent_recourse_EV": ev_lp,
            "independent_recourse_star": star_lp,
        }

        solution: Dict[str, Any] = {
            "nearest_violation_xi": xi.tolist(),
            "delta_xi": delta.tolist(),
            "radius_incumbent": radius,
        }

        if self.config.norm_kind == "2":
            raw = float(model.ObjVal)
            solution["squared_l2_objective"] = raw
            validation["squared_radius_recomputed"] = float(radius * radius)
            validation["objective_minus_squared_radius"] = float(raw - radius * radius)
        else:
            rho = float(h["rho"].X)
            solution["rho_inf"] = rho
            validation["rho_minus_recomputed_inf_radius"] = float(rho - radius)

        if self.config.formulation_kind == "extended":
            ev_vals = self._extract_kkt_values(h["EV"])
            star_vals = self._extract_kkt_values(h["star"])
            solution.update({
                "EV_KKT": {k: v.tolist() for k, v in ev_vals.items()},
                "star_KKT": {k: v.tolist() for k, v in star_vals.items()},
            })
            validation["EV_KKT"] = self._validate_kkt_side("EV", d.x_EV, xi, ev_vals, ev_lp)
            validation["star_KKT"] = self._validate_kkt_side("star", d.x_star, xi, star_vals, star_lp)

            cert_gap = (
                d.first_stage_cost_EV + float(d.d @ ev_vals["y"])
                - d.first_stage_cost_star - float(d.d @ star_vals["y"])
            )
            validation["model_value_gap_from_KKT_y"] = float(cert_gap)
            validation["model_margin_constraint_residual"] = float(cert_gap + self.violation_margin)
        else:
            y_EV = np.asarray([h["y_EV"][j].X for j in range(d.num_second_stage_vars)], dtype=float)
            star_vals = self._extract_kkt_values(h["star"])
            solution["y_EV_primal_certificate"] = y_EV.tolist()
            solution["star_KKT"] = {k: v.tolist() for k, v in star_vals.items()}

            b_EV = d.H @ xi - d.T @ d.x_EV
            ev_primal_resid = d.W @ y_EV - b_EV
            ev_cert_obj = float(d.d @ y_EV)
            validation["EV_primal_certificate"] = {
                "max_primal_violation_positive_part": _positive_part_max(ev_primal_resid),
                "recourse_objective_certificate": ev_cert_obj,
                "certificate_minus_independent_LP": (
                    None if ev_lp["objective"] is None else ev_cert_obj - float(ev_lp["objective"])
                ),
            }
            validation["star_KKT"] = self._validate_kkt_side("star", d.x_star, xi, star_vals, star_lp)

            cert_gap = (
                d.first_stage_cost_EV + ev_cert_obj
                - d.first_stage_cost_star - float(d.d @ star_vals["y"])
            )
            validation["model_certificate_gap"] = float(cert_gap)
            validation["model_margin_constraint_residual"] = float(cert_gap + self.violation_margin)

        return solution, validation

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
        d = self.data
        model, h = self.build_model()

        scenario_tag = self.choice.name
        log_path = self.output_dir / f"gurobi_{scenario_tag}.log"
        result_path = self.output_dir / f"{self.config.slug}_result_{scenario_tag}.json"
        canonical_path = self.output_dir / self.config.canonical_filename

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
            # MPS preserves quadratic objective/SOS information more reliably than LP across versions.
            model.write(str(self.output_dir / f"model_{scenario_tag}.mps"))

        model.update()
        raw_stats = {
            "num_vars": int(_safe_model_attr(model, "NumVars", 0)),
            "num_linear_constraints": int(_safe_model_attr(model, "NumConstrs", 0)),
            "num_quadratic_constraints": int(_safe_model_attr(model, "NumQConstrs", 0)),
            "num_sos": int(_safe_model_attr(model, "NumSOS", 0)),
            "num_binary_vars": int(_safe_model_attr(model, "NumBinVars", 0)),
            "num_integer_vars": int(_safe_model_attr(model, "NumIntVars", 0)),
            "num_continuous_vars": int(_safe_model_attr(model, "NumVars", 0))
            - int(_safe_model_attr(model, "NumIntVars", 0)),
            "num_linear_nonzeros": int(_safe_model_attr(model, "NumNZs", 0)),
            "num_quadratic_objective_nonzeros": int(_safe_model_attr(model, "NumQNZs", 0)),
            "is_mip": bool(_safe_model_attr(model, "IsMIP", True)),
        }

        start = time.perf_counter()
        model.optimize()
        python_wall = float(time.perf_counter() - start)

        status_code = int(model.Status)
        status = _status_name(status_code)
        sol_count = int(model.SolCount)
        has_incumbent = sol_count > 0

        raw_obj = None
        raw_bound = None
        mip_gap_value = None
        try:
            raw_bound = float(model.ObjBound)
        except Exception:
            pass
        if has_incumbent:
            try:
                raw_obj = float(model.ObjVal)
            except Exception:
                pass
            try:
                mip_gap_value = float(model.MIPGap)
            except Exception:
                pass

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
            "objective_raw": raw_obj,
            "best_bound_raw": raw_bound,
            "radius_incumbent": _radius_from_raw_objective(raw_obj, self.config.norm_kind),
            "best_bound_radius": _radius_from_raw_objective(raw_bound, self.config.norm_kind),
            "mip_gap": mip_gap_value,
        }

        solution = None
        validation = None
        if has_incumbent:
            solution, validation = self._validate_incumbent(model, h)
            # Prefer the radius recomputed from xi to avoid any tiny quadratic rounding mismatch.
            perf["radius_incumbent"] = solution["radius_incumbent"]

        formulation_description = {
            "norm": "L2" if self.config.norm_kind == "2" else "L-infinity",
            "norm_implementation": (
                "minimize squared Euclidean distance; report sqrt(objective)"
                if self.config.norm_kind == "2"
                else "minimize rho with |xi_j-xi_hat_j|<=rho for all j"
            ),
            "formulation_kind": self.config.formulation_kind,
            "EV_side": (
                "KKT optimal recourse"
                if self.config.formulation_kind == "extended"
                else "primal-feasible upper certificate only"
            ),
            "star_side": "KKT optimal recourse",
            "complementarity": "SOS1(row_slack,mu) and SOS1(y,reduced_cost) on every KKT side",
            "violation_condition": f"Delta(xi) <= -{self.violation_margin:g}",
            "violation_margin": self.violation_margin,
            "artificial_xi_bounds": False,
            "artificial_dual_bounds": False,
            "big_M_complementarity": False,
            "native_SOS1_preserved_via_PreSOS1BigM_0": bool(keep_native_sos1),
            "nonconvex_bilinear_strong_duality_terms": False,
        }

        result: Dict[str, Any] = {
            "instance_name": d.instance_name,
            "method": self.config.slug,
            "display_name": self.config.display_name,
            "formulation": formulation_description,
            "selected_scenario": {
                "scenario_index_1_based": self.choice.index_1_based,
                "scenario_name": self.choice.name,
                "xi_hat": self.choice.xi_hat.tolist(),
                "initial_cost_gap": self.choice.cost_gap,
                "probability": self.choice.probability,
            },
            "instance_structure": self.structural_metrics,
            "raw_gurobi_model_structure": raw_stats,
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


def run_configured_experiment(
    config: ExperimentConfig,
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
    problem = SOS1NormLocalDominanceProblem(
        config,
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


def build_single_parser(config: ExperimentConfig) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=f"Solve {config.display_name} on one LandS instance.")
    p.add_argument("instance_name", help="e.g. lands_instance_small_1_1_1 or lands_instance_naive")
    p.add_argument("--base-dir", default=None, help="Directory containing instance folders; defaults to this file's directory.")
    p.add_argument("--scenario", default=None, help="Optional positive-gap scenario: 3, scenario_3, or xi_3.")
    p.add_argument(
        "--scenario-policy", choices=["first_positive", "max_gap", "min_positive"],
        default="first_positive", help="Selection rule when --scenario is omitted."
    )
    p.add_argument("--gap-tol", type=float, default=1e-8)
    p.add_argument(
        "--violation-margin", type=float, default=0.0,
        help="epsilon>=0 in Delta(xi)<=-epsilon. Use 0 for the closed boundary problem."
    )
    p.add_argument("--time-limit", type=float, default=300.0)
    p.add_argument("--mip-gap", type=float, default=1e-4)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--int-feas-tol", type=float, default=1e-8)
    p.add_argument("--feasibility-tol", type=float, default=1e-8)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--write-model", action="store_true")
    p.add_argument(
        "--allow-sos1-reformulation", action="store_true",
        help="Do not force PreSOS1BigM=0. Leave off for the native-SOS1 experiment."
    )
    return p


def single_main(config: ExperimentConfig, argv: Optional[Sequence[str]] = None) -> int:
    args = build_single_parser(config).parse_args(argv)
    result = run_configured_experiment(
        config,
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
    print(f"\n=== {config.display_name} ===")
    print(f"Instance:          {result['instance_name']}")
    print(f"Scenario:          {result['selected_scenario']['scenario_name']}")
    print(f"Status:            {perf['status']}")
    print(f"Runtime (s):       {perf['runtime_sec_gurobi']:.3f}")
    print(f"Radius incumbent:  {perf['radius_incumbent']}")
    print(f"Radius best bound: {perf['best_bound_radius']}")
    print(f"Raw objective:     {perf['objective_raw']}")
    print(f"Raw best bound:    {perf['best_bound_raw']}")
    print(f"MIP gap:           {perf['mip_gap']}")
    if result.get("validation") is not None:
        print(f"True gap @ xi:     {result['validation']['true_gap_from_independent_recourse_LPs']}")
    print(f"Saved to:          {result['files']['canonical_result']}")
    return 0
