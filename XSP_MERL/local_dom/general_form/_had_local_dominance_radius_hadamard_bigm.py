"""
Hadamard-Cramer Big-M experiment for the reduced/asymmetric local-dominance model.

Place this file in:
    local_dom/general_form/

Prerequisites for each instance (already produced by run_workflow.py):
    output_data/stochastic_results.json
    output_data/expectedvalue_results.json
    output_data/cost_gap_diagnostics.json

Per-instance outputs are written to:
    <instance_dir>/output_data/hadamard_exp/

Core formulation
----------------
For a selected realized scenario xi_hat with positive cost gap, solve the exact
nearest-violation model using the asymmetric certificate:

    - EV side: only a primal-feasible y_EV is required.
    - x_star side: recourse optimality is imposed by KKT.
    - KKT complementarity is linearized with explicit binary variables and
      mathematically derived Big-M constants.

The star recourse problem is

    min d^T y
    s.t. W y <= H xi - T x_star,
         y >= 0.

Introduce row slacks v >= 0 and write it in standard form

    Wbar u = b_star(xi),
    u = [y; v] >= 0,
    Wbar = [W I],
    qbar = [d; 0].

For the LANDS instances W is integral. Because Wbar contains I, it has full row
rank r and every nonsingular basis determinant is a nonzero integer, hence has
absolute value at least one. Hadamard + Cramer therefore gives the rank-r
bounds used here.

Safe xi domain
--------------
A finite primal Big-M requires a finite bound on the follower RHS. We first
find ANY independently verified violating point xi_bar with Delta(xi_bar) <= 0
and set

    U = ||xi_bar - xi_hat||_1.

Then sum(t) <= U is a safe objective cutoff, and the componentwise xi bounds
xi_hat[k]-U <= xi[k] <= xi_hat[k]+U are implied and added explicitly for
presolve. This cannot remove a global nearest violation.

For every xi in that L1 ball,

    |b_star_i(xi)|
      <= |b_star_i(xi_hat)| + U * ||H_i,:||_infinity.

Let Bmax be the maximum of these rowwise bounds and let

    Wmax = max(1, max_ij |W_ij|),
    qmax = max_j |d_j|.

The dense Hadamard-Cramer factor is

    K_H(r) = r^(r/2).

We use

    M_primal = K_H * Bmax * Wmax^(r-1),

which bounds the components of one optimal basic standard-form recourse
solution u=[y_star; row_slack]. This specialization is sharper than the fully
general optimistic-bilevel primal bound because, in the one-sided local-
dominance certificate, individual components of y_star are not coupled to the
upper level; only the common optimal value d^T y_star is needed. Hence an
optimal basic follower solution is sufficient.

For the dual multiplier pi_star (mu=-pi_star >= 0),

    M_mu = K_H * qmax * Wmax^(r-1).

For each reduced cost rc_j = d_j + W[:,j]^T mu >= 0, we then use the valid
componentwise bound

    M_rc[j] = |d_j| + ||W[:,j]||_1 * M_mu.

The code also reports the corresponding factorial/Cramer bounds for comparison
but DOES NOT use them in the MILP.

Big-M complementarity
---------------------
For every recourse row i, with slack

    v_i = (H xi - T x_star - W y_star)_i >= 0,

use binary z_row[i]:

    v_i  <= M_primal * (1-z_row[i]),
    mu_i <= M_mu      * z_row[i].

For every recourse variable j, with reduced cost rc_j >= 0, use binary z_var[j]:

    y_star[j] <= M_primal * (1-z_var[j]),
    rc_j      <= M_rc[j]  * z_var[j].

No SOS constraints and no bilinear terms remain.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import math
import re
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception as exc:  # pragma: no cover
    gp = None
    GRB = None
    _GUROBI_IMPORT_ERROR = exc
else:
    _GUROBI_IMPORT_ERROR = None

try:  # package import
    from .local_dominance_radius_unbounded import LocalDominanceData
except Exception:
    try:  # direct execution from general_form/
        from local_dominance_radius_unbounded import LocalDominanceData
    except Exception as exc:  # allows pure bound helpers to be unit-tested standalone
        LocalDominanceData = None  # type: ignore
        _LOCAL_DATA_IMPORT_ERROR = exc
    else:
        _LOCAL_DATA_IMPORT_ERROR = None
else:
    _LOCAL_DATA_IMPORT_ERROR = None


class HadamardBigMError(RuntimeError):
    """Raised when the Hadamard Big-M experiment cannot be certified/run."""


Number = Union[int, float]


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
    names = [
        "LOADED", "OPTIMAL", "INFEASIBLE", "INF_OR_UNBD", "UNBOUNDED",
        "CUTOFF", "ITERATION_LIMIT", "NODE_LIMIT", "TIME_LIMIT",
        "SOLUTION_LIMIT", "INTERRUPTED", "NUMERIC", "SUBOPTIMAL",
        "INPROGRESS", "USER_OBJ_LIMIT", "WORK_LIMIT", "MEM_LIMIT",
    ]
    for name in names:
        if hasattr(GRB, name) and status == getattr(GRB, name):
            return name
    return f"STATUS_{status}"


def _safe_model_attr(model: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(model, name)
    except Exception:
        return default


def _max_abs(x: np.ndarray) -> float:
    return 0.0 if x.size == 0 else float(np.max(np.abs(x)))


def _positive_part_max(x: np.ndarray) -> float:
    return 0.0 if x.size == 0 else float(np.max(np.maximum(x, 0.0)))


def _log10_or_none(x: Optional[float]) -> Optional[float]:
    if x is None or x <= 0.0 or not math.isfinite(x):
        return None
    return float(math.log10(x))


def derive_hadamard_big_m_constants(
    W: np.ndarray,
    d: np.ndarray,
    Bmax: float,
    *,
    max_finite_M: float = 1e90,
) -> Dict[str, Any]:
    """Pure Hadamard-Cramer bound calculation (no solver required).

    Assumes the standard-form matrix Wbar=[W I_r] is integral. The caller is
    responsible for verifying integrality of W. Returns the Hadamard constants
    actually used by the MILP and factorial references for comparison.
    """
    W = np.asarray(W, dtype=float)
    d = np.asarray(d, dtype=float)
    if W.ndim != 2:
        raise ValueError("W must be a matrix.")
    r = int(W.shape[0])
    if r <= 0:
        raise ValueError("W must have at least one row.")
    if d.ndim != 1 or d.size != W.shape[1]:
        raise ValueError("d must be a vector with length equal to W columns.")
    if not math.isfinite(Bmax) or Bmax < 0.0:
        raise ValueError("Bmax must be finite and nonnegative.")

    Wbar_max = max(1.0, _max_abs(W))
    qbar_max = _max_abs(d)
    log_KH = 0.5 * r * math.log(float(r)) if r > 1 else 0.0
    if log_KH > math.log(float(max_finite_M)):
        raise HadamardBigMError(
            f"Hadamard factor r^(r/2) exceeds max_finite_M={max_finite_M:.3e}."
        )
    KH = float(math.exp(log_KH))
    KF = float(math.factorial(r))

    Mp = KH * Bmax * (Wbar_max ** max(0, r - 1))
    Mmu = KH * qbar_max * (Wbar_max ** max(0, r - 1))
    W_col_l1 = np.sum(np.abs(W), axis=0)
    Mrc = np.abs(d) + W_col_l1 * Mmu
    MHd_uniform = qbar_max * (1.0 + (float(r) ** (1.0 + 0.5 * r)) * (Wbar_max ** r))

    MpF = KF * Bmax * (Wbar_max ** max(0, r - 1))
    MmuF = KF * qbar_max * (Wbar_max ** max(0, r - 1))
    MBd_uniform = qbar_max * (1.0 + float(r) * KF * (Wbar_max ** r))

    values = [Mp, Mmu, MHd_uniform]
    if Mrc.size:
        values.append(float(np.max(Mrc)))
    if any((not math.isfinite(v) or v >= max_finite_M) for v in values):
        raise HadamardBigMError(
            "A derived Hadamard Big-M is non-finite or exceeds the configured "
            f"numerical safety threshold {max_finite_M:.3e}."
        )

    return {
        "rank_r": r,
        "Wbar_max_abs": Wbar_max,
        "qbar_max_abs": qbar_max,
        "hadamard_factor": KH,
        "factorial_factor": KF,
        "M_primal": Mp,
        "M_mu": Mmu,
        "M_reduced_cost_componentwise": Mrc,
        "M_dual_slack_uniform_writeup": MHd_uniform,
        "factorial_M_primal": MpF,
        "factorial_M_mu": MmuF,
        "factorial_M_dual_slack_uniform": MBd_uniform,
    }


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
    gap_payload = data.cost_gap_diagnostics.get("scenario_cost_gaps", {})
    if not isinstance(gap_payload, Mapping):
        raise HadamardBigMError(
            "cost_gap_diagnostics.json is missing 'scenario_cost_gaps'."
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
            raise HadamardBigMError(
                f"Scenario {idx} is outside 1..{data.num_scenarios}."
            )
        name = f"scenario_{idx}"
        rec = gap_payload.get(name)
        if rec is None:
            raise HadamardBigMError(f"No cost-gap record found for {name}.")
        gap = float(rec["cost_gap"])
        if gap <= gap_tol:
            raise HadamardBigMError(
                f"Selected {name} has cost gap {gap:.8g}, not > {gap_tol:.3g}."
            )
        return ScenarioChoice(
            index_1_based=idx,
            name=name,
            xi_hat=np.asarray(data.xi[idx - 1], dtype=float),
            cost_gap=gap,
            probability=float(rec.get("probability", data.probabilities[idx - 1])),
        )

    if not records:
        raise HadamardBigMError(
            f"Instance {data.instance_name!r} has no positive-gap scenarios."
        )
    if policy == "first_positive":
        return min(records, key=lambda z: z.index_1_based)
    if policy == "max_gap":
        return max(records, key=lambda z: z.cost_gap)
    if policy == "min_positive":
        return min(records, key=lambda z: z.cost_gap)
    raise ValueError("policy must be first_positive, max_gap, or min_positive")


@dataclass
class HadamardBoundCertificate:
    incumbent_radius_U: float
    incumbent_xi: np.ndarray
    incumbent_cost_gap: float
    incumbent_source: str
    search_runtime_sec: float
    bstar_abs_bound_by_row: np.ndarray
    Bmax: float
    Wbar_max_abs: float
    qbar_max_abs: float
    hadamard_factor: float
    factorial_factor: float
    M_primal: float
    M_mu: float
    M_reduced_cost: np.ndarray
    M_dual_slack_uniform_writeup: float
    factorial_M_primal: float
    factorial_M_mu: float
    factorial_M_reduced_cost_uniform: float

    def to_dict(self) -> Dict[str, Any]:
        ratio = (
            self.factorial_factor / self.hadamard_factor
            if self.hadamard_factor > 0.0
            else None
        )
        return {
            "safe_incumbent": {
                "radius_U": self.incumbent_radius_U,
                "xi": self.incumbent_xi,
                "cost_gap": self.incumbent_cost_gap,
                "source": self.incumbent_source,
                "search_runtime_sec": self.search_runtime_sec,
            },
            "rhs_bound": {
                "formula": "|b_i(xi)| <= |b_i(xi_hat)| + U * ||H_i,:||_inf",
                "rowwise_abs_bounds": self.bstar_abs_bound_by_row,
                "Bmax": self.Bmax,
            },
            "hadamard": {
                "Wbar_max_abs": self.Wbar_max_abs,
                "qbar_max_abs": self.qbar_max_abs,
                "factor_r_to_r_over_2": self.hadamard_factor,
                "M_primal_uniform_for_y_star_and_row_slack": self.M_primal,
                "M_mu_uniform": self.M_mu,
                "M_reduced_cost_componentwise": self.M_reduced_cost,
                "M_reduced_cost_max": float(np.max(self.M_reduced_cost)) if self.M_reduced_cost.size else 0.0,
                "M_dual_slack_uniform_exact_writeup_formula": self.M_dual_slack_uniform_writeup,
                "log10_M_primal": _log10_or_none(self.M_primal),
                "log10_M_mu": _log10_or_none(self.M_mu),
                "log10_M_reduced_cost_max": _log10_or_none(
                    float(np.max(self.M_reduced_cost)) if self.M_reduced_cost.size else 0.0
                ),
            },
            "factorial_reference_not_used": {
                "factor_r_factorial": self.factorial_factor,
                "factorial_over_hadamard_factor_ratio": ratio,
                "M_primal": self.factorial_M_primal,
                "M_mu": self.factorial_M_mu,
                "M_reduced_cost_uniform": self.factorial_M_reduced_cost_uniform,
                "log10_M_primal": _log10_or_none(self.factorial_M_primal),
                "log10_M_mu": _log10_or_none(self.factorial_M_mu),
                "log10_M_reduced_cost_uniform": _log10_or_none(self.factorial_M_reduced_cost_uniform),
            },
        }


class HadamardBigMLocalDominanceProblem:
    """Reduced one-sided nearest-violation model with Hadamard Big-M KKT."""

    def __init__(
        self,
        instance_name: str,
        *,
        base_dir: Optional[Union[str, Path]] = None,
        scenario: Optional[Union[int, str]] = None,
        scenario_policy: str = "first_positive",
        gap_tol: float = 1e-8,
        integer_tol: float = 1e-9,
    ):
        if gp is None or GRB is None:
            raise HadamardBigMError(
                "gurobipy could not be imported. Configure Gurobi first. "
                f"Original import error: {_GUROBI_IMPORT_ERROR}"
            )
        if LocalDominanceData is None:
            raise HadamardBigMError(
                "Could not import LocalDominanceData. Place this file beside "
                "local_dominance_radius_unbounded.py in local_dom/general_form/. "
                f"Original import error: {_LOCAL_DATA_IMPORT_ERROR}"
            )

        self.data = LocalDominanceData.load(instance_name, base_dir=base_dir)
        self.choice = choose_positive_gap_scenario(
            self.data, scenario=scenario, policy=scenario_policy, gap_tol=gap_tol
        )
        self.gap_tol = float(gap_tol)
        self.integer_tol = float(integer_tol)
        self.output_dir = self.data.instance_dir / "output_data" / "hadamard_exp"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._Tx_EV = np.asarray(self.data.T @ self.data.x_EV, dtype=float)
        self._Tx_star = np.asarray(self.data.T @ self.data.x_star, dtype=float)

        self._W_row_nz = [
            np.flatnonzero(np.abs(self.data.W[i, :]) > 0.0)
            for i in range(self.data.num_recourse_constraints)
        ]
        self._W_col_nz = [
            np.flatnonzero(np.abs(self.data.W[:, j]) > 0.0)
            for j in range(self.data.num_second_stage_vars)
        ]
        self._H_row_nz = [
            np.flatnonzero(np.abs(self.data.H[i, :]) > 0.0)
            for i in range(self.data.num_recourse_constraints)
        ]

        # The determinant denominator >= 1 argument used by the write-up
        # requires an integer basis matrix. For this standard-form recourse,
        # Wbar=[W I], so it is enough to verify W is integral.
        dev = _max_abs(self.data.W - np.rint(self.data.W))
        if dev > self.integer_tol:
            raise HadamardBigMError(
                "Hadamard-Cramer certificate requires integral W (or an exact "
                "denominator-clearing transformation). This instance has "
                f"max |W-round(W)|={dev:.3e} > integer_tol={self.integer_tol:.3e}."
            )
        self.integer_W_max_deviation = dev

    @property
    def structural_metrics(self) -> Dict[str, Any]:
        d = self.data
        m = d.xi_dim
        ny = d.num_second_stage_vars
        r = d.num_recourse_constraints
        # Explicit Big-M implementation eliminates row_slack/reduced_cost vars
        # but adds one binary per complementarity pair.
        n_cont = 2 * m + 2 * ny + r
        n_bin = ny + r
        n_rows = 2 * m + 4 * r + 3 * ny + 2  # + radius cutoff + certificate
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
            "standard_form_follower_dimension_y_plus_row_slack": ny + r,
            "integer_W_verified": True,
            "integer_W_max_deviation": self.integer_W_max_deviation,
            "theoretical_hadamard_bigm_num_continuous_vars": n_cont,
            "theoretical_hadamard_bigm_num_binary_vars": n_bin,
            "theoretical_hadamard_bigm_num_total_vars": n_cont + n_bin,
            "theoretical_hadamard_bigm_num_linear_constraints": n_rows,
        }

    def _W_row_expr(self, vars_by_j: Any, i: int) -> Any:
        return gp.quicksum(
            float(self.data.W[i, j]) * vars_by_j[int(j)]
            for j in self._W_row_nz[i]
        )

    def _W_col_mu_expr(self, mu: Any, j: int) -> Any:
        return gp.quicksum(
            float(self.data.W[i, j]) * mu[int(i)]
            for i in self._W_col_nz[j]
        )

    def _Hxi_expr(self, xi: Any, i: int) -> Any:
        return gp.quicksum(
            float(self.data.H[i, k]) * xi[int(k)]
            for k in self._H_row_nz[i]
        )

    def _solve_recourse_lp(
        self,
        x_fixed: np.ndarray,
        xi_value: np.ndarray,
        *,
        return_solution: bool = False,
    ) -> Dict[str, Any]:
        d = self.data
        ny = d.num_second_stage_vars
        nr = d.num_recourse_constraints
        rhs = np.asarray(d.H @ xi_value - d.T @ x_fixed, dtype=float)

        lp = gp.Model("hadamard_recourse_eval")
        lp.Params.OutputFlag = 0
        y = lp.addVars(ny, lb=0.0, vtype=GRB.CONTINUOUS, name="y")
        cons = []
        for i in range(nr):
            cons.append(
                lp.addConstr(self._W_row_expr(y, i) <= float(rhs[i]), name=f"rec[{i}]")
            )
        lp.setObjective(
            gp.quicksum(
                float(d.d[j]) * y[j] for j in range(ny) if d.d[j] != 0.0
            ),
            GRB.MINIMIZE,
        )
        lp.optimize()
        out: Dict[str, Any] = {
            "status": _status_name(int(lp.Status)),
            "objective": None,
        }
        if lp.Status == GRB.OPTIMAL:
            out["objective"] = float(lp.ObjVal)
            if return_solution:
                out["y"] = [float(y[j].X) for j in range(ny)]
                out["pi"] = [float(c.Pi) for c in cons]
        lp.dispose()
        return out

    def evaluate_true_gap(self, xi_value: np.ndarray) -> Dict[str, Any]:
        ev = self._solve_recourse_lp(self.data.x_EV, xi_value, return_solution=False)
        star = self._solve_recourse_lp(self.data.x_star, xi_value, return_solution=False)
        if ev["objective"] is None or star["objective"] is None:
            raise HadamardBigMError(
                "Could not evaluate both recourse LPs at candidate xi. "
                f"EV={ev['status']}, star={star['status']}"
            )
        gap = (
            self.data.first_stage_cost_EV
            + float(ev["objective"])
            - self.data.first_stage_cost_star
            - float(star["objective"])
        )
        return {
            "xi": np.asarray(xi_value, dtype=float),
            "cost_gap": float(gap),
            "Q_EV": float(ev["objective"]),
            "Q_star": float(star["objective"]),
        }

    def find_safe_incumbent(
        self,
        *,
        radius_multiplier: float = 10.0,
        growth_factor: float = 2.0,
        bisection_iterations: int = 40,
        num_random_directions: int = 20,
        random_seed: int = 12345,
        max_radius: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Find any independently verified xi_bar with Delta(xi_bar)<=0.

        Priority:
          1) existing realized scenarios in cost_gap_diagnostics.json,
          2) deterministic/rand seeded ray search using independent recourse LPs.

        The returned U is therefore solver-independent preprocessing, not an
        imported incumbent from another local-dominance formulation.
        """
        start = time.perf_counter()
        d = self.data
        xi_hat = np.asarray(self.choice.xi_hat, dtype=float)

        best: Optional[Dict[str, Any]] = None
        evaluated = 0
        failed = 0

        def radius(x: np.ndarray) -> float:
            return float(np.sum(np.abs(x - xi_hat)))

        def update(candidate: Dict[str, Any], source: str) -> None:
            nonlocal best
            if float(candidate["cost_gap"]) <= self.gap_tol:
                U = radius(np.asarray(candidate["xi"], dtype=float))
                rec = {**candidate, "source": source, "radius_U": U}
                if best is None or U < float(best["radius_U"]):
                    best = rec

        # Step 1: saved realized scenarios.
        gap_payload = d.cost_gap_diagnostics.get("scenario_cost_gaps", {})
        if isinstance(gap_payload, Mapping):
            for idx, name in enumerate(d.scenario_names):
                rec = gap_payload.get(name)
                if rec is None:
                    continue
                evaluated += 1
                gap = float(rec["cost_gap"])
                xi_s = np.asarray(d.xi[idx], dtype=float)
                if gap <= self.gap_tol:
                    update(
                        {
                            "xi": xi_s,
                            "cost_gap": gap,
                            "scenario_name": name,
                            "scenario_index_1_based": idx + 1,
                        },
                        "saved_realized_scenario",
                    )

        if best is not None:
            best["search_runtime_sec"] = float(time.perf_counter() - start)
            best["num_candidate_evaluations"] = evaluated
            best["num_failed_evaluations"] = failed
            return best

        # Step 2: ray search.
        directions: List[Tuple[str, np.ndarray]] = []
        for idx, xi_s in enumerate(d.xi):
            vec = np.asarray(xi_s, dtype=float) - xi_hat
            nrm = float(np.sum(np.abs(vec)))
            if nrm > 1e-12:
                directions.append((f"toward_scenario_{idx+1}", vec / nrm))

        xi_mean = sum(
            float(p) * np.asarray(x, dtype=float)
            for p, x in zip(d.probabilities, d.xi)
        )
        vec = xi_mean - xi_hat
        nrm = float(np.sum(np.abs(vec)))
        if nrm > 1e-12:
            directions.append(("toward_expected_xi", vec / nrm))

        for k in range(d.xi_dim):
            e = np.zeros(d.xi_dim)
            e[k] = 1.0
            directions.append((f"positive_coordinate_{k}", e.copy()))
            directions.append((f"negative_coordinate_{k}", -e.copy()))

        ones = np.ones(d.xi_dim)
        directions.append(("positive_all_ones", ones / np.sum(np.abs(ones))))
        directions.append(("negative_all_ones", -ones / np.sum(np.abs(ones))))

        rng = np.random.default_rng(random_seed)
        for k in range(int(num_random_directions)):
            v = rng.normal(size=d.xi_dim)
            nrm = float(np.sum(np.abs(v)))
            if nrm > 1e-12:
                directions.append((f"random_normal_{k}", v / nrm))

        # Remove duplicates.
        unique: List[Tuple[str, np.ndarray]] = []
        seen: List[np.ndarray] = []
        for name, v in directions:
            if any(float(np.max(np.abs(v - old))) <= 1e-10 for old in seen):
                continue
            seen.append(v)
            unique.append((name, v))

        spread = max(
            [float(np.sum(np.abs(np.asarray(x, dtype=float) - xi_hat))) for x in d.xi]
            + [1.0]
        )
        max_radius_eff = (
            float(max_radius)
            if max_radius is not None
            else float(radius_multiplier) * max(1.0, spread)
        )
        initial_step = max(1e-6, min(1.0, spread / 100.0))

        for direction_name, direction in unique:
            lo = 0.0
            hi = initial_step
            found = False
            hi_eval: Optional[Dict[str, Any]] = None

            while hi <= max_radius_eff * (1.0 + 1e-12):
                xi_try = xi_hat + hi * direction
                try:
                    ev = self.evaluate_true_gap(xi_try)
                    evaluated += 1
                except Exception:
                    failed += 1
                    break

                if ev["cost_gap"] <= self.gap_tol:
                    found = True
                    hi_eval = ev
                    break
                lo = hi
                hi *= float(growth_factor)

            if not found or hi_eval is None:
                continue

            # Refine the first crossing on this ray.
            best_on_ray = hi_eval
            hi_ref = hi
            lo_ref = lo
            for _ in range(int(bisection_iterations)):
                mid = 0.5 * (lo_ref + hi_ref)
                xi_mid = xi_hat + mid * direction
                try:
                    ev = self.evaluate_true_gap(xi_mid)
                    evaluated += 1
                except Exception:
                    failed += 1
                    break
                if ev["cost_gap"] <= self.gap_tol:
                    hi_ref = mid
                    best_on_ray = ev
                else:
                    lo_ref = mid

            update(
                {
                    "xi": np.asarray(best_on_ray["xi"], dtype=float),
                    "cost_gap": float(best_on_ray["cost_gap"]),
                    "Q_EV": float(best_on_ray["Q_EV"]),
                    "Q_star": float(best_on_ray["Q_star"]),
                    "direction_name": direction_name,
                },
                "independent_recourse_ray_search",
            )

        if best is None:
            raise HadamardBigMError(
                "Could not find a safe violating incumbent to bound xi and the "
                "star-recouse RHS. Increase --search-radius-multiplier or provide "
                "a larger --search-max-radius."
            )

        best["search_runtime_sec"] = float(time.perf_counter() - start)
        best["num_candidate_evaluations"] = evaluated
        best["num_failed_evaluations"] = failed
        return best

    def compute_hadamard_bounds(
        self,
        *,
        radius_multiplier: float = 10.0,
        growth_factor: float = 2.0,
        bisection_iterations: int = 40,
        num_random_directions: int = 20,
        random_seed: int = 12345,
        max_radius: Optional[float] = None,
        max_finite_M: float = 1e90,
    ) -> HadamardBoundCertificate:
        d = self.data
        r = d.num_recourse_constraints
        if r <= 0:
            raise HadamardBigMError("Recourse must have at least one row.")

        inc = self.find_safe_incumbent(
            radius_multiplier=radius_multiplier,
            growth_factor=growth_factor,
            bisection_iterations=bisection_iterations,
            num_random_directions=num_random_directions,
            random_seed=random_seed,
            max_radius=max_radius,
        )
        U = float(inc["radius_U"])
        if not math.isfinite(U) or U <= 0.0:
            raise HadamardBigMError(f"Invalid safe radius U={U!r}.")

        xi_hat = np.asarray(self.choice.xi_hat, dtype=float)
        b_hat = np.asarray(d.H @ xi_hat - d.T @ d.x_star, dtype=float)
        H_row_inf = np.max(np.abs(d.H), axis=1) if d.H.size else np.zeros(r)
        b_abs_bounds = np.abs(b_hat) + U * H_row_inf
        Bmax = float(np.max(b_abs_bounds)) if b_abs_bounds.size else 0.0

        constants = derive_hadamard_big_m_constants(
            d.W, d.d, Bmax, max_finite_M=max_finite_M
        )
        Wbar_max = float(constants["Wbar_max_abs"])
        qbar_max = float(constants["qbar_max_abs"])
        KH = float(constants["hadamard_factor"])
        KF = float(constants["factorial_factor"])
        M_primal = float(constants["M_primal"])
        M_mu = float(constants["M_mu"])
        M_rc = np.asarray(constants["M_reduced_cost_componentwise"], dtype=float)
        M_Hd_uniform = float(constants["M_dual_slack_uniform_writeup"])
        M_primal_F = float(constants["factorial_M_primal"])
        M_mu_F = float(constants["factorial_M_mu"])
        M_rc_F_uniform = float(constants["factorial_M_dual_slack_uniform"])

        cert = HadamardBoundCertificate(
            incumbent_radius_U=U,
            incumbent_xi=np.asarray(inc["xi"], dtype=float),
            incumbent_cost_gap=float(inc["cost_gap"]),
            incumbent_source=str(inc["source"]),
            search_runtime_sec=float(inc["search_runtime_sec"]),
            bstar_abs_bound_by_row=b_abs_bounds,
            Bmax=Bmax,
            Wbar_max_abs=Wbar_max,
            qbar_max_abs=qbar_max,
            hadamard_factor=KH,
            factorial_factor=KF,
            M_primal=M_primal,
            M_mu=M_mu,
            M_reduced_cost=np.asarray(M_rc, dtype=float),
            M_dual_slack_uniform_writeup=M_Hd_uniform,
            factorial_M_primal=M_primal_F,
            factorial_M_mu=M_mu_F,
            factorial_M_reduced_cost_uniform=M_rc_F_uniform,
        )

        bounds_payload = {
            "instance_name": d.instance_name,
            "selected_scenario": self.choice.name,
            "theorem_checks": {
                "standard_form_Wbar": "[W I_r]",
                "W_integral_verified": True,
                "max_abs_W_minus_round_W": self.integer_W_max_deviation,
                "Wbar_full_row_rank_reason": "identity slack block I_r",
                "determinant_denominator_lower_bound": "|det(B)| >= 1 for every nonsingular integer basis B",
            },
            "specialization_note": (
                "Unlike the general optimistic-bilevel primal bound, this one-sided "
                "local-dominance certificate only needs the optimal recourse value on "
                "the x_star side. Therefore one optimal basic follower solution is "
                "sufficient, allowing a rank-r Hadamard-Cramer primal M after a safe "
                "incumbent radius bounds the RHS."
            ),
            "rank_r": r,
            **cert.to_dict(),
        }
        _write_json(
            self.output_dir / f"hadamard_bounds_{self.choice.name}.json",
            bounds_payload,
        )
        _write_json(self.output_dir / "hadamard_bounds.json", bounds_payload)
        return cert

    def build_model(self, bounds: HadamardBoundCertificate) -> Tuple[Any, Dict[str, Any]]:
        d = self.data
        m = d.xi_dim
        ny = d.num_second_stage_vars
        nr = d.num_recourse_constraints
        U = bounds.incumbent_radius_U
        Mp = bounds.M_primal
        Mmu = bounds.M_mu
        Mrc = bounds.M_reduced_cost

        model = gp.Model(f"local_dom_hadamard_bigm_{d.instance_name}_{self.choice.name}")

        xi = model.addVars(
            m,
            lb={k: float(self.choice.xi_hat[k] - U) for k in range(m)},
            ub={k: float(self.choice.xi_hat[k] + U) for k in range(m)},
            vtype=GRB.CONTINUOUS,
            name="xi",
        )
        t = model.addVars(m, lb=0.0, ub=U, vtype=GRB.CONTINUOUS, name="t")
        y_EV = model.addVars(ny, lb=0.0, vtype=GRB.CONTINUOUS, name="y_EV")
        y_star = model.addVars(ny, lb=0.0, ub=Mp, vtype=GRB.CONTINUOUS, name="y_star")
        mu = model.addVars(nr, lb=0.0, ub=Mmu, vtype=GRB.CONTINUOUS, name="mu")
        z_row = model.addVars(nr, vtype=GRB.BINARY, name="z_row")
        z_var = model.addVars(ny, vtype=GRB.BINARY, name="z_var")

        model.setObjective(gp.quicksum(t[k] for k in range(m)), GRB.MINIMIZE)

        for k in range(m):
            hat = float(self.choice.xi_hat[k])
            model.addConstr(xi[k] - hat <= t[k], name=f"abs_upper[{k}]")
            model.addConstr(hat - xi[k] <= t[k], name=f"abs_lower[{k}]")
        model.addConstr(gp.quicksum(t[k] for k in range(m)) <= U, name="safe_radius_cutoff")

        # EV primal feasibility only.
        for i in range(nr):
            model.addConstr(
                self._W_row_expr(y_EV, i)
                <= self._Hxi_expr(xi, i) - float(self._Tx_EV[i]),
                name=f"primal_EV[{i}]",
            )

        # Star primal feasibility and row complementarity Big-M.
        for i in range(nr):
            slack_expr = (
                self._Hxi_expr(xi, i)
                - float(self._Tx_star[i])
                - self._W_row_expr(y_star, i)
            )
            model.addConstr(slack_expr >= 0.0, name=f"primal_star[{i}]")
            model.addConstr(
                slack_expr <= Mp * (1.0 - z_row[i]),
                name=f"bigM_row_slack[{i}]",
            )
            model.addConstr(
                mu[i] <= Mmu * z_row[i],
                name=f"bigM_mu[{i}]",
            )

        # Dual feasibility and y/reduced-cost complementarity Big-M.
        for j in range(ny):
            rc_expr = float(d.d[j]) + self._W_col_mu_expr(mu, j)
            model.addConstr(rc_expr >= 0.0, name=f"dual_feas_rc[{j}]")
            model.addConstr(
                y_star[j] <= Mp * (1.0 - z_var[j]),
                name=f"bigM_y_star[{j}]",
            )
            model.addConstr(
                rc_expr <= float(Mrc[j]) * z_var[j],
                name=f"bigM_reduced_cost[{j}]",
            )

        lhs = float(d.first_stage_cost_EV) + gp.quicksum(
            float(d.d[j]) * y_EV[j] for j in range(ny) if d.d[j] != 0.0
        )
        rhs = float(d.first_stage_cost_star) + gp.quicksum(
            float(d.d[j]) * y_star[j] for j in range(ny) if d.d[j] != 0.0
        )
        model.addConstr(lhs <= rhs, name="violation_certificate")

        model.update()
        return model, {
            "xi": xi,
            "t": t,
            "y_EV": y_EV,
            "y_star": y_star,
            "mu": mu,
            "z_row": z_row,
            "z_var": z_var,
        }

    def _validate_incumbent(
        self,
        vals: Dict[str, np.ndarray],
        bounds: HadamardBoundCertificate,
    ) -> Dict[str, Any]:
        d = self.data
        xi = vals["xi"]
        t = vals["t"]
        y_EV = vals["y_EV"]
        y_star = vals["y_star"]
        mu = vals["mu"]
        z_row = vals["z_row"]
        z_var = vals["z_var"]

        b_EV = d.H @ xi - d.T @ d.x_EV
        b_star = d.H @ xi - d.T @ d.x_star
        row_slack = b_star - d.W @ y_star
        reduced_cost = d.d + d.W.T @ mu
        primal_EV_resid = d.W @ y_EV - b_EV

        cert_lhs = d.first_stage_cost_EV + float(d.d @ y_EV)
        cert_rhs = d.first_stage_cost_star + float(d.d @ y_star)

        ev_lp = self._solve_recourse_lp(d.x_EV, xi, return_solution=True)
        star_lp = self._solve_recourse_lp(d.x_star, xi, return_solution=True)
        true_gap = None
        if ev_lp["objective"] is not None and star_lp["objective"] is not None:
            true_gap = (
                d.first_stage_cost_EV + float(ev_lp["objective"])
                - d.first_stage_cost_star - float(star_lp["objective"])
            )

        Mrc = bounds.M_reduced_cost
        rc_util = np.divide(
            reduced_cost,
            Mrc,
            out=np.zeros_like(reduced_cost),
            where=Mrc > 0.0,
        )

        return {
            "l1_distance_from_xi_hat": float(np.sum(np.abs(xi - self.choice.xi_hat))),
            "sum_t": float(np.sum(t)),
            "sum_t_minus_l1_distance": float(np.sum(t) - np.sum(np.abs(xi - self.choice.xi_hat))),
            "safe_radius_U": bounds.incumbent_radius_U,
            "sum_t_minus_U": float(np.sum(t) - bounds.incumbent_radius_U),
            "certificate_lhs_minus_rhs": float(cert_lhs - cert_rhs),
            "max_primal_EV_violation_positive_part": _positive_part_max(primal_EV_resid),
            "min_star_row_slack": float(np.min(row_slack)) if row_slack.size else 0.0,
            "min_reduced_cost": float(np.min(reduced_cost)) if reduced_cost.size else 0.0,
            "max_row_complementarity_product_abs": _max_abs(row_slack * mu),
            "max_variable_complementarity_product_abs": _max_abs(y_star * reduced_cost),
            "max_row_binary_integrality_error": _max_abs(z_row - np.rint(z_row)),
            "max_var_binary_integrality_error": _max_abs(z_var - np.rint(z_var)),
            "bigM_utilization": {
                "max_y_star_over_M_primal": float(np.max(y_star) / bounds.M_primal) if y_star.size else 0.0,
                "max_row_slack_over_M_primal": float(np.max(row_slack) / bounds.M_primal) if row_slack.size else 0.0,
                "max_mu_over_M_mu": float(np.max(mu) / bounds.M_mu) if mu.size else 0.0,
                "max_reduced_cost_over_component_M": float(np.max(rc_util)) if rc_util.size else 0.0,
            },
            "independent_recourse_EV": ev_lp,
            "independent_recourse_star": star_lp,
            "true_gap_from_independent_recourse_LPs": true_gap,
            "star_recourse_objective_from_KKT_y_star": float(d.d @ y_star),
            "star_recourse_KKT_minus_independent_LP": (
                None if star_lp["objective"] is None
                else float(d.d @ y_star) - float(star_lp["objective"])
            ),
            "pi_star_recovered_as_minus_mu": (-mu).tolist(),
            "row_slack": row_slack.tolist(),
            "reduced_cost": reduced_cost.tolist(),
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
        numeric_focus: int = 0,
        tee: bool = True,
        write_model: bool = False,
        compute_lp_relaxation: bool = True,
        radius_multiplier: float = 10.0,
        growth_factor: float = 2.0,
        bisection_iterations: int = 40,
        num_random_directions: int = 20,
        random_seed: int = 12345,
        max_radius: Optional[float] = None,
        max_finite_M: float = 1e90,
        extra_gurobi_params: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        bounds_start = time.perf_counter()
        bounds = self.compute_hadamard_bounds(
            radius_multiplier=radius_multiplier,
            growth_factor=growth_factor,
            bisection_iterations=bisection_iterations,
            num_random_directions=num_random_directions,
            random_seed=random_seed,
            max_radius=max_radius,
            max_finite_M=max_finite_M,
        )
        bounds_wall = float(time.perf_counter() - bounds_start)

        model, h = self.build_model(bounds)
        tag = self.choice.name
        log_path = self.output_dir / f"gurobi_{tag}.log"
        result_path = self.output_dir / f"hadamard_result_{tag}.json"
        canonical_path = self.output_dir / "hadamard_result.json"

        model.Params.TimeLimit = float(time_limit)
        model.Params.MIPGap = float(mip_gap)
        model.Params.Threads = int(threads)
        model.Params.Seed = int(seed)
        model.Params.LogFile = str(log_path)
        model.Params.OutputFlag = 1
        model.Params.LogToConsole = 1 if tee else 0
        model.Params.NumericFocus = int(numeric_focus)
        if int_feas_tol is not None:
            model.Params.IntFeasTol = float(int_feas_tol)
        if feasibility_tol is not None:
            model.Params.FeasibilityTol = float(feasibility_tol)
        if extra_gurobi_params:
            for key, value in extra_gurobi_params.items():
                model.setParam(str(key), value)

        if write_model:
            model.write(str(self.output_dir / f"hadamard_model_{tag}.lp"))

        model.update()
        raw_stats = {
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

        lp_relax: Dict[str, Any] = {
            "computed": bool(compute_lp_relaxation),
            "status": None,
            "objective": None,
            "runtime_sec": None,
        }
        if compute_lp_relaxation:
            relax = model.relax()
            relax.Params.OutputFlag = 0
            relax_start = time.perf_counter()
            relax.optimize()
            lp_relax["runtime_sec"] = float(time.perf_counter() - relax_start)
            lp_relax["status"] = _status_name(int(relax.Status))
            if relax.Status == GRB.OPTIMAL:
                lp_relax["objective"] = float(relax.ObjVal)
            relax.dispose()

        solve_start = time.perf_counter()
        model.optimize()
        solve_wall = float(time.perf_counter() - solve_start)

        status_code = int(model.Status)
        status = _status_name(status_code)
        sol_count = int(model.SolCount)
        has_inc = sol_count > 0

        perf: Dict[str, Any] = {
            "status_code": status_code,
            "status": status,
            "has_incumbent": has_inc,
            "solution_count": sol_count,
            "runtime_sec_gurobi": float(_safe_model_attr(model, "Runtime", solve_wall)),
            "runtime_sec_python_wall": solve_wall,
            "bounds_preprocessing_runtime_sec": bounds_wall,
            "total_runtime_including_bounds_sec": bounds_wall + solve_wall,
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

        solution = None
        validation = None
        if has_inc:
            try:
                perf["objective_value"] = float(model.ObjVal)
            except Exception:
                pass
            try:
                perf["mip_gap"] = float(model.MIPGap)
            except Exception:
                pass

            vals = {
                "xi": np.asarray([h["xi"][k].X for k in range(self.data.xi_dim)], dtype=float),
                "t": np.asarray([h["t"][k].X for k in range(self.data.xi_dim)], dtype=float),
                "y_EV": np.asarray([h["y_EV"][j].X for j in range(self.data.num_second_stage_vars)], dtype=float),
                "y_star": np.asarray([h["y_star"][j].X for j in range(self.data.num_second_stage_vars)], dtype=float),
                "mu": np.asarray([h["mu"][i].X for i in range(self.data.num_recourse_constraints)], dtype=float),
                "z_row": np.asarray([h["z_row"][i].X for i in range(self.data.num_recourse_constraints)], dtype=float),
                "z_var": np.asarray([h["z_var"][j].X for j in range(self.data.num_second_stage_vars)], dtype=float),
            }
            row_slack = self.data.H @ vals["xi"] - self.data.T @ self.data.x_star - self.data.W @ vals["y_star"]
            reduced_cost = self.data.d + self.data.W.T @ vals["mu"]
            solution = {
                "gamma_l1_radius_incumbent": float(np.sum(vals["t"])),
                "nearest_violation_xi": vals["xi"].tolist(),
                "absolute_deviation_t": vals["t"].tolist(),
                "y_EV": vals["y_EV"].tolist(),
                "y_star": vals["y_star"].tolist(),
                "mu_equals_minus_pi_star": vals["mu"].tolist(),
                "pi_star": (-vals["mu"]).tolist(),
                "row_slack": row_slack.tolist(),
                "reduced_cost": reduced_cost.tolist(),
                "z_row": vals["z_row"].tolist(),
                "z_var": vals["z_var"].tolist(),
            }
            validation = self._validate_incumbent(vals, bounds)

        result: Dict[str, Any] = {
            "instance_name": self.data.instance_name,
            "method": "reduced_asymmetric_star_KKT_Hadamard_BigM",
            "formulation": {
                "EV_side": "primal feasible y_EV only",
                "star_side": "primal y_star + dual mu=-pi_star + KKT",
                "complementarity": "Fortuny-Amat explicit binaries",
                "big_M_source": "rank-r dense Hadamard-Cramer bounds specialized to standard-form star recourse [W I]",
                "safe_xi_domain_source": "independently verified violating incumbent U",
                "g_equals_Ht_pi_variables": False,
                "SOS1_constraints": False,
                "nonconvex_bilinear_terms": False,
            },
            "selected_scenario": {
                "scenario_index_1_based": self.choice.index_1_based,
                "scenario_name": self.choice.name,
                "xi_hat": self.choice.xi_hat.tolist(),
                "initial_cost_gap": self.choice.cost_gap,
                "probability": self.choice.probability,
            },
            "instance_structure": self.structural_metrics,
            "hadamard_bounds": {
                "rank_r": self.data.num_recourse_constraints,
                **bounds.to_dict(),
            },
            "raw_gurobi_model_structure": raw_stats,
            "continuous_relaxation": lp_relax,
            "solver_configuration": {
                "solver": "gurobi",
                "gurobi_version": list(gp.gurobi.version()),
                "time_limit_sec": float(time_limit),
                "target_mip_gap": float(mip_gap),
                "threads": int(threads),
                "seed": int(seed),
                "int_feas_tol": int_feas_tol,
                "feasibility_tol": feasibility_tol,
                "numeric_focus": int(numeric_focus),
                "extra_gurobi_params": dict(extra_gurobi_params or {}),
            },
            "performance": perf,
            "solution": solution,
            "validation": validation,
            "files": {
                "gurobi_log": str(log_path),
                "scenario_result": str(result_path),
                "canonical_result": str(canonical_path),
                "bounds": str(self.output_dir / "hadamard_bounds.json"),
            },
        }

        _write_json(result_path, result)
        _write_json(canonical_path, result)
        model.dispose()
        return result


def run_hadamard_local_dominance(
    instance_name: str,
    *,
    base_dir: Optional[Union[str, Path]] = None,
    scenario: Optional[Union[int, str]] = None,
    scenario_policy: str = "first_positive",
    gap_tol: float = 1e-8,
    **solve_kwargs: Any,
) -> Dict[str, Any]:
    problem = HadamardBigMLocalDominanceProblem(
        instance_name,
        base_dir=base_dir,
        scenario=scenario,
        scenario_policy=scenario_policy,
        gap_tol=gap_tol,
    )
    return problem.solve(**solve_kwargs)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Solve the reduced local-dominance model with Hadamard-Cramer Big-M KKT."
    )
    p.add_argument("instance_name")
    p.add_argument("--base-dir", default=None)
    p.add_argument("--scenario", default=None)
    p.add_argument(
        "--scenario-policy",
        choices=["first_positive", "max_gap", "min_positive"],
        default="first_positive",
    )
    p.add_argument("--gap-tol", type=float, default=1e-8)
    p.add_argument("--time-limit", type=float, default=300.0)
    p.add_argument("--mip-gap", type=float, default=1e-4)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--int-feas-tol", type=float, default=1e-8)
    p.add_argument("--feasibility-tol", type=float, default=1e-8)
    p.add_argument("--numeric-focus", type=int, choices=[0, 1, 2, 3], default=0)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--write-model", action="store_true")
    p.add_argument("--skip-lp-relaxation", action="store_true")
    p.add_argument("--search-radius-multiplier", type=float, default=10.0)
    p.add_argument("--search-growth-factor", type=float, default=2.0)
    p.add_argument("--search-bisection-iterations", type=int, default=40)
    p.add_argument("--search-random-directions", type=int, default=20)
    p.add_argument("--search-random-seed", type=int, default=12345)
    p.add_argument("--search-max-radius", type=float, default=None)
    p.add_argument("--max-finite-M", type=float, default=1e90)
    return p


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    args = _parser().parse_args(argv)
    result = run_hadamard_local_dominance(
        args.instance_name,
        base_dir=args.base_dir,
        scenario=args.scenario,
        scenario_policy=args.scenario_policy,
        gap_tol=args.gap_tol,
        time_limit=args.time_limit,
        mip_gap=args.mip_gap,
        threads=args.threads,
        seed=args.seed,
        int_feas_tol=args.int_feas_tol,
        feasibility_tol=args.feasibility_tol,
        numeric_focus=args.numeric_focus,
        tee=not args.quiet,
        write_model=args.write_model,
        compute_lp_relaxation=not args.skip_lp_relaxation,
        radius_multiplier=args.search_radius_multiplier,
        growth_factor=args.search_growth_factor,
        bisection_iterations=args.search_bisection_iterations,
        num_random_directions=args.search_random_directions,
        random_seed=args.search_random_seed,
        max_radius=args.search_max_radius,
        max_finite_M=args.max_finite_M,
    )

    perf = result["performance"]
    hb = result["hadamard_bounds"]["hadamard"]
    inc = result["hadamard_bounds"]["safe_incumbent"]
    val = result.get("validation") or {}
    print("\n=== Hadamard Big-M local-dominance result ===")
    print(f"Instance:      {result['instance_name']}")
    print(f"Scenario:      {result['selected_scenario']['scenario_name']}")
    print(f"Safe U:        {inc['radius_U']}")
    print(f"M_primal:      {hb['M_primal_uniform_for_y_star_and_row_slack']}")
    print(f"M_mu:          {hb['M_mu_uniform']}")
    print(f"max M_rc:      {hb['M_reduced_cost_max']}")
    print(f"LP relaxation: {result['continuous_relaxation'].get('objective')}")
    print(f"Status:        {perf['status']}")
    print(f"Runtime (s):   {perf['runtime_sec_gurobi']:.3f}")
    print(f"Incumbent:     {perf.get('objective_value')}")
    print(f"Best bound:    {perf.get('best_bound')}")
    print(f"MIP gap:       {perf.get('mip_gap')}")
    print(f"True gap @ xi: {val.get('true_gap_from_independent_recourse_LPs')}")
    print(f"Saved to:      {result['files']['canonical_result']}")
    return result


if __name__ == "__main__":
    main()
