"""
Isolated directional local-dominance upper-bound workflow.

This module intentionally does not import general_model.py or
local_dominance_radius.py. It assumes the first-stage stochastic and expected-
value solves have already been run and that an instance folder contains:

    instance_data.json
    stochastic_results.json
    expectedvalue_results.json
    cost_gap_diagnostics.json

For a selected positive-gap scenario xi_hat, it computes the coordinate-ray
upper bound

    gamma_bar = min_{i, sigma in {-1,+1}} inf{t >= 0 : Delta(xi_hat + sigma t e_i) <= 0}

using a basis-tracking implementation for the two parametric recourse LPs.
The recourse LP is converted to standard form

    min cbar^T u
    s.t. Abar u = H xi_hat - T x + sigma t H e_i,
         u >= 0,

where Abar = [W I] and cbar = [d 0].  Since only the RHS changes along a ray,
any dual-feasible basis remains optimal as long as its basic solution remains
primal feasible.  At breakpoints, this implementation applies a deterministic
lexicographic basis-selection rule over the dual-feasible standard-form bases.
For small/medium recourse systems this is exact and highly auditable.  The
basis enumeration step is combinatorial; use the reported counts as a scalability
indicator.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import itertools
import json
import math
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

try:
    import numpy as np
except Exception as exc:  # pragma: no cover
    raise RuntimeError("numpy is required for directional_upper_bound.py") from exc

Number = Union[int, float]


class DirectionalDataError(ValueError):
    """Raised when saved data needed by the directional workflow is invalid."""


class DirectionalSolveError(RuntimeError):
    """Raised when the directional basis-tracking solve fails."""


def _base_dir(base_dir: Optional[Union[str, Path]]) -> Path:
    return Path(base_dir).expanduser().resolve() if base_dir is not None else Path(__file__).resolve().parent


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if hasattr(value, "tolist"):
        return _json_ready(value.tolist())
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return "Infinity" if value > 0 else "-Infinity"
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_json_ready(payload), f, indent=2, sort_keys=False)
        f.write("\n")


def _matrix_from_json(name: str, raw: Mapping[str, Any]) -> np.ndarray:
    missing = {key for key in ("rows", "cols", "data") if key not in raw}
    if missing:
        raise DirectionalDataError(f"Matrix {name!r} is missing keys: {sorted(missing)}")
    rows = int(raw["rows"])
    cols = int(raw["cols"])
    data = np.asarray(raw["data"], dtype=float)
    if data.size != rows * cols:
        raise DirectionalDataError(
            f"Matrix {name!r} declares shape ({rows}, {cols}) but contains {data.size} entries."
        )
    return data.reshape((rows, cols))


def _sorted_xi_items(data: Mapping[str, Any]) -> List[Tuple[str, np.ndarray]]:
    pattern = re.compile(r"^xi_(\d+)$")
    items: List[Tuple[int, str, np.ndarray]] = []
    for key, value in data.items():
        match = pattern.match(key)
        if match:
            items.append((int(match.group(1)), key, np.asarray(value, dtype=float)))
    items.sort(key=lambda item: item[0])
    if not items:
        raise DirectionalDataError("No scenario vectors found. Expected keys xi_1, xi_2, ...")
    expected = list(range(1, len(items) + 1))
    actual = [idx for idx, _, _ in items]
    if actual != expected:
        raise DirectionalDataError(f"Scenario xi keys must be consecutive from xi_1. Found {actual}.")
    return [(key, xi) for _, key, xi in items]


def _scenario_name(index_1_based: int) -> str:
    return f"scenario_{int(index_1_based)}"


def _scenario_index_from_name(scenario: Union[int, str]) -> int:
    if isinstance(scenario, int):
        if scenario <= 0:
            raise ValueError("Scenario numbers are 1-based and must be positive.")
        return scenario
    text = str(scenario).strip()
    if text.isdigit():
        return int(text)
    match = re.match(r"^(?:scenario|xi)_(\d+)$", text)
    if match:
        return int(match.group(1))
    raise ValueError(f"Could not parse scenario identifier {scenario!r}.")


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return float(np.asarray(a, dtype=float) @ np.asarray(b, dtype=float))


@dataclass(frozen=True)
class DirectionalData:
    instance_name: str
    instance_dir: Path
    A: np.ndarray
    b: np.ndarray
    c: np.ndarray
    d: np.ndarray
    T: np.ndarray
    W: np.ndarray
    H: np.ndarray
    probabilities: np.ndarray
    xi_keys: List[str]
    xi: List[np.ndarray]
    x_star: np.ndarray
    x_EV: np.ndarray
    first_stage_cost_star: float
    first_stage_cost_EV: float
    stochastic_results: Dict[str, Any]
    expectedvalue_results: Dict[str, Any]
    cost_gap_diagnostics: Dict[str, Any]

    @classmethod
    def load(cls, instance_name: str, base_dir: Optional[Union[str, Path]] = None) -> "DirectionalData":
        root = _base_dir(base_dir)
        instance_dir = root / instance_name
        instance_data = _read_json(instance_dir / "instance_data.json")
        stochastic_results = _read_json(instance_dir / "stochastic_results.json")
        expectedvalue_results = _read_json(instance_dir / "expectedvalue_results.json")
        cost_gap_diagnostics = _read_json(instance_dir / "cost_gap_diagnostics.json")

        required = ["A", "b", "c", "d", "T", "W", "H"]
        missing = [key for key in required if key not in instance_data]
        if missing:
            raise DirectionalDataError(f"instance_data.json is missing required keys: {missing}")

        A = _matrix_from_json("A", instance_data["A"])
        T = _matrix_from_json("T", instance_data["T"])
        W = _matrix_from_json("W", instance_data["W"])
        H = _matrix_from_json("H", instance_data["H"])
        b = np.asarray(instance_data["b"], dtype=float)
        c = np.asarray(instance_data["c"], dtype=float)
        d = np.asarray(instance_data["d"], dtype=float)
        xi_items = _sorted_xi_items(instance_data)
        xi_keys = [key for key, _ in xi_items]
        xi = [vec for _, vec in xi_items]

        if "p_s" in instance_data:
            probabilities = np.asarray(instance_data["p_s"], dtype=float)
        else:
            probabilities = np.asarray([instance_data[f"p_{i + 1}"][0] for i in range(len(xi))], dtype=float)

        x_star_raw = stochastic_results.get("x_star", stochastic_results.get("x_vector"))
        x_ev_raw = expectedvalue_results.get("x_EV", expectedvalue_results.get("x_vector"))
        if x_star_raw is None:
            raise DirectionalDataError("stochastic_results.json must contain x_star or x_vector.")
        if x_ev_raw is None:
            raise DirectionalDataError("expectedvalue_results.json must contain x_EV or x_vector.")
        x_star = np.asarray(x_star_raw, dtype=float)
        x_EV = np.asarray(x_ev_raw, dtype=float)

        first_stage_cost_star = float(stochastic_results.get("first_stage_cost", c @ x_star))
        first_stage_cost_EV = float(expectedvalue_results.get("EEV_first_stage_cost", expectedvalue_results.get("first_stage_cost", c @ x_EV)))

        obj = cls(
            instance_name=instance_name,
            instance_dir=instance_dir,
            A=A,
            b=b,
            c=c,
            d=d,
            T=T,
            W=W,
            H=H,
            probabilities=probabilities,
            xi_keys=xi_keys,
            xi=xi,
            x_star=x_star,
            x_EV=x_EV,
            first_stage_cost_star=first_stage_cost_star,
            first_stage_cost_EV=first_stage_cost_EV,
            stochastic_results=stochastic_results,
            expectedvalue_results=expectedvalue_results,
            cost_gap_diagnostics=cost_gap_diagnostics,
        )
        obj.validate()
        return obj

    @property
    def xi_dim(self) -> int:
        return int(self.H.shape[1])

    @property
    def num_recourse_constraints(self) -> int:
        return int(self.W.shape[0])

    @property
    def num_second_stage_vars(self) -> int:
        return int(self.W.shape[1])

    @property
    def first_stage_gap_EV_minus_star(self) -> float:
        return self.first_stage_cost_EV - self.first_stage_cost_star

    def validate(self) -> None:
        if self.A.ndim != 2 or self.T.ndim != 2 or self.W.ndim != 2 or self.H.ndim != 2:
            raise DirectionalDataError("A, T, W, and H must be matrices.")
        if self.A.shape[0] != self.b.size:
            raise DirectionalDataError(f"A has {self.A.shape[0]} rows but b has length {self.b.size}.")
        if self.A.shape[1] != self.c.size:
            raise DirectionalDataError(f"A has {self.A.shape[1]} columns but c has length {self.c.size}.")
        if self.T.shape[1] != self.c.size:
            raise DirectionalDataError(f"T has {self.T.shape[1]} columns but x has length {self.c.size}.")
        if self.W.shape[0] != self.T.shape[0]:
            raise DirectionalDataError("W and T must have the same number of rows.")
        if self.H.shape[0] != self.W.shape[0]:
            raise DirectionalDataError("H and W must have the same number of rows.")
        if self.W.shape[1] != self.d.size:
            raise DirectionalDataError(f"W has {self.W.shape[1]} columns but d has length {self.d.size}.")
        if self.x_star.size != self.c.size:
            raise DirectionalDataError("x_star length does not match c length.")
        if self.x_EV.size != self.c.size:
            raise DirectionalDataError("x_EV length does not match c length.")
        if len(self.xi) != self.probabilities.size:
            raise DirectionalDataError("Number of xi_s vectors does not match probabilities.")
        if abs(float(self.probabilities.sum()) - 1.0) > 1e-7:
            raise DirectionalDataError(f"Scenario probabilities must sum to 1. Found {self.probabilities.sum()}.")
        for key, vec in zip(self.xi_keys, self.xi):
            if vec.size != self.xi_dim:
                raise DirectionalDataError(f"{key} has length {vec.size}; expected {self.xi_dim}.")


@dataclass(frozen=True)
class BasisRecord:
    """A dual-feasible standard-form basis for Abar u = rhs, u >= 0."""

    indices: Tuple[int, ...]
    B_inv: np.ndarray
    reduced_costs: np.ndarray
    dual_prices: np.ndarray


@dataclass
class BasisState:
    label: str
    x: np.ndarray
    basis: BasisRecord
    rhs: np.ndarray
    u_B: np.ndarray
    q: np.ndarray
    tau: float
    slope: float
    value: float


class DirectionalUpperBoundSolver:
    """Compute the coordinate-ray upper bound using basis tracking."""

    def __init__(
        self,
        instance_name: str,
        base_dir: Optional[Union[str, Path]] = None,
        scenario: Optional[Union[int, str]] = None,
        feasibility_tol: float = 1e-8,
        optimality_tol: float = 1e-8,
        duplicate_tol: float = 1e-9,
        max_basis_candidates: int = 2_000_000,
        max_breakpoints_per_ray: int = 10_000,
    ):
        self.data = DirectionalData.load(instance_name, base_dir=base_dir)
        self.feasibility_tol = float(feasibility_tol)
        self.optimality_tol = float(optimality_tol)
        self.duplicate_tol = float(duplicate_tol)
        self.max_basis_candidates = int(max_basis_candidates)
        self.max_breakpoints_per_ray = int(max_breakpoints_per_ray)

        self.positive_gap_scenarios = self._find_positive_gap_scenarios()
        if scenario is None:
            if not self.positive_gap_scenarios:
                raise DirectionalDataError("No positive-gap scenarios were found in cost_gap_diagnostics.json.")
            self.selected_scenario_index_1_based = int(self.positive_gap_scenarios[0]["scenario_index_1_based"])
        else:
            self.selected_scenario_index_1_based = _scenario_index_from_name(scenario)
        if not (1 <= self.selected_scenario_index_1_based <= len(self.data.xi)):
            raise DirectionalDataError(
                f"Selected scenario index {self.selected_scenario_index_1_based} is outside available range."
            )
        self.selected_scenario_name = _scenario_name(self.selected_scenario_index_1_based)
        self.xi_hat = self.data.xi[self.selected_scenario_index_1_based - 1]
        self.initial_cost_gap = float(
            self.data.cost_gap_diagnostics["scenario_cost_gaps"][self.selected_scenario_name]["cost_gap"]
        )

        self.Abar = np.hstack([self.data.W, np.eye(self.data.num_recourse_constraints)])
        self.cbar = np.concatenate([self.data.d, np.zeros(self.data.num_recourse_constraints)])
        self.var_names = self._standard_variable_names()
        self.dual_feasible_bases = self._enumerate_dual_feasible_bases()

    def _standard_variable_names(self) -> List[str]:
        names = [f"y_{j + 1}" for j in range(self.data.num_second_stage_vars)]
        names.extend([f"s_{r + 1}" for r in range(self.data.num_recourse_constraints)])
        return names

    def _find_positive_gap_scenarios(self) -> List[Dict[str, Any]]:
        gaps = self.data.cost_gap_diagnostics.get("scenario_cost_gaps", {})
        out: List[Dict[str, Any]] = []
        for name, entry in gaps.items():
            idx = _scenario_index_from_name(name)
            gap = float(entry.get("cost_gap", 0.0))
            if gap > self.feasibility_tol:
                out.append(
                    {
                        "scenario": name,
                        "scenario_index_1_based": idx,
                        "xi_key": f"xi_{idx}",
                        "xi": self.data.xi[idx - 1].tolist(),
                        "cost_gap": gap,
                    }
                )
        out.sort(key=lambda item: int(item["scenario_index_1_based"]))
        return out

    def _basis_payload(self, basis: BasisRecord, state: Optional[BasisState] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "indices_0_based": list(basis.indices),
            "indices_1_based": [i + 1 for i in basis.indices],
            "variables": [self.var_names[i] for i in basis.indices],
        }
        if state is not None:
            payload.update(
                {
                    "basic_values": state.u_B.tolist(),
                    "rhs_direction_in_basis": state.q.tolist(),
                    "value": state.value,
                    "slope": state.slope,
                    "tau_to_next_breakpoint": state.tau,
                }
            )
        return payload

    def _enumerate_dual_feasible_bases(self) -> List[BasisRecord]:
        """Enumerate all nonsingular bases with nonnegative reduced costs.

        A basis is dual-feasible iff its reduced costs are nonnegative. Because
        the RHS is the only object that changes along a coordinate ray, every
        dual-feasible basis is optimal whenever it is primal feasible.
        """
        m_rows, n_cols = self.Abar.shape
        total_candidates = math.comb(n_cols, m_rows)
        if total_candidates > self.max_basis_candidates:
            raise DirectionalSolveError(
                f"Basis enumeration would require checking C({n_cols},{m_rows})={total_candidates} candidates, "
                f"which exceeds max_basis_candidates={self.max_basis_candidates}."
            )
        bases: List[BasisRecord] = []
        for basis in itertools.combinations(range(n_cols), m_rows):
            B = self.Abar[:, basis]
            if np.linalg.matrix_rank(B, tol=1e-10) < m_rows:
                continue
            try:
                B_inv = np.linalg.inv(B)
            except np.linalg.LinAlgError:  # pragma: no cover
                continue
            c_B = self.cbar[list(basis)]
            # lambda solves B^T lambda = c_B.
            dual_prices = np.linalg.solve(B.T, c_B)
            reduced_costs = self.cbar - self.Abar.T @ dual_prices
            if np.min(reduced_costs) < -self.optimality_tol:
                continue
            bases.append(
                BasisRecord(
                    indices=tuple(int(i) for i in basis),
                    B_inv=B_inv,
                    reduced_costs=reduced_costs,
                    dual_prices=dual_prices,
                )
            )
        bases.sort(key=lambda b: b.indices)
        if not bases:
            raise DirectionalSolveError("No dual-feasible bases were found for the standard-form recourse LP.")
        return bases

    def _rhs(self, x: np.ndarray, xi: np.ndarray) -> np.ndarray:
        return self.data.H @ xi - self.data.T @ x

    def _rhs0(self, x: np.ndarray) -> np.ndarray:
        return self._rhs(x, self.xi_hat)

    def _value_from_basis(self, basis: BasisRecord, rhs: np.ndarray) -> Tuple[np.ndarray, float]:
        u_B = basis.B_inv @ rhs
        value = float(self.cbar[list(basis.indices)] @ u_B)
        return u_B, value

    def _is_primal_feasible(self, basis: BasisRecord, rhs: np.ndarray) -> bool:
        u_B = basis.B_inv @ rhs
        return bool(np.min(u_B) >= -self.feasibility_tol)

    def _right_direction_feasible(self, basis: BasisRecord, rhs: np.ndarray, direction: np.ndarray) -> bool:
        u_B = basis.B_inv @ rhs
        q = basis.B_inv @ direction
        # Variables that are zero at the breakpoint must not immediately move negative.
        for val, deriv in zip(u_B, q):
            if val <= self.feasibility_tol and deriv < -self.feasibility_tol:
                return False
        return True

    def _choose_lex_basis(
        self,
        rhs: np.ndarray,
        direction: np.ndarray,
        previous_basis: Optional[BasisRecord] = None,
    ) -> Tuple[BasisRecord, str]:
        """Choose the lexicographically first optimal basis at a breakpoint.

        Preference is given to bases that are feasible at the breakpoint and
        also feasible for a right-neighborhood of the ray parameter. This avoids
        zero-length cycling at degenerate breakpoints. If no right-feasible basis
        exists, the lexicographically first primal-feasible dual-feasible basis
        is returned with a diagnostic status.
        """
        primal_feasible: List[BasisRecord] = []
        right_feasible: List[BasisRecord] = []
        for basis in self.dual_feasible_bases:
            if not self._is_primal_feasible(basis, rhs):
                continue
            primal_feasible.append(basis)
            if self._right_direction_feasible(basis, rhs, direction):
                right_feasible.append(basis)
        if right_feasible:
            return right_feasible[0], "lexicographically_first_right_feasible_optimal_basis"
        if primal_feasible:
            return primal_feasible[0], "lexicographically_first_optimal_basis_no_right_feasible_basis"
        raise DirectionalSolveError("No optimal primal-feasible basis exists at the requested RHS.")

    def _make_state(
        self,
        label: str,
        x: np.ndarray,
        basis: BasisRecord,
        rhs: np.ndarray,
        direction: np.ndarray,
    ) -> BasisState:
        u_B, value = self._value_from_basis(basis, rhs)
        if np.min(u_B) < -10 * self.feasibility_tol:
            raise DirectionalSolveError(f"Basis for {label} is not primal feasible; min basic value={np.min(u_B)}")
        q = basis.B_inv @ direction
        c_B = self.cbar[list(basis.indices)]
        slope = float(c_B @ q)
        tau = self._next_tau(u_B, q)
        return BasisState(label=label, x=x, basis=basis, rhs=rhs, u_B=u_B, q=q, tau=tau, slope=slope, value=value)

    def _next_tau(self, u_B: np.ndarray, q: np.ndarray) -> float:
        ratios: List[float] = []
        for val, deriv in zip(u_B, q):
            if deriv < -self.feasibility_tol:
                ratios.append(max(0.0, float(val / (-deriv))))
        if not ratios:
            return math.inf
        tau = min(ratios)
        # Very small negative/positive artifacts near degenerate breakpoints are treated as zero.
        if tau < self.feasibility_tol:
            return 0.0
        return float(tau)

    def _active_leaving_variables(self, state: BasisState, tau: float) -> List[Dict[str, Any]]:
        if math.isinf(tau):
            return []
        out: List[Dict[str, Any]] = []
        for row_pos, (val, deriv) in enumerate(zip(state.u_B, state.q)):
            if deriv < -self.feasibility_tol:
                ratio = max(0.0, float(val / (-deriv)))
                if abs(ratio - tau) <= 50 * self.feasibility_tol * max(1.0, abs(tau)):
                    idx = state.basis.indices[row_pos]
                    out.append(
                        {
                            "basis_row": row_pos,
                            "leaving_index_0_based": idx,
                            "leaving_index_1_based": idx + 1,
                            "leaving_variable": self.var_names[idx],
                            "ratio": ratio,
                            "basic_value": float(val),
                            "direction_value": float(deriv),
                        }
                    )
        out.sort(key=lambda item: (item["ratio"], item["leaving_index_0_based"]))
        return out

    def _refresh_state_at_t(
        self,
        old_state: BasisState,
        t_value: float,
        direction: np.ndarray,
    ) -> Tuple[BasisState, Dict[str, Any]]:
        rhs = self._rhs0(old_state.x) + t_value * direction
        new_basis, selection_status = self._choose_lex_basis(rhs, direction, previous_basis=old_state.basis)
        new_state = self._make_state(old_state.label, old_state.x, new_basis, rhs, direction)
        payload = {
            "x_label": old_state.label,
            "selection_rule": selection_status,
            "old_basis": self._basis_payload(old_state.basis),
            "new_basis": self._basis_payload(new_basis, new_state),
            "basis_changed": old_state.basis.indices != new_basis.indices,
        }
        return new_state, payload

    def solve_single_direction(self, coordinate_1_based: int, sign: int) -> Dict[str, Any]:
        if sign not in (-1, 1):
            raise ValueError("sign must be +1 or -1.")
        i0 = int(coordinate_1_based) - 1
        if not (0 <= i0 < self.data.xi_dim):
            raise ValueError(f"Coordinate {coordinate_1_based} is outside 1..{self.data.xi_dim}.")

        h_i = self.data.H[:, i0]
        direction = float(sign) * h_i
        t0 = 0.0
        first_gap = self.data.first_stage_gap_EV_minus_star

        rhs_ev_0 = self._rhs0(self.data.x_EV)
        rhs_star_0 = self._rhs0(self.data.x_star)
        basis_ev, status_ev = self._choose_lex_basis(rhs_ev_0, direction)
        basis_star, status_star = self._choose_lex_basis(rhs_star_0, direction)
        state_ev = self._make_state("EV", self.data.x_EV, basis_ev, rhs_ev_0, direction)
        state_star = self._make_state("star", self.data.x_star, basis_star, rhs_star_0, direction)

        intervals: List[Dict[str, Any]] = []
        basis_change_events: List[Dict[str, Any]] = []
        basis_history = {
            "EV": [self._basis_payload(state_ev.basis, state_ev)],
            "star": [self._basis_payload(state_star.basis, state_star)],
        }
        initial_phi = first_gap + state_ev.value - state_star.value
        if initial_phi <= self.feasibility_tol:
            return self._direction_result(
                coordinate_1_based,
                sign,
                gamma=0.0,
                termination_reason="initial_point_already_violates_or_ties",
                intervals=intervals,
                basis_change_events=basis_change_events,
                basis_history=basis_history,
                state_ev=state_ev,
                state_star=state_star,
                phi_at_end=initial_phi,
                t_end=0.0,
                initial_basis_selection={"EV": status_ev, "star": status_star},
            )

        for iteration in range(self.max_breakpoints_per_ray):
            phi0 = first_gap + state_ev.value - state_star.value
            beta = state_ev.slope - state_star.slope
            tau = min(state_ev.tau, state_star.tau)
            interval_end = math.inf if math.isinf(tau) else t0 + tau

            if beta < -self.optimality_tol:
                t_cross = t0 + phi0 / (-beta)
                if t_cross <= interval_end + 100 * self.feasibility_tol:
                    # Crossing occurs before the next basis change.
                    xi_cross = self.xi_hat.copy()
                    xi_cross[i0] += sign * max(0.0, t_cross)
                    q_ev_cross = state_ev.value + state_ev.slope * (t_cross - t0)
                    q_star_cross = state_star.value + state_star.slope * (t_cross - t0)
                    intervals.append(
                        {
                            "interval_index": len(intervals),
                            "start_t": t0,
                            "end_t": float(t_cross),
                            "length": float(t_cross - t0),
                            "EV_basis": self._basis_payload(state_ev.basis),
                            "star_basis": self._basis_payload(state_star.basis),
                            "phi_start": phi0,
                            "phi_end": first_gap + q_ev_cross - q_star_cross,
                            "beta": beta,
                            "event_at_end": ["violation_crossing"],
                        }
                    )
                    return self._direction_result(
                        coordinate_1_based,
                        sign,
                        gamma=float(max(0.0, t_cross)),
                        termination_reason="finite_crossing_found_before_next_breakpoint",
                        intervals=intervals,
                        basis_change_events=basis_change_events,
                        basis_history=basis_history,
                        state_ev=state_ev,
                        state_star=state_star,
                        phi_at_end=first_gap + q_ev_cross - q_star_cross,
                        t_end=float(max(0.0, t_cross)),
                        xi_at_end=xi_cross,
                        Q_EV_at_end=q_ev_cross,
                        Q_star_at_end=q_star_cross,
                        initial_basis_selection={"EV": status_ev, "star": status_star},
                    )

            if math.isinf(tau):
                return self._direction_result(
                    coordinate_1_based,
                    sign,
                    gamma=math.inf,
                    termination_reason="no_crossing_and_no_future_breakpoints",
                    intervals=intervals,
                    basis_change_events=basis_change_events,
                    basis_history=basis_history,
                    state_ev=state_ev,
                    state_star=state_star,
                    phi_at_end=phi0,
                    t_end=t0,
                    initial_basis_selection={"EV": status_ev, "star": status_star},
                )

            if tau < -self.feasibility_tol:
                raise DirectionalSolveError(f"Negative tau encountered on direction ({coordinate_1_based}, {sign}).")

            t1 = t0 + max(0.0, tau)
            phi1 = phi0 + beta * max(0.0, tau)
            event_labels: List[str] = []
            refresh_payloads: List[Dict[str, Any]] = []
            active_x_labels: List[str] = []

            ev_active = abs(state_ev.tau - tau) <= 100 * self.feasibility_tol * max(1.0, abs(tau))
            star_active = abs(state_star.tau - tau) <= 100 * self.feasibility_tol * max(1.0, abs(tau))
            if ev_active:
                active_x_labels.append("EV")
            if star_active:
                active_x_labels.append("star")

            intervals.append(
                {
                    "interval_index": len(intervals),
                    "start_t": t0,
                    "end_t": t1,
                    "length": t1 - t0,
                    "EV_basis": self._basis_payload(state_ev.basis),
                    "star_basis": self._basis_payload(state_star.basis),
                    "EV_tau": state_ev.tau,
                    "star_tau": state_star.tau,
                    "phi_start": phi0,
                    "phi_end": phi1,
                    "beta": beta,
                    "event_at_end": active_x_labels,
                    "leaving_variables_at_end": {
                        "EV": self._active_leaving_variables(state_ev, tau) if ev_active else [],
                        "star": self._active_leaving_variables(state_star, tau) if star_active else [],
                    },
                }
            )

            # Advance values to t1 under the old basis before changing bases.
            if ev_active:
                old_basis = state_ev.basis.indices
                state_ev, payload = self._refresh_state_at_t(state_ev, t1, direction)
                refresh_payloads.append(payload)
                if old_basis != state_ev.basis.indices:
                    basis_history["EV"].append(self._basis_payload(state_ev.basis, state_ev))
                    event_labels.append("EV")
            else:
                rhs_ev = self._rhs0(self.data.x_EV) + t1 * direction
                state_ev = self._make_state("EV", self.data.x_EV, state_ev.basis, rhs_ev, direction)

            if star_active:
                old_basis = state_star.basis.indices
                state_star, payload = self._refresh_state_at_t(state_star, t1, direction)
                refresh_payloads.append(payload)
                if old_basis != state_star.basis.indices:
                    basis_history["star"].append(self._basis_payload(state_star.basis, state_star))
                    event_labels.append("star")
            else:
                rhs_star = self._rhs0(self.data.x_star) + t1 * direction
                state_star = self._make_state("star", self.data.x_star, state_star.basis, rhs_star, direction)

            basis_change_events.append(
                {
                    "event_index": len(basis_change_events),
                    "t": t1,
                    "active_breakpoint_for": active_x_labels,
                    "basis_changed_for": event_labels,
                    "updates": refresh_payloads,
                }
            )

            # If no basis can move to the right, stop this ray.  This usually means the ray has reached
            # the boundary of the region where the recourse LP remains finite/feasible.
            if (
                (ev_active and state_ev.tau == 0.0 and not self._right_direction_feasible(state_ev.basis, state_ev.rhs, direction))
                or (star_active and state_star.tau == 0.0 and not self._right_direction_feasible(state_star.basis, state_star.rhs, direction))
            ):
                return self._direction_result(
                    coordinate_1_based,
                    sign,
                    gamma=math.inf,
                    termination_reason="stopped_at_degenerate_breakpoint_without_right_feasible_basis",
                    intervals=intervals,
                    basis_change_events=basis_change_events,
                    basis_history=basis_history,
                    state_ev=state_ev,
                    state_star=state_star,
                    phi_at_end=phi1,
                    t_end=t1,
                    initial_basis_selection={"EV": status_ev, "star": status_star},
                )

            if t1 <= t0 + self.feasibility_tol and not event_labels:
                raise DirectionalSolveError(
                    f"Zero-length breakpoint loop without a basis change on direction ({coordinate_1_based}, {sign})."
                )
            t0 = t1

        raise DirectionalSolveError(
            f"Exceeded max_breakpoints_per_ray={self.max_breakpoints_per_ray} on direction ({coordinate_1_based}, {sign})."
        )

    def _direction_result(
        self,
        coordinate_1_based: int,
        sign: int,
        gamma: float,
        termination_reason: str,
        intervals: List[Dict[str, Any]],
        basis_change_events: List[Dict[str, Any]],
        basis_history: Dict[str, List[Dict[str, Any]]],
        state_ev: BasisState,
        state_star: BasisState,
        phi_at_end: float,
        t_end: float,
        xi_at_end: Optional[np.ndarray] = None,
        Q_EV_at_end: Optional[float] = None,
        Q_star_at_end: Optional[float] = None,
        initial_basis_selection: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        if xi_at_end is None and math.isfinite(t_end):
            xi_at_end = self.xi_hat.copy()
            xi_at_end[int(coordinate_1_based) - 1] += sign * t_end
        if Q_EV_at_end is None:
            Q_EV_at_end = state_ev.value
        if Q_star_at_end is None:
            Q_star_at_end = state_star.value
        return {
            "coordinate_index_1_based": int(coordinate_1_based),
            "coordinate_index_0_based": int(coordinate_1_based) - 1,
            "sign": int(sign),
            "direction_label": f"xi_{coordinate_1_based}_{'plus' if sign > 0 else 'minus'}",
            "gamma_directional": gamma,
            "termination_reason": termination_reason,
            "xi_at_termination_or_crossing": None if xi_at_end is None else xi_at_end.tolist(),
            "phi_at_termination_or_crossing": phi_at_end,
            "Q_EV_at_termination_or_crossing": Q_EV_at_end,
            "Q_star_at_termination_or_crossing": Q_star_at_end,
            "cost_EV_at_termination_or_crossing": self.data.first_stage_cost_EV + float(Q_EV_at_end),
            "cost_star_at_termination_or_crossing": self.data.first_stage_cost_star + float(Q_star_at_end),
            "num_intervals": len(intervals),
            "num_basis_changes": {
                "EV": max(0, len(basis_history["EV"]) - 1),
                "star": max(0, len(basis_history["star"]) - 1),
            },
            "num_bases_explored": {
                "EV": len(basis_history["EV"]),
                "star": len(basis_history["star"]),
            },
            "initial_basis_selection": initial_basis_selection or {},
            "basis_history": basis_history,
            "intervals": intervals,
            "basis_change_events": basis_change_events,
        }

    def solve_all_directions(self) -> Dict[str, Any]:
        directions: List[Dict[str, Any]] = []
        for coord in range(1, self.data.xi_dim + 1):
            for sign in (1, -1):
                directions.append(self.solve_single_direction(coord, sign))

        finite = [d for d in directions if isinstance(d["gamma_directional"], (int, float)) and math.isfinite(float(d["gamma_directional"]))]
        if finite:
            best = min(finite, key=lambda d: float(d["gamma_directional"]))
            gamma_bar = float(best["gamma_directional"])
            best_direction = {
                "coordinate_index_1_based": best["coordinate_index_1_based"],
                "coordinate_index_0_based": best["coordinate_index_0_based"],
                "sign": best["sign"],
                "direction_label": best["direction_label"],
            }
        else:
            gamma_bar = math.inf
            best_direction = None

        result = {
            "instance_name": self.data.instance_name,
            "method": "coordinate_directional_upper_bound_basis_tracking",
            "selected_scenario": self.selected_scenario_name,
            "selected_scenario_index_1_based": self.selected_scenario_index_1_based,
            "xi_hat": self.xi_hat.tolist(),
            "initial_cost_gap_at_xi_hat": self.initial_cost_gap,
            "directional_upper_bound_gamma_bar": gamma_bar,
            "best_direction": best_direction,
            "num_coordinates": self.data.xi_dim,
            "num_directional_problems": 2 * self.data.xi_dim,
            "standard_form_dimensions": {
                "num_rows_m_r": self.data.num_recourse_constraints,
                "num_original_second_stage_variables_n_y": self.data.num_second_stage_vars,
                "num_standard_form_variables_n_y_plus_m_r": self.Abar.shape[1],
                "num_basis_candidates_combination": math.comb(self.Abar.shape[1], self.Abar.shape[0]),
                "num_dual_feasible_bases_enumerated": len(self.dual_feasible_bases),
            },
            "basis_tracking_rule": {
                "description": (
                    "At each breakpoint the implementation selects the lexicographically first "
                    "dual-feasible, primal-feasible basis that is also feasible for a right-neighborhood "
                    "of the ray parameter. This is the implemented anti-cycling/lexicographic rule."
                ),
                "basis_order": "increasing tuple of standard-form column indices, with y variables followed by slacks",
                "standard_form_variables": self.var_names,
            },
            "positive_gap_scenarios": self.positive_gap_scenarios,
            "directions": directions,
            "source_files": {
                "instance_data": "instance_data.json",
                "stochastic_results": "stochastic_results.json",
                "expectedvalue_results": "expectedvalue_results.json",
                "cost_gap_diagnostics": "cost_gap_diagnostics.json",
            },
            "notes": [
                "The coordinate-ray search gives an upper bound on the true l1 local dominance radius.",
                "No coordinate of xi is fixed here; each directional problem perturbs exactly one coordinate by construction.",
                "Directions with no finite violation are reported with gamma_directional = Infinity.",
            ],
        }
        _write_json(self.data.instance_dir / "directional_upper_bound_results.json", result)
        return result


def run_directional_upper_bound(
    instance_name: str,
    base_dir: Optional[Union[str, Path]] = None,
    scenario: Optional[Union[int, str]] = None,
    feasibility_tol: float = 1e-8,
    optimality_tol: float = 1e-8,
    max_basis_candidates: int = 2_000_000,
) -> Dict[str, Any]:
    solver = DirectionalUpperBoundSolver(
        instance_name=instance_name,
        base_dir=base_dir,
        scenario=scenario,
        feasibility_tol=feasibility_tol,
        optimality_tol=optimality_tol,
        max_basis_candidates=max_basis_candidates,
    )
    return solver.solve_all_directions()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Compute coordinate directional upper bound by basis tracking.")
    parser.add_argument("instance_name", help="Instance folder name, e.g., lands_instance")
    parser.add_argument("--base-dir", default=None, help="Folder containing instance folders. Defaults to this file's folder.")
    parser.add_argument("--scenario", default=None, help="Optional scenario, e.g., scenario_3, xi_3, or 3.")
    parser.add_argument("--feasibility-tol", type=float, default=1e-8)
    parser.add_argument("--optimality-tol", type=float, default=1e-8)
    parser.add_argument("--max-basis-candidates", type=int, default=2_000_000)
    args = parser.parse_args(argv)

    result = run_directional_upper_bound(
        args.instance_name,
        base_dir=args.base_dir,
        scenario=args.scenario,
        feasibility_tol=args.feasibility_tol,
        optimality_tol=args.optimality_tol,
        max_basis_candidates=args.max_basis_candidates,
    )
    print(json.dumps(_json_ready(result), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
