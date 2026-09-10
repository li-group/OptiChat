# local_dominance_radius_bounded.py
"""
Isolated local-dominance-radius workflow for the general-form TSSP instance.

This file intentionally does NOT import general_model.py. 

It assumes the first workflow has already been run and that the following files exist:

    input_data/{instance_name}.json
    or input_data/instance_data.json

and:

    output_data/stochastic_results.json
    output_data/expectedvalue_results.json
    output_data/cost_gap_diagnostics.json

    
It then:
    1. finds scenarios with positive cost gap Delta(xi_s) > 0;
    2. writes cost_gap_dominating_scenarios.json;
    3. selects either a user-requested scenario or the smallest positive-gap
       scenario by default;
    4. builds/solves the exact primal-dual single-level nearest-violation model
       for the local dominance radius:

       min ||xi - xi_hat||_1
       s.t. primal feasibility for x_EV and x_star,
            dual feasibility for both recourse LPs,
            strong duality for both recourse LPs,
            c^T x_EV + d^T y_EV <= c^T x_star + d^T y_star.

Default solve path: Pyomo + Gurobi with NonConvex=2.

For validation on machines without Gurobi/Pyomo, this file also contains an
optional small-instance dual-extreme-point enumeration method.  The enumeration
method is not the primary requested workflow; it is an exact LP reformulation
when the common dual recourse polyhedron has finitely enumerable extreme points.
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
    raise RuntimeError("numpy is required for local_dominance_radius.py") from exc

try:
    import pyomo.environ as pyo  # type: ignore
except Exception:  # pragma: no cover - Pyomo may be absent in lightweight test envs
    pyo = None

try:
    from scipy.optimize import linprog  # type: ignore
except Exception:  # pragma: no cover
    linprog = None


Number = Union[int, float]


class LocalDominanceDataError(ValueError):
    """Raised when the saved data needed by the local-dominance workflow is invalid."""


class LocalDominanceSolveError(RuntimeError):
    """Raised when the local-dominance-radius optimization fails."""


def _base_dir(base_dir: Optional[Union[str, Path]]) -> Path:
    return Path(base_dir).expanduser().resolve() if base_dir is not None else Path(__file__).resolve().parent

def _looks_like_instance_dir(instance_dir: Path, instance_name: str) -> bool:
    return any(
        path.exists()
        for path in [
            instance_dir / "input_data" / f"{instance_name}.json",
            instance_dir / "input_data" / "instance_data.json",
            instance_dir / f"{instance_name}.json",
            instance_dir / "instance_data.json",
        ]
    )


def _resolve_instance_dir(instance_name: str, base_dir: Optional[Union[str, Path]] = None) -> Path:
    root = _base_dir(base_dir)

    candidates = [
        root / instance_name,
        root / "lands_generated_instances" / instance_name,
    ]

    for instance_dir in candidates:
        if _looks_like_instance_dir(instance_dir, instance_name):
            return instance_dir

    looked = "\n".join(f"  - {p}" for p in candidates)
    raise FileNotFoundError(
        f"Could not find instance folder for {instance_name!r}. Looked in:\n{looked}"
    )


def _first_existing(description: str, candidates: Sequence[Path]) -> Path:
    for path in candidates:
        if path.exists():
            return path

    looked = "\n".join(f"  - {p}" for p in candidates)
    raise FileNotFoundError(f"Could not find {description}. Looked in:\n{looked}")


def _instance_data_path(instance_dir: Path, instance_name: str) -> Path:
    return _first_existing(
        "instance data JSON",
        [
            instance_dir / "input_data" / f"{instance_name}.json",
            instance_dir / "input_data" / "instance_data.json",
            instance_dir / f"{instance_name}.json",
            instance_dir / "instance_data.json",
        ],
    )


def _output_json_path(instance_dir: Path, filename: str) -> Path:
    return _first_existing(
        filename,
        [
            instance_dir / "output_data" / filename,
            instance_dir / filename,
        ],
    )


def _output_dir_for_instance(instance_dir: Path) -> Path:
    if (instance_dir / "input_data").exists() or (instance_dir / "output_data").exists():
        return instance_dir / "output_data"
    return instance_dir


def _write_instance_output_json(instance_dir: Path, filename: str, payload: Mapping[str, Any]) -> None:
    _write_json(_output_dir_for_instance(instance_dir) / filename, payload)


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_json_ready(payload), f, indent=2, sort_keys=False)
        f.write("\n")


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
        return str(value)
    return value


def _matrix_from_json(name: str, raw: Mapping[str, Any]) -> np.ndarray:
    missing = {key for key in ("rows", "cols", "data") if key not in raw}
    if missing:
        raise LocalDominanceDataError(f"Matrix {name!r} is missing keys: {sorted(missing)}")
    rows = int(raw["rows"])
    cols = int(raw["cols"])
    data = np.asarray(raw["data"], dtype=float)
    if data.size != rows * cols:
        raise LocalDominanceDataError(
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
        raise LocalDominanceDataError("No scenario vectors found. Expected keys xi_1, xi_2, ...")
    expected = list(range(1, len(items) + 1))
    actual = [idx for idx, _, _ in items]
    if actual != expected:
        raise LocalDominanceDataError(f"Scenario xi keys must be consecutive from xi_1. Found {actual}.")
    return [(key, xi) for _, key, xi in items]


def _scenario_name(index_1_based: int) -> str:
    return f"scenario_{int(index_1_based)}"


def _scenario_index_from_name(scenario: Union[int, str]) -> int:
    """Return a 1-based scenario index from 3, '3', 'scenario_3', or 'xi_3'."""
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
    raise ValueError(f"Could not parse scenario identifier {scenario!r}. Use 3, '3', 'scenario_3', or 'xi_3'.")


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return float(np.asarray(a, dtype=float) @ np.asarray(b, dtype=float))


def _max_abs_residual(values: Iterable[float]) -> float:
    vals = [abs(float(v)) for v in values]
    return max(vals) if vals else 0.0



@dataclass(frozen=True)
class LocalDominanceData:
    """All data needed by the isolated local-dominance-radius workflow."""

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
    stochastic_results: Dict[str, Any]
    expectedvalue_results: Dict[str, Any]
    cost_gap_diagnostics: Dict[str, Any]

    @classmethod
    def load(cls, instance_name: str, base_dir: Optional[Union[str, Path]] = None) -> "LocalDominanceData":
        # root = _base_dir(base_dir)
        # instance_dir = root / instance_name
        # instance_data = _read_json(instance_dir / "instance_data.json")
        # stochastic_results = _read_json(instance_dir / "stochastic_results.json")
        # expectedvalue_results = _read_json(instance_dir / "expectedvalue_results.json")
        # cost_gap_diagnostics = _read_json(instance_dir / "cost_gap_diagnostics.json")
        instance_dir = _resolve_instance_dir(instance_name, base_dir)
        instance_data = _read_json(_instance_data_path(instance_dir, instance_name))
        stochastic_results = _read_json(_output_json_path(instance_dir, "stochastic_results.json"))
        expectedvalue_results = _read_json(_output_json_path(instance_dir, "expectedvalue_results.json"))
        cost_gap_diagnostics = _read_json(_output_json_path(instance_dir, "cost_gap_diagnostics.json"))

        required = ["A", "b", "c", "d", "T", "W", "H"]
        missing = [key for key in required if key not in instance_data]
        if missing:
            raise LocalDominanceDataError(f"instance_data.json is missing required keys: {missing}")

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
            raise LocalDominanceDataError("stochastic_results.json must contain 'x_star' or 'x_vector'.")
        if x_ev_raw is None:
            raise LocalDominanceDataError("expectedvalue_results.json must contain 'x_EV' or 'x_vector'.")
        x_star = np.asarray(x_star_raw, dtype=float)
        x_EV = np.asarray(x_ev_raw, dtype=float)

        data = cls(
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
            stochastic_results=stochastic_results,
            expectedvalue_results=expectedvalue_results,
            cost_gap_diagnostics=cost_gap_diagnostics,
        )
        data.check_dimensions()
        return data

    @property
    def num_first_stage_vars(self) -> int:
        return int(self.c.size)

    @property
    def num_second_stage_vars(self) -> int:
        return int(self.d.size)

    @property
    def num_recourse_constraints(self) -> int:
        return int(self.W.shape[0])

    @property
    def xi_dim(self) -> int:
        return int(self.H.shape[1])

    @property
    def num_scenarios(self) -> int:
        return len(self.xi)

    @property
    def scenario_names(self) -> List[str]:
        return [_scenario_name(i + 1) for i in range(self.num_scenarios)]

    @property
    def first_stage_cost_star(self) -> float:
        return _dot(self.c, self.x_star)

    @property
    def first_stage_cost_EV(self) -> float:
        return _dot(self.c, self.x_EV)
    



    def check_dimensions(self, probability_tol: float = 1e-8) -> None:
        errors: List[str] = []
        if self.A.shape[0] != self.b.size:
            errors.append(f"A has {self.A.shape[0]} rows but b has length {self.b.size}.")
        if self.A.shape[1] != self.c.size:
            errors.append(f"A has {self.A.shape[1]} columns but c has length {self.c.size}.")
        if self.T.shape[1] != self.c.size:
            errors.append(f"T has {self.T.shape[1]} columns but c has length {self.c.size}.")
        if self.W.shape[1] != self.d.size:
            errors.append(f"W has {self.W.shape[1]} columns but d has length {self.d.size}.")
        if self.T.shape[0] != self.W.shape[0]:
            errors.append(f"T has {self.T.shape[0]} rows but W has {self.W.shape[0]} rows.")
        if self.H.shape[0] != self.W.shape[0]:
            errors.append(f"H has {self.H.shape[0]} rows but W has {self.W.shape[0]} rows.")
        for key, vec in zip(self.xi_keys, self.xi):
            if vec.size != self.H.shape[1]:
                errors.append(f"{key} has length {vec.size} but H has {self.H.shape[1]} columns.")
        if self.x_star.size != self.c.size:
            errors.append(f"x_star has length {self.x_star.size} but c has length {self.c.size}.")
        if self.x_EV.size != self.c.size:
            errors.append(f"x_EV has length {self.x_EV.size} but c has length {self.c.size}.")
        if self.probabilities.size != self.num_scenarios:
            errors.append(f"There are {self.num_scenarios} scenarios but {self.probabilities.size} probabilities.")
        prob_sum = float(self.probabilities.sum())
        if abs(prob_sum - 1.0) > probability_tol:
            errors.append(f"Scenario probabilities must sum to 1. Found {prob_sum}.")
        if errors:
            raise LocalDominanceDataError("Local-dominance data checks failed:\n- " + "\n- ".join(errors))

    def xi_for_scenario(self, scenario: Union[int, str]) -> Tuple[int, str, np.ndarray]:
        idx_1 = _scenario_index_from_name(scenario)
        if idx_1 < 1 or idx_1 > self.num_scenarios:
            raise ValueError(f"Scenario {idx_1} is outside available range 1..{self.num_scenarios}.")
        return idx_1, _scenario_name(idx_1), np.asarray(self.xi[idx_1 - 1], dtype=float)


def find_cost_gap_dominating_scenarios(
    instance_name: str,
    base_dir: Optional[Union[str, Path]] = None,
    gap_tol: float = 1e-8,
) -> Dict[str, Any]:
    """Find and save all scenarios with positive cost gap Delta(xi_s) > gap_tol."""
    data = LocalDominanceData.load(instance_name, base_dir=base_dir)
    gap_payload = data.cost_gap_diagnostics.get("scenario_cost_gaps", {})
    if not isinstance(gap_payload, Mapping):
        raise LocalDominanceDataError("cost_gap_diagnostics.json must contain a mapping named 'scenario_cost_gaps'.")

    positive_records: List[Dict[str, Any]] = []
    nonpositive_records: List[Dict[str, Any]] = []
    missing_records: List[str] = []

    for idx_1, name in enumerate(data.scenario_names, start=1):
        if name not in gap_payload:
            missing_records.append(name)
            continue
        record = gap_payload[name]
        cost_gap = float(record["cost_gap"])
        entry = {
            "scenario_index_1_based": idx_1,
            "scenario_name": name,
            "xi_key": data.xi_keys[idx_1 - 1],
            "xi_hat": data.xi[idx_1 - 1].tolist(),
            "cost_gap": cost_gap,
            "weighted_cost_gap": float(record.get("weighted_cost_gap", float("nan"))),
            "probability": float(record.get("probability", data.probabilities[idx_1 - 1])),
        }
        if cost_gap > gap_tol:
            positive_records.append(entry)
        else:
            nonpositive_records.append(entry)

    positive_records.sort(key=lambda item: int(item["scenario_index_1_based"]))
    default = positive_records[0] if positive_records else None

    result = {
        "instance_name": instance_name,
        "definition": {
            "dominating_scenario": "A scenario with Delta(xi_s) > 0, i.e., x_star is cheaper than x_EV at that realized scenario.",
            "gap_tolerance": gap_tol,
        },
        "num_positive_gap_scenarios": len(positive_records),
        "cost_gap_dominating_scenarios": positive_records,
        "nonpositive_gap_scenarios": nonpositive_records,
        "missing_scenarios_in_cost_gap_diagnostics": missing_records,
        "default_selected_scenario": None if default is None else default["scenario_name"],
        "default_selected_scenario_index_1_based": None if default is None else default["scenario_index_1_based"],
        "source_file": "cost_gap_diagnostics.json",
    }
    # _write_json(data.instance_dir / "cost_gap_dominating_scenarios.json", result)
    _write_instance_output_json(data.instance_dir, "cost_gap_dominating_scenarios.json", result)
    return result


class LocalDominanceRadiusProblem:
    """Build and solve the exact local-dominance-radius model for one scenario."""

    def __init__(
        self,
        instance_name: str,
        base_dir: Optional[Union[str, Path]] = None,
        scenario: Optional[Union[int, str]] = None,
        gap_tol: float = 1e-8,
    ):
        self.data = LocalDominanceData.load(instance_name, base_dir=base_dir)
        self.dominating = find_cost_gap_dominating_scenarios(instance_name, base_dir=base_dir, gap_tol=gap_tol)
        self.selected_scenario_index_1_based = self._select_scenario(scenario, gap_tol=gap_tol)
        _, self.selected_scenario_name, self.xi_hat = self.data.xi_for_scenario(self.selected_scenario_index_1_based)
        self.initial_cost_gap = float(
            self.data.cost_gap_diagnostics["scenario_cost_gaps"][self.selected_scenario_name]["cost_gap"]
        )

    def _evaluate_cost_gap_at_xi(
        self,
        xi_vec: Sequence[float],
        solver: str = "gurobi",
        backend: str = "auto",
        tee: bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluate

            Delta(xi)
            =
            c^T x_EV + Q(x_EV, xi)
            -
            c^T x_star - Q(x_star, xi).

        A negative or zero value gives an incumbent nearest-violation certificate.
        """
        data = self.data
        xi = np.asarray(xi_vec, dtype=float)

        rec_EV = self._evaluate_recourse_at_xi(
            data.x_EV,
            xi,
            solver=solver,
            backend=backend,
            tee=tee,
        )

        rec_star = self._evaluate_recourse_at_xi(
            data.x_star,
            xi,
            solver=solver,
            backend=backend,
            tee=tee,
        )

        q_EV = float(rec_EV["objective"])
        q_star = float(rec_star["objective"])

        cost_EV = data.first_stage_cost_EV + q_EV
        cost_star = data.first_stage_cost_star + q_star
        gap = cost_EV - cost_star

        return {
            "xi": xi.tolist(),
            "cost_gap": gap,
            "Q_x_EV_xi": q_EV,
            "Q_x_star_xi": q_star,
            "cost_with_x_EV": cost_EV,
            "cost_with_x_star": cost_star,
            "recourse_EV": rec_EV,
            "recourse_star": rec_star,
        }
    
    def _evaluate_recourse_at_xi(
        self,
        x_vec: Sequence[float],
        xi_vec: Sequence[float],
        solver: str = "gurobi",
        backend: str = "auto",
        tee: bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluate

            Q(x, xi) = min d^T y
                    s.t. W y <= H xi - T x,
                        y >= 0.

        First tries Pyomo/Gurobi when available, then falls back to scipy.linprog
        when backend='auto'.
        """
        data = self.data
        x = np.asarray(x_vec, dtype=float)
        xi = np.asarray(xi_vec, dtype=float)
        rhs = data.H @ xi - data.T @ x

        def solve_with_pyomo() -> Dict[str, Any]:
            if pyo is None:
                raise LocalDominanceSolveError("Pyomo is not installed.")

            m = pyo.ConcreteModel(name=f"recourse_eval_{data.instance_name}")
            m.J = pyo.RangeSet(0, data.num_second_stage_vars - 1)
            m.R = pyo.RangeSet(0, data.num_recourse_constraints - 1)

            m.y = pyo.Var(m.J, domain=pyo.NonNegativeReals)

            m.obj = pyo.Objective(
                expr=sum(float(data.d[j]) * m.y[j] for j in m.J),
                sense=pyo.minimize,
            )

            m.recourse = pyo.Constraint(
                m.R,
                rule=lambda model, r: sum(float(data.W[r, j]) * model.y[j] for j in model.J)
                <= float(rhs[r]),
            )

            opt = pyo.SolverFactory(solver)
            if opt is None or not opt.available(False):
                raise LocalDominanceSolveError(f"Pyomo solver {solver!r} is not available.")

            results = opt.solve(m, tee=tee)
            status = str(results.solver.status)
            termination = str(results.solver.termination_condition)

            ok_terms = {"optimal", "globallyOptimal", "locallyOptimal"}
            if termination not in ok_terms:
                raise LocalDominanceSolveError(
                    f"Recourse LP did not solve optimally. status={status}, termination={termination}"
                )

            y = np.asarray([float(pyo.value(m.y[j])) for j in m.J], dtype=float)

            return {
                "objective": _dot(data.d, y),
                "y": y.tolist(),
                "backend": "pyomo",
                "solver": solver,
                "solver_status": status,
                "termination_condition": termination,
            }

        def solve_with_linprog() -> Dict[str, Any]:
            if linprog is None:
                raise LocalDominanceSolveError("scipy.optimize.linprog is not installed.")

            res = linprog(
                c=data.d,
                A_ub=data.W,
                b_ub=rhs,
                bounds=[(0.0, None)] * data.num_second_stage_vars,
                method="highs",
            )

            if not res.success:
                raise LocalDominanceSolveError(
                    f"Recourse LP failed with scipy.linprog status {res.status}: {res.message}"
                )

            y = np.asarray(res.x, dtype=float)

            return {
                "objective": float(res.fun),
                "y": y.tolist(),
                "backend": "scipy",
                "solver": "highs",
                "solver_status": int(res.status),
                "termination_condition": "optimal",
            }

        backend = backend.lower()
        if backend not in {"auto", "pyomo", "scipy"}:
            raise ValueError("backend must be one of: 'auto', 'pyomo', 'scipy'.")

        if backend == "pyomo":
            return solve_with_pyomo()

        if backend == "scipy":
            return solve_with_linprog()

        try:
            return solve_with_pyomo()
        except Exception as pyomo_exc:
            if linprog is None:
                raise pyomo_exc
            return solve_with_linprog()


    def compute_compact_bilevel_xi_bound_diagnostics(
        self,
        solver: str = "gurobi",
        backend: str = "auto",
        tee: bool = False,
        gap_tol: float = 1e-8,
        max_radius: Optional[float] = None,
        radius_multiplier: float = 10.0,
        growth_factor: float = 2.0,
        bisection_iterations: int = 40,
        num_random_directions: int = 20,
        random_seed: int = 12345,
    ) -> Dict[str, Any]:
        """
        Find a feasible incumbent xi_bar with Delta(xi_bar) <= 0.

        If found, define

            U = ||xi_bar - xi_hat||_1.

        Then add the safe bounds

            xi_hat[k] - U <= xi[k] <= xi_hat[k] + U

        and the safe objective cutoff

            sum_k t[k] <= U.

        This does not remove any global optimum because the nearest-violation
        optimum is no larger than any feasible incumbent radius.
        """
        data = self.data
        xi_hat = np.asarray(self.xi_hat, dtype=float)
        xi_dim = data.xi_dim

        evaluated_candidates: List[Dict[str, Any]] = []
        failed_candidates: List[Dict[str, Any]] = []

        best: Optional[Dict[str, Any]] = None

        def l1_radius(xi_vec: np.ndarray) -> float:
            return float(np.abs(xi_vec - xi_hat).sum())

        def maybe_update_best(candidate: Dict[str, Any], source: str) -> None:
            nonlocal best

            xi_candidate = np.asarray(candidate["xi"], dtype=float)
            U = l1_radius(xi_candidate)
            gap = float(candidate["cost_gap"])

            enriched = {
                **candidate,
                "source": source,
                "incumbent_radius_U": U,
            }

            if gap <= gap_tol:
                if best is None or U < float(best["incumbent_radius_U"]):
                    best = enriched

        # ---------------------------------------------------------------
        # Step 1: use already-saved realized-scenario cost gaps.
        # ---------------------------------------------------------------
        gap_payload = data.cost_gap_diagnostics.get("scenario_cost_gaps", {})

        if isinstance(gap_payload, Mapping):
            for idx, scenario_name in enumerate(data.scenario_names):
                if scenario_name not in gap_payload:
                    continue

                saved_gap = float(gap_payload[scenario_name]["cost_gap"])
                xi_s = np.asarray(data.xi[idx], dtype=float)
                U_s = l1_radius(xi_s)

                record = {
                    "source": "saved_realized_scenario",
                    "scenario_name": scenario_name,
                    "scenario_index_1_based": idx + 1,
                    "xi": xi_s.tolist(),
                    "cost_gap": saved_gap,
                    "incumbent_radius_U": U_s,
                }

                evaluated_candidates.append(record)

                if saved_gap <= gap_tol:
                    maybe_update_best(record, source="saved_realized_scenario")

        if best is not None:
            U = float(best["incumbent_radius_U"])
            xi_incumbent = np.asarray(best["xi"], dtype=float)

            xi_lower = xi_hat - U
            xi_upper = xi_hat + U

            componentwise_bounds = [
                {
                    "xi_index_0_based": k,
                    "xi_hat": float(xi_hat[k]),
                    "lower_bound": float(xi_lower[k]),
                    "upper_bound": float(xi_upper[k]),
                    "incumbent_xi_value": float(xi_incumbent[k]),
                }
                for k in range(xi_dim)
            ]

            diagnostics = {
                "instance_name": data.instance_name,
                "selected_scenario": self.selected_scenario_name,
                "selected_scenario_index_1_based": self.selected_scenario_index_1_based,
                "definition": {
                    "purpose": "Find incumbent nearest-violation certificate to bound xi.",
                    "cost_gap": "Delta(xi) = c^T x_EV + Q(x_EV, xi) - c^T x_star - Q(x_star, xi).",
                    "incumbent_condition": "Delta(xi_bar) <= gap_tol.",
                    "incumbent_radius": "U = ||xi_bar - xi_hat||_1.",
                    "safe_xi_bounds": "xi_hat[k] - U <= xi[k] <= xi_hat[k] + U.",
                    "safe_radius_cutoff": "sum_k t[k] <= U.",
                },
                "parameters": {
                    "solver": solver,
                    "backend": backend,
                    "gap_tol": gap_tol,
                    "early_stop_from_saved_scenario": True,
                },
                "dimensions": {
                    "xi_dimension": int(xi_dim),
                    "H_shape": list(data.H.shape),
                    "W_shape": list(data.W.shape),
                    "T_shape": list(data.T.shape),
                },
                "incumbent": {
                    "found": True,
                    "source": best.get("source"),
                    "incumbent_radius_U": U,
                    "xi_hat": xi_hat.tolist(),
                    "incumbent_xi": xi_incumbent.tolist(),
                    "incumbent_cost_gap": float(best["cost_gap"]),
                    "Q_x_EV_xi": best.get("Q_x_EV_xi"),
                    "Q_x_star_xi": best.get("Q_x_star_xi"),
                    "cost_with_x_EV": best.get("cost_with_x_EV"),
                    "cost_with_x_star": best.get("cost_with_x_star"),
                },
                "xi_bounds": {
                    "available": True,
                    "radius_U": U,
                    "lower_bounds": xi_lower.tolist(),
                    "upper_bounds": xi_upper.tolist(),
                    "componentwise_bounds": componentwise_bounds,
                },
                "search_summary": {
                    "num_directions_tried": 0,
                    "num_candidate_evaluations": len(evaluated_candidates),
                    "num_failed_candidate_evaluations": len(failed_candidates),
                    "scenario_spread_l1_from_xi_hat": None,
                    "stopped_after_saved_scenario_incumbent": True,
                },
                "evaluated_candidates": evaluated_candidates,
                "failed_candidates": failed_candidates,
                "notes": [
                    "A saved realized scenario already gave Delta(xi) <= gap_tol.",
                    "Ray search was skipped to avoid unnecessary recourse solves.",
                    "The resulting xi bounds and sum(t) <= U cutoff are safe.",
                ],
            }

            _write_instance_output_json(
                data.instance_dir,
                "compact_bilevel_xi_bound_diagnostics.json",
                diagnostics,
            )

            return diagnostics

        # ---------------------------------------------------------------
        # Step 2: build deterministic search directions.
        # All directions are normalized to have L1 norm 1, so alpha is
        # directly the L1 radius.
        # ---------------------------------------------------------------
        directions: List[Tuple[str, np.ndarray]] = []

        for idx, xi_s in enumerate(data.xi):
            direction = np.asarray(xi_s, dtype=float) - xi_hat
            norm = float(np.abs(direction).sum())
            if norm > 1e-12:
                directions.append((f"toward_scenario_{idx + 1}", direction / norm))

        xi_bar_expected = sum(float(p) * np.asarray(xi, dtype=float) for p, xi in zip(data.probabilities, data.xi))
        direction = xi_bar_expected - xi_hat
        norm = float(np.abs(direction).sum())
        if norm > 1e-12:
            directions.append(("toward_expected_xi", direction / norm))

        for k in range(xi_dim):
            e = np.zeros(xi_dim, dtype=float)
            e[k] = 1.0
            directions.append((f"positive_coordinate_{k}", e.copy()))
            directions.append((f"negative_coordinate_{k}", -e.copy()))

        ones = np.ones(xi_dim, dtype=float)
        directions.append(("positive_all_ones", ones / float(np.abs(ones).sum())))
        directions.append(("negative_all_ones", -ones / float(np.abs(ones).sum())))

        rng = np.random.default_rng(random_seed)

        for q in range(num_random_directions):
            direction = rng.normal(size=xi_dim)
            norm = float(np.abs(direction).sum())
            if norm > 1e-12:
                directions.append((f"random_normal_{q}", direction / norm))

        # Remove near-duplicate directions.
        unique_directions: List[Tuple[str, np.ndarray]] = []
        seen: List[np.ndarray] = []

        for name, direction in directions:
            if any(float(np.linalg.norm(direction - old, ord=np.inf)) <= 1e-10 for old in seen):
                continue
            seen.append(direction)
            unique_directions.append((name, direction))

        scenario_spread = max(
            [float(np.abs(np.asarray(xi_s, dtype=float) - xi_hat).sum()) for xi_s in data.xi] + [1.0]
        )

        if max_radius is None:
            max_radius_eff = radius_multiplier * max(1.0, scenario_spread)
        else:
            max_radius_eff = float(max_radius)

        if best is not None:
            max_radius_eff = min(max_radius_eff, float(best["incumbent_radius_U"]))

        initial_step = max(1e-6, min(1.0, scenario_spread / 100.0))

        def try_evaluate_on_ray(direction_name: str, direction: np.ndarray, alpha: float) -> Optional[Dict[str, Any]]:
            xi_trial = xi_hat + float(alpha) * direction

            try:
                eval_result = self._evaluate_cost_gap_at_xi(
                    xi_trial,
                    solver=solver,
                    backend=backend,
                    tee=tee,
                )

                record = {
                    "source": "ray_search",
                    "direction_name": direction_name,
                    "alpha_l1_radius": float(alpha),
                    "xi": xi_trial.tolist(),
                    "cost_gap": float(eval_result["cost_gap"]),
                    "Q_x_EV_xi": float(eval_result["Q_x_EV_xi"]),
                    "Q_x_star_xi": float(eval_result["Q_x_star_xi"]),
                    "cost_with_x_EV": float(eval_result["cost_with_x_EV"]),
                    "cost_with_x_star": float(eval_result["cost_with_x_star"]),
                }

                evaluated_candidates.append(record)
                return record

            except Exception as exc:
                failed_candidates.append(
                    {
                        "source": "ray_search",
                        "direction_name": direction_name,
                        "alpha_l1_radius": float(alpha),
                        "message": f"{type(exc).__name__}: {exc}",
                    }
                )
                return None

        # ---------------------------------------------------------------
        # Step 3: ray expansion plus bisection.
        # ---------------------------------------------------------------
        for direction_name, direction in unique_directions:
            current_best_radius = (
                float(best["incumbent_radius_U"])
                if best is not None
                else max_radius_eff
            )

            if current_best_radius <= 1e-12:
                break

            alpha_low = 0.0
            gap_low = float(self.initial_cost_gap)

            alpha = min(initial_step, current_best_radius)

            found_high: Optional[Dict[str, Any]] = None

            while alpha <= current_best_radius + 1e-12:
                record = try_evaluate_on_ray(direction_name, direction, alpha)

                if record is not None:
                    gap = float(record["cost_gap"])

                    if gap <= gap_tol:
                        found_high = record
                        break

                    alpha_low = alpha
                    gap_low = gap

                next_alpha = alpha * growth_factor
                if next_alpha <= alpha + 1e-12:
                    break
                alpha = min(next_alpha, current_best_radius)

                if abs(alpha - current_best_radius) <= 1e-12 and next_alpha > current_best_radius:
                    # Make sure the current incumbent radius is tested once.
                    continue

            if found_high is None:
                continue

            alpha_high = float(found_high["alpha_l1_radius"])

            # Bisection between alpha_low, alpha_high.
            best_on_ray = found_high

            for _ in range(bisection_iterations):
                if alpha_high - alpha_low <= 1e-8:
                    break

                alpha_mid = 0.5 * (alpha_low + alpha_high)
                mid_record = try_evaluate_on_ray(direction_name, direction, alpha_mid)

                if mid_record is None:
                    # If recourse fails at midpoint, keep the known feasible high.
                    alpha_low = alpha_mid
                    continue

                gap_mid = float(mid_record["cost_gap"])

                if gap_mid <= gap_tol:
                    alpha_high = alpha_mid
                    best_on_ray = mid_record
                else:
                    alpha_low = alpha_mid
                    gap_low = gap_mid

            maybe_update_best(best_on_ray, source=f"ray_search:{direction_name}")

            if best is not None:
                max_radius_eff = min(max_radius_eff, float(best["incumbent_radius_U"]))

        incumbent_found = best is not None

        if incumbent_found:
            U = float(best["incumbent_radius_U"])
            xi_incumbent = np.asarray(best["xi"], dtype=float)

            xi_lower = xi_hat - U
            xi_upper = xi_hat + U

            componentwise_bounds = [
                {
                    "xi_index_0_based": k,
                    "xi_hat": float(xi_hat[k]),
                    "lower_bound": float(xi_lower[k]),
                    "upper_bound": float(xi_upper[k]),
                    "incumbent_xi_value": float(xi_incumbent[k]),
                }
                for k in range(xi_dim)
            ]
        else:
            U = None
            xi_incumbent = None
            xi_lower = None
            xi_upper = None
            componentwise_bounds = [
                {
                    "xi_index_0_based": k,
                    "xi_hat": float(xi_hat[k]),
                    "lower_bound": None,
                    "upper_bound": None,
                    "incumbent_xi_value": None,
                }
                for k in range(xi_dim)
            ]

        diagnostics = {
            "instance_name": data.instance_name,
            "selected_scenario": self.selected_scenario_name,
            "selected_scenario_index_1_based": self.selected_scenario_index_1_based,

            "definition": {
                "purpose": "Find incumbent nearest-violation certificate to bound xi.",
                "cost_gap": "Delta(xi) = c^T x_EV + Q(x_EV, xi) - c^T x_star - Q(x_star, xi).",
                "incumbent_condition": "Delta(xi_bar) <= gap_tol.",
                "incumbent_radius": "U = ||xi_bar - xi_hat||_1.",
                "safe_xi_bounds": "xi_hat[k] - U <= xi[k] <= xi_hat[k] + U.",
                "safe_radius_cutoff": "sum_k t[k] <= U.",
            },

            "parameters": {
                "solver": solver,
                "backend": backend,
                "gap_tol": gap_tol,
                "max_radius_input": max_radius,
                "max_radius_effective": max_radius_eff,
                "radius_multiplier": radius_multiplier,
                "growth_factor": growth_factor,
                "bisection_iterations": bisection_iterations,
                "num_random_directions": num_random_directions,
                "random_seed": random_seed,
            },

            "dimensions": {
                "xi_dimension": int(xi_dim),
                "H_shape": list(data.H.shape),
                "W_shape": list(data.W.shape),
                "T_shape": list(data.T.shape),
            },

            "incumbent": {
                "found": incumbent_found,
                "source": None if best is None else best.get("source"),
                "incumbent_radius_U": U,
                "xi_hat": xi_hat.tolist(),
                "incumbent_xi": None if xi_incumbent is None else xi_incumbent.tolist(),
                "incumbent_cost_gap": None if best is None else float(best["cost_gap"]),
                "Q_x_EV_xi": None if best is None or "Q_x_EV_xi" not in best else float(best["Q_x_EV_xi"]),
                "Q_x_star_xi": None if best is None or "Q_x_star_xi" not in best else float(best["Q_x_star_xi"]),
                "cost_with_x_EV": None if best is None or "cost_with_x_EV" not in best else float(best["cost_with_x_EV"]),
                "cost_with_x_star": None if best is None or "cost_with_x_star" not in best else float(best["cost_with_x_star"]),
            },

            "xi_bounds": {
                "available": incumbent_found,
                "radius_U": U,
                "lower_bounds": None if xi_lower is None else xi_lower.tolist(),
                "upper_bounds": None if xi_upper is None else xi_upper.tolist(),
                "componentwise_bounds": componentwise_bounds,
            },

            "search_summary": {
                "num_directions_tried": len(unique_directions),
                "num_candidate_evaluations": len(evaluated_candidates),
                "num_failed_candidate_evaluations": len(failed_candidates),
                "scenario_spread_l1_from_xi_hat": scenario_spread,
            },

            "evaluated_candidates": evaluated_candidates,
            "failed_candidates": failed_candidates,

            "notes": [
                "If no incumbent is found, xi remains unbounded in the Pyomo model.",
                "The heuristic is not a proof that no violating xi exists.",
                "Any found incumbent is enough to safely add xi bounds and sum(t) <= U.",
            ],
        }

        _write_instance_output_json(
            data.instance_dir,
            "compact_bilevel_xi_bound_diagnostics.json",
            diagnostics,
        )

        return diagnostics



    def _load_or_compute_compact_xi_bound_diagnostics(
        self,
        solver: str = "gurobi",
        backend: str = "auto",
        tee: bool = False,
    ) -> Dict[str, Any]:
        """
        Load output_data/compact_bilevel_xi_bound_diagnostics.json if present;
        otherwise compute it.
        """
        diagnostics_path = _output_dir_for_instance(self.data.instance_dir) / "compact_bilevel_xi_bound_diagnostics.json"

        if diagnostics_path.exists():
            return _read_json(diagnostics_path)

        # return self.compute_compact_bilevel_xi_bound_diagnostics(
        #     solver=solver,
        #     backend=backend,
        #     tee=tee,
        # )
        return self.compute_compact_bilevel_xi_bound_diagnostics(
            solver=solver,
            backend=backend,
            tee=False,
            bisection_iterations=10,
            num_random_directions=0,
            radius_multiplier=2.0,
        )

    def _select_scenario(self, scenario: Optional[Union[int, str]], gap_tol: float) -> int:
        positive = self.dominating.get("cost_gap_dominating_scenarios", [])
        positive_indices = {int(item["scenario_index_1_based"]) for item in positive}
        if scenario is None:
            if not positive:
                raise LocalDominanceDataError("No positive-gap scenarios were found; local dominance radius is undefined.")
            return int(positive[0]["scenario_index_1_based"])

        chosen = _scenario_index_from_name(scenario)
        if chosen < 1 or chosen > self.data.num_scenarios:
            raise ValueError(f"Scenario {chosen} is outside available range 1..{self.data.num_scenarios}.")
        if chosen not in positive_indices:
            name = _scenario_name(chosen)
            gap = float(self.data.cost_gap_diagnostics["scenario_cost_gaps"][name]["cost_gap"])
            raise ValueError(
                f"Selected {name} has cost gap {gap}, which is not > {gap_tol}. "
                "Choose one of the scenarios in cost_gap_dominating_scenarios.json."
            )
        return chosen

    def _initial_y(self, which: str) -> Optional[np.ndarray]:
        """Return saved recourse y at xi_hat if available, for model initialization only."""
        name = self.selected_scenario_name
        if which == "EV":
            payload = self.data.expectedvalue_results.get("recourse_evaluations", {}).get(name, {})
        elif which == "star":
            payload = self.data.stochastic_results.get("recourse_evaluations", {}).get(name, {})
        else:  # pragma: no cover
            raise ValueError(which)
        y = payload.get("y")
        if y is None:
            return None
        arr = np.asarray(y, dtype=float)
        return arr if arr.size == self.data.num_second_stage_vars else None
    

    def compute_compact_bilevel_h_trans_pi_diagnostics(
        self,
        feasibility_tol: float = 1e-9,
        solver: str = "gurobi",
        tee: bool = False,
    ) -> Dict[str, Any]:
        """
        Compute componentwise lower/upper bounds on

            g_star = H^T pi_star

        over the star-side dual feasible region

            W^T pi_star <= d,
            pi_star <= 0.

        For each k, solve two LPs:

            lower_bound[k] = min_pi H[:, k]^T pi
            upper_bound[k] = max_pi H[:, k]^T pi

        The function first tries Pyomo with the requested solver, usually Gurobi.
        If that attempt is unavailable or fails to solve optimally, it falls back
        to scipy.optimize.linprog with HiGHS.

        Bounds that are not found are stored as None, which becomes JSON null.
        """
        data = self.data

        pi_dim = data.num_recourse_constraints
        g_dim = data.xi_dim

        def solve_bound_with_pyomo(
            h_col: np.ndarray,
            sense_name: str,
        ) -> Dict[str, Any]:
            """
            sense_name is either 'min' or 'max'.
            """
            if pyo is None:
                return {
                    "attempted": False,
                    "success": False,
                    "backend": "pyomo",
                    "solver": solver,
                    "value": None,
                    "solver_status": None,
                    "termination_condition": None,
                    "message": "Pyomo is not installed.",
                }

            try:
                opt = pyo.SolverFactory(solver)
                if opt is None or not opt.available(False):
                    return {
                        "attempted": True,
                        "success": False,
                        "backend": "pyomo",
                        "solver": solver,
                        "value": None,
                        "solver_status": None,
                        "termination_condition": None,
                        "message": f"Pyomo solver {solver!r} is not available.",
                    }

                m = pyo.ConcreteModel(
                    name=f"g_bound_{sense_name}_{data.instance_name}_{self.selected_scenario_name}"
                )

                m.R = pyo.RangeSet(0, pi_dim - 1)
                m.J = pyo.RangeSet(0, data.num_second_stage_vars - 1)

                m.pi_star = pyo.Var(
                    m.R,
                    domain=pyo.Reals,
                    bounds=(None, 0.0),
                    initialize=0.0,
                )

                m.dual_star = pyo.Constraint(
                    m.J,
                    rule=lambda model, j: sum(
                        float(data.W[r, j]) * model.pi_star[r]
                        for r in model.R
                    ) <= float(data.d[j]),
                )

                obj_expr = sum(float(h_col[r]) * m.pi_star[r] for r in m.R)

                m.obj = pyo.Objective(
                    expr=obj_expr,
                    sense=pyo.minimize if sense_name == "min" else pyo.maximize,
                )

                results = opt.solve(m, tee=tee)

                status = str(results.solver.status)
                termination = str(results.solver.termination_condition)

                ok_terms = {"optimal", "globallyOptimal", "locallyOptimal"}

                if termination not in ok_terms:
                    return {
                        "attempted": True,
                        "success": False,
                        "backend": "pyomo",
                        "solver": solver,
                        "value": None,
                        "solver_status": status,
                        "termination_condition": termination,
                        "message": (
                            f"Pyomo/Gurobi did not return an optimal bound. "
                            f"status={status}, termination={termination}"
                        ),
                    }

                value = float(pyo.value(m.obj))

                return {
                    "attempted": True,
                    "success": True,
                    "backend": "pyomo",
                    "solver": solver,
                    "value": value,
                    "solver_status": status,
                    "termination_condition": termination,
                    "message": "optimal",
                }

            except Exception as exc:
                return {
                    "attempted": True,
                    "success": False,
                    "backend": "pyomo",
                    "solver": solver,
                    "value": None,
                    "solver_status": None,
                    "termination_condition": None,
                    "message": f"{type(exc).__name__}: {exc}",
                }

        def solve_bound_with_linprog(
            h_col: np.ndarray,
            sense_name: str,
        ) -> Dict[str, Any]:
            """
            sense_name is either 'min' or 'max'.
            """
            if linprog is None:
                return {
                    "attempted": False,
                    "success": False,
                    "backend": "scipy",
                    "solver": "highs",
                    "value": None,
                    "solver_status": None,
                    "termination_condition": None,
                    "message": "scipy.optimize.linprog is not installed.",
                }

            A_ub = np.vstack(
                [
                    data.W.T,
                    np.eye(pi_dim),
                ]
            )

            b_ub = np.concatenate(
                [
                    data.d,
                    np.zeros(pi_dim),
                ]
            )

            bounds = [(None, None)] * pi_dim

            c_vec = h_col if sense_name == "min" else -h_col

            try:
                res = linprog(
                    c=c_vec,
                    A_ub=A_ub,
                    b_ub=b_ub,
                    bounds=bounds,
                    method="highs",
                )

                if res.success:
                    value = float(res.fun if sense_name == "min" else -res.fun)

                    return {
                        "attempted": True,
                        "success": True,
                        "backend": "scipy",
                        "solver": "highs",
                        "value": value,
                        "solver_status": int(res.status),
                        "termination_condition": "optimal",
                        "message": str(res.message),
                    }

                return {
                    "attempted": True,
                    "success": False,
                    "backend": "scipy",
                    "solver": "highs",
                    "value": None,
                    "solver_status": int(res.status),
                    "termination_condition": "not_optimal",
                    "message": str(res.message),
                }

            except Exception as exc:
                return {
                    "attempted": True,
                    "success": False,
                    "backend": "scipy",
                    "solver": "highs",
                    "value": None,
                    "solver_status": None,
                    "termination_condition": None,
                    "message": f"{type(exc).__name__}: {exc}",
                }

        def solve_one_bound(
            h_col: np.ndarray,
            sense_name: str,
        ) -> Dict[str, Any]:
            """
            Try Pyomo/Gurobi first. If it does not produce an optimal bound,
            fall back to scipy.linprog.
            """
            gurobi_attempt = solve_bound_with_pyomo(h_col, sense_name)

            if gurobi_attempt["success"]:
                return {
                    "success": True,
                    "value": gurobi_attempt["value"],
                    "final_backend": gurobi_attempt["backend"],
                    "final_solver": gurobi_attempt["solver"],
                    "final_status": "optimal",
                    "gurobi_attempt": gurobi_attempt,
                    "linprog_fallback_attempt": None,
                }

            linprog_attempt = solve_bound_with_linprog(h_col, sense_name)

            if linprog_attempt["success"]:
                return {
                    "success": True,
                    "value": linprog_attempt["value"],
                    "final_backend": linprog_attempt["backend"],
                    "final_solver": linprog_attempt["solver"],
                    "final_status": "optimal_after_fallback",
                    "gurobi_attempt": gurobi_attempt,
                    "linprog_fallback_attempt": linprog_attempt,
                }

            return {
                "success": False,
                "value": None,
                "final_backend": None,
                "final_solver": None,
                "final_status": "not_found",
                "gurobi_attempt": gurobi_attempt,
                "linprog_fallback_attempt": linprog_attempt,
            }

        per_dimension: List[Dict[str, Any]] = []

        lb_found_indices: List[int] = []
        ub_found_indices: List[int] = []
        lb_missing_indices: List[int] = []
        ub_missing_indices: List[int] = []

        lb_gurobi_indices: List[int] = []
        ub_gurobi_indices: List[int] = []
        lb_fallback_indices: List[int] = []
        ub_fallback_indices: List[int] = []

        for k in range(g_dim):
            h_col = np.asarray(data.H[:, k], dtype=float)

            lb_result = solve_one_bound(h_col, "min")
            ub_result = solve_one_bound(h_col, "max")

            if lb_result["success"]:
                lb_value: Optional[float] = float(lb_result["value"])
                lb_status = str(lb_result["final_status"])
                lb_found_indices.append(k)

                if lb_result["final_backend"] == "pyomo":
                    lb_gurobi_indices.append(k)
                elif lb_result["final_backend"] == "scipy":
                    lb_fallback_indices.append(k)
            else:
                lb_value = None
                lb_status = "not_found"
                lb_missing_indices.append(k)

            if ub_result["success"]:
                ub_value: Optional[float] = float(ub_result["value"])
                ub_status = str(ub_result["final_status"])
                ub_found_indices.append(k)

                if ub_result["final_backend"] == "pyomo":
                    ub_gurobi_indices.append(k)
                elif ub_result["final_backend"] == "scipy":
                    ub_fallback_indices.append(k)
            else:
                ub_value = None
                ub_status = "not_found"
                ub_missing_indices.append(k)

            per_dimension.append(
                {
                    "g_index_0_based": k,
                    "H_column_index_0_based": k,
                    "objective_vector_H_col": h_col.tolist(),

                    "lower_bound": lb_value,
                    "lower_bound_status": lb_status,
                    "lower_bound_final_backend": lb_result["final_backend"],
                    "lower_bound_final_solver": lb_result["final_solver"],
                    "lower_bound_gurobi_attempt": lb_result["gurobi_attempt"],
                    "lower_bound_linprog_fallback_attempt": lb_result["linprog_fallback_attempt"],

                    "upper_bound": ub_value,
                    "upper_bound_status": ub_status,
                    "upper_bound_final_backend": ub_result["final_backend"],
                    "upper_bound_final_solver": ub_result["final_solver"],
                    "upper_bound_gurobi_attempt": ub_result["gurobi_attempt"],
                    "upper_bound_linprog_fallback_attempt": ub_result["linprog_fallback_attempt"],
                }
            )

        diagnostics = {
            "instance_name": data.instance_name,
            "selected_scenario": self.selected_scenario_name,
            "selected_scenario_index_1_based": self.selected_scenario_index_1_based,

            "definition": {
                "dual_region": "Pi = {pi_star : W^T pi_star <= d, pi_star <= 0}",
                "projected_dual_sensitivity": "g_star = H^T pi_star",
                "bound_computation": (
                    "For each k, lower_bound[k] = min H[:, k]^T pi_star over Pi, "
                    "and upper_bound[k] = max H[:, k]^T pi_star over Pi."
                ),
                "solver_order": (
                    f"First try Pyomo with solver={solver!r}; if unavailable or not optimal, "
                    "fall back to scipy.optimize.linprog with HiGHS."
                ),
                "missing_bound_encoding": "None in Python, null in JSON.",
            },

            "dimensions": {
                "H_shape": list(data.H.shape),
                "pi_star_dimension": int(pi_dim),
                "H_transpose_shape": list(data.H.T.shape),
                "g_star_dimension": int(g_dim),
                "W_shape": list(data.W.shape),
                "d_dimension": int(data.d.size),
                "dual_constraint_matrix_W_transpose_shape": list(data.W.T.shape),
            },

            "summary": {
                "num_g_dimensions": int(g_dim),

                "num_lower_bounds_found": len(lb_found_indices),
                "num_upper_bounds_found": len(ub_found_indices),

                "lower_bound_found_indices_0_based": lb_found_indices,
                "upper_bound_found_indices_0_based": ub_found_indices,

                "lower_bound_missing_indices_0_based": lb_missing_indices,
                "upper_bound_missing_indices_0_based": ub_missing_indices,

                "lower_bounds_found_by_gurobi_indices_0_based": lb_gurobi_indices,
                "upper_bounds_found_by_gurobi_indices_0_based": ub_gurobi_indices,

                "lower_bounds_found_by_linprog_fallback_indices_0_based": lb_fallback_indices,
                "upper_bounds_found_by_linprog_fallback_indices_0_based": ub_fallback_indices,

                "all_lower_bounds_found": len(lb_missing_indices) == 0,
                "all_upper_bounds_found": len(ub_missing_indices) == 0,
            },

            "bounds_by_g_dimension": per_dimension,

            "notes": [
                "These are bounds on g_star = H^T pi_star, not direct bounds on each pi_star component.",
                "The compact one-sided model uses products g_star[k] * xi[k].",
                "If a lower or upper bound is missing, the corresponding Pyomo Var bound should be left as None.",
                "A missing bound usually means the LP was infeasible, unbounded, or neither solver returned an optimal solution.",
            ],
        }

        _write_instance_output_json(
            data.instance_dir,
            "compact_bilevel_h_trans_pie_diagnostics.json",
            diagnostics,
        )

        return diagnostics
    
    def _load_or_compute_compact_g_bounds_diagnostics(self) -> Dict[str, Any]:
        """
        Load output_data/compact_bilevel_h_trans_pie_diagnostics.json if present;
        otherwise compute it.
        """
        data = self.data

        diagnostics_path = _output_dir_for_instance(data.instance_dir) / "compact_bilevel_h_trans_pie_diagnostics.json"

        if diagnostics_path.exists():
            return _read_json(diagnostics_path)

        return self.compute_compact_bilevel_h_trans_pi_diagnostics()

    def build_compact_bilevel_pyomo_model_with_g_bounds(
        self,
        recourse_solver: str = "gurobi",
        recourse_backend: str = "auto",
        tee: bool = False,
    ) -> Any:
        """
        Build the compact one-sided nearest-violation model with bounds on

            g_star = H^T pi_star.

        The bilinear certificate is written as

            c^T x_EV + d^T y_EV
            <=
            c^T x_star + g_star^T xi - pi_star^T T x_star.

        The bilinear products are now g_star[k] * xi[k].
        Bounds on g_star are read from:

            output_data/compact_bilevel_h_trans_pie_diagnostics.json

        Missing lower/upper bounds are handled as None.
        """
        if pyo is None:
            raise LocalDominanceSolveError("Pyomo is not installed. Install pyomo and use solver='gurobi'.")

        data = self.data
        diagnostics = self._load_or_compute_compact_g_bounds_diagnostics()

        xi_diagnostics = self._load_or_compute_compact_xi_bound_diagnostics(
            solver=recourse_solver,
            backend=recourse_backend,
            tee=tee,
        )

        bounds_by_dim = diagnostics.get("bounds_by_g_dimension", [])
        if len(bounds_by_dim) != data.xi_dim:
            raise LocalDominanceDataError(
                f"g-bound diagnostics dimension mismatch: expected {data.xi_dim}, "
                f"found {len(bounds_by_dim)}."
            )

        g_bounds: Dict[int, Tuple[Optional[float], Optional[float]]] = {}
        for item in bounds_by_dim:
            k = int(item["g_index_0_based"])
            lb_raw = item.get("lower_bound")
            ub_raw = item.get("upper_bound")

            lb = None if lb_raw is None else float(lb_raw)
            ub = None if ub_raw is None else float(ub_raw)

            if lb is not None and ub is not None and lb > ub + 1e-8:
                raise LocalDominanceDataError(
                    f"Invalid g_star bounds for index {k}: lower_bound={lb}, upper_bound={ub}."
                )

            g_bounds[k] = (lb, ub)

        m = pyo.ConcreteModel(
            name=f"compact_local_dominance_radius_g_bounded_{data.instance_name}_{self.selected_scenario_name}"
        )

        m.K = pyo.RangeSet(0, data.xi_dim - 1)
        m.J = pyo.RangeSet(0, data.num_second_stage_vars - 1)
        m.R = pyo.RangeSet(0, data.num_recourse_constraints - 1)
        m.I = pyo.RangeSet(0, data.num_first_stage_vars - 1)

        # m.xi = pyo.Var(
        #     m.K,
        #     domain=pyo.Reals,
        #     initialize=lambda model, k: float(self.xi_hat[k]),
        # )

        xi_bounds: Dict[int, Tuple[Optional[float], Optional[float]]] = {}

        xi_bounds_payload = xi_diagnostics.get("xi_bounds", {})
        xi_bounds_available = bool(xi_bounds_payload.get("available", False))

        incumbent_radius_U: Optional[float] = None

        if xi_bounds_available:
            incumbent_radius_U = float(xi_bounds_payload["radius_U"])

            componentwise_xi_bounds = xi_bounds_payload.get("componentwise_bounds", [])

            if len(componentwise_xi_bounds) != data.xi_dim:
                raise LocalDominanceDataError(
                    f"xi-bound diagnostics dimension mismatch: expected {data.xi_dim}, "
                    f"found {len(componentwise_xi_bounds)}."
                )

            for item in componentwise_xi_bounds:
                k = int(item["xi_index_0_based"])
                lb_raw = item.get("lower_bound")
                ub_raw = item.get("upper_bound")

                lb = None if lb_raw is None else float(lb_raw)
                ub = None if ub_raw is None else float(ub_raw)

                if lb is not None and ub is not None and lb > ub + 1e-8:
                    raise LocalDominanceDataError(
                        f"Invalid xi bounds for index {k}: lower_bound={lb}, upper_bound={ub}."
                    )

                xi_bounds[k] = (lb, ub)
        else:
            for k in range(data.xi_dim):
                xi_bounds[k] = (None, None)

        def xi_bound_rule(model: Any, k: int) -> Tuple[Optional[float], Optional[float]]:
            return xi_bounds[int(k)]

        m.xi = pyo.Var(
            m.K,
            domain=pyo.Reals,
            bounds=xi_bound_rule,
            initialize=lambda model, k: float(self.xi_hat[k]),
        )

        m.t = pyo.Var(
            m.K,
            domain=pyo.NonNegativeReals,
            initialize=0.0,
        )

        m.y_EV = pyo.Var(
            m.J,
            domain=pyo.NonNegativeReals,
            initialize=0.0,
        )

        m.pi_star = pyo.Var(
            m.R,
            domain=pyo.Reals,
            bounds=(None, 0.0),
            initialize=0.0,
        )

        def g_bound_rule(model: Any, k: int) -> Tuple[Optional[float], Optional[float]]:
            return g_bounds[int(k)]

        m.g_star = pyo.Var(
            m.K,
            domain=pyo.Reals,
            bounds=g_bound_rule,
            initialize=0.0,
        )

        y_ev_init = self._initial_y("EV")
        if y_ev_init is not None:
            for j in range(data.num_second_stage_vars):
                m.y_EV[j].set_value(max(0.0, float(y_ev_init[j])))

        def Hxi(model: Any, r: int) -> Any:
            return sum(float(data.H[r, k]) * model.xi[k] for k in model.K)

        def Tx(x_vec: np.ndarray, r: int) -> float:
            return float(
                sum(
                    float(data.T[r, i]) * float(x_vec[i])
                    for i in range(data.num_first_stage_vars)
                )
            )

        def rhs_EV(model: Any, r: int) -> Any:
            return Hxi(model, r) - Tx(data.x_EV, r)

        def rhs_star(model: Any, r: int) -> Any:
            return Hxi(model, r) - Tx(data.x_star, r)

        def Tx_star_component(r: int) -> float:
            return Tx(data.x_star, r)

        m.objective = pyo.Objective(
            expr=sum(m.t[k] for k in m.K),
            sense=pyo.minimize,
        )

        m.abs_upper = pyo.Constraint(
            m.K,
            rule=lambda model, k: model.xi[k] - float(self.xi_hat[k]) <= model.t[k],
        )

        m.abs_lower = pyo.Constraint(
            m.K,
            rule=lambda model, k: float(self.xi_hat[k]) - model.xi[k] <= model.t[k],
        )

        if incumbent_radius_U is not None:
            m.incumbent_radius_bound = pyo.Constraint(
                expr=sum(m.t[k] for k in m.K) <= float(incumbent_radius_U)
        )

        # EV primal feasibility: W y_EV <= H xi - T x_EV
        m.primal_EV = pyo.Constraint(
            m.R,
            rule=lambda model, r: sum(float(data.W[r, j]) * model.y_EV[j] for j in model.J)
            <= rhs_EV(model, r),
        )

        # Star dual feasibility: W^T pi_star <= d
        m.dual_star = pyo.Constraint(
            m.J,
            rule=lambda model, j: sum(float(data.W[r, j]) * model.pi_star[r] for r in model.R)
            <= float(data.d[j]),
        )

        # Link projected dual sensitivity:
        #
        #     g_star = H^T pi_star
        m.g_star_link = pyo.Constraint(
            m.K,
            rule=lambda model, k: model.g_star[k]
            == sum(float(data.H[r, k]) * model.pi_star[r] for r in model.R),
        )

        # Bounded one-sided violation certificate:
        #
        #     c^T x_EV + d^T y_EV
        #     <=
        #     c^T x_star + g_star^T xi - pi_star^T T x_star
        #
        # Since rhs_star = H xi - T x_star,
        # pi_star^T rhs_star = pi_star^T H xi - pi_star^T T x_star
        #                    = g_star^T xi - pi_star^T T x_star.
        m.violation_cert = pyo.Constraint(
            expr=float(data.first_stage_cost_EV)
            + sum(float(data.d[j]) * m.y_EV[j] for j in m.J)
            <= float(data.first_stage_cost_star)
            + sum(m.g_star[k] * m.xi[k] for k in m.K)
            - sum(m.pi_star[r] * Tx_star_component(r) for r in m.R)
        )

        m._local_rhs_EV = rhs_EV  # type: ignore[attr-defined]
        m._local_rhs_star = rhs_star  # type: ignore[attr-defined]
        # m._compact_g_bounds_diagnostics = diagnostics  # type: ignore[attr-defined]
        m._compact_g_bounds_diagnostics = diagnostics  # type: ignore[attr-defined]
        m._compact_xi_bounds_diagnostics = xi_diagnostics  # type: ignore[attr-defined]

        return m

    def solve_compact_pyomo_with_g_bounds(
        self,
        solver: str = "gurobi",
        tee: bool = False,
        solver_options: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Solve the compact one-sided Pyomo model with bounded projected dual
        sensitivity variables g_star = H^T pi_star.
        """
        if pyo is None:
            raise LocalDominanceSolveError("Pyomo is not installed.")

        # model = self.build_compact_bilevel_pyomo_model_with_g_bounds()
        model = self.build_compact_bilevel_pyomo_model_with_g_bounds(
            recourse_solver=solver,
            recourse_backend="auto",
            tee=tee,
        )

        opt = pyo.SolverFactory(solver)
        if opt is None or not opt.available(False):
            raise LocalDominanceSolveError(f"Solver {solver!r} is not available to Pyomo.")

        if solver.lower().startswith("gurobi"):
            opt.options["NonConvex"] = 2

        if solver_options:
            for key, value in solver_options.items():
                opt.options[str(key)] = value
         
        tee=True
        print(f"Started solving compact bilevel Pyomo model with solver={solver!r} and options={solver_options}...")
        results = opt.solve(model, tee=tee)

        status = str(results.solver.status)
        termination = str(results.solver.termination_condition)

        ok_terms = {"optimal", "globallyOptimal", "locallyOptimal"}
        if termination not in ok_terms:
            raise LocalDominanceSolveError(
                f"Bounded compact local dominance radius solve did not terminate optimally. "
                f"status={status}, termination={termination}"
            )

        return self._extract_compact_pyomo_solution(
            model,
            solver=solver,
            solver_status=status,
            termination_condition=termination,
        )
    
    def _extract_compact_pyomo_solution(
        self,
        model: Any,
        solver: str,
        solver_status: str,
        termination_condition: str,
    ) -> Dict[str, Any]:
        """
        Extract solution from either compact one-sided formulation:

            Unbounded/direct compact model:
                xi, t, y_EV, pi_star

            Bounded projected-sensitivity compact model:
                xi, t, y_EV, pi_star, g_star

        If g_star exists, the certificate is interpreted as:

            c^T x_EV + d^T y_EV
            <=
            c^T x_star + g_star^T xi - pi_star^T T x_star

        Otherwise, the certificate is interpreted as:

            c^T x_EV + d^T y_EV
            <=
            c^T x_star + pi_star^T(H xi - T x_star)
        """
        data = self.data

        xi = np.asarray([float(pyo.value(model.xi[k])) for k in model.K], dtype=float)
        t = np.asarray([float(pyo.value(model.t[k])) for k in model.K], dtype=float)
        y_EV = np.asarray([float(pyo.value(model.y_EV[j])) for j in model.J], dtype=float)
        pi_star = np.asarray([float(pyo.value(model.pi_star[r])) for r in model.R], dtype=float)

        has_g_star = hasattr(model, "g_star")
        if has_g_star:
            g_star = np.asarray([float(pyo.value(model.g_star[k])) for k in model.K], dtype=float)
        else:
            g_star = None

        gamma = float(sum(t))
        l1_distance = float(np.abs(xi - self.xi_hat).sum())

        rhs_EV = data.H @ xi - data.T @ data.x_EV
        rhs_star = data.H @ xi - data.T @ data.x_star
        Tx_star = data.T @ data.x_star

        q_EV_primal_candidate = _dot(data.d, y_EV)

        # Direct pi form:
        #     pi_star^T(H xi - T x_star)
        q_star_dual_certificate_from_pi = float(pi_star @ rhs_star)

        # Projected g form, when available:
        #     g_star^T xi - pi_star^T T x_star
        if g_star is not None:
            q_star_dual_certificate_from_g = float(g_star @ xi - pi_star @ Tx_star)
            q_star_dual_certificate_used = q_star_dual_certificate_from_g
        else:
            q_star_dual_certificate_from_g = None
            q_star_dual_certificate_used = q_star_dual_certificate_from_pi

        cost_EV_primal_candidate = data.first_stage_cost_EV + q_EV_primal_candidate
        cost_star_dual_certificate = data.first_stage_cost_star + q_star_dual_certificate_used

        certified_gap = cost_EV_primal_candidate - cost_star_dual_certificate

        primal_EV_resid = data.W @ y_EV - rhs_EV
        dual_star_resid = data.W.T @ pi_star - data.d
        pi_star_upper_resid = pi_star

        if g_star is not None:
            g_star_link_resid = g_star - data.H.T @ pi_star
            q_star_g_pi_abs_difference = abs(
                q_star_dual_certificate_from_pi - q_star_dual_certificate_from_g
            )
        else:
            g_star_link_resid = None
            q_star_g_pi_abs_difference = None

        # Check g bounds if diagnostics were attached to the model.
        g_bounds_used = None
        max_g_lower_bound_violation = None
        max_g_upper_bound_violation = None

        diagnostics = getattr(model, "_compact_g_bounds_diagnostics", None)

        if g_star is not None and isinstance(diagnostics, Mapping):
            bounds_by_dim = diagnostics.get("bounds_by_g_dimension", [])
            g_bounds_used = []

            lower_violations: List[float] = []
            upper_violations: List[float] = []

            for item in bounds_by_dim:
                k = int(item["g_index_0_based"])
                lb_raw = item.get("lower_bound")
                ub_raw = item.get("upper_bound")

                lb = None if lb_raw is None else float(lb_raw)
                ub = None if ub_raw is None else float(ub_raw)

                g_val = float(g_star[k])

                lb_violation = 0.0 if lb is None else max(lb - g_val, 0.0)
                ub_violation = 0.0 if ub is None else max(g_val - ub, 0.0)

                lower_violations.append(lb_violation)
                upper_violations.append(ub_violation)

                g_bounds_used.append(
                    {
                        "g_index_0_based": k,
                        "lower_bound": lb,
                        "upper_bound": ub,
                        "g_value": g_val,
                        "lower_bound_violation_positive_part": lb_violation,
                        "upper_bound_violation_positive_part": ub_violation,
                    }
                )

            max_g_lower_bound_violation = max(lower_violations) if lower_violations else 0.0
            max_g_upper_bound_violation = max(upper_violations) if upper_violations else 0.0

        xi_bounds_used = None
        max_xi_lower_bound_violation = None
        max_xi_upper_bound_violation = None
        incumbent_radius_bound_violation = None

        xi_diagnostics = getattr(model, "_compact_xi_bounds_diagnostics", None)

        if isinstance(xi_diagnostics, Mapping):
            xi_bounds_payload = xi_diagnostics.get("xi_bounds", {})
            componentwise_xi_bounds = xi_bounds_payload.get("componentwise_bounds", [])

            if componentwise_xi_bounds:
                xi_bounds_used = []

                lower_violations: List[float] = []
                upper_violations: List[float] = []

                for item in componentwise_xi_bounds:
                    k = int(item["xi_index_0_based"])
                    lb_raw = item.get("lower_bound")
                    ub_raw = item.get("upper_bound")

                    lb = None if lb_raw is None else float(lb_raw)
                    ub = None if ub_raw is None else float(ub_raw)

                    xi_val = float(xi[k])

                    lb_violation = 0.0 if lb is None else max(lb - xi_val, 0.0)
                    ub_violation = 0.0 if ub is None else max(xi_val - ub, 0.0)

                    lower_violations.append(lb_violation)
                    upper_violations.append(ub_violation)

                    xi_bounds_used.append(
                        {
                            "xi_index_0_based": k,
                            "lower_bound": lb,
                            "upper_bound": ub,
                            "xi_value": xi_val,
                            "lower_bound_violation_positive_part": lb_violation,
                            "upper_bound_violation_positive_part": ub_violation,
                        }
                    )

                max_xi_lower_bound_violation = max(lower_violations) if lower_violations else 0.0
                max_xi_upper_bound_violation = max(upper_violations) if upper_violations else 0.0

            incumbent = xi_diagnostics.get("incumbent", {})
            U_raw = incumbent.get("incumbent_radius_U")

            if U_raw is not None:
                incumbent_radius_bound_violation = max(float(gamma) - float(U_raw), 0.0)

        method_name = (
            "compact_one_sided_pyomo_with_g_bounds_certificate"
            if has_g_star
            else "compact_one_sided_pyomo_primal_dual_certificate"
        )

        output_filename = (
            "local_dominance_radius_compact_g_bounded_xi_bounded.json"
            if has_g_star
            else "local_dominance_radius_compact.json"
        )

        result = {
            "instance_name": data.instance_name,
            "method": method_name,
            "solver": solver,
            "solver_status": solver_status,
            "termination_condition": termination_condition,

            "selected_scenario": self.selected_scenario_name,
            "selected_scenario_index_1_based": self.selected_scenario_index_1_based,
            "xi_hat": self.xi_hat.tolist(),
            "initial_cost_gap_at_xi_hat": self.initial_cost_gap,

            "gamma_l1_radius": gamma,
            "nearest_violation_xi": xi.tolist(),
            "absolute_deviation_t": t.tolist(),
            "l1_distance_check": l1_distance,

            "x_star": data.x_star.tolist(),
            "x_EV": data.x_EV.tolist(),
            "first_stage_cost_star": data.first_stage_cost_star,
            "first_stage_cost_EV": data.first_stage_cost_EV,

            "y_EV": y_EV.tolist(),
            "pi_star": pi_star.tolist(),
            "g_star": None if g_star is None else g_star.tolist(),

            "Q_x_EV_primal_candidate_at_nearest_violation": q_EV_primal_candidate,

            "Q_x_star_dual_certificate_at_nearest_violation": q_star_dual_certificate_used,
            "Q_x_star_dual_certificate_from_pi_at_nearest_violation": q_star_dual_certificate_from_pi,
            "Q_x_star_dual_certificate_from_g_at_nearest_violation": q_star_dual_certificate_from_g,

            "cost_with_x_EV_primal_candidate_at_nearest_violation": cost_EV_primal_candidate,
            "cost_with_x_star_dual_certificate_at_nearest_violation": cost_star_dual_certificate,

            "certified_gap_at_nearest_violation": certified_gap,

            "g_bounds_used": g_bounds_used,
            "xi_bounds_used": xi_bounds_used,

            "residual_checks": {
                "max_primal_EV_violation_positive_part": float(
                    np.maximum(primal_EV_resid, 0.0).max(initial=0.0)
                ),
                "max_dual_star_violation_positive_part": float(
                    np.maximum(dual_star_resid, 0.0).max(initial=0.0)
                ),
                "max_pi_star_upper_bound_violation_positive_part": float(
                    np.maximum(pi_star_upper_resid, 0.0).max(initial=0.0)
                ),
                "max_g_star_link_abs_residual": (
                    None
                    if g_star_link_resid is None
                    else float(np.max(np.abs(g_star_link_resid)))
                ),
                "max_g_star_lower_bound_violation_positive_part": max_g_lower_bound_violation,
                "max_g_star_upper_bound_violation_positive_part": max_g_upper_bound_violation,
                "q_star_dual_certificate_g_form_abs_difference": q_star_g_pi_abs_difference,
                "violation_certificate_lhs_minus_rhs": certified_gap,
                "l1_epigraph_gap_sum_t_minus_abs_distance": float(gamma - l1_distance),
                "max_xi_lower_bound_violation_positive_part": max_xi_lower_bound_violation,
                "max_xi_upper_bound_violation_positive_part": max_xi_upper_bound_violation,
                "incumbent_radius_bound_violation_positive_part": incumbent_radius_bound_violation,
            },

            "source_files": {
                "instance_data": "input_data/instance_data.json or input_data/{instance_name}.json",
                "stochastic_results": "output_data/stochastic_results.json",
                "expectedvalue_results": "output_data/expectedvalue_results.json",
                "cost_gap_diagnostics": "output_data/cost_gap_diagnostics.json",
                "cost_gap_dominating_scenarios": "output_data/cost_gap_dominating_scenarios.json",
                "compact_g_bounds_diagnostics": (
                    "output_data/compact_bilevel_h_trans_pie_diagnostics.json"
                    if has_g_star
                    else None
                ),
                "compact_xi_bounds_diagnostics": "output_data/compact_bilevel_xi_bound_diagnostics.json",
            },

            "notes": [
                "This is the compact one-sided nearest-violation formulation.",
                "The model uses y_EV as a primal feasible recourse decision for x_EV.",
                "The model uses pi_star as a dual feasible certificate for the x_star recourse problem.",
                "No y_star variables are present in this compact formulation.",
                "No pi_EV variables are present in this compact formulation.",
                "If g_star is present, the model uses g_star = H^T pi_star and the bilinear term g_star^T xi.",
                "The compact model is nonconvex because of the bilinear projected sensitivity term.",
            ],
        }

        _write_instance_output_json(
            data.instance_dir,
            output_filename,
            result,
        )

        return result


def run_local_dominance_radius(
    instance_name: str,
    base_dir: Optional[Union[str, Path]] = None,
    scenario: Optional[Union[int, str]] = None,
    solver: str = "gurobi",
    tee: bool = False,
    method: str = "pyomo",
    gap_tol: float = 1e-8,
) -> Dict[str, Any]:
    """Run the isolated local-dominance-radius workflow for one selected scenario."""
    problem = LocalDominanceRadiusProblem(
        instance_name,
        base_dir=base_dir,
        scenario=scenario,
        gap_tol=gap_tol,
    )

    if method == "compact_pyomo_with_g_bounds":
        return problem.solve_compact_pyomo_with_g_bounds(solver=solver, tee=tee)    

    raise ValueError("method must be 'solve_compact_pyomo_with_g_bounds' for this script.")

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Solve the local dominance radius problem for a saved TSSP instance.")
    parser.add_argument("instance_name", help="Instance folder name, e.g., lands_instance")
    parser.add_argument("--base-dir", default=None, help="Folder containing instance folders. Defaults to this script's folder.")
    parser.add_argument("--scenario", default=None, help="Optional scenario identifier: 3, scenario_3, or xi_3.")
    parser.add_argument("--solver", default="gurobi", help="Pyomo solver name. Default: gurobi")
    parser.add_argument(
    "--method",
    choices=["pyomo", "compact_pyomo", "compact_pyomo_with_g_bounds", "enumeration"],
    default="pyomo",
    help="Default uses the full Pyomo/Gurobi formulation. Use compact_pyomo for the one-sided compact formulation.",
    )
    parser.add_argument("--gap-tol", type=float, default=1e-8, help="Positive-gap tolerance for selecting scenarios.")
    parser.add_argument("--tee", action="store_true", help="Stream solver output.")
    args = parser.parse_args(argv)

    result = run_local_dominance_radius(
        instance_name=args.instance_name,
        base_dir=args.base_dir,
        scenario=args.scenario,
        solver=args.solver,
        tee=args.tee,
        method=args.method,
        gap_tol=args.gap_tol,
    )
    print(json.dumps(_json_ready(result), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
