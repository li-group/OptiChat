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

    def build_original_pyomo_model(self) -> Any:
        """
        Build the nonconvex quadratic Pyomo model for the exact primal-dual reformulation.

        Variables:
            xi[k]       free realized-scenario vector at nearest violation
            t[k]        absolute-value epigraph variables for ||xi - xi_hat||_1
            y_EV[j]     primal recourse variables for fixed x_EV
            y_star[j]   primal recourse variables for fixed x_star
            pi_EV[r]    dual recourse variables for fixed x_EV, pi_EV <= 0
            pi_star[r]  dual recourse variables for fixed x_star, pi_star <= 0
        """
        if pyo is None:
            raise LocalDominanceSolveError("Pyomo is not installed. Install pyomo and use solver='gurobi'.")

        data = self.data
        m = pyo.ConcreteModel(name=f"local_dominance_radius_{data.instance_name}_{self.selected_scenario_name}")
        m.K = pyo.RangeSet(0, data.xi_dim - 1)                       # xi components
        m.J = pyo.RangeSet(0, data.num_second_stage_vars - 1)         # y variables
        m.R = pyo.RangeSet(0, data.num_recourse_constraints - 1)      # recourse rows / dual variables
        m.I = pyo.RangeSet(0, data.num_first_stage_vars - 1)          # first-stage variables, fixed here

        m.xi = pyo.Var(m.K, domain=pyo.Reals, initialize=lambda model, k: float(self.xi_hat[k]))
        m.t = pyo.Var(m.K, domain=pyo.NonNegativeReals, initialize=0.0)
        m.y_EV = pyo.Var(m.J, domain=pyo.NonNegativeReals, initialize=0.0)
        m.y_star = pyo.Var(m.J, domain=pyo.NonNegativeReals, initialize=0.0)
        m.pi_EV = pyo.Var(m.R, domain=pyo.Reals, bounds=(None, 0.0), initialize=0.0)
        m.pi_star = pyo.Var(m.R, domain=pyo.Reals, bounds=(None, 0.0), initialize=0.0)

        DUAL_LB = -1000.0   # example lower bound
        m.pi_EV_manual_lb = pyo.Constraint(
            m.R,
            rule=lambda model, r: model.pi_EV[r] >= DUAL_LB
        )

        m.pi_star_manual_lb = pyo.Constraint(
            m.R,
            rule=lambda model, r: model.pi_star[r] >= DUAL_LB
        )

        ###############################################################
        # In lannds instance only \xi_5 (5th elemnent, k = 4) is allowed to vary for the local dominance radius, so we fix all other xi_k to 0.
        if data.instance_name == "lands_instancex":
        # if data.instance_name == "lands_instance":  
            if len(m.K) != 7:
                raise ValueError(
                    f"Expected m.K to have 7 elements for {data.instance_name}, "
                    f"but got {len(m.K)}."
                )

            # Since m.K = RangeSet(0, 6), the fifth element is k = 4.
            fifth_k = list(m.K)[4]
            def fix_t_except_fifth_rule(model, k):
                if k == fifth_k:
                    return pyo.Constraint.Skip
                return model.t[k] == 0.0

            m.fix_t_except_fifth = pyo.Constraint(m.K, rule=fix_t_except_fifth_rule)
        ###############################################################

        y_ev_init = self._initial_y("EV")
        y_star_init = self._initial_y("star")
        if y_ev_init is not None:
            for j in range(data.num_second_stage_vars):
                m.y_EV[j].set_value(max(0.0, float(y_ev_init[j])))
        if y_star_init is not None:
            for j in range(data.num_second_stage_vars):
                m.y_star[j].set_value(max(0.0, float(y_star_init[j])))

        def Hxi(model: Any, r: int) -> Any:
            return sum(float(data.H[r, k]) * model.xi[k] for k in model.K)

        def Tx(x_vec: np.ndarray, r: int) -> float:
            return float(sum(float(data.T[r, i]) * float(x_vec[i]) for i in range(data.num_first_stage_vars)))

        def rhs_EV(model: Any, r: int) -> Any:
            return Hxi(model, r) - Tx(data.x_EV, r)

        def rhs_star(model: Any, r: int) -> Any:
            return Hxi(model, r) - Tx(data.x_star, r)

        m.objective = pyo.Objective(expr=sum(m.t[k] for k in m.K), sense=pyo.minimize)

        m.abs_upper = pyo.Constraint(
            m.K,
            rule=lambda model, k: model.xi[k] - float(self.xi_hat[k]) <= model.t[k],
        )
        m.abs_lower = pyo.Constraint(
            m.K,
            rule=lambda model, k: float(self.xi_hat[k]) - model.xi[k] <= model.t[k],
        )

        m.primal_EV = pyo.Constraint(
            m.R,
            rule=lambda model, r: sum(float(data.W[r, j]) * model.y_EV[j] for j in model.J) <= rhs_EV(model, r),
        )
        m.primal_star = pyo.Constraint(
            m.R,
            rule=lambda model, r: sum(float(data.W[r, j]) * model.y_star[j] for j in model.J) <= rhs_star(model, r),
        )

        m.dual_EV = pyo.Constraint(
            m.J,
            rule=lambda model, j: sum(float(data.W[r, j]) * model.pi_EV[r] for r in model.R) <= float(data.d[j]),
        )
        m.dual_star = pyo.Constraint(
            m.J,
            rule=lambda model, j: sum(float(data.W[r, j]) * model.pi_star[r] for r in model.R) <= float(data.d[j]),
        )

        m.strong_duality_EV = pyo.Constraint(
            expr=sum(float(data.d[j]) * m.y_EV[j] for j in m.J)
            == sum(m.pi_EV[r] * rhs_EV(m, r) for r in m.R)
        )
        m.strong_duality_star = pyo.Constraint(
            expr=sum(float(data.d[j]) * m.y_star[j] for j in m.J)
            == sum(m.pi_star[r] * rhs_star(m, r) for r in m.R)
        )

        m.violation = pyo.Constraint(
            expr=float(data.first_stage_cost_EV) + sum(float(data.d[j]) * m.y_EV[j] for j in m.J)
            <= float(data.first_stage_cost_star) + sum(float(data.d[j]) * m.y_star[j] for j in m.J)
        )

        # Attach these helpers for cleaner extraction; they are not Pyomo components.
        m._local_rhs_EV = rhs_EV  # type: ignore[attr-defined]
        m._local_rhs_star = rhs_star  # type: ignore[attr-defined]
        return m
    

    def solve_pyomo(self, solver: str = "gurobi", tee: bool = False, solver_options: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        """Solve the Pyomo model. For Gurobi, NonConvex=2 is set automatically."""
        if pyo is None:
            raise LocalDominanceSolveError("Pyomo is not installed.")
        model = self.build_original_pyomo_model()
        opt = pyo.SolverFactory(solver)
        if opt is None or not opt.available(False):
            raise LocalDominanceSolveError(f"Solver {solver!r} is not available to Pyomo.")
        if solver.lower().startswith("gurobi"):
            opt.options["NonConvex"] = 2
        if solver_options:
            for key, value in solver_options.items():
                opt.options[str(key)] = value
        tee= True
        results = opt.solve(model, tee=tee)
        status = str(results.solver.status)
        termination = str(results.solver.termination_condition)
        ok_terms = {"optimal", "globallyOptimal", "locallyOptimal"}
        if termination not in ok_terms:
            raise LocalDominanceSolveError(
                f"Local dominance radius solve did not terminate optimally. status={status}, termination={termination}"
            )
        return self._extract_pyomo_solution(model, solver=solver, solver_status=status, termination_condition=termination)

    def _extract_pyomo_solution(self, model: Any, solver: str, solver_status: str, termination_condition: str) -> Dict[str, Any]:
        data = self.data
        xi = np.asarray([float(pyo.value(model.xi[k])) for k in model.K], dtype=float)
        t = np.asarray([float(pyo.value(model.t[k])) for k in model.K], dtype=float)
        y_EV = np.asarray([float(pyo.value(model.y_EV[j])) for j in model.J], dtype=float)
        y_star = np.asarray([float(pyo.value(model.y_star[j])) for j in model.J], dtype=float)
        pi_EV = np.asarray([float(pyo.value(model.pi_EV[r])) for r in model.R], dtype=float)
        pi_star = np.asarray([float(pyo.value(model.pi_star[r])) for r in model.R], dtype=float)
        gamma = float(sum(t))

        q_EV = _dot(data.d, y_EV)
        q_star = _dot(data.d, y_star)
        cost_EV = data.first_stage_cost_EV + q_EV
        cost_star = data.first_stage_cost_star + q_star
        gap_at_nearest = cost_EV - cost_star

        primal_EV_resid = data.W @ y_EV - (data.H @ xi - data.T @ data.x_EV)
        primal_star_resid = data.W @ y_star - (data.H @ xi - data.T @ data.x_star)
        dual_EV_resid = data.W.T @ pi_EV - data.d
        dual_star_resid = data.W.T @ pi_star - data.d
        sd_EV_resid = q_EV - float(pi_EV @ (data.H @ xi - data.T @ data.x_EV))
        sd_star_resid = q_star - float(pi_star @ (data.H @ xi - data.T @ data.x_star))

        result = {
            "instance_name": data.instance_name,
            "method": "pyomo_primal_dual_strong_duality",
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
            "l1_distance_check": float(np.abs(xi - self.xi_hat).sum()),
            "x_star": data.x_star.tolist(),
            "x_EV": data.x_EV.tolist(),
            "first_stage_cost_star": data.first_stage_cost_star,
            "first_stage_cost_EV": data.first_stage_cost_EV,
            "y_star": y_star.tolist(),
            "y_EV": y_EV.tolist(),
            "pi_star": pi_star.tolist(),
            "pi_EV": pi_EV.tolist(),
            "Q_x_star_at_nearest_violation": q_star,
            "Q_x_EV_at_nearest_violation": q_EV,
            "cost_with_x_star_at_nearest_violation": cost_star,
            "cost_with_x_EV_at_nearest_violation": cost_EV,
            "gap_at_nearest_violation": gap_at_nearest,
            "residual_checks": {
                "max_primal_EV_violation_positive_part": float(np.maximum(primal_EV_resid, 0.0).max(initial=0.0)),
                "max_primal_star_violation_positive_part": float(np.maximum(primal_star_resid, 0.0).max(initial=0.0)),
                "max_dual_EV_violation_positive_part": float(np.maximum(dual_EV_resid, 0.0).max(initial=0.0)),
                "max_dual_star_violation_positive_part": float(np.maximum(dual_star_resid, 0.0).max(initial=0.0)),
                "strong_duality_EV_abs_residual": abs(sd_EV_resid),
                "strong_duality_star_abs_residual": abs(sd_star_resid),
                "violation_constraint_lhs_minus_rhs": gap_at_nearest,
            },
            "source_files": {
                "instance_data": "instance_data.json",
                "stochastic_results": "stochastic_results.json",
                "expectedvalue_results": "expectedvalue_results.json",
                "cost_gap_diagnostics": "cost_gap_diagnostics.json",
                "cost_gap_dominating_scenarios": "cost_gap_dominating_scenarios.json",
            },
            "notes": [
                "Positive initial gap means x_star dominates x_EV at xi_hat.",
                "The returned radius is the nearest l1 distance to any xi where the dominance is no longer strict, i.e., Delta(xi) <= 0.",
                "The model uses Wy <= H xi - T x, matching T x + W y <= H xi.",
            ],
        }
        # _write_json(data.instance_dir / "local_dominance_radius.json", result)
        _write_instance_output_json(data.instance_dir, "local_dominance_radius.json", result)
        return result
    
    def build_compact_bilevel_pyomo_model_without_bounds(self) -> Any:
        """
        Build the compact one-sided nearest-violation Pyomo model.

        This implements:

            min sum_k t[k]

            s.t.  -t[k] <= xi[k] - xi_hat[k] <= t[k]
                t[k] >= 0

                y_EV >= 0
                W y_EV <= H xi - T x_EV

                W^T pi_star <= d
                pi_star <= 0

                c^T x_EV + d^T y_EV
                <=
                c^T x_star + pi_star^T(H xi - T x_star)

        Notes:
            - No y_star variables are used.
            - No pi_EV variables are used.
            - pi_star is the dual certificate for the x_star recourse problem.
            - This model is still nonconvex because of pi_star^T H xi.
            - No artificial bounds are imposed on xi or pi_star.
        """
        if pyo is None:
            raise LocalDominanceSolveError("Pyomo is not installed. Install pyomo and use solver='gurobi'.")

        data = self.data

        m = pyo.ConcreteModel(
            name=f"compact_local_dominance_radius_{data.instance_name}_{self.selected_scenario_name}"
        )

        m.K = pyo.RangeSet(0, data.xi_dim - 1)
        m.J = pyo.RangeSet(0, data.num_second_stage_vars - 1)
        m.R = pyo.RangeSet(0, data.num_recourse_constraints - 1)
        m.I = pyo.RangeSet(0, data.num_first_stage_vars - 1)

        # Scenario perturbation variables
        m.xi = pyo.Var(
            m.K,
            domain=pyo.Reals,
            initialize=lambda model, k: float(self.xi_hat[k]),
        )

        # L1 epigraph variables
        m.t = pyo.Var(
            m.K,
            domain=pyo.NonNegativeReals,
            initialize=0.0,
        )

        # EV-side primal recourse variables
        m.y_EV = pyo.Var(
            m.J,
            domain=pyo.NonNegativeReals,
            initialize=0.0,
        )

        # Star-side dual certificate variables
        m.pi_star = pyo.Var(
            m.R,
            domain=pyo.Reals,
            bounds=(None, 0.0),
            initialize=0.0,
        )

        # Optional initialization from saved EV recourse solution
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

        m.objective = pyo.Objective(
            expr=sum(m.t[k] for k in m.K),
            sense=pyo.minimize,
        )

        # L1-distance linearization: -t_k <= xi_k - xi_hat_k <= t_k
        m.abs_upper = pyo.Constraint(
            m.K,
            rule=lambda model, k: model.xi[k] - float(self.xi_hat[k]) <= model.t[k],
        )

        m.abs_lower = pyo.Constraint(
            m.K,
            rule=lambda model, k: float(self.xi_hat[k]) - model.xi[k] <= model.t[k],
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

        # One-sided violation certificate:
        #
        # c^T x_EV + d^T y_EV
        # <=
        # c^T x_star + pi_star^T(H xi - T x_star)
        m.violation_cert = pyo.Constraint(
            expr=float(data.first_stage_cost_EV)
            + sum(float(data.d[j]) * m.y_EV[j] for j in m.J)
            <= float(data.first_stage_cost_star)
            + sum(m.pi_star[r] * rhs_star(m, r) for r in m.R)
        )

        # Attach helpers for extraction/debugging; these are not Pyomo components.
        m._local_rhs_EV = rhs_EV  # type: ignore[attr-defined]
        m._local_rhs_star = rhs_star  # type: ignore[attr-defined]

        return m
    
    def solve_compact_pyomo(
        self,
        solver: str = "gurobi",
        tee: bool = False,
        solver_options: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Solve the compact one-sided Pyomo model.

        This uses build_compact_bilevel_pyomo_model_without_bounds(), which has:
            - xi, t
            - y_EV
            - pi_star

        It does not contain:
            - y_star
            - pi_EV
        """
        if pyo is None:
            raise LocalDominanceSolveError("Pyomo is not installed.")

        model = self.build_compact_bilevel_pyomo_model_without_bounds()

        opt = pyo.SolverFactory(solver)
        if opt is None or not opt.available(False):
            raise LocalDominanceSolveError(f"Solver {solver!r} is not available to Pyomo.")

        if solver.lower().startswith("gurobi"):
            opt.options["NonConvex"] = 2

        if solver_options:
            for key, value in solver_options.items():
                opt.options[str(key)] = value

        results = opt.solve(model, tee=tee)

        status = str(results.solver.status)
        termination = str(results.solver.termination_condition)

        ok_terms = {"optimal", "globallyOptimal", "locallyOptimal"}
        if termination not in ok_terms:
            raise LocalDominanceSolveError(
                f"Compact local dominance radius solve did not terminate optimally. "
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
        Extract solution from the compact one-sided formulation.

        The compact formulation has only:
            y_EV      primal feasible recourse decision for x_EV
            pi_star   dual feasible certificate for x_star

        Therefore this extractor does not read y_star or pi_EV.
        """
        data = self.data

        xi = np.asarray([float(pyo.value(model.xi[k])) for k in model.K], dtype=float)
        t = np.asarray([float(pyo.value(model.t[k])) for k in model.K], dtype=float)
        y_EV = np.asarray([float(pyo.value(model.y_EV[j])) for j in model.J], dtype=float)
        pi_star = np.asarray([float(pyo.value(model.pi_star[r])) for r in model.R], dtype=float)

        gamma = float(sum(t))

        rhs_EV = data.H @ xi - data.T @ data.x_EV
        rhs_star = data.H @ xi - data.T @ data.x_star

        q_EV_primal_candidate = _dot(data.d, y_EV)
        q_star_dual_certificate = float(pi_star @ rhs_star)

        cost_EV_primal_candidate = data.first_stage_cost_EV + q_EV_primal_candidate
        cost_star_dual_certificate = data.first_stage_cost_star + q_star_dual_certificate

        certified_gap = cost_EV_primal_candidate - cost_star_dual_certificate

        primal_EV_resid = data.W @ y_EV - rhs_EV
        dual_star_resid = data.W.T @ pi_star - data.d
        pi_star_upper_resid = pi_star

        l1_distance = float(np.abs(xi - self.xi_hat).sum())

        result = {
            "instance_name": data.instance_name,
            "method": "compact_one_sided_pyomo_primal_dual_certificate",
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

            "Q_x_EV_primal_candidate_at_nearest_violation": q_EV_primal_candidate,
            "Q_x_star_dual_certificate_at_nearest_violation": q_star_dual_certificate,

            "cost_with_x_EV_primal_candidate_at_nearest_violation": cost_EV_primal_candidate,
            "cost_with_x_star_dual_certificate_at_nearest_violation": cost_star_dual_certificate,

            "certified_gap_at_nearest_violation": certified_gap,

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
                "violation_certificate_lhs_minus_rhs": certified_gap,
                "l1_epigraph_gap_sum_t_minus_abs_distance": float(gamma - l1_distance),
            },

            "source_files": {
                "instance_data": "input_data/instance_data.json or input_data/{instance_name}.json",
                "stochastic_results": "output_data/stochastic_results.json",
                "expectedvalue_results": "output_data/expectedvalue_results.json",
                "cost_gap_diagnostics": "output_data/cost_gap_diagnostics.json",
                "cost_gap_dominating_scenarios": "output_data/cost_gap_dominating_scenarios.json",
            },

            "notes": [
                "This is the compact one-sided nearest-violation formulation.",
                "The model uses y_EV as a primal feasible recourse decision for x_EV.",
                "The model uses pi_star as a dual feasible certificate for the x_star recourse problem.",
                "No y_star variables are present in this compact formulation.",
                "No pi_EV variables are present in this compact formulation.",
                "The violation certificate is c^T x_EV + d^T y_EV <= c^T x_star + pi_star^T(H xi - T x_star).",
                "The model is nonconvex because pi_star^T H xi is bilinear.",
            ],
        }

        _write_instance_output_json(
            data.instance_dir,
            "local_dominance_radius_compact.json",
            result,
        )

        return result

    # ------------------------------------------------------------------
    # Optional validation method for small recourse LPs.
    # ------------------------------------------------------------------
    def solve_by_dual_extreme_enumeration(
        self,
        feasibility_tol: float = 1e-8,
        duplicate_tol: float = 1e-7,
        max_extreme_points: int = 5000,
    ) -> Dict[str, Any]:
        """
        Optional exact LP cross-check for small instances.

        The common dual feasible region is
            P = {pi : W^T pi <= d, pi <= 0}.
        The method enumerates extreme points of P. For every pair of extreme
        points that can be optimal for the EV and stochastic recourse LPs, it
        solves a linear nearest-violation LP in (xi,t).  This is useful for
        testing and validation but is not intended to replace the requested
        Pyomo/Gurobi primal-dual model for large instances.
        """
        if linprog is None:
            raise LocalDominanceSolveError("scipy.optimize.linprog is required for the enumeration cross-check.")
        data = self.data
        extreme_points = self._enumerate_dual_extreme_points(feasibility_tol, duplicate_tol, max_extreme_points)
        if not extreme_points:
            raise LocalDominanceSolveError("No dual extreme points were enumerated; cannot run cross-check.")
        E = np.asarray(extreme_points, dtype=float)

        m_dim = data.xi_dim
        n_vars = 2 * m_dim
        obj = np.concatenate([np.zeros(m_dim), np.ones(m_dim)])
        bounds = [(None, None)] * m_dim + [(0.0, None)] * m_dim

        base_A: List[np.ndarray] = []
        base_b: List[float] = []
        for k in range(m_dim):
            row = np.zeros(n_vars)
            row[k] = 1.0
            row[m_dim + k] = -1.0
            base_A.append(row)
            base_b.append(float(self.xi_hat[k]))

            row = np.zeros(n_vars)
            row[k] = -1.0
            row[m_dim + k] = -1.0
            base_A.append(row)
            base_b.append(float(-self.xi_hat[k]))

        first_stage_gap = data.first_stage_cost_EV - data.first_stage_cost_star
        best_res: Optional[Any] = None
        best_pair: Optional[Tuple[int, int]] = None
        feasible_pair_count = 0

        Tx_EV = data.T @ data.x_EV
        Tx_star = data.T @ data.x_star

        for ev_idx, pi_EV in enumerate(E):
            for star_idx, pi_star in enumerate(E):
                A_ub = list(base_A)
                b_ub = list(base_b)

                # pi_EV optimal for rhs_EV = H xi - T x_EV:
                # (pi_EV - pi)^T (H xi - T x_EV) >= 0 for all pi in E.
                for pi in E:
                    a = pi_EV - pi
                    row = np.zeros(n_vars)
                    row[:m_dim] = -(a @ data.H)
                    A_ub.append(row)
                    b_ub.append(float(-(a @ Tx_EV)))

                # pi_star optimal for rhs_star = H xi - T x_star.
                for pi in E:
                    a = pi_star - pi
                    row = np.zeros(n_vars)
                    row[:m_dim] = -(a @ data.H)
                    A_ub.append(row)
                    b_ub.append(float(-(a @ Tx_star)))

                # Violation: c^T x_EV + Q_EV(xi) <= c^T x_star + Q_star(xi).
                row = np.zeros(n_vars)
                row[:m_dim] = (pi_EV - pi_star) @ data.H
                rhs = -first_stage_gap + float(pi_EV @ Tx_EV) - float(pi_star @ Tx_star)
                A_ub.append(row)
                b_ub.append(rhs)

                res = linprog(
                    c=obj,
                    A_ub=np.asarray(A_ub, dtype=float),
                    b_ub=np.asarray(b_ub, dtype=float),
                    bounds=bounds,
                    method="highs",
                )
                if not res.success:
                    continue
                feasible_pair_count += 1
                if best_res is None or float(res.fun) < float(best_res.fun) - 1e-9:
                    best_res = res
                    best_pair = (ev_idx, star_idx)

        if best_res is None or best_pair is None:
            raise LocalDominanceSolveError("The enumeration cross-check found no feasible nearest-violation LP.")

        xi = np.asarray(best_res.x[:m_dim], dtype=float)
        t = np.asarray(best_res.x[m_dim:], dtype=float)
        pi_EV = E[best_pair[0]]
        pi_star = E[best_pair[1]]
        q_EV = float(np.max(E @ (data.H @ xi - data.T @ data.x_EV)))
        q_star = float(np.max(E @ (data.H @ xi - data.T @ data.x_star)))
        cost_EV = data.first_stage_cost_EV + q_EV
        cost_star = data.first_stage_cost_star + q_star
        gap_at_nearest = cost_EV - cost_star

        result = {
            "instance_name": data.instance_name,
            "method": "dual_extreme_point_enumeration_lp_cross_check",
            "selected_scenario": self.selected_scenario_name,
            "selected_scenario_index_1_based": self.selected_scenario_index_1_based,
            "xi_hat": self.xi_hat.tolist(),
            "initial_cost_gap_at_xi_hat": self.initial_cost_gap,
            "gamma_l1_radius": float(best_res.fun),
            "nearest_violation_xi": xi.tolist(),
            "absolute_deviation_t": t.tolist(),
            "l1_distance_check": float(np.abs(xi - self.xi_hat).sum()),
            "first_stage_cost_star": data.first_stage_cost_star,
            "first_stage_cost_EV": data.first_stage_cost_EV,
            "Q_x_star_at_nearest_violation": q_star,
            "Q_x_EV_at_nearest_violation": q_EV,
            "cost_with_x_star_at_nearest_violation": cost_star,
            "cost_with_x_EV_at_nearest_violation": cost_EV,
            "gap_at_nearest_violation": gap_at_nearest,
            "num_dual_extreme_points": int(E.shape[0]),
            "num_feasible_dual_extreme_point_pairs": feasible_pair_count,
            "selected_dual_extreme_point_indices": {
                "EV": int(best_pair[0]),
                "star": int(best_pair[1]),
            },
            "pi_EV": pi_EV.tolist(),
            "pi_star": pi_star.tolist(),
            "notes": [
                "This is an optional cross-check for small instances, not the primary Pyomo/Gurobi workflow.",
                "The primary local-dominance model is built by build_original_pyomo_model() and solved by solve_pyomo().",
            ],
        }
        # _write_json(data.instance_dir / "local_dominance_radius_enumeration_check.json", result)
        _write_instance_output_json(
            data.instance_dir,
            "local_dominance_radius_enumeration_check.json",
            result,
        )
        return result

    def _enumerate_dual_extreme_points(
        self,
        feasibility_tol: float,
        duplicate_tol: float,
        max_extreme_points: int,
    ) -> List[np.ndarray]:
        data = self.data
        r_dim = data.num_recourse_constraints
        # Dual feasible polyhedron: W^T pi <= d, pi <= 0.
        A_dual = np.vstack([data.W.T, np.eye(r_dim)])
        b_dual = np.concatenate([data.d, np.zeros(r_dim)])
        n_constraints = A_dual.shape[0]
        extreme_points: List[np.ndarray] = []

        for active_set in itertools.combinations(range(n_constraints), r_dim):
            active = A_dual[list(active_set), :]
            if np.linalg.matrix_rank(active, tol=1e-10) < r_dim:
                continue
            try:
                pi = np.linalg.solve(active, b_dual[list(active_set)])
            except np.linalg.LinAlgError:  # pragma: no cover
                continue
            if not np.all(A_dual @ pi <= b_dual + feasibility_tol):
                continue
            if any(np.linalg.norm(pi - old, ord=np.inf) <= duplicate_tol for old in extreme_points):
                continue
            extreme_points.append(pi)
            if len(extreme_points) > max_extreme_points:
                raise LocalDominanceSolveError(
                    f"More than {max_extreme_points} dual extreme points found; enumeration is not appropriate."
                )
        return extreme_points



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

    if method == "pyomo":
        return problem.solve_pyomo(solver=solver, tee=tee)

    if method == "compact_pyomo":
        return problem.solve_compact_pyomo(solver=solver, tee=tee)

    if method == "enumeration":
        return problem.solve_by_dual_extreme_enumeration()

    raise ValueError("method must be 'pyomo', 'compact_pyomo', or 'enumeration'.")

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Solve the local dominance radius problem for a saved TSSP instance.")
    parser.add_argument("instance_name", help="Instance folder name, e.g., lands_instance")
    parser.add_argument("--base-dir", default=None, help="Folder containing instance folders. Defaults to this script's folder.")
    parser.add_argument("--scenario", default=None, help="Optional scenario identifier: 3, scenario_3, or xi_3.")
    parser.add_argument("--solver", default="gurobi", help="Pyomo solver name. Default: gurobi")
    parser.add_argument(
    "--method",
    choices=["pyomo", "compact_pyomo", "enumeration"],
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
