"""
Reusable two-stage stochastic programming workflow in deterministic-equivalent form.

The module supports the instance structure used by instance_name/instance_name.json 

and is written in terms of general matrices/vectors:

    min_x,y_s c^T x + sum_s p_s d^T y_s
    s.t.    A x >= b
            x >= 0
            T x + W y_s <= H xi_s, y_s >= 0, for all scenarios s.

Primary solve path: Pyomo + the requested solver, e.g., Gurobi.
Fallback solve path: scipy.optimize.linprog, useful for unit testing on machines
without Pyomo/Gurobi. Set backend="pyomo" to require Pyomo/Gurobi strictly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from unittest import result

try:
    import pyomo.environ as pyo  # type: ignore
except Exception:  # pragma: no cover - exercised only when Pyomo is absent
    pyo = None

try:
    import numpy as np
except Exception as exc:  # pragma: no cover
    raise RuntimeError("numpy is required for this workflow") from exc

try:
    from scipy.optimize import linprog  # type: ignore
except Exception:  # pragma: no cover
    linprog = None


Number = float


class DataConsistencyError(ValueError):
    # """Raised when instance_data.json is dimensionally inconsistent."""
    """Raised when {instance_name}.json/instance_data.json is dimensionally inconsistent."""


class SolveError(RuntimeError):
    """Raised when an optimization model does not solve to optimality."""

def _resolve_instance_dir(instance_name: str, base_dir: Optional[str | Path] = None) -> Path:
    root = _as_path(base_dir)

    candidates = [
        root / instance_name,
        root / "lands_generated_instances" / instance_name,
    ]

    for instance_dir in candidates:
        input_dir = instance_dir / "input_data"
        if (input_dir / f"{instance_name}.json").exists() or (input_dir / "instance_data.json").exists():
            return instance_dir

    looked = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"Could not find instance {instance_name!r}. Looked in:\n{looked}"
    )

def _as_path(base_dir: Optional[str | Path]) -> Path:
    return Path(base_dir).expanduser().resolve() if base_dir is not None else Path(__file__).resolve().parent


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_json_ready(payload), f, indent=2, sort_keys=False)
        f.write("\n")


def _json_ready(value: Any) -> Any:
    """Convert numpy/scalar values into JSON-friendly Python values."""
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
    missing = {k for k in ("rows", "cols", "data") if k not in raw}
    if missing:
        raise DataConsistencyError(f"Matrix {name!r} is missing keys: {sorted(missing)}")
    rows, cols = int(raw["rows"]), int(raw["cols"])
    data = np.asarray(raw["data"], dtype=float)
    if data.size != rows * cols:
        raise DataConsistencyError(
            f"Matrix {name!r} declares shape ({rows}, {cols}) but has {data.size} entries."
        )
    return data.reshape((rows, cols))


def _sorted_xi_items(data: Mapping[str, Any]) -> List[Tuple[str, np.ndarray]]:
    pattern = re.compile(r"^xi_(\d+)$")
    found: List[Tuple[int, str, np.ndarray]] = []
    for key, value in data.items():
        match = pattern.match(key)
        if match:
            found.append((int(match.group(1)), key, np.asarray(value, dtype=float)))
    found.sort(key=lambda item: item[0])
    if not found:
        raise DataConsistencyError("No scenario vectors were found. Expected keys xi_1, xi_2, ...")
    expected = list(range(1, len(found) + 1))
    actual = [idx for idx, _, _ in found]
    if actual != expected:
        raise DataConsistencyError(f"Scenario xi keys must be consecutive starting at xi_1. Found {actual}.")
    return [(key, xi) for _, key, xi in found]


def _scenario_label(index0: int) -> str:
    return f"scenario_{index0 + 1}"


def _dot(vector: np.ndarray, values: np.ndarray) -> float:
    return float(np.asarray(vector, dtype=float) @ np.asarray(values, dtype=float))


def _max_abs_diff(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        return float("inf")
    if not left:
        return 0.0
    return float(max(abs(float(a) - float(b)) for a, b in zip(left, right)))


@dataclass(frozen=True)
class TSSPInstance:
    """A validated TSSP instance loaded from instance_data.json or from {instance_name}.json."""

    name: str
    instance_dir: Path
    raw_data: Dict[str, Any]
    A: np.ndarray
    b: np.ndarray
    c: np.ndarray
    d: np.ndarray
    T: np.ndarray
    W: np.ndarray
    H: np.ndarray
    probabilities: np.ndarray
    xi: List[np.ndarray]
    xi_keys: List[str]

    @classmethod
    def load(cls, instance_name: str, base_dir: Optional[str | Path] = None) -> "TSSPInstance":
        instance_dir = _resolve_instance_dir(instance_name, base_dir)

        data_path = instance_dir / "input_data" / f"{instance_name}.json"
        if not data_path.exists():
            data_path = instance_dir / "input_data" / "instance_data.json"

        if not data_path.exists():
            raise FileNotFoundError(f"Could not find instance data at {data_path}")

        data = _read_json(data_path)

        required = ["A", "b", "c", "d", "T", "W", "H"]
        missing = [key for key in required if key not in data]
        if missing:
            raise DataConsistencyError(f"{instance_name}.json/instance_data.json is missing required keys: {missing}")

        A = _matrix_from_json("A", data["A"])
        T = _matrix_from_json("T", data["T"])
        W = _matrix_from_json("W", data["W"])
        H = _matrix_from_json("H", data["H"])
        b = np.asarray(data["b"], dtype=float)
        c = np.asarray(data["c"], dtype=float)
        d = np.asarray(data["d"], dtype=float)

        xi_items = _sorted_xi_items(data)
        xi_keys = [key for key, _ in xi_items]
        xi = [value for _, value in xi_items]

        if "p_s" in data:
            probabilities = np.asarray(data["p_s"], dtype=float)
        else:
            probabilities = np.asarray([data[f"p_{i + 1}"][0] for i in range(len(xi))], dtype=float)

        inst = cls(
            name=instance_name,
            instance_dir=instance_dir,
            raw_data=data,
            A=A,
            b=b,
            c=c,
            d=d,
            T=T,
            W=W,
            H=H,
            probabilities=probabilities,
            xi=xi,
            xi_keys=xi_keys,
        )
        inst.check_consistency()
        return inst

    @property
    def num_first_stage_vars(self) -> int:
        return int(self.c.size)

    @property
    def num_second_stage_vars(self) -> int:
        return int(self.d.size)

    @property
    def num_scenarios(self) -> int:
        return len(self.xi)

    @property
    def scenario_names(self) -> List[str]:
        return [_scenario_label(i) for i in range(self.num_scenarios)]

    @property
    def expected_xi(self) -> np.ndarray:
        return sum(float(p) * xi for p, xi in zip(self.probabilities, self.xi))

    def check_consistency(self, probability_tol: float = 1e-8) -> None:
        """Check the matrix/vector conformability required by the workflow."""
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
        if self.H.shape[0] != self.T.shape[0]:
            errors.append(f"H has {self.H.shape[0]} rows but T/W have {self.T.shape[0]} rows.")
        if self.H.shape[1] == 0:
            errors.append("H must have at least one column.")
        for key, xi in zip(self.xi_keys, self.xi):
            if xi.size != self.H.shape[1]:
                errors.append(f"{key} has length {xi.size} but H has {self.H.shape[1]} columns.")
        if self.probabilities.size != self.num_scenarios:
            errors.append(
                f"There are {self.num_scenarios} scenarios but {self.probabilities.size} probabilities."
            )
        if np.any(self.probabilities < -probability_tol):
            errors.append("Scenario probabilities must be nonnegative.")
        prob_sum = float(self.probabilities.sum())
        if abs(prob_sum - 1.0) > probability_tol:
            errors.append(f"Scenario probabilities must sum to 1. Found {prob_sum}.")
        if self.num_first_stage_vars <= 0 or self.num_second_stage_vars <= 0:
            errors.append("Both first-stage and second-stage variable dimensions must be positive.")

        if errors:
            raise DataConsistencyError("Data consistency checks failed:\n- " + "\n- ".join(errors))

    def consistency_report(self) -> Dict[str, Any]:
        return {
            "status": "passed",
            "num_first_stage_vars": self.num_first_stage_vars,
            "num_second_stage_vars": self.num_second_stage_vars,
            "num_scenarios": self.num_scenarios,
            "A_shape": list(self.A.shape),
            "T_shape": list(self.T.shape),
            "W_shape": list(self.W.shape),
            "H_shape": list(self.H.shape),
            "probability_sum": float(self.probabilities.sum()),
            "constraint_form_used": "T x + W y_s <= H xi_s",
        }


class _SolverMixin:
    def _choose_backend(self, backend: str) -> str:
        backend = backend.lower()
        if backend not in {"auto", "pyomo", "scipy"}:
            raise ValueError("backend must be one of: 'auto', 'pyomo', 'scipy'")
        if backend == "pyomo":
            if pyo is None:
                raise SolveError("backend='pyomo' requested, but Pyomo is not installed.")
            return "pyomo"
        if backend == "scipy":
            if linprog is None:
                raise SolveError("backend='scipy' requested, but scipy is not installed.")
            return "scipy"
        if pyo is not None:
            return "pyomo"
        if linprog is not None:
            return "scipy"
        raise SolveError("Neither Pyomo nor scipy.optimize.linprog is available.")

    @staticmethod
    def _check_linprog_result(result: Any, label: str) -> None:
        if not result.success:
            raise SolveError(f"{label} failed with scipy.linprog status {result.status}: {result.message}")

    @staticmethod
    def _solve_pyomo_model(model: Any, solver: str, tee: bool = False) -> Any:
        if pyo is None:  # pragma: no cover
            raise SolveError("Pyomo is not installed.")
        opt = pyo.SolverFactory(solver)
        if opt is None or not opt.available(False):
            raise SolveError(f"Pyomo solver {solver!r} is not available.")
        result = opt.solve(model, tee=tee)
        status = result.solver.status
        termination = result.solver.termination_condition
        if status != pyo.SolverStatus.ok or termination != pyo.TerminationCondition.optimal:
            raise SolveError(f"Solver did not report optimality. status={status}, termination={termination}")
        return result


class RecourseEvaluator(_SolverMixin):
    """Evaluate Q(x, xi_s) = min_y d^T y s.t. T x + W y <= H xi_s, y >= 0."""

    def __init__(self, instance: TSSPInstance):
        self.instance = instance

    def build_pyomo_model(self, x: Sequence[float], xi: Sequence[float]) -> Any:
        if pyo is None:
            raise SolveError("Pyomo is not installed.")
        inst = self.instance
        x_vec = np.asarray(x, dtype=float)
        xi_vec = np.asarray(xi, dtype=float)
        rhs = inst.H @ xi_vec - inst.T @ x_vec

        m = pyo.ConcreteModel(name=f"recourse_{inst.name}")
        m.J = pyo.RangeSet(0, inst.num_second_stage_vars - 1)
        m.R = pyo.RangeSet(0, inst.W.shape[0] - 1)
        m.y = pyo.Var(m.J, domain=pyo.NonNegativeReals)
        m.obj = pyo.Objective(expr=sum(float(inst.d[j]) * m.y[j] for j in m.J), sense=pyo.minimize)
        m.recourse = pyo.Constraint(
            m.R,
            rule=lambda model, r: sum(float(inst.W[r, j]) * model.y[j] for j in model.J) <= float(rhs[r]),
        )
        return m

    def evaluate(
        self,
        x: Sequence[float],
        xi: Sequence[float],
        solver: str = "gurobi",
        backend: str = "auto",
        tee: bool = False,
    ) -> Dict[str, Any]:
        chosen = self._choose_backend(backend)
        if chosen == "pyomo":
            try:
                return self._evaluate_pyomo(x, xi, solver=solver, tee=tee)
            except Exception:
                if backend == "auto" and linprog is not None:
                    return self._evaluate_scipy(x, xi)
                raise
        return self._evaluate_scipy(x, xi)

    def _evaluate_pyomo(self, x: Sequence[float], xi: Sequence[float], solver: str, tee: bool) -> Dict[str, Any]:
        if pyo is None:  # pragma: no cover
            raise SolveError("Pyomo is not installed.")
        m = self.build_pyomo_model(x, xi)
        self._solve_pyomo_model(m, solver=solver, tee=tee)
        y = np.asarray([pyo.value(m.y[j]) for j in m.J], dtype=float)
        return {"objective": _dot(self.instance.d, y), "y": y.tolist(), "backend": "pyomo", "solver": solver}

    def _evaluate_scipy(self, x: Sequence[float], xi: Sequence[float]) -> Dict[str, Any]:
        if linprog is None:  # pragma: no cover
            raise SolveError("scipy.optimize.linprog is not installed.")
        inst = self.instance
        x_vec = np.asarray(x, dtype=float)
        xi_vec = np.asarray(xi, dtype=float)
        rhs = inst.H @ xi_vec - inst.T @ x_vec
        res = linprog(
            c=inst.d,
            A_ub=inst.W,
            b_ub=rhs,
            bounds=[(0.0, None)] * inst.num_second_stage_vars,
            method="highs",
        )
        self._check_linprog_result(res, "Recourse evaluation")
        y = np.asarray(res.x, dtype=float)
        return {"objective": float(res.fun), "y": y.tolist(), "backend": "scipy", "solver": "highs"}


class StochasticProgramDE(_SolverMixin):
    """Deterministic equivalent of the stochastic/RP model."""

    def __init__(self, instance_name: str, base_dir: Optional[str | Path] = None):
        self.instance = TSSPInstance.load(instance_name, base_dir=base_dir)

    def build_pyomo_model(self) -> Any:
        if pyo is None:
            raise SolveError("Pyomo is not installed.")
        inst = self.instance
        m = pyo.ConcreteModel(name=f"stochastic_de_{inst.name}")
        m.I = pyo.RangeSet(0, inst.num_first_stage_vars - 1)
        m.J = pyo.RangeSet(0, inst.num_second_stage_vars - 1)
        m.S = pyo.RangeSet(0, inst.num_scenarios - 1)
        m.Arows = pyo.RangeSet(0, inst.A.shape[0] - 1)
        m.R = pyo.RangeSet(0, inst.T.shape[0] - 1)

        m.x = pyo.Var(m.I, domain=pyo.NonNegativeReals)
        m.y = pyo.Var(m.S, m.J, domain=pyo.NonNegativeReals)

        m.obj = pyo.Objective(
            expr=sum(float(inst.c[i]) * m.x[i] for i in m.I)
            + sum(
                float(inst.probabilities[s]) * float(inst.d[j]) * m.y[s, j]
                for s in m.S
                for j in m.J
            ),
            sense=pyo.minimize,
        )
        m.first_stage = pyo.Constraint(
            m.Arows,
            rule=lambda model, r: sum(float(inst.A[r, i]) * model.x[i] for i in model.I) >= float(inst.b[r]),
        )
        m.recourse = pyo.Constraint(
            m.S,
            m.R,
            rule=lambda model, s, r: sum(float(inst.T[r, i]) * model.x[i] for i in model.I)
            + sum(float(inst.W[r, j]) * model.y[s, j] for j in model.J)
            <= float((inst.H @ inst.xi[s])[r]),
        )
        return m

    def solve(self, solver: str = "gurobi", backend: str = "auto", tee: bool = False) -> Dict[str, Any]:
        chosen = self._choose_backend(backend)
        if chosen == "pyomo":
            try:
                return self._solve_pyomo(solver=solver, tee=tee)
            except Exception:
                if backend == "auto" and linprog is not None:
                    return self._solve_scipy()
                raise
        return self._solve_scipy()

    def _solve_pyomo(self, solver: str, tee: bool) -> Dict[str, Any]:
        if pyo is None:  # pragma: no cover
            raise SolveError("Pyomo is not installed.")
        inst = self.instance
        m = self.build_pyomo_model()
        self._solve_pyomo_model(m, solver=solver, tee=tee)
        x = np.asarray([pyo.value(m.x[i]) for i in m.I], dtype=float)
        y_by_s = {
            _scenario_label(s): np.asarray([pyo.value(m.y[s, j]) for j in m.J], dtype=float).tolist()
            for s in range(inst.num_scenarios)
        }
        return self._summarize_solution(x=x, y_by_s=y_by_s, backend="pyomo", solver=solver)

    def _solve_scipy(self) -> Dict[str, Any]:
        if linprog is None:  # pragma: no cover
            raise SolveError("scipy.optimize.linprog is not installed.")
        inst = self.instance
        nx, ny, ns = inst.num_first_stage_vars, inst.num_second_stage_vars, inst.num_scenarios
        obj = np.concatenate([inst.c] + [float(inst.probabilities[s]) * inst.d for s in range(ns)])

        lhs_blocks: List[np.ndarray] = []
        rhs_blocks: List[np.ndarray] = []

        first = np.zeros((inst.A.shape[0], nx + ns * ny))
        first[:, :nx] = -inst.A
        lhs_blocks.append(first)
        rhs_blocks.append(-inst.b)

        for s, xi_s in enumerate(inst.xi):
            block = np.zeros((inst.T.shape[0], nx + ns * ny))
            block[:, :nx] = inst.T
            block[:, nx + s * ny : nx + (s + 1) * ny] = inst.W
            lhs_blocks.append(block)
            rhs_blocks.append(inst.H @ xi_s)

        res = linprog(
            c=obj,
            A_ub=np.vstack(lhs_blocks),
            b_ub=np.concatenate(rhs_blocks),
            bounds=[(0.0, None)] * (nx + ns * ny),
            method="highs",
        )
        self._check_linprog_result(res, "Stochastic deterministic equivalent")
        x = np.asarray(res.x[:nx], dtype=float)
        y_by_s = {
            _scenario_label(s): np.asarray(res.x[nx + s * ny : nx + (s + 1) * ny], dtype=float).tolist()
            for s in range(ns)
        }
        return self._summarize_solution(x=x, y_by_s=y_by_s, backend="scipy", solver="highs")

    def _summarize_solution(
        self,
        x: np.ndarray,
        y_by_s: Mapping[str, Sequence[float]],
        backend: str,
        solver: str,
    ) -> Dict[str, Any]:
        inst = self.instance
        first_stage_cost = _dot(inst.c, x)
        scenario_second_stage_costs = {
            name: _dot(inst.d, np.asarray(y, dtype=float)) for name, y in y_by_s.items()
        }
        weighted_second_stage_costs = {
            _scenario_label(s): float(inst.probabilities[s]) * scenario_second_stage_costs[_scenario_label(s)]
            for s in range(inst.num_scenarios)
        }
        expected_second_stage_cost = float(sum(weighted_second_stage_costs.values()))
        return {
            "label": "RP/stochastic optimum",
            "instance_name": inst.name,
            "backend": backend,
            "solver": solver,
            "constraint_form_used": "T x + W y_s <= H xi_s",
            "objective": first_stage_cost + expected_second_stage_cost,
            "first_stage_cost": first_stage_cost,
            "expected_second_stage_cost": expected_second_stage_cost,
            "total_second_stage_cost": expected_second_stage_cost,
            "scenario_second_stage_costs": scenario_second_stage_costs,
            "weighted_second_stage_costs": weighted_second_stage_costs,
            "x_star": x.tolist(),
            "x_vector": x.tolist(),
            "y_s": {name: list(map(float, y)) for name, y in y_by_s.items()},
            "probabilities": inst.probabilities.tolist(),
            "scenario_names": inst.scenario_names,
            "consistency_report": inst.consistency_report(),
        }

    def solve_and_save(self, solver: str = "gurobi", backend: str = "auto", tee: bool = False) -> Dict[str, Any]:
        result = self.solve(solver=solver, backend=backend, tee=tee)
        evaluator = RecourseEvaluator(self.instance)
        recourse_evaluations: Dict[str, Any] = {}
        for s, xi_s in enumerate(self.instance.xi):
            name = _scenario_label(s)
            rec = evaluator.evaluate(result["x_star"], xi_s, solver=solver, backend=backend, tee=tee)
            recourse_evaluations[name] = {
                "Q_value": rec["objective"],
                "y": rec["y"],
                "backend": rec["backend"],
                "solver": rec["solver"],
            }
        result["recourse_evaluations"] = recourse_evaluations
        # output_dir = self.instance.instance_dir / "output_data"
        output_dir = self.instance.instance_dir / "output_data"
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(output_dir / "stochastic_results.json", result)
        # _write_json(self.instance.instance_dir / "output_data" / "stochastic_results.json", result)
        return result


class ExpectedValueProgramDE(_SolverMixin):
    """Deterministic equivalent of the expected-value problem with one averaged scenario."""

    def __init__(self, instance_name: str, base_dir: Optional[str | Path] = None):
        self.instance = TSSPInstance.load(instance_name, base_dir=base_dir)

    def build_pyomo_model(self) -> Any:
        if pyo is None:
            raise SolveError("Pyomo is not installed.")
        inst = self.instance
        xi_bar = inst.expected_xi
        m = pyo.ConcreteModel(name=f"expected_value_de_{inst.name}")
        m.I = pyo.RangeSet(0, inst.num_first_stage_vars - 1)
        m.J = pyo.RangeSet(0, inst.num_second_stage_vars - 1)
        m.Arows = pyo.RangeSet(0, inst.A.shape[0] - 1)
        m.R = pyo.RangeSet(0, inst.T.shape[0] - 1)

        m.x = pyo.Var(m.I, domain=pyo.NonNegativeReals)
        m.y = pyo.Var(m.J, domain=pyo.NonNegativeReals)

        m.obj = pyo.Objective(
            expr=sum(float(inst.c[i]) * m.x[i] for i in m.I)
            + sum(float(inst.d[j]) * m.y[j] for j in m.J),
            sense=pyo.minimize,
        )
        m.first_stage = pyo.Constraint(
            m.Arows,
            rule=lambda model, r: sum(float(inst.A[r, i]) * model.x[i] for i in model.I) >= float(inst.b[r]),
        )
        m.recourse = pyo.Constraint(
            m.R,
            rule=lambda model, r: sum(float(inst.T[r, i]) * model.x[i] for i in model.I)
            + sum(float(inst.W[r, j]) * model.y[j] for j in model.J)
            <= float((inst.H @ xi_bar)[r]),
        )
        return m

    def solve(self, solver: str = "gurobi", backend: str = "auto", tee: bool = False) -> Dict[str, Any]:
        chosen = self._choose_backend(backend)
        if chosen == "pyomo":
            try:
                return self._solve_pyomo(solver=solver, tee=tee)
            except Exception:
                if backend == "auto" and linprog is not None:
                    return self._solve_scipy()
                raise
        return self._solve_scipy()

    def _solve_pyomo(self, solver: str, tee: bool) -> Dict[str, Any]:
        if pyo is None:  # pragma: no cover
            raise SolveError("Pyomo is not installed.")
        inst = self.instance
        m = self.build_pyomo_model()
        self._solve_pyomo_model(m, solver=solver, tee=tee)
        x = np.asarray([pyo.value(m.x[i]) for i in m.I], dtype=float)
        y = np.asarray([pyo.value(m.y[j]) for j in m.J], dtype=float)
        return self._summarize_solution(x=x, y=y, backend="pyomo", solver=solver)

    def _solve_scipy(self) -> Dict[str, Any]:
        if linprog is None:  # pragma: no cover
            raise SolveError("scipy.optimize.linprog is not installed.")
        inst = self.instance
        nx, ny = inst.num_first_stage_vars, inst.num_second_stage_vars
        xi_bar = inst.expected_xi
        obj = np.concatenate([inst.c, inst.d])

        first = np.zeros((inst.A.shape[0], nx + ny))
        first[:, :nx] = -inst.A
        rec = np.zeros((inst.T.shape[0], nx + ny))
        rec[:, :nx] = inst.T
        rec[:, nx:] = inst.W

        res = linprog(
            c=obj,
            A_ub=np.vstack([first, rec]),
            b_ub=np.concatenate([-inst.b, inst.H @ xi_bar]),
            bounds=[(0.0, None)] * (nx + ny),
            method="highs",
        )
        self._check_linprog_result(res, "Expected-value deterministic equivalent")
        x = np.asarray(res.x[:nx], dtype=float)
        y = np.asarray(res.x[nx:], dtype=float)
        return self._summarize_solution(x=x, y=y, backend="scipy", solver="highs")

    def _summarize_solution(
        self,
        x: np.ndarray,
        y: np.ndarray,
        backend: str,
        solver: str,
    ) -> Dict[str, Any]:
        inst = self.instance
        first_stage_cost = _dot(inst.c, x)
        q_at_expected = _dot(inst.d, y)
        ev_objective = first_stage_cost + q_at_expected
        return {
            "label": "EV solution under expected scenario",
            "instance_name": inst.name,
            "backend": backend,
            "solver": solver,
            "constraint_form_used": "T x + W y <= H xi_bar",
            "objective": ev_objective,
            "EV_objective": ev_objective,
            "first_stage_cost": first_stage_cost,
            "expected_scenario_second_stage_cost": q_at_expected,
            "total_second_stage_cost_under_expected_scenario": q_at_expected,
            "scenario_second_stage_costs": {"expected_value_scenario": q_at_expected},
            "x_EV": x.tolist(),
            "x_vector": x.tolist(),
            "y": y.tolist(),
            "expected_xi": inst.expected_xi.tolist(),
            "probabilities": inst.probabilities.tolist(),
            "scenario_names": inst.scenario_names,
            "consistency_report": inst.consistency_report(),
        }

    def solve_and_save(self, solver: str = "gurobi", backend: str = "auto", tee: bool = False) -> Dict[str, Any]:
        result = self.solve(solver=solver, backend=backend, tee=tee)
        evaluator = RecourseEvaluator(self.instance)
        recourse_evaluations: Dict[str, Any] = {}
        scenario_second_stage_costs: Dict[str, float] = {}
        weighted_second_stage_costs: Dict[str, float] = {}
        for s, xi_s in enumerate(self.instance.xi):
            name = _scenario_label(s)
            rec = evaluator.evaluate(result["x_EV"], xi_s, solver=solver, backend=backend, tee=tee)
            q_val = float(rec["objective"])
            recourse_evaluations[name] = {
                "Q_value": q_val,
                "y": rec["y"],
                "backend": rec["backend"],
                "solver": rec["solver"],
            }
            scenario_second_stage_costs[name] = q_val
            weighted_second_stage_costs[name] = float(self.instance.probabilities[s]) * q_val

        eev_second_stage = float(sum(weighted_second_stage_costs.values()))
        eev_objective = float(result["first_stage_cost"] + eev_second_stage)
        result["recourse_evaluations"] = recourse_evaluations
        result["EEV_objective"] = eev_objective
        result["EEV"] = eev_objective
        result["EEV_first_stage_cost"] = result["first_stage_cost"]
        result["EEV_expected_second_stage_cost"] = eev_second_stage
        result["EEV_total_second_stage_cost"] = eev_second_stage
        result["EEV_scenario_second_stage_costs"] = scenario_second_stage_costs
        result["EEV_weighted_second_stage_costs"] = weighted_second_stage_costs
        output_dir = self.instance.instance_dir / "output_data"
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(output_dir / "expectedvalue_results.json", result)
        return result

def create_cost_gap_diagnostics(instance_name: str, base_dir: Optional[str | Path] = None) -> Dict[str, Any]:
    """Create cost_gap_diagnostics.json from saved JSONs only; no model re-solve."""
    instance_dir = _resolve_instance_dir(instance_name, base_dir)
    stochastic_path = instance_dir / "output_data" / "stochastic_results.json"
    ev_path = instance_dir / "output_data" / "expectedvalue_results.json"
    if not stochastic_path.exists() or not ev_path.exists():
        raise FileNotFoundError(
            "Run stochastic and expected-value solves first. Missing one of: "
            f"{stochastic_path}, {ev_path}"
        )
    stochastic = _read_json(stochastic_path)
    ev = _read_json(ev_path)

    probabilities = stochastic["probabilities"]
    scenario_names = stochastic["scenario_names"]
    cTx_star = float(stochastic["first_stage_cost"])
    cTx_ev = float(ev["first_stage_cost"])

    scenario_cost_gaps: Dict[str, Any] = {}
    weighted_sum = 0.0
    for idx, name in enumerate(scenario_names):
        q_star = float(stochastic["recourse_evaluations"][name]["Q_value"])
        q_ev = float(ev["recourse_evaluations"][name]["Q_value"])
        gap = (cTx_ev + q_ev) - (cTx_star + q_star)
        weighted_gap = float(probabilities[idx]) * gap
        weighted_sum += weighted_gap
        scenario_cost_gaps[name] = {
            "probability": float(probabilities[idx]),
            "Q_x_star_xi": q_star,
            "Q_x_EV_xi": q_ev,
            "cost_with_x_star": cTx_star + q_star,
            "cost_with_x_EV": cTx_ev + q_ev,
            "cost_gap": gap,
            "weighted_cost_gap": weighted_gap,
        }

    rp = float(stochastic["objective"])
    ev_objective = float(ev["EV_objective"])
    eev = float(ev["EEV_objective"])
    vss = eev - rp

    payload = {
        "instance_name": instance_name,
        "definition": {
            "cost_gap": "Delta(xi) = (c^T x_EV + Q(x_EV, xi)) - (c^T x_star + Q(x_star, xi)).",
            "sign_convention": "Positive means x_EV is more expensive than x_star for that realized scenario; negative means x_EV is cheaper.",
        },
        "metrics": {
            "RP": rp,
            "EV_objective": ev_objective,
            "EEV": eev,
            "VSS": vss,
            "expected_cost_gap": weighted_sum,
            "first_stage_cost_difference_cTxEV_minus_cTxStar": cTx_ev - cTx_star,
        },
        "scenario_cost_gaps": scenario_cost_gaps,
        "source_files": {
            "stochastic_results": str(stochastic_path.name),
            "expectedvalue_results": str(ev_path.name),
        },
        "note": "This file was computed only from saved result JSONs; no optimization model was re-solved.",
    }
    output_dir = instance_dir / "output_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "cost_gap_diagnostics.json", payload)
    # _write_json(instance_dir / "cost_gap_diagnostics.json", payload)
    return payload


def run_full_workflow(
    instance_name: str,
    base_dir: Optional[str | Path] = None,
    solver: str = "gurobi",
    backend: str = "auto",
    tee: bool = False,
) -> Dict[str, Any]:
    """Run validation, stochastic solve, EV solve, and cost-gap diagnostics."""
    root = _as_path(base_dir)
    instance = TSSPInstance.load(instance_name, base_dir=root)
    stochastic = StochasticProgramDE(instance_name, base_dir=root).solve_and_save(
        solver=solver, backend=backend, tee=tee
    )
    expected_value = ExpectedValueProgramDE(instance_name, base_dir=root).solve_and_save(
        solver=solver, backend=backend, tee=tee
    )
    diagnostics = create_cost_gap_diagnostics(instance_name, base_dir=root)
    return {
        "instance_name": instance_name,
        "instance_dir": str(instance.instance_dir),
        "consistency_report": instance.consistency_report(),
        "stochastic_results": stochastic,
        "expectedvalue_results": expected_value,
        "cost_gap_diagnostics": diagnostics,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run the general-form TSSP workflow.")
    parser.add_argument("instance_name", help="Instance folder name, e.g., lands_instance")
    parser.add_argument("--base-dir", default=None, help="Folder containing instance folders. Defaults to this file's folder.")
    parser.add_argument("--solver", default="gurobi", help="Pyomo solver name. Default: gurobi")
    parser.add_argument(
        "--backend",
        choices=["auto", "pyomo", "scipy"],
        default="auto",
        help="auto uses Pyomo when available and scipy otherwise; pyomo requires the requested solver.",
    )
    parser.add_argument("--tee", action="store_true", help="Show Pyomo solver output.")
    args = parser.parse_args(argv)

    summary = run_full_workflow(
        instance_name=args.instance_name,
        base_dir=args.base_dir,
        solver=args.solver,
        backend=args.backend,
        tee=args.tee,
    )
    print(json.dumps(_json_ready(summary["cost_gap_diagnostics"]["metrics"]), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
