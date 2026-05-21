"""Pyomo deterministic equivalent for the LandS two-stage stochastic LP.

Run:
    python lands_model.py

This script solves the stochastic deterministic equivalent, solves the
expected-value (EV) problem, evaluates the expected result of using the EV
solution (EEV), computes the value of the stochastic solution (VSS), and saves a
JSON diagnostics file.

"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, Optional

import pyomo.environ as pyo


NumberDict = Dict[str, float]


def _clean_float(value: float, tol: float = 1e-10) -> float:
    """Return a plain float, replacing tiny numerical noise by zero."""
    value = float(value)
    if abs(value) <= tol:
        return 0.0
    return value


class LandSDeterministicEquivalent:
    """Build and solve a deterministic equivalent of the LandS model.

    The implementation is data-driven. The sets of plants/technologies I,
    demand modes J, and scenarios S are read from JSON rather than hard-coded in
    the Pyomo equations.
    """

    def __init__(self, data: Dict[str, Any]):
        self.data = data
        self.model: Optional[pyo.ConcreteModel] = None
        self.results = None

    @classmethod
    def from_json(cls, path: str | Path) -> "LandSDeterministicEquivalent":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data)

    @property
    def I(self):
        return list(self.data["sets"]["I"])

    @property
    def J(self):
        return list(self.data["sets"]["J"])

    @property
    def S(self):
        return list(self.data["sets"]["S"])

    def build_model(self, fixed_x: Optional[NumberDict] = None) -> pyo.ConcreteModel:
        """Build the Pyomo model.

        Args:
            fixed_x: Optional first-stage capacity vector. If supplied, the x
                variables are fixed to these values. This is used to compute EEV,
                i.e., the expected cost of using the EV first-stage solution in
                the original stochastic problem.
        """
        data = self.data
        I, J, S = self.I, self.J, self.S
        p = data["probability"]
        c = data["investment_cost"]
        f = data["operating_cost"]
        d = data["demand"]
        min_total_capacity = float(data["min_total_capacity"])
        budget = float(data["budget"])

        m = pyo.ConcreteModel(name=data.get("name", "LandS"))

        # Sets
        m.I = pyo.Set(initialize=I, ordered=True)
        m.J = pyo.Set(initialize=J, ordered=True)
        m.S = pyo.Set(initialize=S, ordered=True)

        # First-stage variables: capacities, chosen before demand is known.
        m.x = pyo.Var(m.I, domain=pyo.NonNegativeReals)

        # Second-stage variables: scenario-dependent dispatch/allocation.
        m.y = pyo.Var(m.S, m.I, m.J, domain=pyo.NonNegativeReals)

        if fixed_x is not None:
            missing = sorted(set(I) - set(fixed_x))
            extra = sorted(set(fixed_x) - set(I))
            if missing or extra:
                raise ValueError(
                    f"fixed_x keys must match I. Missing={missing}, extra={extra}"
                )
            for i in I:
                m.x[i].fix(float(fixed_x[i]))

        # Objective: investment cost + expected operating/dispatch cost.
        m.objective = pyo.Objective(
            expr=sum(float(c[i]) * m.x[i] for i in m.I)
            + sum(
                float(p[s])
                * sum(float(f[i][j]) * m.y[s, i, j] for i in m.I for j in m.J)
                for s in m.S
            ),
            sense=pyo.minimize,
        )

        # First-stage constraints.
        m.minimum_total_capacity = pyo.Constraint(
            expr=sum(m.x[i] for i in m.I) >= min_total_capacity
        )
        m.budget = pyo.Constraint(expr=sum(float(c[i]) * m.x[i] for i in m.I) <= budget)

        # Second-stage constraints, repeated for every scenario.
        def capacity_rule(model, s, i):
            return sum(model.y[s, i, j] for j in model.J) <= model.x[i]

        m.capacity_link = pyo.Constraint(m.S, m.I, rule=capacity_rule)

        def demand_rule(model, s, j):
            return sum(model.y[s, i, j] for i in model.I) >= float(d[s][j])

        m.demand_satisfaction = pyo.Constraint(m.S, m.J, rule=demand_rule)

        self.model = m
        return m

    def solve(self, solver_name: str = "gurobi", tee: bool = False):
        if self.model is None:
            self.build_model()

        solver = pyo.SolverFactory(solver_name)
        if not solver.available(False):
            raise RuntimeError(
                f"Solver '{solver_name}' is not available. For this LP, install a high-performance LP solver like Gurobi or HiGHS. For example, with Gurobi: "
                "pip install gurobipy"
            )

        self.results = solver.solve(self.model, tee=tee)
        term = self.results.solver.termination_condition
        if term != pyo.TerminationCondition.optimal:
            raise RuntimeError(f"Solver did not return optimal termination. Got: {term}")
        return self.results

    def objective_value(self) -> float:
        if self.model is None:
            raise RuntimeError("Build and solve the model first.")
        return _clean_float(pyo.value(self.model.objective))

    def x_values(self) -> NumberDict:
        if self.model is None:
            raise RuntimeError("Build and solve the model first.")
        return {i: _clean_float(pyo.value(self.model.x[i])) for i in self.model.I}

    def y_values(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        if self.model is None:
            raise RuntimeError("Build and solve the model first.")
        return {
            s: {
                i: {
                    j: _clean_float(pyo.value(self.model.y[s, i, j]))
                    for j in self.model.J
                }
                for i in self.model.I
            }
            for s in self.model.S
        }

    def first_stage_cost(self) -> float:
        x = self.x_values()
        c = self.data["investment_cost"]
        return _clean_float(sum(float(c[i]) * x[i] for i in self.I))

    def scenario_second_stage_costs(self) -> NumberDict:
        y = self.y_values()
        f = self.data["operating_cost"]
        costs = {}
        for s in self.S:
            costs[s] = _clean_float(
                sum(float(f[i][j]) * y[s][i][j] for i in self.I for j in self.J)
            )
        return costs

    def expected_second_stage_cost(self) -> float:
        p = self.data["probability"]
        q = self.scenario_second_stage_costs()
        return _clean_float(sum(float(p[s]) * q[s] for s in self.S))

    def expected_demand(self) -> NumberDict:
        p = self.data["probability"]
        d = self.data["demand"]
        return {
            j: _clean_float(sum(float(p[s]) * float(d[s][j]) for s in self.S))
            for j in self.J
        }

    def expected_value_data(
        self, scenario_name: str = "expected_value_scenario"
    ) -> Dict[str, Any]:
        """Return data for the EV problem with demand replaced by E[d]."""
        ev_data = copy.deepcopy(self.data)
        ev_data["name"] = self.data.get("name", "LandS") + " - expected value problem"
        ev_data["sets"]["S"] = [scenario_name]
        ev_data["probability"] = {scenario_name: 1.0}
        ev_data["demand"] = {scenario_name: self.expected_demand()}
        return ev_data

    def solve_expected_value_problem(
        self, solver_name: str = "gurobi", tee: bool = False
    ) -> "LandSDeterministicEquivalent":
        """Solve the deterministic EV problem obtained by using expected demand."""
        ev_problem = LandSDeterministicEquivalent(self.expected_value_data())
        ev_problem.solve(solver_name=solver_name, tee=tee)
        return ev_problem

    def evaluate_first_stage_solution(
        self,
        fixed_x: NumberDict,
        solver_name: str = "gurobi",
        tee: bool = False,
    ) -> "LandSDeterministicEquivalent":
        """Evaluate a fixed first-stage x in the original stochastic problem."""
        evaluation_problem = LandSDeterministicEquivalent(copy.deepcopy(self.data))
        evaluation_problem.build_model(fixed_x=fixed_x)
        evaluation_problem.solve(solver_name=solver_name, tee=tee)
        return evaluation_problem

    def scenario_data(self, scenario_name: str) -> Dict[str, Any]:
        """Return data restricted to one original demand scenario.

        This is useful for explicitly evaluating the recourse function
        Q(x, xi) scenario by scenario. The single scenario is assigned
        probability 1.0; the original scenario probability is stored separately
        in cost-gap diagnostics.
        """
        if scenario_name not in self.S:
            raise ValueError(f"Unknown scenario {scenario_name!r}. Available scenarios: {self.S}")

        scenario_data = copy.deepcopy(self.data)
        scenario_data["name"] = (
            self.data.get("name", "LandS") + f" - recourse evaluation for {scenario_name}"
        )
        scenario_data["sets"]["S"] = [scenario_name]
        scenario_data["probability"] = {scenario_name: 1.0}
        scenario_data["demand"] = {scenario_name: copy.deepcopy(self.data["demand"][scenario_name])}
        return scenario_data

    def evaluate_recourse_for_scenario(
        self,
        fixed_x: NumberDict,
        scenario_name: str,
        solver_name: str = "gurobi",
        tee: bool = False,
    ) -> "LandSDeterministicEquivalent":
        """Solve the recourse LP Q(x, xi) for one fixed x and scenario xi.

        The returned model has one scenario with probability 1.0 and the
        first-stage vector fixed to ``fixed_x``. Its
        ``scenario_second_stage_costs()[scenario_name]`` entry is Q(x, xi).
        """
        recourse_problem = LandSDeterministicEquivalent(self.scenario_data(scenario_name))
        recourse_problem.build_model(fixed_x=fixed_x)
        recourse_problem.solve(solver_name=solver_name, tee=tee)
        return recourse_problem

    def _x_vector(self) -> list[Dict[str, Any]]:
        x = self.x_values()
        return [{"i": i, "value": x[i]} for i in self.I]

    def _y_vector(self) -> list[Dict[str, Any]]:
        y = self.y_values()
        return [
            {"s": s, "i": i, "j": j, "value": y[s][i][j]}
            for s in self.S
            for i in self.I
            for j in self.J
        ]

    def feasibility_diagnostics(self) -> Dict[str, Any]:
        """Return slack/surplus diagnostics for the solved model."""
        x = self.x_values()
        y = self.y_values()
        c = self.data["investment_cost"]
        d = self.data["demand"]
        min_total_capacity = float(self.data["min_total_capacity"])
        budget = float(self.data["budget"])

        total_capacity = _clean_float(sum(x[i] for i in self.I))
        budget_used = _clean_float(sum(float(c[i]) * x[i] for i in self.I))

        scenario_diagnostics = {}
        for s in self.S:
            capacity_by_i = {}
            for i in self.I:
                used = _clean_float(sum(y[s][i][j] for j in self.J))
                capacity_by_i[i] = {
                    "used": used,
                    "capacity": x[i],
                    "slack": _clean_float(x[i] - used),
                }

            demand_by_j = {}
            for j in self.J:
                supplied = _clean_float(sum(y[s][i][j] for i in self.I))
                rhs = float(d[s][j])
                demand_by_j[j] = {
                    "supplied": supplied,
                    "demand": rhs,
                    "surplus": _clean_float(supplied - rhs),
                }

            scenario_diagnostics[s] = {
                "capacity_by_i": capacity_by_i,
                "demand_by_j": demand_by_j,
            }

        return {
            "minimum_total_capacity": {
                "lhs_total_capacity": total_capacity,
                "rhs_minimum_capacity": min_total_capacity,
                "slack": _clean_float(total_capacity - min_total_capacity),
            },
            "budget": {
                "lhs_budget_used": budget_used,
                "rhs_budget": budget,
                "slack": _clean_float(budget - budget_used),
            },
            "scenario_diagnostics": scenario_diagnostics,
        }

    def solution_report(self, label: str) -> Dict[str, Any]:
        """Return a JSON-serializable report for a solved model."""
        return {
            "label": label,
            "objective": self.objective_value(),
            "first_stage_cost": self.first_stage_cost(),
            "expected_second_stage_cost": self.expected_second_stage_cost(),
            "scenario_second_stage_costs": self.scenario_second_stage_costs(),
            "first_stage": {
                "x_by_i": self.x_values(),
                "x_vector_order": self.I,
                "x_vector": self._x_vector(),
            },
            "second_stage": {
                "y_by_s_i_j": self.y_values(),
                "y_vector_order": ["s", "i", "j"],
                "y_vector": self._y_vector(),
            },
            "feasibility": self.feasibility_diagnostics(),
        }

    def compute_diagnostics(
        self, solver_name: str = "gurobi", tee: bool = False
    ) -> Dict[str, Any]:
        """Solve RP, EV, and EEV models and return diagnostics.

        Definitions for this minimization problem:
            RP  = optimal stochastic-program objective.
            EV  = deterministic problem where uncertain demand is replaced by E[d].
            EEV = expected objective obtained by fixing x = x_EV in the original
                  stochastic problem and re-optimizing only recourse decisions.
            VSS = EEV - RP.
        """
        # RP: full stochastic deterministic equivalent.
        rp_problem = LandSDeterministicEquivalent(copy.deepcopy(self.data))
        rp_problem.solve(solver_name=solver_name, tee=tee)

        # EV: deterministic problem using expected demand.
        ev_problem = rp_problem.solve_expected_value_problem(
            solver_name=solver_name, tee=tee
        )
        x_ev = ev_problem.x_values()

        # EEV: evaluate the EV first-stage decision in the original stochastic model.
        eev_problem = rp_problem.evaluate_first_stage_solution(
            x_ev, solver_name=solver_name, tee=tee
        )

        rp_objective = rp_problem.objective_value()
        ev_objective = ev_problem.objective_value()
        eev_objective = eev_problem.objective_value()
        vss = _clean_float(eev_objective - rp_objective)

        return {
            "model_name": self.data.get("name", "LandS"),
            "sets": copy.deepcopy(self.data["sets"]),
            "expected_demand": rp_problem.expected_demand(),
            "definitions": {
                "RP": "Optimal objective of the original stochastic program.",
                "EV": "Optimal objective of the deterministic problem with demand replaced by expected demand.",
                "EEV": "Expected objective of using the EV first-stage solution in the original stochastic problem.",
                "VSS": "Value of the stochastic solution for minimization, computed as EEV - RP.",
            },
            "metrics": {
                "RP": rp_objective,
                "EV_objective": ev_objective,
                "EEV": eev_objective,
                "VSS": vss,
            },
            "stochastic_solution": rp_problem.solution_report("RP/stochastic optimum"),
            "expected_value_solution": ev_problem.solution_report(
                "EV solution under expected demand"
            ),
            "eev_evaluation": eev_problem.solution_report(
                "EEV evaluation with x fixed to x_EV"
            ),
        }

    def compute_cost_gap_diagnostics(
        self, solver_name: str = "gurobi", tee: bool = False
    ) -> Dict[str, Any]:
        """Compute scenario-wise cost gaps between x_EV and x_star.

        For each original scenario xi, this evaluates

            Delta(xi) = [c^T x_EV + Q(x_EV, xi)]
                        - [c^T x_star + Q(x_star, xi)].

        A positive gap means that the EV first-stage solution is more expensive
        than the stochastic first-stage solution for that realized scenario. A
        negative gap means the EV first-stage solution is cheaper for that
        realized scenario.
        """
        # Solve the stochastic problem to obtain x_star.
        rp_problem = LandSDeterministicEquivalent(copy.deepcopy(self.data))
        rp_problem.solve(solver_name=solver_name, tee=tee)
        x_star = rp_problem.x_values()
        first_stage_cost_star = rp_problem.first_stage_cost()

        # Solve the expected-value problem to obtain x_EV.
        ev_problem = rp_problem.solve_expected_value_problem(
            solver_name=solver_name, tee=tee
        )
        x_ev = ev_problem.x_values()
        first_stage_cost_ev = ev_problem.first_stage_cost()

        scenarios: Dict[str, Any] = {}
        expected_cost_gap = 0.0

        for s in rp_problem.S:
            probability = float(rp_problem.data["probability"][s])

            # Explicit one-scenario recourse evaluations for Q(x_star, xi)
            # and Q(x_EV, xi).
            star_recourse = rp_problem.evaluate_recourse_for_scenario(
                x_star, s, solver_name=solver_name, tee=tee
            )
            ev_recourse = rp_problem.evaluate_recourse_for_scenario(
                x_ev, s, solver_name=solver_name, tee=tee
            )

            q_star = star_recourse.scenario_second_stage_costs()[s]
            q_ev = ev_recourse.scenario_second_stage_costs()[s]
            total_star = _clean_float(first_stage_cost_star + q_star)
            total_ev = _clean_float(first_stage_cost_ev + q_ev)
            cost_gap = _clean_float(total_ev - total_star)
            weighted_cost_gap = _clean_float(probability * cost_gap)
            expected_cost_gap += weighted_cost_gap

            scenarios[s] = {
                "probability": probability,
                "demand": copy.deepcopy(rp_problem.data["demand"][s]),
                "x_star_evaluation": {
                    "first_stage_cost": first_stage_cost_star,
                    "recourse_cost_Q_x_xi": q_star,
                    "total_scenario_cost": total_star,
                    "y_by_i_j": star_recourse.y_values()[s],
                    "feasibility": star_recourse.feasibility_diagnostics()[
                        "scenario_diagnostics"
                    ][s],
                },
                "x_EV_evaluation": {
                    "first_stage_cost": first_stage_cost_ev,
                    "recourse_cost_Q_x_xi": q_ev,
                    "total_scenario_cost": total_ev,
                    "y_by_i_j": ev_recourse.y_values()[s],
                    "feasibility": ev_recourse.feasibility_diagnostics()[
                        "scenario_diagnostics"
                    ][s],
                },
                "cost_gap": cost_gap,
                "weighted_cost_gap": weighted_cost_gap,
            }

        expected_cost_gap = _clean_float(expected_cost_gap)

        # This should agree with VSS = EEV - RP because it is the same
        # difference decomposed scenario by scenario.
        eev_problem = rp_problem.evaluate_first_stage_solution(
            x_ev, solver_name=solver_name, tee=tee
        )
        rp_objective = rp_problem.objective_value()
        eev_objective = eev_problem.objective_value()
        vss = _clean_float(eev_objective - rp_objective)

        return {
            "model_name": self.data.get("name", "LandS"),
            "sets": copy.deepcopy(self.data["sets"]),
            "definition": {
                "cost_gap": "Delta(xi) = (c^T x_EV + Q(x_EV, xi)) - (c^T x_star + Q(x_star, xi)).",
                "sign_convention": "Positive means x_EV is more expensive than x_star for that realized scenario; negative means x_EV is cheaper for that realized scenario.",
                "recourse_value": "Q(x, xi) is computed by fixing x and solving the second-stage allocation LP for the realized scenario xi.",
            },
            "first_stage_solutions": {
                "x_star": x_star,
                "x_EV": x_ev,
                "first_stage_cost_x_star": first_stage_cost_star,
                "first_stage_cost_x_EV": first_stage_cost_ev,
            },
            "scenario_cost_gaps": scenarios,
            "summary": {
                "expected_cost_gap": expected_cost_gap,
                "RP": rp_objective,
                "EEV": eev_objective,
                "VSS": vss,
                "expected_cost_gap_minus_VSS": _clean_float(expected_cost_gap - vss),
            },
        }

    def save_cost_gap_diagnostics(
        self,
        path: str | Path,
        solver_name: str = "gurobi",
        tee: bool = False,
    ) -> Dict[str, Any]:
        """Compute scenario-wise cost-gap diagnostics and save them as JSON."""
        diagnostics = self.compute_cost_gap_diagnostics(
            solver_name=solver_name, tee=tee
        )
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(diagnostics, f, indent=2)
            f.write("\n")
        return diagnostics

    def save_diagnostics(
        self,
        path: str | Path,
        solver_name: str = "gurobi",
        tee: bool = False,
    ) -> Dict[str, Any]:
        """Compute diagnostics and save them as pretty-printed JSON."""
        diagnostics = self.compute_diagnostics(solver_name=solver_name, tee=tee)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(diagnostics, f, indent=2)
            f.write("\n")
        return diagnostics


if __name__ == "__main__":
    data_path = Path(__file__).with_name("lands_data.json")
    diagnostics_path = Path(__file__).with_name("lands_diagnostics.json")
    cost_gap_diagnostics_path = Path(__file__).with_name("cost_gap_diagnostics.json")

    problem = LandSDeterministicEquivalent.from_json(data_path)
    diagnostics = problem.save_diagnostics(diagnostics_path, solver_name="gurobi")
    cost_gap_diagnostics = problem.save_cost_gap_diagnostics(
        cost_gap_diagnostics_path, solver_name="gurobi"
    )

    print("Stochastic objective/RP:", diagnostics["metrics"]["RP"])
    print("EV objective:", diagnostics["metrics"]["EV_objective"])
    print("EEV:", diagnostics["metrics"]["EEV"])
    print("VSS:", diagnostics["metrics"]["VSS"])
    print("Stochastic x:")
    for i, value in diagnostics["stochastic_solution"]["first_stage"]["x_by_i"].items():
        print(f"  x[{i}] = {value:.12g}")
    print("EV x:")
    for i, value in diagnostics["expected_value_solution"]["first_stage"]["x_by_i"].items():
        print(f"  x_EV[{i}] = {value:.12g}")
    print("Scenario-wise cost gaps Delta(xi):")
    for s, values in cost_gap_diagnostics["scenario_cost_gaps"].items():
        print(f"  {s}: {values['cost_gap']:.12g}")
    print(
        "Expected cost gap:",
        cost_gap_diagnostics["summary"]["expected_cost_gap"],
    )
    print(f"Saved diagnostics to: {diagnostics_path}")
    print(f"Saved cost-gap diagnostics to: {cost_gap_diagnostics_path}")
