"""
Dummy demand two-stage stochastic program (TSSP) in Pyomo.

Problem description
-------------------
First stage:
    Buy x units of one product today at unit cost c.

Second stage:
    Tomorrow demand d_s is revealed. If x is not enough, buy emergency
    recourse y_s at unit cost r to cover the shortage.

Scenarios:
    Ten fixed equally likely demand scenarios: 10, 20, ..., 100.

This script solves:
    1. The full stochastic extensive-form model, giving x_star.
    2. The expected-value model, giving x_EV.
    3. The stochastic evaluation of both x_star and x_EV.

Outputs:
    - Printed terminal summary.
    - A text summary file.
    - CSV files with scenario data, model results, recourse evaluations,
      and comparison metrics.

Run:
    python dummy_demand_TSSP_updated.py

Optional:
    python dummy_demand_TSSP_updated.py --output-dir my_results
    python dummy_demand_TSSP_updated.py --solver gurobi --tee
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pyomo.environ as pyo


# -----------------------------------------------------------------------------
# Data configuration
# -----------------------------------------------------------------------------

N_SCENARIOS = 10
DEMANDS = list(range(10, 101, 10))

FIRST_STAGE_COST = 1.0
RECOURSE_COST = 5.0

DEFAULT_SOLVER = "gurobi"
DEFAULT_OUTPUT_DIR = "tssp_results"


@dataclass(frozen=True)
class TSSPData:
    scenarios: List[int]
    demand: Dict[int, float]
    probability: Dict[int, float]
    expected_demand: float
    first_stage_cost: float
    recourse_cost: float


# -----------------------------------------------------------------------------
# Data helpers
# -----------------------------------------------------------------------------


def build_data(
    demands: Iterable[float] = DEMANDS,
    first_stage_cost: float = FIRST_STAGE_COST,
    recourse_cost: float = RECOURSE_COST,
) -> TSSPData:
    """Build fixed equally likely scenario data."""
    demand_values = [float(d) for d in demands]
    if len(demand_values) == 0:
        raise ValueError("At least one demand scenario is required.")

    scenarios = list(range(len(demand_values)))
    probability = {s: 1.0 / len(scenarios) for s in scenarios}
    demand = {s: demand_values[s] for s in scenarios}
    expected_demand = sum(probability[s] * demand[s] for s in scenarios)

    return TSSPData(
        scenarios=scenarios,
        demand=demand,
        probability=probability,
        expected_demand=expected_demand,
        first_stage_cost=float(first_stage_cost),
        recourse_cost=float(recourse_cost),
    )


# -----------------------------------------------------------------------------
# Pyomo models
# -----------------------------------------------------------------------------


def build_stochastic_extensive_form(data: TSSPData) -> pyo.ConcreteModel:
    """
    Build the full two-stage stochastic extensive-form model.

    min    c*x + sum_s p_s*r*y_s
    s.t.   x + y_s >= d_s       for all scenarios s
           x >= 0, y_s >= 0

    x is the first-stage decision shared across all scenarios.
    y_s is the second-stage recourse decision for scenario s.
    """
    m = pyo.ConcreteModel("TwoStageStochasticExtensiveForm")

    m.S = pyo.Set(initialize=data.scenarios, ordered=True)
    m.d = pyo.Param(m.S, initialize=data.demand)
    m.p = pyo.Param(m.S, initialize=data.probability)

    # There is no reason to buy more than the maximum possible demand in this toy model.
    max_demand = max(data.demand.values())
    m.x = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0.0, max_demand))
    m.y = pyo.Var(m.S, domain=pyo.NonNegativeReals)

    def demand_coverage_rule(m: pyo.ConcreteModel, s: int) -> pyo.Constraint:
        return m.x + m.y[s] >= m.d[s]

    m.demand_coverage = pyo.Constraint(m.S, rule=demand_coverage_rule)

    def objective_rule(m: pyo.ConcreteModel) -> pyo.Expression:
        first_stage = data.first_stage_cost * m.x
        expected_recourse = sum(
            m.p[s] * data.recourse_cost * m.y[s]
            for s in m.S
        )
        return first_stage + expected_recourse

    m.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    return m



def build_expected_value_model(data: TSSPData) -> pyo.ConcreteModel:
    """
    Build the expected-value model.

    Replace random demand D by E[D].

    min    c*x + r*y
    s.t.   x + y >= E[D]
           x >= 0, y >= 0

    The resulting first-stage decision is x_EV.
    """
    m = pyo.ConcreteModel("ExpectedValueModel")

    max_demand = max(data.demand.values())
    m.expected_demand = pyo.Param(initialize=data.expected_demand)

    m.x = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0.0, max_demand))
    m.y = pyo.Var(domain=pyo.NonNegativeReals)

    m.demand_coverage = pyo.Constraint(expr=m.x + m.y >= m.expected_demand)

    m.obj = pyo.Objective(
        expr=data.first_stage_cost * m.x + data.recourse_cost * m.y,
        sense=pyo.minimize,
    )

    return m


# -----------------------------------------------------------------------------
# Solve and evaluation helpers
# -----------------------------------------------------------------------------


def solve_with_solver(
    model: pyo.ConcreteModel,
    solver_name: str = DEFAULT_SOLVER,
    tee: bool = False,
) -> pyo.SolverResults:
    """Solve a Pyomo model and raise a clear error if the solve fails."""
    solver = pyo.SolverFactory(solver_name)

    if solver is None or not solver.available(exception_flag=False):
        raise RuntimeError(
            f"Solver '{solver_name}' is not available. Install/configure it, "
            f"or pass another solver with --solver."
        )

    results = solver.solve(model, tee=tee)

    status = results.solver.status
    termination = results.solver.termination_condition

    ok_status = status == pyo.SolverStatus.ok
    ok_termination = termination in {
        pyo.TerminationCondition.optimal,
        pyo.TerminationCondition.locallyOptimal,
    }

    if not (ok_status and ok_termination):
        raise RuntimeError(
            f"Solve failed for model '{model.name}'. "
            f"Status: {status}; termination condition: {termination}."
        )

    return results



def evaluate_first_stage_solution(
    data: TSSPData,
    x_value: float,
    label: str,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """
    Evaluate a fixed first-stage solution x on all stochastic scenarios.

    Q_s(x) = r * max(d_s - x, 0).
    """
    first_stage_cost = data.first_stage_cost * x_value
    expected_recourse_cost = 0.0
    rows: List[Dict[str, Any]] = []

    for s in data.scenarios:
        shortage = max(data.demand[s] - x_value, 0.0)
        recourse_quantity = shortage
        q_s_x = data.recourse_cost * recourse_quantity
        weighted_q_s_x = data.probability[s] * q_s_x
        total_cost_in_scenario = first_stage_cost + q_s_x
        weighted_total_cost = data.probability[s] * total_cost_in_scenario

        expected_recourse_cost += weighted_q_s_x

        rows.append(
            {
                "label": label,
                "scenario": s,
                "demand": data.demand[s],
                "probability": data.probability[s],
                "x": x_value,
                "first_stage_cost": first_stage_cost,
                "recourse_quantity": recourse_quantity,
                "shortage": shortage,
                "Q_s_x": q_s_x,
                "p_s_Q_s_x": weighted_q_s_x,
                "scenario_total_cost": total_cost_in_scenario,
                "p_s_total_cost": weighted_total_cost,
            }
        )

    expected_total_cost = first_stage_cost + expected_recourse_cost

    summary = {
        "x": x_value,
        "first_stage_cost": first_stage_cost,
        "expected_recourse_cost": expected_recourse_cost,
        "expected_total_cost": expected_total_cost,
    }

    return summary, rows



def extract_stochastic_solution_rows(
    data: TSSPData,
    model: pyo.ConcreteModel,
) -> List[Dict[str, Any]]:
    """Extract scenario-level second-stage values from the extensive-form model."""
    x_star = pyo.value(model.x)
    first_stage_cost = data.first_stage_cost * x_star

    rows: List[Dict[str, Any]] = []
    for s in data.scenarios:
        y_s = pyo.value(model.y[s])
        q_s = data.recourse_cost * y_s
        rows.append(
            {
                "scenario": s,
                "demand": data.demand[s],
                "probability": data.probability[s],
                "x_star": x_star,
                "first_stage_cost": first_stage_cost,
                "y_star_s": y_s,
                "Q_s_x_star": q_s,
                "p_s_Q_s_x_star": data.probability[s] * q_s,
            }
        )
    return rows


# -----------------------------------------------------------------------------
# Output helpers
# -----------------------------------------------------------------------------


def format_money(value: float) -> str:
    return f"{value:.4f}"



def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Write a list of dictionaries as a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)



def make_scenario_rows(data: TSSPData) -> List[Dict[str, Any]]:
    return [
        {
            "scenario": s,
            "demand": data.demand[s],
            "probability": data.probability[s],
        }
        for s in data.scenarios
    ]



def make_model_summary_rows(
    stoch_summary: Dict[str, float],
    ev_summary: Dict[str, float],
    comparison: Dict[str, float],
    stoch_model_objective: float,
    ev_model_objective: float,
) -> List[Dict[str, Any]]:
    return [
        {
            "quantity": "x_star",
            "value": stoch_summary["x"],
            "description": "First-stage solution from the stochastic extensive-form model",
        },
        {
            "quantity": "stochastic_model_objective",
            "value": stoch_model_objective,
            "description": "Objective reported by the stochastic extensive-form model",
        },
        {
            "quantity": "x_star_first_stage_cost",
            "value": stoch_summary["first_stage_cost"],
            "description": "First-stage cost c*x_star",
        },
        {
            "quantity": "x_star_expected_recourse_cost",
            "value": stoch_summary["expected_recourse_cost"],
            "description": "Expected recourse cost E[Q(x_star, D)]",
        },
        {
            "quantity": "x_star_expected_total_cost_RP",
            "value": stoch_summary["expected_total_cost"],
            "description": "Recourse problem objective RP",
        },
        {
            "quantity": "x_EV",
            "value": ev_summary["x"],
            "description": "First-stage solution from the expected-value model",
        },
        {
            "quantity": "expected_value_model_objective",
            "value": ev_model_objective,
            "description": "Objective reported by the expected-value model",
        },
        {
            "quantity": "x_EV_first_stage_cost",
            "value": ev_summary["first_stage_cost"],
            "description": "First-stage cost c*x_EV",
        },
        {
            "quantity": "x_EV_expected_recourse_cost",
            "value": ev_summary["expected_recourse_cost"],
            "description": "Expected recourse cost E[Q(x_EV, D)] evaluated on full scenarios",
        },
        {
            "quantity": "x_EV_expected_total_cost_EEV",
            "value": ev_summary["expected_total_cost"],
            "description": "Expected result of using the expected-value solution, EEV",
        },
        {
            "quantity": "VSS",
            "value": comparison["VSS"],
            "description": "Value of the stochastic solution: EEV - RP",
        },
    ]



def print_scenario_table(data: TSSPData) -> None:
    print("\nScenarios")
    print("-" * 45)
    print(f"{'scenario':>8} {'demand':>12} {'probability':>14}")
    for s in data.scenarios:
        print(f"{s:>8} {data.demand[s]:>12.2f} {data.probability[s]:>14.4f}")
    print(f"\nExpected demand = {data.expected_demand:.4f}")



def print_evaluation(label: str, summary: Dict[str, float], rows: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 80)
    print(f"Evaluation of {label}")
    print("=" * 80)
    print(f"x = {summary['x']:.4f}")
    print(f"First-stage cost c*x = {summary['first_stage_cost']:.4f}")
    print(f"Expected recourse cost E[Q(x,D)] = {summary['expected_recourse_cost']:.4f}")
    print(f"Expected total cost = {summary['expected_total_cost']:.4f}")

    print("\nScenario recourse costs")
    print(
        f"{'s':>3} "
        f"{'demand':>8} "
        f"{'prob':>8} "
        f"{'shortage':>10} "
        f"{'Q_s(x)':>12} "
        f"{'p_s Q_s(x)':>14} "
        f"{'scenario total':>16}"
    )
    for row in rows:
        print(
            f"{row['scenario']:>3} "
            f"{row['demand']:>8.2f} "
            f"{row['probability']:>8.4f} "
            f"{row['shortage']:>10.2f} "
            f"{row['Q_s_x']:>12.2f} "
            f"{row['p_s_Q_s_x']:>14.2f} "
            f"{row['scenario_total_cost']:>16.2f}"
        )



def write_text_summary(
    path: Path,
    data: TSSPData,
    x_star: float,
    stochastic_objective: float,
    stoch_second_stage_rows: List[Dict[str, Any]],
    x_ev: float,
    ev_model_objective: float,
    ev_model_y: float,
    stoch_eval_summary: Dict[str, float],
    stoch_eval_rows: List[Dict[str, Any]],
    ev_eval_summary: Dict[str, float],
    ev_eval_rows: List[Dict[str, Any]],
    comparison: Dict[str, float],
) -> None:
    """Write a complete plain-text run summary."""
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    add = lines.append

    add("Dummy demand two-stage stochastic program")
    add("=" * 80)
    add("")
    add("Data")
    add("----")
    add(f"First-stage unit cost c = {data.first_stage_cost:.4f}")
    add(f"Recourse unit cost r = {data.recourse_cost:.4f}")
    add(f"Number of scenarios = {len(data.scenarios)}")
    add(f"Demand scenarios = {[data.demand[s] for s in data.scenarios]}")
    add(f"Scenario probability = {1.0 / len(data.scenarios):.4f}")
    add(f"Expected demand = {data.expected_demand:.4f}")
    add("")

    add("Scenario table")
    add("--------------")
    add(f"{'scenario':>8} {'demand':>12} {'probability':>14}")
    for s in data.scenarios:
        add(f"{s:>8} {data.demand[s]:>12.2f} {data.probability[s]:>14.4f}")
    add("")

    add("Stochastic extensive-form solution")
    add("-----------------------------------")
    add(f"x_star = {x_star:.4f}")
    add(f"Stochastic model objective = {stochastic_objective:.4f}")
    add(f"First-stage cost c*x_star = {data.first_stage_cost * x_star:.4f}")
    add("")
    add(f"{'s':>3} {'demand':>8} {'y_star_s':>12} {'Q_s(x_star)':>14} {'p_s Q_s':>12}")
    for row in stoch_second_stage_rows:
        add(
            f"{row['scenario']:>3} "
            f"{row['demand']:>8.2f} "
            f"{row['y_star_s']:>12.4f} "
            f"{row['Q_s_x_star']:>14.4f} "
            f"{row['p_s_Q_s_x_star']:>12.4f}"
        )
    add("")

    add("Expected-value model solution")
    add("-----------------------------")
    add(f"x_EV = {x_ev:.4f}")
    add(f"Expected-value model objective = {ev_model_objective:.4f}")
    add(f"Expected-value model y = {ev_model_y:.4f}")
    add(f"First-stage cost c*x_EV = {data.first_stage_cost * x_ev:.4f}")
    add("")

    for label, summary, rows in [
        ("stochastic solution x_star", stoch_eval_summary, stoch_eval_rows),
        ("expected-value solution x_EV", ev_eval_summary, ev_eval_rows),
    ]:
        add(f"Evaluation of {label}")
        add("-" * 80)
        add(f"x = {summary['x']:.4f}")
        add(f"First-stage cost c*x = {summary['first_stage_cost']:.4f}")
        add(f"Expected recourse cost E[Q(x,D)] = {summary['expected_recourse_cost']:.4f}")
        add(f"Expected total cost = {summary['expected_total_cost']:.4f}")
        add("")
        add(
            f"{'s':>3} "
            f"{'demand':>8} "
            f"{'prob':>8} "
            f"{'shortage':>10} "
            f"{'Q_s(x)':>12} "
            f"{'p_s Q_s(x)':>14} "
            f"{'scenario total':>16}"
        )
        for row in rows:
            add(
                f"{row['scenario']:>3} "
                f"{row['demand']:>8.2f} "
                f"{row['probability']:>8.4f} "
                f"{row['shortage']:>10.2f} "
                f"{row['Q_s_x']:>12.2f} "
                f"{row['p_s_Q_s_x']:>14.2f} "
                f"{row['scenario_total_cost']:>16.2f}"
            )
        add("")

    add("Comparison")
    add("----------")
    add(f"RP  = stochastic optimum expected cost      = {comparison['RP']:.4f}")
    add(f"EEV = expected cost using x_EV in scenarios = {comparison['EEV']:.4f}")
    add(f"VSS = EEV - RP                              = {comparison['VSS']:.4f}")
    add("")
    add("Note")
    add("----")
    add(
        "With these fixed data and costs, the stochastic objective can have multiple "
        "optimal first-stage solutions. Gurobi may return any optimal extreme point. "
        "The reported x_star, Q_s(x_star), and VSS are computed from the value returned "
        "by the solver."
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")



def save_outputs(
    output_dir: Path,
    data: TSSPData,
    stoch_second_stage_rows: List[Dict[str, Any]],
    stoch_eval_summary: Dict[str, float],
    stoch_eval_rows: List[Dict[str, Any]],
    ev_eval_summary: Dict[str, float],
    ev_eval_rows: List[Dict[str, Any]],
    comparison: Dict[str, float],
    stochastic_objective: float,
    ev_model_objective: float,
    x_star: float,
    x_ev: float,
) -> None:
    """Write CSV outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    write_csv(output_dir / "scenario_data.csv", make_scenario_rows(data))
    write_csv(output_dir / "stochastic_solution_by_scenario.csv", stoch_second_stage_rows)
    write_csv(output_dir / "evaluation_x_star.csv", stoch_eval_rows)
    write_csv(output_dir / "evaluation_x_EV.csv", ev_eval_rows)
    write_csv(output_dir / "all_solution_evaluations.csv", stoch_eval_rows + ev_eval_rows)

    summary_rows = make_model_summary_rows(
        stoch_summary=stoch_eval_summary,
        ev_summary=ev_eval_summary,
        comparison=comparison,
        stoch_model_objective=stochastic_objective,
        ev_model_objective=ev_model_objective,
    )
    write_csv(output_dir / "model_summary.csv", summary_rows)

    comparison_rows = [
        {"metric": "x_star", "value": x_star},
        {"metric": "x_EV", "value": x_ev},
        {"metric": "RP", "value": comparison["RP"]},
        {"metric": "EEV", "value": comparison["EEV"]},
        {"metric": "VSS", "value": comparison["VSS"]},
    ]
    write_csv(output_dir / "comparison_summary.csv", comparison_rows)


# -----------------------------------------------------------------------------
# Main run
# -----------------------------------------------------------------------------


def run_tssp(solver_name: str, output_dir: Path, tee: bool = False) -> Dict[str, Any]:
    """Run the full workflow and return the main results."""
    data = build_data()

    print("\n" + "=" * 80)
    print("Dummy demand two-stage stochastic program")
    print("=" * 80)
    print(f"First-stage unit cost c = {data.first_stage_cost:.4f}")
    print(f"Recourse unit cost r = {data.recourse_cost:.4f}")
    print_scenario_table(data)

    # Full stochastic extensive form
    stoch_model = build_stochastic_extensive_form(data)
    solve_with_solver(stoch_model, solver_name=solver_name, tee=tee)

    x_star = pyo.value(stoch_model.x)
    stochastic_objective = pyo.value(stoch_model.obj)
    stoch_second_stage_rows = extract_stochastic_solution_rows(data, stoch_model)

    print("\n" + "=" * 80)
    print("Stochastic extensive-form solution")
    print("=" * 80)
    print(f"x_star = {x_star:.4f}")
    print(f"Stochastic model objective = {stochastic_objective:.4f}")
    print(f"First-stage cost c*x_star = {data.first_stage_cost * x_star:.4f}")
    print("\nSecond-stage solution by scenario")
    print(f"{'s':>3} {'demand':>8} {'y_star_s':>12} {'Q_s(x_star)':>14} {'p_s Q_s':>12}")
    for row in stoch_second_stage_rows:
        print(
            f"{row['scenario']:>3} "
            f"{row['demand']:>8.2f} "
            f"{row['y_star_s']:>12.4f} "
            f"{row['Q_s_x_star']:>14.4f} "
            f"{row['p_s_Q_s_x_star']:>12.4f}"
        )

    # Expected-value model
    ev_model = build_expected_value_model(data)
    solve_with_solver(ev_model, solver_name=solver_name, tee=tee)

    x_ev = pyo.value(ev_model.x)
    ev_model_y = pyo.value(ev_model.y)
    ev_model_objective = pyo.value(ev_model.obj)

    print("\n" + "=" * 80)
    print("Expected-value model solution")
    print("=" * 80)
    print(f"x_EV = {x_ev:.4f}")
    print(f"Expected-value model objective = {ev_model_objective:.4f}")
    print(f"Expected-value model y = {ev_model_y:.4f}")
    print(f"First-stage cost c*x_EV = {data.first_stage_cost * x_ev:.4f}")

    # Evaluate both x_star and x_EV under the stochastic scenarios
    stoch_eval_summary, stoch_eval_rows = evaluate_first_stage_solution(
        data=data,
        x_value=x_star,
        label="x_star",
    )
    ev_eval_summary, ev_eval_rows = evaluate_first_stage_solution(
        data=data,
        x_value=x_ev,
        label="x_EV",
    )

    print_evaluation("stochastic solution x_star", stoch_eval_summary, stoch_eval_rows)
    print_evaluation("expected-value solution x_EV", ev_eval_summary, ev_eval_rows)

    RP = stoch_eval_summary["expected_total_cost"]
    EEV = ev_eval_summary["expected_total_cost"]
    VSS = EEV - RP
    comparison = {"RP": RP, "EEV": EEV, "VSS": VSS}

    print("\n" + "=" * 80)
    print("Comparison")
    print("=" * 80)
    print(f"RP  = stochastic optimum expected cost      = {RP:.4f}")
    print(f"EEV = expected cost using x_EV in scenarios = {EEV:.4f}")
    print(f"VSS = EEV - RP                              = {VSS:.4f}")

    save_outputs(
        output_dir=output_dir,
        data=data,
        stoch_second_stage_rows=stoch_second_stage_rows,
        stoch_eval_summary=stoch_eval_summary,
        stoch_eval_rows=stoch_eval_rows,
        ev_eval_summary=ev_eval_summary,
        ev_eval_rows=ev_eval_rows,
        comparison=comparison,
        stochastic_objective=stochastic_objective,
        ev_model_objective=ev_model_objective,
        x_star=x_star,
        x_ev=x_ev,
    )

    text_summary_path = output_dir / "run_summary.txt"
    write_text_summary(
        path=text_summary_path,
        data=data,
        x_star=x_star,
        stochastic_objective=stochastic_objective,
        stoch_second_stage_rows=stoch_second_stage_rows,
        x_ev=x_ev,
        ev_model_objective=ev_model_objective,
        ev_model_y=ev_model_y,
        stoch_eval_summary=stoch_eval_summary,
        stoch_eval_rows=stoch_eval_rows,
        ev_eval_summary=ev_eval_summary,
        ev_eval_rows=ev_eval_rows,
        comparison=comparison,
    )

    print("\n" + "=" * 80)
    print("Files written")
    print("=" * 80)
    print(f"Output directory: {output_dir.resolve()}")
    print(f"Text summary:     {(output_dir / 'run_summary.txt').resolve()}")
    print(f"CSV summary:      {(output_dir / 'model_summary.csv').resolve()}")
    print(f"All evaluations:  {(output_dir / 'all_solution_evaluations.csv').resolve()}")

    return {
        "data": data,
        "x_star": x_star,
        "x_EV": x_ev,
        "RP": RP,
        "EEV": EEV,
        "VSS": VSS,
        "output_dir": output_dir,
    }



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Solve a dummy demand two-stage stochastic program with Pyomo."
    )
    parser.add_argument(
        "--solver",
        default=DEFAULT_SOLVER,
        help=f"Pyomo solver name. Default: {DEFAULT_SOLVER}",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory where txt/csv results are written. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--tee",
        action="store_true",
        help="Print solver log output.",
    )
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    run_tssp(
        solver_name=args.solver,
        output_dir=Path(args.output_dir),
        tee=args.tee,
    )


if __name__ == "__main__":
    main()
