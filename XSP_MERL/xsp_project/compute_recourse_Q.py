from pathlib import Path
from types import SimpleNamespace
import argparse
import csv

import pyomo.environ as pyo
from pyomo.opt import TerminationCondition
from pyomo.repn.standard_repn import generate_standard_repn

import mpisppy.problem_io.smps_module as smps_module


# ============================================================
# Settings
# ============================================================

SOLVER_NAME = "gurobi"
DATA_FOLDER_NAME = "dcap233_200"
RESULTS_FOLDER_NAME = "results_xstar_xev"


# ============================================================
# Paths
# ============================================================

project_dir = Path(__file__).resolve().parent
smps_dir = project_dir / DATA_FOLDER_NAME
results_dir = project_dir / RESULTS_FOLDER_NAME
out_dir = project_dir / "results_recourse_Q"
out_dir.mkdir(exist_ok=True)

cfg = SimpleNamespace(smps_dir=str(smps_dir))


# ============================================================
# Helpers
# ============================================================

def to_pyomo_name(mps_name: str) -> str:
    """
    mpi-sppy/mps_reader converts names like x(1) to x_1_.
    """
    return mps_name.replace("(", "_").replace(")", "_")


def load_x_solution(csv_path: Path):
    """
    Load x_star.csv or x_EV.csv from results_xstar_xev.

    Returns a dictionary keyed by both original MPS name and Pyomo name.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find solution file: {csv_path}")

    x = {}

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            orig = row["original_mps_name"]
            pyomo = row["pyomo_name"]
            value_raw = row["value"]

            if value_raw in ["", "None", None]:
                raise ValueError(
                    f"Missing value for first-stage variable {orig} in {csv_path}"
                )

            value = float(value_raw)

            x[orig] = value
            x[pyomo] = value

    return x


def get_single_active_objective(model):
    objs = list(model.component_data_objects(pyo.Objective, active=True))

    if len(objs) != 1:
        raise RuntimeError(f"Expected one active objective, found {len(objs)}")

    return objs[0]


def replace_objective_with_recourse_only(model, first_stage_pyomo_names):
    """
    The scenario model objective is usually:

        c^T x + q_s^T y_s

    For Q_s(x), we only want:

        q_s^T y_s

    So we remove all terms involving first-stage variables.
    """
    original_obj = get_single_active_objective(model)
    repn = generate_standard_repn(original_obj.expr, compute_values=False)

    recourse_expr = 0

    if repn.constant is not None:
        recourse_expr += repn.constant

    for var, coef in zip(repn.linear_vars, repn.linear_coefs):
        if var.name not in first_stage_pyomo_names:
            recourse_expr += coef * var

    original_sense = original_obj.sense
    original_obj.deactivate()

    model.RecourseObjective = pyo.Objective(
        expr=recourse_expr,
        sense=original_sense,
    )


def fix_first_stage_variables(model, x_values, first_stage_var_names):
    """
    Fix first-stage variables in the scenario model to a supplied vector x.
    """
    fixed = []

    for orig_name in first_stage_var_names:
        pyomo_name = to_pyomo_name(orig_name)

        var = model.find_component(pyomo_name)

        if var is None:
            raise RuntimeError(
                f"Could not find first-stage variable {orig_name} "
                f"with Pyomo name {pyomo_name} in scenario model."
            )

        if pyomo_name in x_values:
            value = x_values[pyomo_name]
        elif orig_name in x_values:
            value = x_values[orig_name]
        else:
            raise RuntimeError(f"No supplied x value for {orig_name}")

        var.fix(value)
        fixed.append((orig_name, pyomo_name, value))

    return fixed


def solve_Q_for_scenario(scenario_name, x_values, first_stage_var_names, tee=False):
    """
    Build and solve:

        Q_s(x) = min_y q_s^T y_s
                 s.t. T_s x + W_s y_s >= h_s,
                      y_s in Y_s

    for a given scenario and fixed first-stage vector x.
    """
    model = smps_module.scenario_creator(scenario_name, cfg=cfg)

    first_stage_pyomo_names = {
        to_pyomo_name(name) for name in first_stage_var_names
    }

    # Important: replace the objective before fixing x, so first-stage
    # objective terms are cleanly removed rather than becoming constants.
    replace_objective_with_recourse_only(model, first_stage_pyomo_names)

    fixed_vars = fix_first_stage_variables(
        model,
        x_values,
        first_stage_var_names,
    )

    solver = pyo.SolverFactory(SOLVER_NAME)
    results = solver.solve(model, tee=tee)

    term = results.solver.termination_condition

    if term != TerminationCondition.optimal:
        return {
            "scenario": scenario_name,
            "termination_condition": str(term),
            "Q_value": None,
            "num_fixed_first_stage_vars": len(fixed_vars),
        }

    q_value = pyo.value(model.RecourseObjective)

    return {
        "scenario": scenario_name,
        "termination_condition": str(term),
        "Q_value": q_value,
        "num_fixed_first_stage_vars": len(fixed_vars),
    }


def write_Q_results(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scenario",
                "termination_condition",
                "Q_value",
                "num_fixed_first_stage_vars",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--x",
        choices=["x_star", "x_EV"],
        required=True,
        help="Which first-stage solution to use.",
    )

    parser.add_argument(
        "--scenario",
        default=None,
        help="Scenario name. If omitted, the first scenario is used.",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Compute Q_s(x) for all scenarios.",
    )

    parser.add_argument(
        "--list-scenarios",
        action="store_true",
        help="Print scenario names and exit.",
    )

    parser.add_argument(
        "--tee",
        action="store_true",
        help="Show solver output.",
    )

    args = parser.parse_args()

    parsed = smps_module._ensure_parsed(cfg.smps_dir)

    scenarios = parsed["scenarios"]
    scenario_names = [s["name"] for s in scenarios]

    stages = parsed["stages"]
    vars_by_stage = parsed["vars_by_stage"]

    root_stage_name = stages[0][0]
    first_stage_var_names = vars_by_stage[root_stage_name]

    print("SMPS directory:", smps_dir)
    print("Number of scenarios:", len(scenario_names))
    print("Root stage:", root_stage_name)
    print("Number of first-stage variables:", len(first_stage_var_names))

    if args.list_scenarios:
        print("\nScenario names:")
        for name in scenario_names:
            print(name)
        return

    if args.x == "x_star":
        x_path = results_dir / "x_star.csv"
    else:
        x_path = results_dir / "x_EV.csv"

    x_values = load_x_solution(x_path)

    print("\nUsing x solution:", x_path)

    if args.all:
        selected_scenarios = scenario_names
    else:
        selected_scenarios = [args.scenario or scenario_names[0]]

    print("Selected scenarios:", selected_scenarios[:10])
    if len(selected_scenarios) > 10:
        print("...")

    rows = []

    for scen in selected_scenarios:
        print(f"\nSolving Q for scenario {scen} using {args.x}...")

        row = solve_Q_for_scenario(
            scenario_name=scen,
            x_values=x_values,
            first_stage_var_names=first_stage_var_names,
            tee=args.tee,
        )

        print("Termination:", row["termination_condition"])
        print("Q value:", row["Q_value"])

        rows.append(row)

    if args.all:
        out_path = out_dir / f"Q_all_scenarios_{args.x}.csv"
    else:
        safe_scen = selected_scenarios[0].replace("/", "_")
        out_path = out_dir / f"Q_{safe_scen}_{args.x}.csv"

    write_Q_results(out_path, rows)

    print("\nWrote results to:")
    print(out_path)


if __name__ == "__main__":
    main()