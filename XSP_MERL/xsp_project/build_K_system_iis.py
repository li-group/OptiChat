# build_K_system_iis.py
"""
Build the compensation-adjusted K-system for a minimum bad-scenario set,
verify the recourse/budget accounting, compute a Gurobi IIS, and export
IIS rows/bounds for explainability.

Main design choices:
  1. Extract the recourse objective BEFORE fixing x_EV. This prevents fixed
     first-stage costs c^T x_EV from leaking into the K-budget as constants.
  2. Include any true objective constant that exists before fixing x, because
     compute_recourse_Q.py includes that constant when it computes Q_s(x).
     For the DCAP instance this should usually be zero.
  3. Run a recourse consistency check before computing the IIS:
        min sum_{s in K} p_s Q_s(x_EV)
     over the same K-system blocks must match the stored Q_x_EV values.
"""

from pathlib import Path
from types import SimpleNamespace
import csv
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pyomo.environ as pyo
from pyomo.opt import TerminationCondition
from pyomo.repn.standard_repn import generate_standard_repn

import gurobipy as gp
from gurobipy import GRB

import mpisppy.problem_io.smps_module as smps_module


# ============================================================
# Settings
# ============================================================

# DATA_FOLDER_NAME = "dcap233_200"
DATA_FOLDER_NAME = "dummy_demand_tssp"

# General numerical tolerances
TOL = 1e-8
CONSISTENCY_TOL = 1e-6
CONSTANT_TOL = 1e-9

# Match compute_recourse_Q.py:
# It keeps the objective constant that exists BEFORE x is fixed.
# This should usually be 0 for this DCAP model, but keeping it makes
# the K-system consistent with the stored Q values if a true constant exists.
INCLUDE_TRUE_OBJECTIVE_CONSTANT = True

SOLVER_NAME = "gurobi"

RESULTS_XSTAR_XEV = "results_xstar_xev"
RESULTS_GOOD_BAD = "results_good_bad_partition"
RESULTS_FIRST_STAGE_COST = "results_first_stage_cost"
RESULTS_K = "results_min_bad_set_K"
OUTPUT_FOLDER = "results_K_system_IIS"


# ============================================================
# Paths
# ============================================================

project_dir = Path(__file__).resolve().parent
smps_dir = project_dir / DATA_FOLDER_NAME

x_ev_path = project_dir / f"{DATA_FOLDER_NAME}_results" / RESULTS_XSTAR_XEV / "x_EV.csv"
scenario_summary_path = project_dir / f"{DATA_FOLDER_NAME}_results" /  RESULTS_XSTAR_XEV / "scenario_summary.csv"
good_bad_path = project_dir / f"{DATA_FOLDER_NAME}_results" / RESULTS_GOOD_BAD / "good_bad_partition.csv"
first_stage_cost_path = project_dir / f"{DATA_FOLDER_NAME}_results" / RESULTS_FIRST_STAGE_COST / "first_stage_cost_summary.csv"
K_path = project_dir / f"{DATA_FOLDER_NAME}_results" / RESULTS_K / "K_min_bad_scenarios.csv"

out_dir = project_dir / f"{DATA_FOLDER_NAME}_results" / OUTPUT_FOLDER
out_dir.mkdir(parents=True, exist_ok=True)

lp_path = out_dir / "K_system.lp"
iis_ilp_path = out_dir / "K_system_iis.ilp"
iis_constraints_path = out_dir / "K_system_iis_constraints.csv"
iis_bounds_path = out_dir / "K_system_iis_bounds.csv"
summary_path = out_dir / "K_system_iis_summary.txt"
diagnostics_path = out_dir / "K_system_diagnostics.csv"
recourse_diag_path = out_dir / "K_system_recourse_objective_diagnostics.csv"

cfg = SimpleNamespace(smps_dir=str(smps_dir))


# ============================================================
# Data containers
# ============================================================

@dataclass
class RecourseObjectiveDiagnostics:
    scenario: str
    objective_constant_before_fixing: float
    included_objective_constant: bool
    num_first_stage_terms_removed: int
    first_stage_objective_coeff_sum_abs: float
    num_recourse_terms_kept: int
    recourse_coeff_sum_abs: float


# ============================================================
# Helpers: names, sorting, reads/writes
# ============================================================

def to_pyomo_name(mps_name: str) -> str:
    """
    mpi-sppy/mps_reader converts names like x(1) to x_1_.
    """
    return mps_name.replace("(", "_").replace(")", "_")


def scenario_sort_key(name: str):
    match = re.search(r"\d+", name)
    if match:
        return int(match.group())
    return name


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def read_x_solution(path: Path) -> Dict[str, float]:
    require_file(path)

    x = {}

    with open(path, newline="") as f:
        reader = csv.DictReader(f)

        required = {"original_mps_name", "pyomo_name", "value"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise RuntimeError(f"{path} must contain columns {sorted(required)}")

        for row in reader:
            orig = row["original_mps_name"]
            pyomo = row["pyomo_name"]
            raw_val = row["value"]

            if raw_val in ["", "None", None]:
                raise RuntimeError(f"Missing x value for {orig} / {pyomo} in {path}")

            val = float(raw_val)

            x[orig] = val
            x[pyomo] = val

    return x


def read_probabilities(path: Path) -> Dict[str, float]:
    require_file(path)

    probs = {}

    with open(path, newline="") as f:
        reader = csv.DictReader(f)

        required = {"scenario", "probability"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise RuntimeError(f"{path} must contain columns {sorted(required)}")

        for row in reader:
            probs[row["scenario"]] = float(row["probability"])

    return probs


def read_good_bad_partition(path: Path) -> List[Dict[str, object]]:
    require_file(path)

    rows = []

    with open(path, newline="") as f:
        reader = csv.DictReader(f)

        required = {"scenario", "Q_x_EV", "Q_x_star", "Delta", "classification"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise RuntimeError(f"{path} must contain columns {sorted(required)}")

        for row in reader:
            rows.append({
                "scenario": row["scenario"],
                "Q_x_EV": float(row["Q_x_EV"]),
                "Q_x_star": float(row["Q_x_star"]),
                "Delta": float(row["Delta"]),
                "classification": row["classification"],
            })

    return rows


def read_first_stage_costs(path: Path) -> Dict[str, float]:
    require_file(path)

    costs = {}

    with open(path, newline="") as f:
        reader = csv.DictReader(f)

        required = {"solution", "first_stage_cost"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise RuntimeError(f"{path} must contain columns {sorted(required)}")

        for row in reader:
            costs[row["solution"]] = float(row["first_stage_cost"])

    missing = {"x_star", "x_EV"} - set(costs)
    if missing:
        raise RuntimeError(f"{path} is missing rows for: {sorted(missing)}")

    return costs


def read_K_scenarios(path: Path) -> List[str]:
    """
    Expected CSV from select_min_bad_set_K.py:
        scenario,probability,Q_x_EV,Q_x_star,Delta,p_Delta,classification

    Falls back gracefully if the file is a one-column scenario list.
    """
    require_file(path)

    with open(path, newline="") as f:
        sample = f.read(4096)
        f.seek(0)

        if "scenario" in sample.splitlines()[0]:
            reader = csv.DictReader(f)
            if "scenario" not in (reader.fieldnames or []):
                raise RuntimeError(f"{path} must contain a 'scenario' column.")
            K = [row["scenario"] for row in reader if row.get("scenario")]
        else:
            K = [line.strip() for line in f if line.strip()]

    return sorted(K, key=scenario_sort_key)


def write_dict_rows(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# Helpers: Pyomo model/objective handling
# ============================================================

def get_single_active_objective(model):
    objs = list(model.component_data_objects(pyo.Objective, active=True))

    if len(objs) != 1:
        raise RuntimeError(f"Expected exactly one active objective, found {len(objs)}")

    return objs[0]


def recourse_objective_expr(
    model,
    first_stage_var_names: List[str],
    scenario_name: str,
    include_true_constant: bool = INCLUDE_TRUE_OBJECTIVE_CONSTANT,
) -> Tuple[pyo.Expression, RecourseObjectiveDiagnostics]:
    """
    Extract q_s^T y_s from the scenario objective by dropping terms involving
    first-stage variables.

    IMPORTANT:
      This must be called BEFORE fixing first-stage variables. Otherwise
      c^T x_EV can be converted into a constant and accidentally included
      in the K-budget.

    Constants:
      A constant that exists before fixing x is a true objective constant.
      compute_recourse_Q.py includes such constants in Q_s(x), so by default
      this function includes them as well for consistency.
    """
    first_stage_pyomo_names = {
        to_pyomo_name(name) for name in first_stage_var_names
    }

    obj = get_single_active_objective(model)
    repn = generate_standard_repn(obj.expr, compute_values=False)

    expr = 0
    true_constant = float(pyo.value(repn.constant or 0.0))

    if include_true_constant:
        expr += true_constant

    num_first_stage_terms_removed = 0
    first_stage_objective_coeff_sum_abs = 0.0
    num_recourse_terms_kept = 0
    recourse_coeff_sum_abs = 0.0

    for var, coef in zip(repn.linear_vars, repn.linear_coefs):
        coef_val = float(pyo.value(coef))

        if var.name in first_stage_pyomo_names:
            num_first_stage_terms_removed += 1
            first_stage_objective_coeff_sum_abs += abs(coef_val)
        else:
            expr += coef * var
            num_recourse_terms_kept += 1
            recourse_coeff_sum_abs += abs(coef_val)

    obj.deactivate()

    diag = RecourseObjectiveDiagnostics(
        scenario=scenario_name,
        objective_constant_before_fixing=true_constant,
        included_objective_constant=include_true_constant,
        num_first_stage_terms_removed=num_first_stage_terms_removed,
        first_stage_objective_coeff_sum_abs=first_stage_objective_coeff_sum_abs,
        num_recourse_terms_kept=num_recourse_terms_kept,
        recourse_coeff_sum_abs=recourse_coeff_sum_abs,
    )

    return expr, diag


def fix_first_stage_vars(model, x_values: Dict[str, float], first_stage_var_names: List[str]):
    """
    Fix first-stage variables in a scenario model to x_EV.
    """
    fixed = []

    for orig_name in first_stage_var_names:
        pyomo_name = to_pyomo_name(orig_name)
        var = model.find_component(pyomo_name)

        if var is None:
            raise RuntimeError(
                f"Could not find first-stage variable {orig_name} / {pyomo_name}"
            )

        if pyomo_name in x_values:
            val = x_values[pyomo_name]
        elif orig_name in x_values:
            val = x_values[orig_name]
        else:
            raise RuntimeError(f"No x_EV value found for {orig_name}")

        var.fix(val)
        fixed.append((orig_name, pyomo_name, val))

    return fixed


def check_no_active_objectives(model, label: str) -> None:
    objs = list(model.component_data_objects(pyo.Objective, active=True))
    if objs:
        names = [obj.name for obj in objs]
        raise RuntimeError(f"{label} unexpectedly has active objectives: {names}")


# ============================================================
# Helpers: IIS classification
# ============================================================

def classify_iis_constraint_name(name: str):
    """
    Identify whether an IIS row is the K-budget row or belongs to a scenario.
    """
    if "K_budget" in name:
        return "K_budget", ""

    match = re.search(r"Scenario_(SCEN\d+)", name)
    if match:
        return "scenario_constraint", match.group(1)

    return "other_constraint", ""


def classify_iis_var_name(name: str):
    match = re.search(r"Scenario_(SCEN\d+)", name)
    if match:
        return match.group(1)
    return ""


# ============================================================
# Helpers: diagnostics and consistency checks
# ============================================================

def solve_recourse_consistency_check(
    base_model,
    K_scenarios: List[str],
    probabilities: Dict[str, float],
    stored_sum_K_p_QEV: float,
) -> Tuple[float, str]:
    """
    Solve the same scenario blocks without the K-budget and verify that the
    minimum weighted recourse cost equals the stored sum_K p_s Q_s(x_EV).
    """
    print("\nChecking K-system recourse objective consistency...")

    M_check = base_model.clone()

    # There should be no active objectives at this point.
    check_no_active_objectives(M_check, "Recourse consistency model")

    K_recourse_check_expr = sum(
        probabilities[s] * getattr(M_check, f"Scenario_{s}").RecourseOnlyObjective
        for s in K_scenarios
    )

    M_check.CheckObjective = pyo.Objective(
        expr=K_recourse_check_expr,
        sense=pyo.minimize,
    )

    solver = pyo.SolverFactory(SOLVER_NAME)
    check_results = solver.solve(M_check, tee=False)

    term = check_results.solver.termination_condition
    print("Recourse consistency check termination:", term)

    if term != TerminationCondition.optimal:
        raise RuntimeError(
            f"Recourse consistency check did not solve optimally. Termination: {term}"
        )

    check_val = float(pyo.value(M_check.CheckObjective))
    diff = check_val - stored_sum_K_p_QEV

    print("Stored sum_K p_s Q_s(x_EV):", stored_sum_K_p_QEV)
    print("Solved K recourse objective:", check_val)
    print("Difference:", diff)

    if abs(diff) > CONSISTENCY_TOL:
        raise RuntimeError(
            "K-system recourse expression does not match stored Q_EV values. "
            "Check objective constants, first-stage leakage, and probability scaling."
        )

    return check_val, str(term)


def build_budget_diagnostics(
    K_budget_body,
    budget_rhs: float,
    K_scenarios: List[str],
    probabilities: Dict[str, float],
    cTx_EV: float,
    recourse_diags: Dict[str, RecourseObjectiveDiagnostics],
):
    """
    Diagnose constants in the K-budget body.

    If the body constant matches sum_p_K * c^T x_EV, that is a signature of
    fixed first-stage cost leakage. After the extraction-before-fixing fix,
    this should not happen.

    If the body constant matches the weighted true objective constants
    extracted before fixing x, that is expected when true objective constants
    are present and INCLUDE_TRUE_OBJECTIVE_CONSTANT=True.
    """
    body_repn = generate_standard_repn(K_budget_body, compute_values=True)
    body_constant = float(pyo.value(body_repn.constant or 0.0))

    sum_p_K = sum(probabilities[s] for s in K_scenarios)
    expected_first_stage_leak = sum_p_K * cTx_EV

    weighted_true_constants = sum(
        probabilities[s] * recourse_diags[s].objective_constant_before_fixing
        for s in K_scenarios
        if recourse_diags[s].included_objective_constant
    )

    effective_lp_rhs = budget_rhs - body_constant

    print("\nK-budget diagnostic")
    print("-------------------")
    print("Printed mathematical budget RHS:", budget_rhs)
    print("K-budget body constant:", body_constant)
    print("LP effective RHS after moving constants:", effective_lp_rhs)
    print("sum_{s in K} p_s:", sum_p_K)
    print("sum_p_K * c^T x_EV:", expected_first_stage_leak)
    print("weighted true objective constants included:", weighted_true_constants)

    if abs(body_constant) > CONSTANT_TOL:
        print("\nNOTE: K-budget body has a nonzero constant.")
        print("      This is OK only if it comes from true objective constants,")
        print("      not from fixed first-stage cost leakage.")

    if abs(body_constant - weighted_true_constants) > CONSISTENCY_TOL:
        print("\nWARNING: K-budget body constant does not match included true objective constants.")

    if abs(body_constant - expected_first_stage_leak) <= CONSISTENCY_TOL:
        print("\nWARNING: K-budget constant numerically matches sum_p_K * c^T x_EV.")
        print("         This is the signature of first-stage cost leakage.")
        print("         Since recourse extraction occurs before fixing x, this should not happen.")

    return {
        "budget_rhs": budget_rhs,
        "body_constant": body_constant,
        "effective_lp_rhs": effective_lp_rhs,
        "sum_p_K": sum_p_K,
        "expected_first_stage_leak": expected_first_stage_leak,
        "weighted_true_constants_included": weighted_true_constants,
    }


# ============================================================
# Main
# ============================================================

def main():
    print("Loading data...")

    parsed = smps_module._ensure_parsed(str(smps_dir))

    stages = parsed["stages"]
    vars_by_stage = parsed["vars_by_stage"]

    root_stage_name = stages[0][0]
    first_stage_var_names = vars_by_stage[root_stage_name]

    x_EV = read_x_solution(x_ev_path)
    probabilities = read_probabilities(scenario_summary_path)
    partition_rows = read_good_bad_partition(good_bad_path)
    costs = read_first_stage_costs(first_stage_cost_path)
    K_scenarios = read_K_scenarios(K_path)

    if not K_scenarios:
        raise RuntimeError(
            "K_scenarios is empty. This K-system/IIS construction expects a nonempty K."
        )

    partition_by_scenario = {
        row["scenario"]: row
        for row in partition_rows
    }

    missing_probs = [s for s in K_scenarios if s not in probabilities]
    missing_partition = [s for s in K_scenarios if s not in partition_by_scenario]

    if missing_probs:
        raise RuntimeError(f"Missing probabilities for K scenarios: {missing_probs}")

    if missing_partition:
        raise RuntimeError(f"Missing partition rows for K scenarios: {missing_partition}")

    cTx_star = costs["x_star"]
    cTx_EV = costs["x_EV"]
    Delta_0 = cTx_EV - cTx_star

    good_scenarios = [
        row["scenario"]
        for row in partition_rows
        if row["classification"] == "good"
    ]

    C_comp = Delta_0 + sum(
        probabilities[s] * partition_by_scenario[s]["Delta"]
        for s in good_scenarios
    )

    sum_K_p_Qstar = sum(
        probabilities[s] * partition_by_scenario[s]["Q_x_star"]
        for s in K_scenarios
    )

    sum_K_p_QEV = sum(
        probabilities[s] * partition_by_scenario[s]["Q_x_EV"]
        for s in K_scenarios
    )

    budget_rhs = sum_K_p_Qstar - C_comp

    sum_K_p_Delta = sum(
        probabilities[s] * partition_by_scenario[s]["Delta"]
        for s in K_scenarios
    )

    infeasibility_margin = sum_K_p_Delta + C_comp

    print("\nK scenarios:", K_scenarios)
    print("|K|:", len(K_scenarios))
    print("c^T x_star:", cTx_star)
    print("c^T x_EV:", cTx_EV)
    print("Delta_0:", Delta_0)
    print("C_comp:", C_comp)
    print("sum_K p_s Q_s(x_star):", sum_K_p_Qstar)
    print("sum_K p_s Q_s(x_EV):", sum_K_p_QEV)
    print("K budget RHS:", budget_rhs)
    print("sum_K p_s Delta_s + C_comp:", infeasibility_margin)

    if infeasibility_margin <= TOL:
        print("\nWARNING:")
        print("The infeasibility margin is very small or nonpositive under the chosen tolerance.")
        print("Gurobi may not report the K system as infeasible.")

    # ============================================================
    # Build K-system Pyomo model
    # ============================================================

    print("\nBuilding K-system Pyomo model...")

    M = pyo.ConcreteModel(name="K_system_for_IIS")

    scenario_blocks = {}
    recourse_exprs = {}
    recourse_diags = {}

    for scen in K_scenarios:
        scen_model = smps_module.scenario_creator(scen, cfg=cfg)

        # CRITICAL: Extract q_s^T y_s BEFORE fixing x_EV.
        # This prevents fixed first-stage terms from becoming constants.
        rec_expr, rec_diag = recourse_objective_expr(
            scen_model,
            first_stage_var_names,
            scenario_name=scen,
            include_true_constant=INCLUDE_TRUE_OBJECTIVE_CONSTANT,
        )

        fixed = fix_first_stage_vars(
            scen_model,
            x_EV,
            first_stage_var_names,
        )

        scen_model.RecourseOnlyObjective = pyo.Expression(expr=rec_expr)

        block_name = f"Scenario_{scen}"
        M.add_component(block_name, scen_model)

        scenario_blocks[scen] = scen_model
        recourse_exprs[scen] = scen_model.RecourseOnlyObjective
        recourse_diags[scen] = rec_diag

        if abs(rec_diag.objective_constant_before_fixing) > CONSTANT_TOL:
            print(
                f"  NOTE {scen}: true objective constant before fixing x = "
                f"{rec_diag.objective_constant_before_fixing}"
            )

        print(
            f"  Added {scen}; fixed {len(fixed)} first-stage vars; "
            f"removed {rec_diag.num_first_stage_terms_removed} first-stage objective terms; "
            f"kept {rec_diag.num_recourse_terms_kept} recourse terms."
        )

    check_no_active_objectives(M, "K-system before adding objectives")

    # Save recourse objective diagnostics
    write_dict_rows(
        recourse_diag_path,
        [
            {
                "scenario": d.scenario,
                "objective_constant_before_fixing": d.objective_constant_before_fixing,
                "included_objective_constant": d.included_objective_constant,
                "num_first_stage_terms_removed": d.num_first_stage_terms_removed,
                "first_stage_objective_coeff_sum_abs": d.first_stage_objective_coeff_sum_abs,
                "num_recourse_terms_kept": d.num_recourse_terms_kept,
                "recourse_coeff_sum_abs": d.recourse_coeff_sum_abs,
            }
            for d in recourse_diags.values()
        ],
        [
            "scenario",
            "objective_constant_before_fixing",
            "included_objective_constant",
            "num_first_stage_terms_removed",
            "first_stage_objective_coeff_sum_abs",
            "num_recourse_terms_kept",
            "recourse_coeff_sum_abs",
        ],
    )

    # Consistency check before adding the infeasibility-causing K-budget.
    recourse_check_value, recourse_check_status = solve_recourse_consistency_check(
        base_model=M,
        K_scenarios=K_scenarios,
        probabilities=probabilities,
        stored_sum_K_p_QEV=sum_K_p_QEV,
    )

    # Budget constraint:
    #   sum_{s in K} p_s q_s^T y_s <= sum_{s in K} p_s Q_s(x_star) - C_comp
    K_budget_body = sum(
        probabilities[s] * recourse_exprs[s]
        for s in K_scenarios
    )

    budget_diag = build_budget_diagnostics(
        K_budget_body=K_budget_body,
        budget_rhs=budget_rhs,
        K_scenarios=K_scenarios,
        probabilities=probabilities,
        cTx_EV=cTx_EV,
        recourse_diags=recourse_diags,
    )

    write_dict_rows(
        diagnostics_path,
        [
            {"name": "K_size", "value": len(K_scenarios)},
            {"name": "cTx_star", "value": cTx_star},
            {"name": "cTx_EV", "value": cTx_EV},
            {"name": "Delta_0", "value": Delta_0},
            {"name": "C_comp", "value": C_comp},
            {"name": "sum_K_p_Qstar", "value": sum_K_p_Qstar},
            {"name": "sum_K_p_QEV", "value": sum_K_p_QEV},
            {"name": "budget_rhs", "value": budget_rhs},
            {"name": "sum_K_p_Delta", "value": sum_K_p_Delta},
            {"name": "infeasibility_margin", "value": infeasibility_margin},
            {"name": "recourse_check_value", "value": recourse_check_value},
            {"name": "recourse_check_difference", "value": recourse_check_value - sum_K_p_QEV},
            {"name": "budget_body_constant", "value": budget_diag["body_constant"]},
            {"name": "budget_effective_lp_rhs", "value": budget_diag["effective_lp_rhs"]},
            {"name": "sum_p_K", "value": budget_diag["sum_p_K"]},
            {"name": "sum_p_K_times_cTx_EV", "value": budget_diag["expected_first_stage_leak"]},
            {"name": "weighted_true_objective_constants_included", "value": budget_diag["weighted_true_constants_included"]},
        ],
        ["name", "value"],
    )

    M.K_budget = pyo.Constraint(expr=K_budget_body <= budget_rhs)

    # Pure feasibility model, but Gurobi wants an objective.
    M.DummyObjective = pyo.Objective(expr=0.0, sense=pyo.minimize)

    # ============================================================
    # Write LP
    # ============================================================

    print("\nWriting K-system LP...")

    M.write(
        str(lp_path),
        io_options={"symbolic_solver_labels": True},
    )

    print("Wrote LP to:", lp_path)

    # ============================================================
    # Gurobi IIS
    # ============================================================

    print("\nReading LP with Gurobi and computing IIS...")

    grb = gp.read(str(lp_path))

    # Helps distinguish infeasible from infeasible-or-unbounded.
    grb.Params.DualReductions = 0

    # Tighten feasibility tolerance because this is an explanation model.
    grb.Params.FeasibilityTol = 1e-9

    grb.optimize()

    if grb.Status != GRB.INFEASIBLE:
        raise RuntimeError(
            f"Gurobi status is {grb.Status}, not INFEASIBLE. "
            "Check budget RHS, Q values, C_comp, and numerical tolerances."
        )

    print("\nModel is infeasible. Computing IIS...")

    grb.computeIIS()
    grb.write(str(iis_ilp_path))

    print("Wrote IIS ILP to:", iis_ilp_path)

    # ============================================================
    # Export IIS constraints
    # ============================================================

    iis_constraint_rows = []

    for c in grb.getConstrs():
        if c.IISConstr:
            kind, scen = classify_iis_constraint_name(c.ConstrName)

            iis_constraint_rows.append({
                "constraint_name": c.ConstrName,
                "kind": kind,
                "scenario": scen,
                "sense": c.Sense,
                "rhs": c.RHS,
                "row_expression": str(grb.getRow(c)),
            })

    write_dict_rows(
        iis_constraints_path,
        iis_constraint_rows,
        [
            "constraint_name",
            "kind",
            "scenario",
            "sense",
            "rhs",
            "row_expression",
        ],
    )

    # ============================================================
    # Export IIS variable bounds
    # ============================================================

    iis_bound_rows = []

    for v in grb.getVars():
        scen = classify_iis_var_name(v.VarName)

        if v.IISLB:
            iis_bound_rows.append({
                "variable_name": v.VarName,
                "scenario": scen,
                "bound_type": "LB",
                "bound_value": v.LB,
            })

        if v.IISUB:
            iis_bound_rows.append({
                "variable_name": v.VarName,
                "scenario": scen,
                "bound_type": "UB",
                "bound_value": v.UB,
            })

    write_dict_rows(
        iis_bounds_path,
        iis_bound_rows,
        [
            "variable_name",
            "scenario",
            "bound_type",
            "bound_value",
        ],
    )

    # ============================================================
    # Summary
    # ============================================================

    num_budget_rows = sum(
        1 for r in iis_constraint_rows
        if r["kind"] == "K_budget"
    )

    scenario_iis_counts = {}

    for r in iis_constraint_rows:
        scen = r["scenario"]
        if scen:
            scenario_iis_counts[scen] = scenario_iis_counts.get(scen, 0) + 1

    bound_iis_counts = {}

    for r in iis_bound_rows:
        scen = r["scenario"]
        if scen:
            bound_iis_counts[scen] = bound_iis_counts.get(scen, 0) + 1

    with open(summary_path, "w") as f:
        f.write("K-system IIS summary\n")
        f.write("====================\n\n")

        f.write(f"K scenarios: {K_scenarios}\n")
        f.write(f"|K|: {len(K_scenarios)}\n\n")

        f.write(f"c^T x_star: {cTx_star}\n")
        f.write(f"c^T x_EV: {cTx_EV}\n")
        f.write(f"Delta_0: {Delta_0}\n")
        f.write(f"C_comp: {C_comp}\n")
        f.write(f"sum_K p_s Q_s(x_star): {sum_K_p_Qstar}\n")
        f.write(f"sum_K p_s Q_s(x_EV): {sum_K_p_QEV}\n")
        f.write(f"K budget RHS: {budget_rhs}\n")
        f.write(f"sum_K p_s Delta_s + C_comp: {infeasibility_margin}\n\n")

        f.write("Recourse consistency check\n")
        f.write("--------------------------\n")
        f.write(f"Termination: {recourse_check_status}\n")
        f.write(f"Stored sum_K p_s Q_s(x_EV): {sum_K_p_QEV}\n")
        f.write(f"Solved K recourse objective: {recourse_check_value}\n")
        f.write(f"Difference: {recourse_check_value - sum_K_p_QEV}\n\n")

        f.write("K-budget diagnostics\n")
        f.write("--------------------\n")
        f.write(f"Budget RHS before LP constant shift: {budget_diag['budget_rhs']}\n")
        f.write(f"K-budget body constant: {budget_diag['body_constant']}\n")
        f.write(f"Effective LP RHS after constant shift: {budget_diag['effective_lp_rhs']}\n")
        f.write(f"sum_p_K: {budget_diag['sum_p_K']}\n")
        f.write(f"sum_p_K * c^T x_EV: {budget_diag['expected_first_stage_leak']}\n")
        f.write(
            "Weighted true objective constants included: "
            f"{budget_diag['weighted_true_constants_included']}\n\n"
        )

        f.write("Recourse objective extraction diagnostics\n")
        f.write("-----------------------------------------\n")
        for scen in K_scenarios:
            d = recourse_diags[scen]
            f.write(
                f"{scen}: constant_before_fixing={d.objective_constant_before_fixing}, "
                f"included_constant={d.included_objective_constant}, "
                f"first_stage_terms_removed={d.num_first_stage_terms_removed}, "
                f"recourse_terms_kept={d.num_recourse_terms_kept}\n"
            )
        f.write("\n")

        f.write("IIS contents\n")
        f.write("------------\n")
        f.write(f"Number of IIS constraints: {len(iis_constraint_rows)}\n")
        f.write(f"Number of IIS variable bounds: {len(iis_bound_rows)}\n")
        f.write(f"K_budget rows in IIS: {num_budget_rows}\n\n")

        f.write("IIS scenario constraint counts:\n")
        for scen, count in sorted(scenario_iis_counts.items(), key=lambda x: scenario_sort_key(x[0])):
            f.write(f"  {scen}: {count}\n")

        f.write("\nIIS scenario bound counts:\n")
        for scen, count in sorted(bound_iis_counts.items(), key=lambda x: scenario_sort_key(x[0])):
            f.write(f"  {scen}: {count}\n")

        f.write("\nOutput files:\n")
        f.write(f"  {lp_path}\n")
        f.write(f"  {iis_ilp_path}\n")
        f.write(f"  {iis_constraints_path}\n")
        f.write(f"  {iis_bounds_path}\n")
        f.write(f"  {diagnostics_path}\n")
        f.write(f"  {recourse_diag_path}\n")

    print("\n============================================================")
    print("IIS construction complete")
    print("============================================================")
    print("IIS constraints:", iis_constraints_path)
    print("IIS bounds     :", iis_bounds_path)
    print("IIS ILP file   :", iis_ilp_path)
    print("Diagnostics    :", diagnostics_path)
    print("Recourse diag  :", recourse_diag_path)
    print("Summary        :", summary_path)
    print("============================================================")


if __name__ == "__main__":
    main()
