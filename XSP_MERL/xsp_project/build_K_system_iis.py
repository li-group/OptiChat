from pathlib import Path
from types import SimpleNamespace
import csv
import re

import pyomo.environ as pyo
from pyomo.repn.standard_repn import generate_standard_repn

import gurobipy as gp
from gurobipy import GRB

import mpisppy.problem_io.smps_module as smps_module


# ============================================================
# Settings
# ============================================================

DATA_FOLDER_NAME = "dcap233_200"

TOL = 1e-8

RESULTS_XSTAR_XEV = "results_xstar_xev"
RESULTS_RECOURSE = "results_recourse_Q"
RESULTS_GOOD_BAD = "results_good_bad_partition"
RESULTS_FIRST_STAGE_COST = "results_first_stage_cost"
RESULTS_K = "results_min_bad_set_K"
OUTPUT_FOLDER = "results_K_system_IIS"


# ============================================================
# Paths
# ============================================================

project_dir = Path(__file__).resolve().parent
smps_dir = project_dir / DATA_FOLDER_NAME

x_ev_path = project_dir / RESULTS_XSTAR_XEV / "x_EV.csv"
scenario_summary_path = project_dir / RESULTS_XSTAR_XEV / "scenario_summary.csv"
good_bad_path = project_dir / RESULTS_GOOD_BAD / "good_bad_partition.csv"
first_stage_cost_path = project_dir / RESULTS_FIRST_STAGE_COST / "first_stage_cost_summary.csv"
K_path = project_dir / RESULTS_K / "K_min_bad_scenarios.csv"

out_dir = project_dir / OUTPUT_FOLDER
out_dir.mkdir(exist_ok=True)

lp_path = out_dir / "K_system.lp"
iis_ilp_path = out_dir / "K_system_iis.ilp"
iis_constraints_path = out_dir / "K_system_iis_constraints.csv"
iis_bounds_path = out_dir / "K_system_iis_bounds.csv"
summary_path = out_dir / "K_system_iis_summary.txt"

cfg = SimpleNamespace(smps_dir=str(smps_dir))


# ============================================================
# Helpers
# ============================================================

def to_pyomo_name(mps_name: str) -> str:
    return mps_name.replace("(", "_").replace(")", "_")


def scenario_sort_key(name):
    match = re.search(r"\d+", name)
    if match:
        return int(match.group())
    return name


def read_x_solution(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")

    x = {}

    with open(path, newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            orig = row["original_mps_name"]
            pyomo = row["pyomo_name"]
            val = float(row["value"])

            x[orig] = val
            x[pyomo] = val

    return x


def read_probabilities(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")

    probs = {}

    with open(path, newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            probs[row["scenario"]] = float(row["probability"])

    return probs


def read_good_bad_partition(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")

    rows = []

    with open(path, newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            rows.append({
                "scenario": row["scenario"],
                "Q_x_EV": float(row["Q_x_EV"]),
                "Q_x_star": float(row["Q_x_star"]),
                "Delta": float(row["Delta"]),
                "classification": row["classification"],
            })

    return rows


def read_first_stage_costs(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")

    costs = {}

    with open(path, newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            costs[row["solution"]] = float(row["first_stage_cost"])

    return costs


def read_K_scenarios(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")

    K = []

    with open(path, newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            K.append(row["scenario"])

    return sorted(K, key=scenario_sort_key)


def get_single_active_objective(model):
    objs = list(model.component_data_objects(pyo.Objective, active=True))

    if len(objs) != 1:
        raise RuntimeError(f"Expected exactly one active objective, found {len(objs)}")

    return objs[0]


def recourse_objective_expr(model, first_stage_var_names):
    """
    Extract q_s^T y_s from the scenario objective by dropping terms
    involving first-stage variables.
    """
    first_stage_pyomo_names = {
        to_pyomo_name(name) for name in first_stage_var_names
    }

    obj = get_single_active_objective(model)
    repn = generate_standard_repn(obj.expr, compute_values=False)

    expr = 0

    if repn.constant is not None:
        expr += repn.constant

    for var, coef in zip(repn.linear_vars, repn.linear_coefs):
        if var.name not in first_stage_pyomo_names:
            expr += coef * var

    obj.deactivate()

    return expr


def fix_first_stage_vars(model, x_values, first_stage_var_names):
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


def classify_iis_constraint_name(name):
    """
    Try to identify whether an IIS row is the K budget row or belongs to a scenario.
    """
    if "K_budget" in name:
        return "K_budget", ""

    match = re.search(r"Scenario_(SCEN\d+)", name)
    if match:
        return "scenario_constraint", match.group(1)

    return "other_constraint", ""


def classify_iis_var_name(name):
    match = re.search(r"Scenario_(SCEN\d+)", name)
    if match:
        return match.group(1)
    return ""


# ============================================================
# Load data
# ============================================================

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

partition_by_scenario = {
    row["scenario"]: row
    for row in partition_rows
}

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

for scen in K_scenarios:
    scen_model = smps_module.scenario_creator(scen, cfg=cfg)

    fixed = fix_first_stage_vars(
        scen_model,
        x_EV,
        first_stage_var_names,
    )

    rec_expr = recourse_objective_expr(
        scen_model,
        first_stage_var_names,
    )

    scen_model.RecourseOnlyObjective = pyo.Expression(expr=rec_expr)

    block_name = f"Scenario_{scen}"
    M.add_component(block_name, scen_model)

    scenario_blocks[scen] = scen_model
    recourse_exprs[scen] = scen_model.RecourseOnlyObjective

    print(f"  Added {scen}; fixed {len(fixed)} first-stage vars.")


# Budget constraint:
#   sum_{s in K} p_s q_s^T y_s <= sum_{s in K} p_s Q_s(x_star) - C_comp

M.K_budget = pyo.Constraint(
    expr=sum(probabilities[s] * recourse_exprs[s] for s in K_scenarios)
    <= budget_rhs
)

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

# Tighten feasibility tolerance a bit because this is an explanation model.
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

with open(iis_constraints_path, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "constraint_name",
            "kind",
            "scenario",
            "sense",
            "rhs",
            "row_expression",
        ],
    )
    writer.writeheader()
    writer.writerows(iis_constraint_rows)


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

with open(iis_bounds_path, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "variable_name",
            "scenario",
            "bound_type",
            "bound_value",
        ],
    )
    writer.writeheader()
    writer.writerows(iis_bound_rows)


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
    f.write(f"K budget RHS: {budget_rhs}\n")
    f.write(f"sum_K p_s Delta_s + C_comp: {infeasibility_margin}\n\n")

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


print("\n============================================================")
print("IIS construction complete")
print("============================================================")
print("IIS constraints:", iis_constraints_path)
print("IIS bounds     :", iis_bounds_path)
print("IIS ILP file   :", iis_ilp_path)
print("Summary        :", summary_path)
print("============================================================")