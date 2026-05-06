# solve_xstar_xev.py
from pathlib import Path
from types import SimpleNamespace
import csv
import shutil
import tempfile

import pyomo.environ as pyo
from pyomo.opt import TerminationCondition

from mpisppy.opt.ef import ExtensiveForm
import mpisppy.problem_io.smps_module as smps_module
import mpisppy.problem_io.mps_reader as mps_reader


# ============================================================
# User settings
# ============================================================

SOLVER_NAME = "gurobi"
DATA_FOLDER_NAME = "dcap233_200"


# ============================================================
# Paths
# ============================================================

project_dir = Path(__file__).resolve().parent
smps_dir = project_dir / DATA_FOLDER_NAME
out_dir = project_dir / "results_xstar_xev"
out_dir.mkdir(exist_ok=True)

print("Project directory:", project_dir)
print("SMPS directory:", smps_dir)
print("Output directory:", out_dir)


# ============================================================
# Small helpers
# ============================================================

def to_pyomo_name(mps_name: str) -> str:
    """
    mpi-sppy/mps_reader converts names like x(1) to x_1_.
    This mirrors the conversion used in mps_reader.py.
    """
    return mps_name.replace("(", "_").replace(")", "_")


def copy_cor_to_temp_mps(cor_path: str, tmpdir: str) -> str:
    """
    The python-mip reader expects a .mps extension.
    The .cor file is MPS format, so we copy it to a temporary .mps file.
    """
    mps_path = Path(tmpdir) / "core.mps"
    shutil.copy2(cor_path, mps_path)
    return str(mps_path)


def get_active_objective(model):
    objs = list(model.component_data_objects(pyo.Objective, active=True))
    if len(objs) != 1:
        raise RuntimeError(f"Expected exactly one active objective, found {len(objs)}")
    return objs[0]


def check_optimal(results, label: str):
    term = results.solver.termination_condition
    print(f"{label} termination condition:", term)

    if term != TerminationCondition.optimal:
        raise RuntimeError(f"{label} did not solve to optimality. Termination condition: {term}")


def extract_first_stage_solution_from_pyomo_model(model, first_stage_var_names):
    """
    Extract first-stage variable values from a Pyomo model.
    """
    rows = []

    for orig_name in first_stage_var_names:
        pyomo_name = to_pyomo_name(orig_name)
        var = model.find_component(pyomo_name)

        if var is None:
            value = None
        else:
            value = pyo.value(var, exception=False)

        rows.append({
            "original_mps_name": orig_name,
            "pyomo_name": pyomo_name,
            "value": value,
        })

    return rows


def write_solution_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["original_mps_name", "pyomo_name", "value"],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_raw_dict_csv(path, data_dict):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "value"])
        for k, v in sorted(data_dict.items()):
            writer.writerow([k, v])


# ============================================================
# Parse SMPS data
# ============================================================

cfg = SimpleNamespace(smps_dir=str(smps_dir))

parsed = smps_module._ensure_parsed(cfg.smps_dir)

cor_path = parsed["cor_path"]
rhs_name = parsed["rhs_name"]
bounds_name = parsed["bounds_name"]
bound_types = parsed["bound_types"]
scenarios = parsed["scenarios"]
stages = parsed["stages"]
vars_by_stage = parsed["vars_by_stage"]

scenario_names = [s["name"] for s in scenarios]

root_stage_name = stages[0][0]
first_stage_var_names = vars_by_stage[root_stage_name]

print("\nParsed SMPS instance.")
print("COR path:", cor_path)
print("RHS name:", rhs_name)
print("BOUNDS name:", bounds_name)
print("Number of scenarios:", len(scenario_names))
print("Probability sum:", sum(s["probability"] for s in scenarios))
print("Stages:", stages)
print("Root stage:", root_stage_name)
print("Number of first-stage variables:", len(first_stage_var_names))
print("First-stage variables:", first_stage_var_names)


# Save basic scenario data
with open(out_dir / "scenario_summary.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["scenario", "parent", "stage", "probability", "num_modifications"])
    for s in scenarios:
        writer.writerow([
            s["name"],
            s["parent"],
            s["stage"],
            s["probability"],
            len(s["modifications"]),
        ])


with open(out_dir / "first_stage_variables.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["original_mps_name", "pyomo_name"])
    for name in first_stage_var_names:
        writer.writerow([name, to_pyomo_name(name)])


# ============================================================
# PART A: Solve stochastic extensive form to get x_star
# ============================================================

print("\n============================================================")
print("Solving stochastic extensive form for x_star")
print("============================================================")

ef_options = {
    "solver": SOLVER_NAME,
}

ef_obj = ExtensiveForm(
    options=ef_options,
    all_scenario_names=scenario_names,
    scenario_creator=smps_module.scenario_creator,
    scenario_creator_kwargs={"cfg": cfg},
    model_name="dcap233_200_EF",
    suppress_warnings=False,
)

ef_results = ef_obj.solve_extensive_form(tee=True)
check_optimal(ef_results, "Stochastic EF")

z_sp = ef_obj.get_objective_value()
root_solution = ef_obj.get_root_solution()

print("\nz_SP:", z_sp)
print("Number of root solution entries:", len(root_solution))

# Save raw root solution exactly as mpi-sppy returns it
write_raw_dict_csv(out_dir / "x_star_root_solution_raw.csv", root_solution)

# Map x_star to original first-stage MPS names where possible
x_star_rows = []

for orig_name in first_stage_var_names:
    pyomo_name = to_pyomo_name(orig_name)

    value = None

    if pyomo_name in root_solution:
        value = root_solution[pyomo_name]
    elif orig_name in root_solution:
        value = root_solution[orig_name]
    else:
        # Fallback search
        matches = [
            (k, v)
            for k, v in root_solution.items()
            if k.endswith("." + pyomo_name) or k.endswith(pyomo_name)
        ]

        if len(matches) == 1:
            value = matches[0][1]

    x_star_rows.append({
        "original_mps_name": orig_name,
        "pyomo_name": pyomo_name,
        "value": value,
    })

write_solution_csv(out_dir / "x_star.csv", x_star_rows)

print("Wrote x_star to:", out_dir / "x_star.csv")


# ============================================================
# PART B: Build expected-value model and solve for x_EV
# ============================================================

print("\n============================================================")
print("Building expected-value deterministic model for x_EV")
print("============================================================")


def get_base_value_from_core_mip(mip_model, key):
    """
    Return the nominal/core value for a stochastic entry.

    key = (col_name, row_name)

    Cases:
      1. RHS modification:       col_name == rhs_name
      2. Bound modification:     row_name == bounds_name
      3. Matrix coefficient:     otherwise
    """
    col_name, row_name = key

    # RHS modification
    if col_name == rhs_name:
        constr = mip_model.constr_by_name(row_name)
        if constr is None:
            raise RuntimeError(f"Core constraint not found for RHS key {key}")
        return float(constr.rhs)

    # Bound modification
    if bounds_name is not None and row_name == bounds_name:
        var = mip_model.var_by_name(col_name)
        if var is None:
            raise RuntimeError(f"Core variable not found for bound key {key}")

        btype = bound_types.get(col_name, "UP")

        if btype == "UP":
            return float(var.ub)
        elif btype == "LO":
            return float(var.lb)
        elif btype == "FX":
            return float(var.lb)
        else:
            # Fallback for less common bound types.
            # For your DCAP instance this should probably not be needed.
            if var.ub is not None:
                return float(var.ub)
            return float(var.lb)

    # Matrix coefficient modification
    constr = mip_model.constr_by_name(row_name)
    if constr is None:
        raise RuntimeError(
            f"Core constraint not found for matrix coefficient key {key}. "
            "This could indicate an objective-coefficient modification, "
            "which this simple EV builder does not yet handle."
        )

    var = mip_model.var_by_name(col_name)
    if var is None:
        raise RuntimeError(f"Core variable not found for matrix coefficient key {key}")

    for v, coef in constr.expr.expr.items():
        if v.name == col_name:
            return float(coef)

    # If coefficient is absent in the nominal core, its base value is zero.
    return 0.0


with tempfile.TemporaryDirectory() as tmpdir:
    core_mps_path = copy_cor_to_temp_mps(cor_path, tmpdir)

    # Read one nominal core model to look up base values.
    core_mip_for_base = mps_reader.read_mps_to_mip_model(core_mps_path)

    # Convert each scenario's modifications to a dict:
    #   (col_name, row_name) -> scenario value
    scenario_mod_maps = []
    all_stochastic_keys = set()

    for s in scenarios:
        mod_map = {}
        for col_name, row_name, value in s["modifications"]:
            key = (col_name, row_name)
            mod_map[key] = float(value)
            all_stochastic_keys.add(key)

        scenario_mod_maps.append((s["name"], s["probability"], mod_map))

    print("Number of distinct stochastic entries:", len(all_stochastic_keys))

    # Compute expected replacement value for every stochastic entry.
    expected_modifications = []

    for key in sorted(all_stochastic_keys):
        base_value = get_base_value_from_core_mip(core_mip_for_base, key)

        expected_value = 0.0

        for scen_name, prob, mod_map in scenario_mod_maps:
            scen_value = mod_map.get(key, base_value)
            expected_value += prob * scen_value

        col_name, row_name = key
        expected_modifications.append((col_name, row_name, expected_value))

    # Save expected modifications for inspection
    with open(out_dir / "expected_value_modifications.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["column_or_rhs", "row_or_bounds", "expected_value"])
        for col_name, row_name, expected_value in expected_modifications:
            writer.writerow([col_name, row_name, expected_value])

    print("Wrote expected-value modifications to:",
          out_dir / "expected_value_modifications.csv")

    # Build a fresh core model and apply expected modifications.
    ev_mip = mps_reader.read_mps_to_mip_model(core_mps_path)

    smps_module._apply_modifications_to_mip(
        ev_mip,
        expected_modifications,
        rhs_name,
        bounds_name,
        bound_types,
    )

    ev_model = mps_reader.mip_model_to_pyomo(ev_mip, mps_path=core_mps_path)


print("\nSolving expected-value deterministic model...")

ev_solver = pyo.SolverFactory(SOLVER_NAME)
ev_results = ev_solver.solve(ev_model, tee=True)
check_optimal(ev_results, "Expected-value model")

ev_obj = get_active_objective(ev_model)
z_ev_deterministic = pyo.value(ev_obj)

print("\nExpected-value deterministic objective:", z_ev_deterministic)

x_ev_rows = extract_first_stage_solution_from_pyomo_model(
    ev_model,
    first_stage_var_names,
)

write_solution_csv(out_dir / "x_EV.csv", x_ev_rows)

print("Wrote x_EV to:", out_dir / "x_EV.csv")


# ============================================================
# Final summary
# ============================================================

summary_path = out_dir / "summary.txt"

with open(summary_path, "w") as f:
    f.write("DCAP x_star and x_EV solve summary\n")
    f.write("=================================\n\n")
    f.write(f"SMPS directory: {smps_dir}\n")
    f.write(f"Number of scenarios: {len(scenario_names)}\n")
    f.write(f"Probability sum: {sum(s['probability'] for s in scenarios)}\n")
    f.write(f"Root stage: {root_stage_name}\n")
    f.write(f"Number of first-stage variables: {len(first_stage_var_names)}\n\n")
    f.write(f"z_SP: {z_sp}\n")
    f.write(f"Expected-value deterministic objective: {z_ev_deterministic}\n\n")
    f.write("Files written:\n")
    f.write("  x_star.csv\n")
    f.write("  x_star_root_solution_raw.csv\n")
    f.write("  x_EV.csv\n")
    f.write("  first_stage_variables.csv\n")
    f.write("  scenario_summary.csv\n")
    f.write("  expected_value_modifications.csv\n")

print("\n============================================================")
print("Done.")
print("Summary written to:", summary_path)
print("Main outputs:")
print("  x_star:", out_dir / "x_star.csv")
print("  x_EV  :", out_dir / "x_EV.csv")
print("============================================================")