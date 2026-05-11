from pathlib import Path
from types import SimpleNamespace

import pyomo.environ as pyo
import mpisppy.problem_io.smps_module as smps_module
import mpisppy.utils.sputils as sputils


# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
# DATA_FOLDER_NAME = "dcap233_200"
###############################################################

DATA_FOLDER_NAME = "dummy_demand_tssp" 

project_dir = Path(__file__).resolve().parent
smps_dir = project_dir / DATA_FOLDER_NAME
out_dir = project_dir / f"{DATA_FOLDER_NAME}_results" / f"{DATA_FOLDER_NAME}_pyomo_ef"
out_dir.mkdir(parents=True, exist_ok=True)

cfg = SimpleNamespace(smps_dir=str(smps_dir))

# ------------------------------------------------------------------
# Parse scenario names
# ------------------------------------------------------------------

parsed = smps_module._ensure_parsed(cfg.smps_dir)
scenario_names = [s["name"] for s in parsed["scenarios"]]

print("Number of scenarios::::::::", len(scenario_names))
print("First five scenarios:", scenario_names[:5])


# ------------------------------------------------------------------
# Wrapper around mpi-sppy's SMPS scenario creator
# ------------------------------------------------------------------

def scenario_creator(scenario_name, cfg=None):
    return smps_module.scenario_creator(scenario_name, cfg=cfg)


# ------------------------------------------------------------------
# Create Pyomo extensive form
# ------------------------------------------------------------------

print("\nBuilding Pyomo extensive form...")

ef = sputils.create_EF(
    scenario_names,
    scenario_creator,
    scenario_creator_kwargs={"cfg": cfg},
    EF_name=f"{DATA_FOLDER_NAME}_EF_from_pyomo",
    suppress_warnings=False,
)

# Implement TSSP structure
# Convert DAta to JSON
# Have a way to add doc strings to params and variables.

print("EF Pyomo model built.")
print("Model type:", type(ef))


# ------------------------------------------------------------------
# Write the EF to LP so you can inspect it
# ------------------------------------------------------------------
lp_path = out_dir / f"{DATA_FOLDER_NAME}_EF_from_pyomo.lp"

# ef.write(
#     str(lp_path),
#     io_options={"symbolic_solver_labels": True},
# )

print("\nWrote EF LP to:")
print(lp_path)


# ------------------------------------------------------------------
# Print model summary
# ------------------------------------------------------------------

num_vars = sum(1 for _ in ef.component_data_objects(pyo.Var, active=True))
num_cons = sum(1 for _ in ef.component_data_objects(pyo.Constraint, active=True))
num_objs = sum(1 for _ in ef.component_data_objects(pyo.Objective, active=True))

print("\nEF summary:")
print("Variables:", num_vars)
print("Constraints:", num_cons)
print("Objectives:", num_objs)

# ------------------------------------------------------------------
# Solve with Gurobi
# ------------------------------------------------------------------

solve_now = True

if solve_now:
    print("\nSolving EF with Gurobi...")

    solver = pyo.SolverFactory("gurobi")
    results = solver.solve(ef, tee=True)

    print("\nSolver status:", results.solver.status)
    print("Termination condition:", results.solver.termination_condition)

    objs = list(ef.component_data_objects(pyo.Objective, active=True))
    if objs:
        obj_val = pyo.value(objs[0])
        print("Objective value:", obj_val)


# ------------------------------------------------------------------
# Export solution values
# ------------------------------------------------------------------
sol_path = out_dir / "ef_solution_values.csv"

with open(sol_path, "w") as f:
    f.write("variable,value\n")

    for v in ef.component_data_objects(pyo.Var, active=True):
        val = pyo.value(v, exception=False)
        f.write(f"{v.name},{val}\n")

print("\nWrote EF solution values to:")
print(sol_path)


print("Number of scenarios::::::::", len(scenario_names))
print("First five scenarios:", scenario_names[:5])

print("\nEF summary:")
print("Variables:", num_vars)
print("Constraints:", num_cons)
print("Objectives:", num_objs)


###########################
# Run below to look at constraints.
###########################

from pyomo.repn.standard_repn import generate_standard_repn
import pyomo.environ as pyo

# Print constraints involving z variables
for c in ef.component_data_objects(pyo.Constraint, active=True):
    repn = generate_standard_repn(c.body)

    if repn.linear_vars is None:
        continue

    var_names = [v.name for v in repn.linear_vars]

    if any(".z_" in name or "_z_" in name for name in var_names):
        print("\nCONSTRAINT:", c.name)
        print(c.expr)
