from pathlib import Path
from types import SimpleNamespace
import csv

import pyomo.environ as pyo
import mpisppy.problem_io.smps_module as smps_module


# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

project_dir = Path(__file__).resolve().parent
smps_dir = project_dir / "dcap233_200"
out_dir = project_dir / "dcap233_200_inspection"
out_dir.mkdir(exist_ok=True)

cfg = SimpleNamespace(smps_dir=str(smps_dir))


# ------------------------------------------------------------------
# Parse SMPS files using mpi-sppy internals
# ------------------------------------------------------------------

parsed = smps_module._ensure_parsed(cfg.smps_dir)

print("Parsed SMPS instance")
print("SMPS directory:", parsed["smps_dir"])
print("COR path:", parsed["cor_path"])
print("RHS name:", parsed["rhs_name"])
print("BOUNDS name:", parsed["bounds_name"])

print("\nStages:")
for stage in parsed["stages"]:
    print("  ", stage)

print("\nVariables by stage:")
for stage_name, vars_ in parsed["vars_by_stage"].items():
    print(f"  {stage_name}: {len(vars_)} variables")

print("\nScenarios:")
scenarios = parsed["scenarios"]
print("  Number of scenarios:", len(scenarios))
print("  Probability sum:", sum(s["probability"] for s in scenarios))
print("  First five scenario names:", [s["name"] for s in scenarios[:5]])


# ------------------------------------------------------------------
# Export variable-stage map
# ------------------------------------------------------------------

with open(out_dir / "vars_by_stage.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["stage", "variable_name"])
    for stage_name, vars_ in parsed["vars_by_stage"].items():
        for v in vars_:
            writer.writerow([stage_name, v])


# ------------------------------------------------------------------
# Export scenario summary
# ------------------------------------------------------------------

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


# ------------------------------------------------------------------
# Export all stochastic modifications
# Each row says: in this scenario, this coefficient/RHS/bound changes.
# ------------------------------------------------------------------

with open(out_dir / "scenario_modifications.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["scenario", "column_or_rhs", "row_or_bounds", "value"])

    for s in scenarios:
        for col_name, row_name, value in s["modifications"]:
            writer.writerow([s["name"], col_name, row_name, value])


# ------------------------------------------------------------------
# Build one scenario as a Pyomo model and inspect it
# ------------------------------------------------------------------

first_scenario_name = scenarios[0]["name"]
print("\nBuilding Pyomo model for scenario:", first_scenario_name)

model = smps_module.scenario_creator(first_scenario_name, cfg=cfg)

print("Scenario model is Pyomo object:", type(model))

with open(out_dir / "one_scenario_model_pprint.txt", "w") as f:
    model.pprint(ostream=f)


# ------------------------------------------------------------------
# Export Pyomo variables for one scenario
# ------------------------------------------------------------------

with open(out_dir / "one_scenario_variables.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["name", "domain", "lb", "ub", "fixed", "value"])

    for v in model.component_data_objects(pyo.Var, active=True):
        lb = pyo.value(v.lb, exception=False) if v.lb is not None else ""
        ub = pyo.value(v.ub, exception=False) if v.ub is not None else ""
        val = pyo.value(v, exception=False)
        writer.writerow([v.name, str(v.domain), lb, ub, v.fixed, val])


# ------------------------------------------------------------------
# Export Pyomo constraints for one scenario
# ------------------------------------------------------------------

with open(out_dir / "one_scenario_constraints.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["name", "lower", "body", "upper", "expr"])

    for c in model.component_data_objects(pyo.Constraint, active=True):
        lower = str(c.lower) if c.has_lb() else ""
        upper = str(c.upper) if c.has_ub() else ""
        writer.writerow([c.name, lower, str(c.body), upper, str(c.expr)])


# ------------------------------------------------------------------
# Export objective for one scenario
# ------------------------------------------------------------------

with open(out_dir / "one_scenario_objectives.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["name", "sense", "expr"])

    for obj in model.component_data_objects(pyo.Objective, active=True):
        writer.writerow([obj.name, obj.sense, str(obj.expr)])


print("\nInspection files written to:")
print(out_dir)


