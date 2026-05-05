from pathlib import Path
from types import SimpleNamespace
import csv

import pyomo.environ as pyo
from pyomo.repn.standard_repn import generate_standard_repn

import mpisppy.problem_io.smps_module as smps_module


# ============================================================
# Settings
# ============================================================

DATA_FOLDER_NAME = "dcap233_200"
RESULTS_FOLDER_NAME = "results_xstar_xev"


# ============================================================
# Paths
# ============================================================

project_dir = Path(__file__).resolve().parent
smps_dir = project_dir / DATA_FOLDER_NAME
results_dir = project_dir / RESULTS_FOLDER_NAME
out_dir = project_dir / "results_first_stage_cost"
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
    Load x_star.csv or x_EV.csv.

    Returns:
        dict keyed by Pyomo variable name
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing solution file: {csv_path}")

    x = {}

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            orig_name = row["original_mps_name"]
            pyomo_name = row["pyomo_name"]
            value_raw = row["value"]

            if value_raw in ["", "None", None]:
                raise ValueError(
                    f"Missing value for variable {orig_name} in {csv_path}"
                )

            value = float(value_raw)

            x[orig_name] = value
            x[pyomo_name] = value

    return x


def get_single_active_objective(model):
    objs = list(model.component_data_objects(pyo.Objective, active=True))

    if len(objs) != 1:
        raise RuntimeError(f"Expected exactly one active objective, found {len(objs)}")

    return objs[0]


def extract_first_stage_objective_coefficients(model, first_stage_var_names):
    """
    Extract c_i for first-stage variables from the objective expression.

    This gives the coefficients in:

        c^T x + q_s^T y_s

    We keep only terms involving first-stage variables.
    """
    first_stage_pyomo_names = {
        to_pyomo_name(name) for name in first_stage_var_names
    }

    obj = get_single_active_objective(model)
    repn = generate_standard_repn(obj.expr, compute_values=False)

    c = {name: 0.0 for name in first_stage_pyomo_names}

    for var, coef in zip(repn.linear_vars, repn.linear_coefs):
        if var.name in first_stage_pyomo_names:
            c[var.name] = float(coef)

    return c


def compute_cTx(c_coeffs, x_values):
    """
    Compute c^T x.
    """
    total = 0.0
    rows = []

    for var_name, coef in sorted(c_coeffs.items()):
        if var_name not in x_values:
            raise RuntimeError(f"No value found for first-stage variable {var_name}")

        x_val = x_values[var_name]
        contribution = coef * x_val
        total += contribution

        rows.append({
            "pyomo_name": var_name,
            "c_i": coef,
            "x_i": x_val,
            "c_i_x_i": contribution,
        })

    return total, rows


def write_contributions(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["pyomo_name", "c_i", "x_i", "c_i_x_i"],
        )
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# Main
# ============================================================

parsed = smps_module._ensure_parsed(str(smps_dir))

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
print("First-stage variables:", first_stage_var_names)


# ------------------------------------------------------------
# Build one scenario model only to read the objective coefficients.
# Since c is first-stage cost, it should be the same across scenarios.
# ------------------------------------------------------------

scenario_name = scenario_names[0]
print("\nUsing scenario model to extract c coefficients:", scenario_name)

model = smps_module.scenario_creator(scenario_name, cfg=cfg)

c_coeffs = extract_first_stage_objective_coefficients(
    model,
    first_stage_var_names,
)

print("\nFirst-stage objective coefficients c:")
for name, coef in sorted(c_coeffs.items()):
    print(f"  {name}: {coef}")


# ------------------------------------------------------------
# Load x_star and x_EV
# ------------------------------------------------------------

x_star_path = results_dir / "x_star.csv"
x_ev_path = results_dir / "x_EV.csv"

x_star = load_x_solution(x_star_path)
x_ev = load_x_solution(x_ev_path)


# ------------------------------------------------------------
# Compute c^T x_star and c^T x_EV
# ------------------------------------------------------------

first_stage_cost_x_star, rows_star = compute_cTx(c_coeffs, x_star)
first_stage_cost_x_ev, rows_ev = compute_cTx(c_coeffs, x_ev)

print("\n============================================================")
print("First-stage costs")
print("============================================================")
print("c^T x_star:", first_stage_cost_x_star)
print("c^T x_EV  :", first_stage_cost_x_ev)


# ------------------------------------------------------------
# Save detailed outputs
# ------------------------------------------------------------

write_contributions(
    out_dir / "first_stage_cost_x_star_contributions.csv",
    rows_star,
)

write_contributions(
    out_dir / "first_stage_cost_x_EV_contributions.csv",
    rows_ev,
)

with open(out_dir / "first_stage_cost_summary.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["solution", "first_stage_cost"])
    writer.writerow(["x_star", first_stage_cost_x_star])
    writer.writerow(["x_EV", first_stage_cost_x_ev])

with open(out_dir / "c_coefficients.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["pyomo_name", "c_i"])
    for name, coef in sorted(c_coeffs.items()):
        writer.writerow([name, coef])

print("\nWrote outputs to:")
print(out_dir)
print("\nMain file:")
print(out_dir / "first_stage_cost_summary.csv")