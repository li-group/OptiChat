from pathlib import Path
from smps.read import StochasticModel
import gurobipy as gp


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

project_dir = Path(__file__).resolve().parent
data_dir = project_dir / "dcap233_200"

stem = data_dir / "dcap233_200"

cor_file = stem.with_suffix(".cor")
tim_file = stem.with_suffix(".tim")
sto_file = stem.with_suffix(".sto")
lp_file = stem.with_suffix(".lp")


# ------------------------------------------------------------
# Check that files exist
# ------------------------------------------------------------

for file in [cor_file, tim_file, sto_file, lp_file]:
    if not file.exists():
        raise FileNotFoundError(f"Missing file: {file}")

print("All files found.")
print("COR:", cor_file)
print("TIM:", tim_file)
print("STO:", sto_file)
print("LP :", lp_file)


# ------------------------------------------------------------
# Read the full stochastic SMPS model
# Uses:
#   dcap233_200.cor
#   dcap233_200.tim
#   dcap233_200.sto
# ------------------------------------------------------------

print("\nReading SMPS stochastic model...")

sm = StochasticModel(str(stem))

print("SMPS model loaded.")

try:
    scenario_names = list(sm.scenario.keys())
    print("Number of scenarios:", len(scenario_names))
    print("First few scenarios:", scenario_names[:5])
except Exception as e:
    print("Could not print scenario information directly.")
    print("Reason:", e)


# ------------------------------------------------------------
# Generate deterministic equivalent
# ------------------------------------------------------------

print("\nGenerating deterministic equivalent...")

det_eq = sm.generate_deterministic_equivalent()

print("Deterministic equivalent created.")
print("Variables:", det_eq.NumVars)
print("Constraints:", det_eq.NumConstrs)


# ------------------------------------------------------------
# Write deterministic equivalent to an LP file
# ------------------------------------------------------------

out_file = data_dir / "dcap233_200_DE.lp"
det_eq.write(str(out_file))

print("\nWrote deterministic equivalent to:")
print(out_file)


# ------------------------------------------------------------
# Optional: solve deterministic equivalent
# ------------------------------------------------------------

solve_now = False

if solve_now:
    print("\nSolving deterministic equivalent...")
    det_eq.optimize()

    if det_eq.status == gp.GRB.OPTIMAL:
        print("Optimal objective:", det_eq.ObjVal)
    else:
        print("Solver status:", det_eq.status)


# ------------------------------------------------------------
# Read only the original LP core model
# This ignores the stochastic scenarios.
# ------------------------------------------------------------

print("\nReading LP core model only...")

core = gp.read(str(lp_file))

print("Core model loaded.")
print("Core variables:", core.NumVars)
print("Core constraints:", core.NumConstrs)

print("\nDone.")