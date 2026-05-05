from pathlib import Path
from smps.read import StochasticModel

folder = Path("/path/to/your/dcap233_folder")
stem = folder / "dcap233_200"   # no extension

sm = StochasticModel(str(stem))

print("Loaded SMPS model")
print("Scenarios:", len(sm.scenario))
print("First scenario names:", list(sm.scenario.keys())[:5])

det_eq = sm.generate_deterministic_equivalent()

print("Deterministic equivalent:")
print("Variables:", det_eq.NumVars)
print("Constraints:", det_eq.NumConstrs)

det_eq.write(str(folder / "dcap233_200_DE.lp"))