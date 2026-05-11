# partition_good_bad_scenarios.py
from pathlib import Path
import csv
import re


# ============================================================
# Settings
# ============================================================

# DATA_FOLDER_NAME = "dcap233_200"
DATA_FOLDER_NAME = "dummy_demand_tssp"
RESULTS_RECOURSE_FOLDER = "results_recourse_Q"
OUTPUT_FOLDER = "results_good_bad_partition"

TOL = 1e-6


# ============================================================
# Paths
# ============================================================

project_dir = Path(__file__).resolve().parent
recourse_dir = project_dir / f"{DATA_FOLDER_NAME}_results" / RESULTS_RECOURSE_FOLDER
out_dir = project_dir / f"{DATA_FOLDER_NAME}_results" / OUTPUT_FOLDER
out_dir.mkdir(exist_ok=True)

q_star_path = recourse_dir / "Q_all_scenarios_x_star.csv"
q_ev_path = recourse_dir / "Q_all_scenarios_x_EV.csv"


# ============================================================
# Helpers
# ============================================================

def scenario_sort_key(name):
    """
    Sort SCEN1, SCEN2, ..., SCEN200 in numerical order.
    """
    match = re.search(r"\d+", name)
    if match:
        return int(match.group())
    return name


def read_Q_file(path):
    """
    Reads a Q_all_scenarios_*.csv file.

    Expected columns:
        scenario
        termination_condition
        Q_value
        num_fixed_first_stage_vars
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}\n"
            "Run:\n"
            "  python compute_recourse_Q.py --x x_star --all\n"
            "  python compute_recourse_Q.py --x x_EV --all"
        )

    data = {}

    with open(path, newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            scen = row["scenario"]
            term = row["termination_condition"]
            q_raw = row["Q_value"]

            if term.lower() != "optimal":
                raise RuntimeError(
                    f"Scenario {scen} in {path.name} did not solve optimally. "
                    f"Termination condition: {term}"
                )

            if q_raw in ["", "None", None]:
                raise RuntimeError(
                    f"Scenario {scen} in {path.name} has missing Q_value."
                )

            data[scen] = float(q_raw)

    return data


def write_partition_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scenario",
                "Q_x_EV",
                "Q_x_star",
                "Delta",
                "classification",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_scenario_list(path, scenario_names):
    with open(path, "w") as f:
        for scen in scenario_names:
            f.write(f"{scen}\n")


# ============================================================
# Main
# ============================================================

print("Reading recourse values...")

Q_star = read_Q_file(q_star_path)
Q_EV = read_Q_file(q_ev_path)

scenarios_star = set(Q_star.keys())
scenarios_ev = set(Q_EV.keys())

if scenarios_star != scenarios_ev:
    only_star = sorted(scenarios_star - scenarios_ev, key=scenario_sort_key)
    only_ev = sorted(scenarios_ev - scenarios_star, key=scenario_sort_key)

    raise RuntimeError(
        "Scenario sets do not match.\n"
        f"Only in x_star file: {only_star}\n"
        f"Only in x_EV file: {only_ev}"
    )

scenario_names = sorted(scenarios_star, key=scenario_sort_key)

print("Number of scenarios:", len(scenario_names))


rows = []
good_scenarios = []
bad_scenarios = []
near_zero_scenarios = []

for scen in scenario_names:
    q_ev = Q_EV[scen]
    q_star = Q_star[scen]

    delta = q_ev - q_star

    # Mathematically:
    #   good: Delta_s <= 0
    #   bad : Delta_s > 0
    #
    # Numerically, use tolerance.
    if delta > TOL:
        classification = "bad"
        bad_scenarios.append(scen)
    else:
        classification = "good"
        good_scenarios.append(scen)

        if abs(delta) <= TOL:
            near_zero_scenarios.append(scen)

    rows.append({
        "scenario": scen,
        "Q_x_EV": q_ev,
        "Q_x_star": q_star,
        "Delta": delta,
        "classification": classification,
    })


# ============================================================
# Write outputs
# ============================================================

partition_path = out_dir / "good_bad_partition.csv"
good_path = out_dir / "good_scenarios.txt"
bad_path = out_dir / "bad_scenarios.txt"
summary_path = out_dir / "good_bad_summary.txt"

write_partition_csv(partition_path, rows)
write_scenario_list(good_path, good_scenarios)
write_scenario_list(bad_path, bad_scenarios)

with open(summary_path, "w") as f:
    f.write("Good/bad scenario partition\n")
    f.write("===========================\n\n")

    f.write(f"Tolerance used: {TOL}\n")
    f.write(f"Number of scenarios: {len(scenario_names)}\n")
    f.write(f"Number of good scenarios: {len(good_scenarios)}\n")
    f.write(f"Number of bad scenarios: {len(bad_scenarios)}\n")
    f.write(f"Number of near-zero good scenarios: {len(near_zero_scenarios)}\n\n")

    f.write("Good scenarios:\n")
    f.write(", ".join(good_scenarios))
    f.write("\n\n")

    f.write("Bad scenarios:\n")
    f.write(", ".join(bad_scenarios))
    f.write("\n\n")

    f.write("Near-zero scenarios, classified as good by Delta <= 0 rule with tolerance:\n")
    f.write(", ".join(near_zero_scenarios))
    f.write("\n")


# ============================================================
# Print console summary
# ============================================================

print("\n============================================================")
print("Good/bad scenario partition complete")
print("============================================================")
print("Tolerance:", TOL)
print("Good scenarios:", len(good_scenarios))
print("Bad scenarios :", len(bad_scenarios))

print("\nGood scenario names:")
print(good_scenarios)

print("\nBad scenario names:")
print(bad_scenarios)

print("\nWrote:")
print(" ", partition_path)
print(" ", good_path)
print(" ", bad_path)
print(" ", summary_path)