# select_min_bad_set_K.py
from pathlib import Path
import csv
import re


# ============================================================
# Settings
# ============================================================

TOL = 1e-6

# DATA_FOLDER_NAME = "dcap233_200"
DATA_FOLDER_NAME = "dummy_demand_tssp"

GOOD_BAD_FOLDER = "results_good_bad_partition"
FIRST_STAGE_COST_FOLDER = "results_first_stage_cost"
XSTAR_XEV_FOLDER = "results_xstar_xev"
OUTPUT_FOLDER = "results_min_bad_set_K"


# ============================================================
# Paths
# ============================================================

project_dir = Path(__file__).resolve().parent

good_bad_path = (
    project_dir
    / f"{DATA_FOLDER_NAME}_results"
    / GOOD_BAD_FOLDER
    / "good_bad_partition.csv"
)

first_stage_cost_path = (
    project_dir
    / f"{DATA_FOLDER_NAME}_results"
    / FIRST_STAGE_COST_FOLDER
    / "first_stage_cost_summary.csv"
)

scenario_summary_path = (
    project_dir
    / f"{DATA_FOLDER_NAME}_results"
    / XSTAR_XEV_FOLDER
    / "scenario_summary.csv"
)

out_dir = project_dir / f"{DATA_FOLDER_NAME}_results" / OUTPUT_FOLDER
out_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# Helpers
# ============================================================

def scenario_sort_key(name):
    """
    Sort SCEN1, SCEN2, ..., SCEN200 numerically.
    """
    match = re.search(r"\d+", name)
    if match:
        return int(match.group())
    return name


def read_first_stage_costs(path):
    """
    Reads:
        solution,first_stage_cost
        x_star,...
        x_EV,...
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}\n"
            "Run:\n"
            "  python compute_first_stage_cost.py"
        )

    costs = {}

    with open(path, newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            solution = row["solution"]
            cost = float(row["first_stage_cost"])
            costs[solution] = cost

    if "x_star" not in costs or "x_EV" not in costs:
        raise RuntimeError(
            f"{path} must contain rows for x_star and x_EV."
        )

    return costs


def read_scenario_probabilities(path):
    """
    Reads scenario probabilities from results_xstar_xev/scenario_summary.csv.

    Expected columns:
        scenario,parent,stage,probability,num_modifications
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}\n"
            "Run:\n"
            "  python solve_xstar_xev.py"
        )

    probs = {}

    with open(path, newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            scen = row["scenario"]
            probs[scen] = float(row["probability"])

    return probs


def read_good_bad_partition(path):
    """
    Reads:
        scenario,Q_x_EV,Q_x_star,Delta,classification
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}\n"
            "Run:\n"
            "  python partition_good_bad_scenarios.py"
        )

    rows = []

    with open(path, newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            scen = row["scenario"]
            q_ev = float(row["Q_x_EV"])
            q_star = float(row["Q_x_star"])
            delta = float(row["Delta"])
            classification = row["classification"]

            rows.append({
                "scenario": scen,
                "Q_x_EV": q_ev,
                "Q_x_star": q_star,
                "Delta": delta,
                "classification": classification,
            })

    return rows


def write_rows(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_scenario_list(path, scenario_names):
    with open(path, "w") as f:
        for scen in scenario_names:
            f.write(f"{scen}\n")


# ============================================================
# Main
# ============================================================

print("Reading inputs...")

costs = read_first_stage_costs(first_stage_cost_path)
probabilities = read_scenario_probabilities(scenario_summary_path)
partition_rows = read_good_bad_partition(good_bad_path)

cTx_star = costs["x_star"]
cTx_EV = costs["x_EV"]

Delta_0 = cTx_EV - cTx_star

print("\nc^T x_star:", cTx_star)
print("c^T x_EV  :", cTx_EV)
print("Delta_0   :", Delta_0)


# ============================================================
# Combine scenario Delta with probabilities
# ============================================================

combined_rows = []

for row in partition_rows:
    scen = row["scenario"]

    if scen not in probabilities:
        raise RuntimeError(f"No probability found for scenario {scen}")

    p_s = probabilities[scen]
    delta_s = row["Delta"]
    weighted_delta = p_s * delta_s

    combined_rows.append({
        "scenario": scen,
        "probability": p_s,
        "Q_x_EV": row["Q_x_EV"],
        "Q_x_star": row["Q_x_star"],
        "Delta": delta_s,
        "p_Delta": weighted_delta,
        "classification": row["classification"],
    })


good_rows = [
    r for r in combined_rows
    if r["classification"] == "good"
]

bad_rows = [
    r for r in combined_rows
    if r["classification"] == "bad"
]


# ============================================================
# Compute compensation and VSS decomposition
# ============================================================

sum_good_weighted_delta = sum(r["p_Delta"] for r in good_rows)
sum_bad_weighted_delta = sum(r["p_Delta"] for r in bad_rows)

C_comp = Delta_0 + sum_good_weighted_delta
VSS_decomposed = sum_bad_weighted_delta + C_comp

print("\nNumber of good scenarios:", len(good_rows))
print("Number of bad scenarios :", len(bad_rows))

print("\nSum good p_s Delta_s:", sum_good_weighted_delta)
print("Sum bad  p_s Delta_s:", sum_bad_weighted_delta)
print("C_comp:", C_comp)
print("VSS from decomposition:", VSS_decomposed)


# ============================================================
# Select minimum-cardinality K
# ============================================================

# Condition:
#     sum_{s in K} p_s Delta_s + C_comp > 0
#
# Equivalent:
#     sum_{s in K} p_s Delta_s > -C_comp
#
# Since every bad scenario has positive p_s Delta_s,
# the minimum-cardinality set is obtained by sorting descending.

bad_sorted = sorted(
    bad_rows,
    key=lambda r: r["p_Delta"],
    reverse=True,
)

K_rows = []
cumulative = 0.0

if C_comp > TOL:
    # Mathematically, the empty set already satisfies the condition.
    K_rows = []
    cumulative = 0.0
    condition_satisfied = True
else:
    condition_satisfied = False

    for r in bad_sorted:
        K_rows.append(r)
        cumulative += r["p_Delta"]

        if cumulative + C_comp > TOL:
            condition_satisfied = True
            break


if not condition_satisfied:
    raise RuntimeError(
        "Could not find a bad-scenario subset K satisfying the condition.\n"
        "This usually means VSS <= 0, probabilities/Delta values are inconsistent, "
        "or the numerical tolerance is too strict."
    )


K_names = [r["scenario"] for r in K_rows]

print("\n============================================================")
print("Minimum bad-scenario set K")
print("============================================================")
print("|K|:", len(K_rows))
print("sum_{s in K} p_s Delta_s:", cumulative)
print("sum_{s in K} p_s Delta_s + C_comp:", cumulative + C_comp)
print("K scenarios:", K_names)


# ============================================================
# Write outputs
# ============================================================

combined_path = out_dir / "scenario_regret_with_probabilities.csv"
bad_sorted_path = out_dir / "bad_scenarios_ranked.csv"
K_path = out_dir / "K_min_bad_scenarios.csv"
K_list_path = out_dir / "K_min_bad_scenarios.txt"
summary_path = out_dir / "K_selection_summary.txt"

fieldnames = [
    "scenario",
    "probability",
    "Q_x_EV",
    "Q_x_star",
    "Delta",
    "p_Delta",
    "classification",
]

write_rows(combined_path, combined_rows, fieldnames)
write_rows(bad_sorted_path, bad_sorted, fieldnames)
write_rows(K_path, K_rows, fieldnames)
write_scenario_list(K_list_path, K_names)

with open(summary_path, "w") as f:
    f.write("Minimum bad-scenario set K selection\n")
    f.write("====================================\n\n")

    f.write(f"Tolerance: {TOL}\n\n")

    f.write(f"c^T x_star: {cTx_star}\n")
    f.write(f"c^T x_EV:   {cTx_EV}\n")
    f.write(f"Delta_0:    {Delta_0}\n\n")

    f.write(f"Number of scenarios: {len(combined_rows)}\n")
    f.write(f"Number of good scenarios: {len(good_rows)}\n")
    f.write(f"Number of bad scenarios: {len(bad_rows)}\n\n")

    f.write(f"sum_good p_s Delta_s: {sum_good_weighted_delta}\n")
    f.write(f"sum_bad  p_s Delta_s: {sum_bad_weighted_delta}\n")
    f.write(f"C_comp: {C_comp}\n")
    f.write(f"VSS decomposition value: {VSS_decomposed}\n\n")

    f.write("K selection condition:\n")
    f.write("  sum_{s in K} p_s Delta_s + C_comp > 0\n\n")

    f.write(f"|K|: {len(K_rows)}\n")
    f.write(f"sum_K p_s Delta_s: {cumulative}\n")
    f.write(f"sum_K p_s Delta_s + C_comp: {cumulative + C_comp}\n\n")

    if len(K_rows) == 0:
        f.write(
            "The empty set satisfies the condition because C_comp > 0.\n"
            "If you want a nonempty explanatory set, impose |K| >= 1 separately.\n\n"
        )

    f.write("K scenarios:\n")
    f.write(", ".join(K_names))
    f.write("\n")

print("\nWrote outputs to:")
print(out_dir)

print("\nMain files:")
print(" ", K_path)
print(" ", K_list_path)
print(" ", summary_path)